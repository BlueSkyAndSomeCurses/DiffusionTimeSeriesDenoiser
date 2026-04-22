from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl
import torch
from torch.utils.data import DataLoader, Dataset


def build_group_series(
	parquet_path: Path | str,
	feature_columns,
	time_column: str | None,
) -> tuple[list[Any], list[Any | None]]:
	currency_klines = pl.read_parquet(parquet_path)
	group_keys = [c for c in ("Symbol", "Interval") if c in currency_klines.columns]

	groups = currency_klines.partition_by(group_keys, maintain_order=True) if group_keys else [currency_klines]
	value_groups = []
	time_groups = []

	for group in groups:
		value_groups.append(group.select(feature_columns).to_numpy())
		time_groups.append(group.get_column(time_column).to_numpy())

	return value_groups, time_groups


def build_window_index(group_arrays: list[Any], window_size: int, step_size: int) -> list[tuple[int, int]]:
	index = []

	for group_idx, values in enumerate(group_arrays):
		series_len = values.shape[0]
		if series_len < window_size:
			continue

		max_start = series_len - window_size

		for start in range(0, max_start + 1, step_size):
			index.append((group_idx, start))

	return index


def split_groups_by_time(
	value_groups: list[Any],
	time_groups: list[Any | None],
	train_ratio: float,
	window_size: int,
) -> tuple[list[Any], list[Any | None], list[Any], list[Any | None]]:
	train_values = []
	train_times = []
	val_values = []
	val_times = []

	for values, times in zip(value_groups, time_groups):
		series_len = values.shape[0]
		if series_len < window_size:
			continue

		raw_split = int(series_len * train_ratio)
		split_idx = max(window_size, raw_split)
		split_idx = min(split_idx, series_len)

		train_values.append(values[:split_idx])
		train_times.append(times[:split_idx] if times is not None else None)

		val_values.append(values[split_idx:])
		val_times.append(times[split_idx:] if times is not None else None)

	return train_values, train_times, val_values, val_times


def fit_normalization_stats(value_groups: list[Any], eps: float) -> tuple[torch.Tensor, torch.Tensor]:
	train_parts = [arr for arr in value_groups if arr.size > 0]

	stacked = torch.cat([torch.as_tensor(arr, dtype=torch.float32) for arr in train_parts], dim=0)
	mean = stacked.mean(dim=0, keepdim=True)
	std = stacked.std(dim=0, keepdim=True, unbiased=False).clamp_min(eps)
	return mean, std


def apply_normalization(value_groups: list[Any], mean: torch.Tensor, std: torch.Tensor) -> list[Any]:
	return [((torch.as_tensor(arr, dtype=torch.float32) - mean) / std).cpu().numpy() for arr in value_groups]


class CandleWindowArrayDataset(Dataset[Any]):
	def __init__(
		self,
		group_arrays: list[Any],
		group_time_arrays: list[Any | None],
		feature_columns,
		window_size: int,
		step_size: int,
		time_column: str | None = "OpenTime",
		include_condition: bool = False,
		time_dtype: torch.dtype = torch.float32,
		normalize_time: bool = True,
		dtype: torch.dtype = torch.float32,
	) -> None:

		self.window_size = window_size
		self.step_size = step_size
		self.feature_columns = list(feature_columns)
		self.time_column = time_column
		self.include_condition = include_condition
		self.time_dtype = time_dtype
		self.normalize_time = normalize_time
		self.dtype = dtype

		self._group_arrays = group_arrays
		self._group_time_arrays = group_time_arrays
		self._index = build_window_index(group_arrays, window_size=window_size, step_size=step_size)

	def __len__(self) -> int:
		return len(self._index)

	def __getitem__(self, idx: int) -> Any:
		group_idx, start = self._index[idx]
		window = self._group_arrays[group_idx][start : start + self.window_size]
		tensor = torch.as_tensor(window, dtype=self.dtype)
		if tensor.shape[-1] == 1:
			tensor = tensor.squeeze(-1)

		time_values = self._group_time_arrays[group_idx]
		if time_values is None:
			time_index = torch.arange(self.window_size, dtype=self.time_dtype)
		else:
			time_window = time_values[start : start + self.window_size]
			time_index = torch.as_tensor(time_window, dtype=self.time_dtype)
			if self.normalize_time:
				time_index = time_index - time_index[0]

		result = {"x0": tensor}
		if self.include_condition:
			result["condition"] = {
				"x_co": tensor.clone(),
				"m_co": torch.ones_like(tensor),
				"time_index": time_index,
			}
		else:
			result["condition"] = {"time_index": time_index}

		return result

def create_candle_train_val_dataloaders(
	parquet_path: Path | str,
	feature_columns,
	window_size: int,
	step_size: int,
	train_ratio: float = 0.8,
	batch_size: int = 64,
	num_workers: int = 0,
	drop_last_train: bool = False,
	drop_last_val: bool = False,
	time_column: str | None = "OpenTime",
	include_condition: bool = False,
	time_dtype: torch.dtype = torch.float32,
	normalize_time: bool = True,
	dtype: torch.dtype = torch.float32,
	normalize_features: bool = True,
	normalization_eps: float = 1e-6,
) -> tuple[DataLoader[Any], DataLoader[Any], dict[str, torch.Tensor] | None]:
	value_groups, time_groups = build_group_series(
		parquet_path,
		feature_columns,
		time_column=time_column,
	)

	train_values, train_times, val_values, val_times = split_groups_by_time(
		value_groups,
		time_groups,
		train_ratio=train_ratio,
		window_size=window_size,
	)

	normalization_stats = None

	if normalize_features:
		mean, std = fit_normalization_stats(train_values, eps=normalization_eps)
		train_values = apply_normalization(train_values, mean, std)
		val_values = apply_normalization(val_values, mean, std)
		normalization_stats = {
			"mean": mean.squeeze(0).to(dtype=dtype),
			"std": std.squeeze(0).to(dtype=dtype),
		}

	train_dataset = CandleWindowArrayDataset(
		group_arrays=train_values,
		group_time_arrays=train_times,
		feature_columns=feature_columns,
		window_size=window_size,
		step_size=step_size,
		time_column=time_column,
		include_condition=include_condition,
		time_dtype=time_dtype,
		normalize_time=normalize_time,
		dtype=dtype,
	)
	val_dataset = CandleWindowArrayDataset(
		group_arrays=val_values,
		group_time_arrays=val_times,
		feature_columns=feature_columns,
		window_size=window_size,
		step_size=step_size,
		time_column=time_column,
		include_condition=include_condition,
		time_dtype=time_dtype,
		normalize_time=normalize_time,
		dtype=dtype,
	)

	train_loader = DataLoader(
		train_dataset,
		batch_size=batch_size,
		shuffle=True,
		num_workers=num_workers,
		drop_last=drop_last_train,
	)
	val_loader = DataLoader(
		val_dataset,
		batch_size=batch_size,
		shuffle=False,
		num_workers=num_workers,
		drop_last=drop_last_val,
	)

	return train_loader, val_loader, normalization_stats

