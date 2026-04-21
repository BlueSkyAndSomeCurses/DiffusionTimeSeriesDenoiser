from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl
import torch
from torch.utils.data import DataLoader, Dataset


PathLike = str | Path


class CandleWindowDataset(Dataset[Any]):
	def __init__(
		self,
		parquet_path: Path,
		feature_columns,
		window_size: int,
		step_size: int,
		*,
		sort_column: str = "OpenTime",
		time_column: str | None = "OpenTime",
		return_dict: bool = False,
		include_condition: bool = False,
		time_dtype: torch.dtype = torch.float32,
		normalize_time: bool = True,
		dtype: torch.dtype = torch.float32,
	) -> None:

		self.window_size = window_size
		self.step_size = step_size
		self.feature_columns = list(feature_columns)
		self.time_column = time_column
		self.return_dict = return_dict
		self.include_condition = include_condition
		self.time_dtype = time_dtype
		self.normalize_time = normalize_time
		self.dtype = dtype

		currency_klines = pl.read_parquet(parquet_path)
		group_keys = [c for c in ("Symbol", "Interval") if c in currency_klines.columns]

		self._group_arrays = []
		self._group_time_arrays = []
		self._index = []

		groups = currency_klines.partition_by(group_keys, maintain_order=True) if group_keys else [currency_klines]
		for group_idx, group in enumerate(groups):
			if sort_column in group.columns:
				group = group.sort(sort_column)

			values = group.select(self.feature_columns).to_numpy()
			if self.time_column is not None and self.time_column in group.columns:
				time_values = group.get_column(self.time_column).to_numpy()
			else:
				time_values = None

			series_len = values.shape[0]
			if series_len < self.window_size:
				continue

			self._group_arrays.append(values)
			self._group_time_arrays.append(time_values)
			saved_group_idx = len(self._group_arrays) - 1
			max_start = series_len - self.window_size
			for start in range(0, max_start + 1, self.step_size):
				self._index.append((saved_group_idx, start))

	def __len__(self) -> int:
		return len(self._index)

	def __getitem__(self, idx: int) -> Any:
		group_idx, start = self._index[idx]
		window = self._group_arrays[group_idx][start : start + self.window_size]
		tensor = torch.as_tensor(window, dtype=self.dtype)
		if tensor.shape[-1] == 1:
			tensor = tensor.squeeze(-1)

		if not self.return_dict:
			return tensor

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


def create_candle_dataloader(
	parquet_path: PathLike,
	feature_columns,
	window_size: int,
	step_size: int,
	*,
	batch_size: int = 32,
	shuffle: bool = False,
	num_workers: int = 0,
	drop_last: bool = False,
	pin_memory: bool = False,
	sort_column: str = "OpenTime",
	time_column: str | None = "OpenTime",
	return_dict: bool = False,
	include_condition: bool = False,
	time_dtype: torch.dtype = torch.float32,
	normalize_time: bool = True,
	dtype: torch.dtype = torch.float32,
) -> DataLoader[Any]:
	dataset = CandleWindowDataset(
		parquet_path=parquet_path,
		feature_columns=feature_columns,
		window_size=window_size,
		step_size=step_size,
		sort_column=sort_column,
		time_column=time_column,
		return_dict=return_dict,
		include_condition=include_condition,
		time_dtype=time_dtype,
		normalize_time=normalize_time,
		dtype=dtype,
	)
	return DataLoader(
		dataset,
		batch_size=batch_size,
		shuffle=shuffle,
		num_workers=num_workers,
		drop_last=drop_last,
		pin_memory=pin_memory,
	)

