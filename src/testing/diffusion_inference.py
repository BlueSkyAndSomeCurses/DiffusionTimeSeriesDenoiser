from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import torch

from diffusion_model.csdi import DiffusionModel
from diffusion_model.data_loader import (
	apply_normalization,
	build_group_series,
	split_groups_by_time,
	fit_normalization_stats,
)

@dataclass
class ValidationInferenceGroupResult:
	group_key: dict[str, Any]
	subsequence_start: int
	subsequence_time_index: torch.Tensor
	original_subsequence: torch.Tensor
	denoised_subsequence: torch.Tensor
	future_time_index: torch.Tensor
	next_k_points: torch.Tensor

	@property
	def denoised_with_next_k(self) -> torch.Tensor:
		return torch.cat([self.denoised_subsequence, self.next_k_points], dim=0)


@dataclass
class ValidationInferenceSplitResult:
	original_subsequence: np.ndarray
	denoised_subsequence: np.ndarray
	next_k_points: np.ndarray
	group_keys: list[dict[str, Any]]
	window_size: int
	k_future_points: int

	def to_group_results(self) -> list[ValidationInferenceGroupResult]:
		results = []
		for sample_idx in range(self.group_indices.shape[0]):
			group_idx = int(self.group_indices[sample_idx])
			start = int(self.subsequence_start[sample_idx])
			future_len = int(self.future_lengths[sample_idx])

			time_values = self.split_times[group_idx] if 0 <= group_idx < len(self.split_times) else None
			if time_values is None:
				subsequence_time_index = torch.arange(start, start + self.window_size, dtype=torch.float32)
				future_time_index = torch.arange(
					start + self.window_size,
					start + self.window_size + future_len,
					dtype=torch.float32,
				)
			else:
				subsequence_time_index = torch.from_numpy(time_values[start : start + self.window_size]).to(torch.float32)
				future_time_index = torch.from_numpy(
					time_values[start + self.window_size : start + self.window_size + future_len],
				).to(torch.float32)

			next_width = future_len * self.n_features
			next_k_points = self.next_k_points[sample_idx, :next_width]

			group_key = (
				self.group_keys[group_idx]
				if 0 <= group_idx < len(self.group_keys)
				else {"group_index": group_idx}
			)
			results.append(
				ValidationInferenceGroupResult(
					group_key=group_key,
					subsequence_start=start,
					subsequence_time_index=subsequence_time_index,
					original_subsequence=torch.from_numpy(self.original_subsequence[sample_idx].copy()),
					denoised_subsequence=torch.from_numpy(self.denoised_subsequence[sample_idx].copy()),
					future_time_index=future_time_index,
					next_k_points=torch.from_numpy(next_k_points.copy()),
				)
			)
		return results


@dataclass
class ValidationInferenceResult:
	model_path: Path
	feature_columns: list[str]
	train_split: ValidationInferenceSplitResult
	validation_split: ValidationInferenceSplitResult


class FinancialDiffusionInferenceRunner:
	def __init__(
		self,
		timesteps: int = 1000,
		window_size: int = 60,
		step_size: int = 20,
		inference_batch_size: int = 64,
		train_ratio: float = 0.8,
		time_column: str | None = "OpenTime",
		normalize_time: bool = True,
		normalize_features: bool = True,
		normalization_eps: float = 1e-6,
		dtype: torch.dtype = torch.float32,
		device: torch.device | None = None,
	) -> None:
		self.timesteps = timesteps
		self.window_size = window_size
		self.step_size = step_size
		self.inference_batch_size = max(1, inference_batch_size)
		self.train_ratio = train_ratio
		self.time_column = time_column
		self.normalize_time = normalize_time
		self.normalize_features = normalize_features
		self.normalization_eps = normalization_eps
		self.dtype = dtype
		self.device = (
			device
			if device is not None
			else torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
		)

	def run_validation_inference(
		self,
		model_params_path: str | Path,
		fin_time_series: pl.DataFrame,
		feature_columns: list[str],
		k_future_points: int,
		N: int,
		T_prime: int,
		corrector_steps: int,
		f_cutoff: float,
		eta: float = 0.01,
		s: int = 5,
		guidance_scale: float = 3.0,
		langevin_snr: float = 0.16,
		seq_dim: int = -1,
	) -> ValidationInferenceResult:
		model_path = Path(model_params_path)
		model = self.load_model(model_path)

		group_keys = [c for c in ("Symbol", "Interval") if c in fin_time_series.columns]
		grouped = fin_time_series.partition_by(group_keys, maintain_order=True) if group_keys else [fin_time_series]
		eligible_group_keys = self._eligible_group_keys(grouped)

		value_groups, time_groups = build_group_series(
			fin_time_series,
			feature_columns=feature_columns,
			time_column=self.time_column,
		)

		train_values, train_times, val_values, val_times = split_groups_by_time(
			value_groups,
			time_groups,
			train_ratio=self.train_ratio,
			window_size=self.window_size,

		)

		train_values_raw = [arr.copy() for arr in train_values]
		val_values_raw = [arr.copy() for arr in val_values]	

		if self.normalize_features:
			norm_mean, norm_std = fit_normalization_stats(train_values, eps=self.normalization_eps)
			train_values = apply_normalization(train_values, norm_mean, norm_std)
			val_values = apply_normalization(val_values, norm_mean, norm_std)


		train_results = self.run_split_inference(
			model=model,
			split_values=train_values,
			split_values_raw=train_values_raw,
			split_times=train_times,
			feature_columns=feature_columns,
			k_future_points=k_future_points,
			N=N,
			T_prime=T_prime,
			corrector_steps=corrector_steps,
			f_cutoff=f_cutoff,
			eta=eta,
			s=s,
			guidance_scale=guidance_scale,
			langevin_snr=langevin_snr,
			seq_dim=seq_dim,
			eligible_group_keys=eligible_group_keys,
		)

		val_results = self.run_split_inference(
			model=model,
			split_values=val_values,
			split_values_raw=val_values_raw,
			split_times=val_times,
			feature_columns=feature_columns,
			k_future_points=k_future_points,
			N=N,
			T_prime=T_prime,
			corrector_steps=corrector_steps,
			f_cutoff=f_cutoff,
			eta=eta,
			s=s,
			guidance_scale=guidance_scale,
			langevin_snr=langevin_snr,
			seq_dim=seq_dim,
			eligible_group_keys=eligible_group_keys,
		)

		return ValidationInferenceResult(
			model_path=model_path,
			feature_columns=feature_columns,
			train_split=train_results,
			validation_split=val_results,
		)

	def empty_split_result(
		self,
		window_size: int,
		k_future_points: int,
	) -> ValidationInferenceSplitResult:
		return ValidationInferenceSplitResult(
			original_subsequence=np.empty((0, window_size ), dtype=np.float32),
			denoised_subsequence=np.empty((0, window_size ), dtype=np.float32),
			next_k_points=np.empty((0, k_future_points ), dtype=np.float32),
			group_keys=[],
			window_size=window_size,
			k_future_points=k_future_points,
		)

	def load_model(self, model_params_path: Path) -> DiffusionModel:
		model = DiffusionModel(timesteps=self.timesteps).to(self.device)
		checkpoint = torch.load(model_params_path, map_location=self.device)

		state_dict = checkpoint
		if isinstance(checkpoint, dict):
			if "model_state_dict" in checkpoint:
				state_dict = checkpoint["model_state_dict"]
			elif "state_dict" in checkpoint:
				state_dict = checkpoint["state_dict"]

		model.load_state_dict(state_dict)
		model.eval()
		return model

	def window_time_index(self, time_values: Any | None, start: int, size: int) -> torch.Tensor:
		if time_values is None:
			return torch.arange(size, dtype=self.dtype)

		time_window = torch.as_tensor(time_values[start : start + size], dtype=self.dtype)
		if self.normalize_time:
			time_window = time_window - time_window[0]
		return time_window


	def as_2d(self, tensor: torch.Tensor, n_features: int) -> torch.Tensor:
		if tensor.dim() == 1:
			return tensor.unsqueeze(-1)
		if tensor.dim() == 2:
			if tensor.shape[-1] == n_features:
				return tensor
			if tensor.shape[0] == n_features:
				return tensor.transpose(0, 1)
			return tensor.unsqueeze(-1)
		if tensor.dim() == 3 and tensor.shape[0] == 1:
			return self.as_2d(tensor.squeeze(0), n_features)
		raise ValueError(f"Unsupported tensor shape for conversion to [T, F]: {tuple(tensor.shape)}")

	def run_split_inference(
		self,
		model: DiffusionModel,
		split_values: list[Any],
		split_values_raw: list[Any],
		split_times: list[Any | None],
		feature_columns: list[str],
		k_future_points: int,
		N: int,
		T_prime: int,
		corrector_steps: int,
		f_cutoff: float,
		eta: float,
		s: int,
		guidance_scale: float,
		langevin_snr: float,
		seq_dim: int,
		eligible_group_keys: list[pl.DataFrame],
	) -> ValidationInferenceSplitResult:
		original_rows = []
		denoised_rows = []
		next_k_rows = []

		for idx, (split_group, split_group_raw, split_time_group) in enumerate(zip(split_values, split_values_raw, split_times, strict=False)):
			split_len = int(split_group.shape[0])
			if split_len < self.window_size:
				continue

			starts = list(range(0, split_len - self.window_size + 1, self.step_size))
			if not starts:
				continue

			for batch_offset in range(0, len(starts), self.inference_batch_size):
				batch_starts = starts[batch_offset : batch_offset + self.inference_batch_size]

				window_batch_2d = torch.stack(
					[
						torch.as_tensor(
							split_group[start : start + self.window_size],
							dtype=self.dtype,
							device=self.device,
						)
						for start in batch_starts
					],
					dim=0,
				)

				x0_batch = window_batch_2d[:, :, 0] if window_batch_2d.shape[-1] == 1 else window_batch_2d
				time_index_batch = torch.stack(
					[
						self.window_time_index(
							split_time_group,
							start=start,
							size=self.window_size,
						)
						for start in batch_starts
					],
					dim=0,
				).to(device=self.device, dtype=self.dtype)
				condition = {
					"x_co": x0_batch.clone(),
					"m_co": torch.ones_like(x0_batch),
					"time_index": time_index_batch,
				}

				denoised_batch = model.financial_time_series_inference(
					sample={"x0": x0_batch, "condition": condition},
					device=self.device,
					N=N,
					T_prime=T_prime,
					corrector_steps=corrector_steps,
					f_cutoff=f_cutoff,
					eta=eta,
					s=s,
					guidance_scale=guidance_scale,
					langevin_snr=langevin_snr,
					seq_dim=seq_dim,
				).detach().cpu()

				for batch_idx, start in enumerate(batch_starts):
					denoised_window_2d = self.as_2d(denoised_batch[batch_idx], len(feature_columns))
					original_window_2d = np.asarray(split_group[start : start + self.window_size], dtype=np.float32)
					original_window_raw_2d = np.asarray(split_group_raw[start : start + self.window_size], dtype=np.float32)

					next_start = start + self.window_size
					next_end = min(next_start + k_future_points, split_len)
					future_len = max(0, next_end - next_start)

					next_k_log_return_2d = np.full((k_future_points, len(feature_columns)), np.nan, dtype=np.float32)
					next_future_raw_2d = None

					if future_len > 0:
						next_future_raw_2d = np.asarray(split_group_raw[next_start:next_end], dtype=np.float32)

					if future_len > 0 and next_future_raw_2d is not None:
						base_values = original_window_raw_2d[-1]
						base_matrix = np.broadcast_to(base_values, next_future_raw_2d.shape)
						valid_mask = (base_matrix > 0.0) & (next_future_raw_2d > 0.0)

						future_log_returns = np.full(next_future_raw_2d.shape, np.nan, dtype=np.float32)
						future_log_returns[valid_mask] = np.where(np.log(next_future_raw_2d[valid_mask] / base_matrix[valid_mask]) > 0, 1, -1).astype(np.float32)
						next_k_log_return_2d[:future_len] = future_log_returns

					original_flat = original_window_2d.reshape(-1).astype(np.float32)
					denoised_flat = denoised_window_2d.numpy().reshape(-1).astype(np.float32)
					next_k_flat = next_k_log_return_2d.reshape(-1).astype(np.float32)

					original_rows.append(original_flat)
					denoised_rows.append(denoised_flat)
					next_k_rows.append(next_k_flat)

		group_keys = [self.resolve_group_key(eligible_group_keys, i) for i in range(len(split_values))]
		if not original_rows:
			return self.empty_split_result(
				window_size=self.window_size,
				k_future_points=k_future_points,
			)

		return ValidationInferenceSplitResult(
			original_subsequence=np.stack(original_rows, axis=0).astype(np.float32),
			denoised_subsequence=np.stack(denoised_rows, axis=0).astype(np.float32),
			next_k_points=np.stack(next_k_rows, axis=0).astype(np.float32),
			group_keys=group_keys,
			window_size=self.window_size,
			k_future_points=k_future_points,
		)

	def resolve_group_key(self, grouped_frames: list[pl.DataFrame], idx: int) -> dict[str, Any]:
		if idx >= len(grouped_frames):
			return {"group_index": idx}

		frame = grouped_frames[idx]
		key: dict[str, Any] = {"group_index": idx}
		if "Symbol" in frame.columns and frame.height > 0:
			key["Symbol"] = frame[0, "Symbol"]
		if "Interval" in frame.columns and frame.height > 0:
			key["Interval"] = frame[0, "Interval"]
		return key

	def _eligible_group_keys(self, grouped_frames: list[pl.DataFrame]) -> list[pl.DataFrame]:
		eligible = []
		for frame in grouped_frames:
			if frame.height >= self.window_size:
				eligible.append(frame)
		return eligible


def run_validation_financial_inference(
	model_params_path: str | Path,
	fin_time_series: pl.DataFrame,
	feature_columns: list[str],
	k_future_points: int,
	N: int,
	T_prime: int,
	corrector_steps: int,
	f_cutoff: float,
	eta: float = 0.01,
	s: int = 5,
	guidance_scale: float = 3.0,
	langevin_snr: float = 0.16,
	seq_dim: int = -1,
	timesteps: int = 1000,
	window_size: int = 60,
	step_size: int = 20,
	inference_batch_size: int = 64,
	train_ratio: float = 0.8,
	time_column: str | None = "OpenTime",
	normalize_time: bool = True,
	normalize_features: bool = True,
	normalization_eps: float = 1e-6,
	dtype: torch.dtype = torch.float32,
	device: torch.device | None = None,
) -> ValidationInferenceResult:
	runner = FinancialDiffusionInferenceRunner(
		timesteps=timesteps,
		window_size=window_size,
		step_size=step_size,
		inference_batch_size=inference_batch_size,
		train_ratio=train_ratio,
		time_column=time_column,
		normalize_time=normalize_time,
		normalize_features=normalize_features,
		normalization_eps=normalization_eps,
		dtype=dtype,
		device=device,
	)

	return runner.run_validation_inference(
		model_params_path=model_params_path,
		fin_time_series=fin_time_series,
		feature_columns=feature_columns,
		k_future_points=k_future_points,
		N=N,
		T_prime=T_prime,
		corrector_steps=corrector_steps,
		f_cutoff=f_cutoff,
		eta=eta,
		s=s,
		guidance_scale=guidance_scale,
		langevin_snr=langevin_snr,
		seq_dim=seq_dim,
	)

