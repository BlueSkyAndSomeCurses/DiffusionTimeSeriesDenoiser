from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef

from xgboost import XGBClassifier

from testing.diffusion_inference import ValidationInferenceResult


@dataclass
class PerHorizonGBTModel:
	backend: str
	estimators: list[Any | None]
	constant_labels: list[int | None]


@dataclass
class GradientBoostedTreeTrainingResult:
	model: PerHorizonGBTModel
	feature_source: str
	price_feature_index: int
	n_train_samples: int
	n_validation_samples: int
	acc: list[float]
	f1: list[float]
	mcc: list[float]


def drop_null_rows_from_xy(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
	X_arr = np.asarray(X)
	y_arr = np.asarray(y)

	if X_arr.ndim != 2:
		raise ValueError(f"Expected 2D X matrix, got shape={X_arr.shape}")
	if y_arr.ndim == 1:
		y_2d = y_arr.reshape(-1, 1)
	elif y_arr.ndim == 2:
		y_2d = y_arr
	else:
		raise ValueError(f"Expected 1D or 2D y array, got shape={y_arr.shape}")

	if X_arr.shape[0] != y_2d.shape[0]:
		raise ValueError(
			f"X and y row mismatch: X has {X_arr.shape[0]} rows, y has {y_2d.shape[0]} rows"
		)

	if y_2d.dtype.kind in ("f", "c"):
		null_mask = np.isnan(y_2d)
	else:
		is_null = np.vectorize(
			lambda v: v is None or (isinstance(v, (float, np.floating)) and np.isnan(v)),
			otypes=[bool],
		)
		null_mask = is_null(y_2d)

	row_mask = ~np.any(null_mask, axis=1)
	X_filtered = X_arr[row_mask]
	y_filtered = y_2d[row_mask]

	if y_arr.ndim == 1:
		y_filtered = y_filtered.reshape(-1)

	if X_filtered.shape[0] == 0:
		raise ValueError("All rows were dropped because y contains only null values")

	return X_filtered, y_filtered

def build_estimator(
	backend: str,
	random_state: int,
	n_estimators: int,
	learning_rate: float,
	max_depth: int,
	xgb_n_jobs: int,
	tree_method: str,
	subsample: float,
	colsample_bytree: float,
) -> Any:
	if backend == "xgboost":
		if XGBClassifier is None:
			raise ValueError("XGBoost backend requested but xgboost is not available")
		return XGBClassifier(
			n_estimators=n_estimators,
			learning_rate=learning_rate,
			max_depth=max_depth,
			objective="binary:logistic",
			eval_metric="logloss",
			n_jobs=xgb_n_jobs,
			tree_method=tree_method,
			subsample=subsample,
			colsample_bytree=colsample_bytree,
			max_bin=256,
			random_state=random_state,
		)

	return HistGradientBoostingClassifier(
		learning_rate=learning_rate,
		max_depth=max_depth,
		max_iter=n_estimators,
		random_state=random_state,
	)


def _fit_predict_per_horizon(
	X_train: np.ndarray,
	y_train_xgb: np.ndarray,
	X_val: np.ndarray,
	backend: str,
	random_state: int,
	n_estimators: int,
	learning_rate: float,
	max_depth: int,
	xgb_n_jobs: int,
	tree_method: str,
	subsample: float,
	colsample_bytree: float,
	future_steps_num: int
) -> tuple[PerHorizonGBTModel, np.ndarray]:
	estimators = []
	constant_labels = []
	pred_cols = []

	classes = np.unique(y_train_xgb)

	if classes.size < 2:
		label = int(classes[0]) if classes.size == 1 else 0
		estimators.append(None)
		constant_labels.append(label)
		pred_cols.append(np.full((X_val.shape[0],), label, dtype=np.int64))

	estimator = build_estimator(
		backend=backend,
		random_state=random_state,
		n_estimators=n_estimators,
		learning_rate=learning_rate,
		max_depth=max_depth,
		xgb_n_jobs=xgb_n_jobs,
		tree_method=tree_method,
		subsample=subsample,
		colsample_bytree=colsample_bytree,
	)
	estimator.fit(X_train, y_train_xgb)
	pred = estimator.predict(X_val)
	estimators.append(estimator)
	constant_labels.append(None)
	pred_cols.append(np.asarray(pred, dtype=np.int64).reshape(-1))

	pred_matrix = np.column_stack(pred_cols) if pred_cols else np.empty((X_val.shape[0], 0), dtype=np.int64)
	return PerHorizonGBTModel(backend=backend, estimators=estimators, constant_labels=constant_labels), pred_matrix


def train_gbt_from_validation_result(
	validation_result: ValidationInferenceResult,
	use_denoised_features: bool,
	future_time_points_num: int,
	price_feature_index: int = 0,
	random_state: int = 42,
	n_estimators: int = 200,
	learning_rate: float = 0.05,
	max_depth: int = 3,
	xgb_n_jobs: int = 1,
	tree_method: str = "hist",
	subsample: float = 0.8,
	colsample_bytree: float = 0.8,
	backend: str = "hist_gbt",
) -> GradientBoostedTreeTrainingResult:
	X_train = validation_result.train_split.original_subsequence if use_denoised_features is False else validation_result.train_split.denoised_subsequence
	y_train = validation_result.train_split.next_k_points[:, future_time_points_num]

	X_val = validation_result.validation_split.original_subsequence if use_denoised_features is False else validation_result.validation_split.denoised_subsequence
	y_val = validation_result.validation_split.next_k_points[:, future_time_points_num]

	X_train, y_train = drop_null_rows_from_xy(X_train, y_train)
	X_val, y_val = drop_null_rows_from_xy(X_val, y_val)
	
	model, y_pred = _fit_predict_per_horizon(
		X_train,
		y_train,
		X_val,
		backend=backend,
		random_state=random_state,
		n_estimators=n_estimators,
		learning_rate=learning_rate,
		max_depth=max_depth,
		xgb_n_jobs=xgb_n_jobs,
		tree_method=tree_method,
		subsample=subsample,
		colsample_bytree=colsample_bytree,
		future_steps_num=future_time_points_num
	)

	acc = accuracy_score(y_val, y_pred)
	f1 = f1_score(y_val, y_pred)
	mcc = float(matthews_corrcoef(y_val, y_pred))

	return GradientBoostedTreeTrainingResult(
		model=model,
		feature_source="denoised" if use_denoised_features else "original",
		price_feature_index=price_feature_index,
		n_train_samples=X_train.shape[0],
		n_validation_samples=X_val.shape[0],
		acc=acc,
		f1=f1,
		mcc=mcc,
	)
