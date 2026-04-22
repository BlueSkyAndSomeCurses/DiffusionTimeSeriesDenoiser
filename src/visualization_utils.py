from __future__ import annotations

import random

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import torch
from torch.utils.data import DataLoader

from diffusion_model.csdi import DiffusionModel


def plot_random_denoise_samples(
    model: DiffusionModel,
    dataloader: DataLoader,
    n_samples: int = 3,
    device: torch.device | None = None,
) -> go.Figure:
    if device is None:
        device = next(model.parameters()).device
    model = model.to(device)
    model.eval()

    def _sample_condition(condition_batch: dict | None, idx: int) -> dict[str, torch.Tensor]:
        if not isinstance(condition_batch, dict):
            return {}

        sampled: dict[str, torch.Tensor] = {}
        for key, value in condition_batch.items():
            if not isinstance(value, torch.Tensor):
                continue
            sampled[key] = value[idx]
        return sampled

    def _to_1d_series(x: torch.Tensor) -> torch.Tensor:
        series = x.detach().cpu()
        if series.dim() >= 2 and series.shape[0] == 1:
            series = series.squeeze(0)
        if series.dim() == 2:
            series = series[:, 0]
        return series.reshape(-1)

    sampled_items: list[tuple[torch.Tensor, dict[str, torch.Tensor]]] = []
    with torch.no_grad():
        for batch in dataloader:
            x_batch = batch.get("x0")
            if not isinstance(x_batch, torch.Tensor) or x_batch.shape[0] == 0:
                continue

            condition_batch = batch.get("condition")
            batch_indices = list(range(x_batch.shape[0]))
            random.shuffle(batch_indices)

            for batch_idx in batch_indices:
                sampled_items.append((x_batch[batch_idx], _sample_condition(condition_batch, batch_idx)))
                if len(sampled_items) >= n_samples:
                    break

            if len(sampled_items) >= n_samples:
                break

    if not sampled_items:
        raise ValueError("Dataloader is empty; cannot sample sequences for plotting.")

    n_samples = len(sampled_items)

    figure = make_subplots(
        rows=n_samples,
        cols=1,
        shared_xaxes=True,
        subplot_titles=[f"Sample {i + 1}" for i in range(n_samples)],
    )

    with torch.no_grad():
        for row, (x0_single, sample_condition) in enumerate(sampled_items, start=1):
            x0 = x0_single.unsqueeze(0).to(device)

            time_index = sample_condition.get("time_index")
            if not isinstance(time_index, torch.Tensor):
                time_index = torch.arange(x0_single.shape[0], dtype=x0_single.dtype)

            self_condition = {
                "x_co": x0_single.clone().to(device),
                "m_co": torch.ones_like(x0_single, device=device),
                "time_index": time_index.to(device=device, dtype=x0.dtype),
            }

            t_start = torch.full((1,), model.timesteps - 1, device=device, dtype=torch.long)
            x_noised, _ = model.add_noise(x0, t_start)

            x_denoised = x_noised
            for step in reversed(range(model.timesteps)):
                t_step = torch.full((1,), step, device=device, dtype=torch.long)
                x_denoised = model.denoise_step(
                    x_denoised,
                    t_step,
                    condition=self_condition,
                    guidance_scale=3.0,
                )

            original_series = _to_1d_series(x0)
            noised_series = _to_1d_series(x_noised)
            denoised_series = _to_1d_series(x_denoised)

            x_axis = list(range(original_series.shape[0]))
            show_legend = row == 1

            figure.add_trace(
                go.Scatter(
                    x=x_axis,
                    y=original_series,
                    name="original",
                    line=dict(color="blue"),
                    showlegend=show_legend,
                ),
                row=row,
                col=1,
            )
            figure.add_trace(
                go.Scatter(
                    x=x_axis,
                    y=noised_series,
                    name="noised",
                    line=dict(color="gray", dash="dot"),
                    showlegend=show_legend,
                ),
                row=row,
                col=1,
            )
            figure.add_trace(
                go.Scatter(
                    x=x_axis,
                    y=denoised_series,
                    name="denoised",
                    line=dict(color="red"),
                    showlegend=show_legend,
                ),
                row=row,
                col=1,
            )

    figure.update_layout(
        title_text=f"Original vs. Noised and Denoised Samples ({n_samples})",
        height=320 * n_samples,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    return figure
