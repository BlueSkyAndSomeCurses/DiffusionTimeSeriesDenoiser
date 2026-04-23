from __future__ import annotations

import random

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import torch
from torch.utils.data import DataLoader

from diffusion_model.csdi import DiffusionModel

def sample_condition_flat(condition_batch: dict | None, idx: int) -> dict[str, torch.Tensor]:
    sampled: dict[str, torch.Tensor] = {}
    for key, value in condition_batch.items():
        if not isinstance(value, torch.Tensor):
            continue
        sampled[key] = value[idx]
    return sampled

def to_1d_series(x: torch.Tensor) -> torch.Tensor:
    series = x.detach().cpu()
    if series.dim() >= 2 and series.shape[0] == 1:
        series = series.squeeze(0)
    if series.dim() == 2:
        series = series[:, 0]
    return series.reshape(-1)

def plot_random_denoise_samples(
    model: DiffusionModel,
    dataloader: DataLoader,
    n_samples: int = 3,
    device: torch.device | None = None,
) -> go.Figure:
    sampled_items = []
    with torch.no_grad():
        for batch in dataloader:
            x_batch = batch.get("x0")
            if not isinstance(x_batch, torch.Tensor) or x_batch.shape[0] == 0:
                continue

            condition_batch = batch.get("condition")
            batch_indices = list(range(x_batch.shape[0]))
            random.shuffle(batch_indices)

            for batch_idx in batch_indices:
                sampled_items.append((x_batch[batch_idx], sample_condition_flat(condition_batch, batch_idx)))
                if len(sampled_items) >= n_samples:
                    break

            if len(sampled_items) >= n_samples:
                break

    n_samples = len(sampled_items)

    figure = make_subplots(
        rows=n_samples,
        cols=1,
        shared_xaxes=True,
        subplot_titles=[f"Sample {i + 1}" for i in range(n_samples)],
    )

    with torch.no_grad():
        for row, (x0_single, sample_condition) in enumerate(sampled_items, start=1):
            x_inference_denoised = model.financial_time_series_inference(
                {
                    "x0" : x0_single,
                    "condition": sample_condition
                },
                device=device,
                N = 10,
                T_prime=100,
                corrector_steps=8,
                f_cutoff=0.1,
            )

            x0 = x0_single.unsqueeze(0).to(device)



            x0 = to_1d_series(x0)
            x_inference_denoised = to_1d_series(x_inference_denoised)

            x_axis = list(range(x0.shape[0]))
            show_legend = row == 1

            figure.add_trace(
                go.Scatter(
                    x=x_axis,
                    y=x0,
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
                    y=x_inference_denoised,
                    name="denoised_smart",
                    line=dict(color="green"),
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

