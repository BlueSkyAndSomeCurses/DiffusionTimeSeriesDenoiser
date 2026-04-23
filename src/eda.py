import marimo

__generated_with = "0.23.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import polars as pl

    from diffusion_model.csdi import DiffusionModel
    from visualization_utils import plot_random_denoise_samples
    from diffusion_model.data_loader import create_candle_train_val_dataloaders
    import torch

    return (
        DiffusionModel,
        create_candle_train_val_dataloaders,
        plot_random_denoise_samples,
        torch,
    )


@app.cell
def _(torch):
    TIMESTAMPS = 200
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu")
    print(device)
    return TIMESTAMPS, device


@app.cell
def _(DiffusionModel, TIMESTAMPS, device, torch):
    model_12h = DiffusionModel(timesteps=TIMESTAMPS).to(device)
    model_12h.load_state_dict(torch.load("models/bnb_12h_denoiser"))
    return (model_12h,)


@app.cell
def _(create_candle_train_val_dataloaders, torch):
    train_dataloader_12h, val_dataloader_12h, _ = create_candle_train_val_dataloaders(
        parquet_path="data/bnbusdt_candles_2025_12h.parquet",
        feature_columns=["Close"],
        window_size=60,
        step_size=20,
        batch_size=64,
        train_ratio=0.8,
        include_condition=True,
        time_column="OpenTime",
        normalize_time=True,
        normalize_features=True,
        dtype=torch.float32,
    )
    return (val_dataloader_12h,)


@app.cell
def _(device, model_12h, plot_random_denoise_samples, val_dataloader_12h):
    plot_random_denoise_samples(model_12h, val_dataloader_12h, device=device)
    return


@app.cell
def _(
    DiffusionModel,
    TIMESTAMPS,
    create_candle_train_val_dataloaders,
    device,
    plot_random_denoise_samples,
    torch,
):
    model_1h = DiffusionModel(timesteps=TIMESTAMPS).to(device)
    model_1h.load_state_dict(torch.load("models/bnb_1h_denoiser"))

    train_dataloader_1h, val_dataloader_1h, _ = create_candle_train_val_dataloaders(
        parquet_path="data/bnbusdt_candles_2025_1h.parquet",
        feature_columns=["Close"],
        window_size=60,
        step_size=20,
        batch_size=64,
        train_ratio=0.8,
        include_condition=True,
        time_column="OpenTime",
        normalize_time=True,
        normalize_features=True,
        dtype=torch.float32,
    )

    plot_random_denoise_samples(model_1h, val_dataloader_1h, device=device)
    return


@app.cell
def _(
    DiffusionModel,
    TIMESTAMPS,
    create_candle_train_val_dataloaders,
    device,
    plot_random_denoise_samples,
    torch,
):
    model_15m = DiffusionModel(timesteps=TIMESTAMPS).to(device)
    model_15m.load_state_dict(torch.load("models/bnb_15m_denoiser"))

    train_dataloader_15m, val_dataloader_15m, _ = create_candle_train_val_dataloaders(
        parquet_path="data/bnbusdt_candles_2025_15m.parquet",
        feature_columns=["Close"],
        window_size=60,
        step_size=20,
        batch_size=64,
        train_ratio=0.8,
        include_condition=True,
        time_column="OpenTime",
        normalize_time=True,
        normalize_features=True,
        dtype=torch.float32,
    )

    plot_random_denoise_samples(model_15m, val_dataloader_15m, device=device)
    return


@app.cell
def _(
    DiffusionModel,
    TIMESTAMPS,
    create_candle_train_val_dataloaders,
    device,
    plot_random_denoise_samples,
    torch,
):
    model_1m = DiffusionModel(timesteps=TIMESTAMPS).to(device)
    model_1m.load_state_dict(torch.load("models/bnb_1m_denoiser"))

    train_dataloader_1m, val_dataloader_1m, _ = create_candle_train_val_dataloaders(
        parquet_path="data/bnbusdt_candles_2025_1m.parquet",
        feature_columns=["Close"],
        window_size=60,
        step_size=20,
        batch_size=64,
        train_ratio=0.8,
        include_condition=True,
        time_column="OpenTime",
        normalize_time=True,
        normalize_features=True,
        dtype=torch.float32,
    )

    plot_random_denoise_samples(model_1m, val_dataloader_1m, device=device)
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
