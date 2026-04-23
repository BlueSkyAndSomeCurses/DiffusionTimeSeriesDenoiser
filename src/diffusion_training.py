import marimo

__generated_with = "0.23.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import sys
    from pathlib import Path

    import polars as pl
    import torch


    from diffusion_model.csdi import DiffusionModel
    from diffusion_model.data_loader import create_candle_train_val_dataloaders

    return DiffusionModel, Path, create_candle_train_val_dataloaders, torch


@app.cell
def _():
    TIME_COLUMN = "OpenTime"
    VALUE_COLUMN = "Close"
    return TIME_COLUMN, VALUE_COLUMN


@app.cell
def _(DiffusionModel, Path, create_candle_train_val_dataloaders, torch):
    def train_diffusion_from_parquet(
        parquet_path: str | Path,
        model_save_path: str,
        value_column: str = "Close",
        time_column: str = "OpenTime",
        window_size: int = 60,
        step_size: int = 20,
        batch_size: int = 64,
        timesteps: int = 1000,
        epochs: int = 2,
        lr: float = 1e-4,
        num_workers: int = 0,
    ) -> tuple[DiffusionModel, list[float]]:
        train_dataloader, val_dataloader, _ = create_candle_train_val_dataloaders(
            parquet_path=parquet_path,
            feature_columns=[value_column],
            window_size=window_size,
            step_size=step_size,
            batch_size=batch_size,
            num_workers=num_workers,
            train_ratio=0.8,
            include_condition=True,
            time_column=time_column,
            normalize_time=True,
            normalize_features=True,
            dtype=torch.float32,
        )

        device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.mps.is_available()
            else "cpu"
        )
        print(device)
        model = DiffusionModel(timesteps=timesteps).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        losses = model.train_ddpm(
            train_dataloader,
            val_dataloader=val_dataloader,
            model_save_path=model_save_path,
            epochs=epochs,
            optimizer=optimizer,
            device=device,
        )
        return model, losses


    return (train_diffusion_from_parquet,)


@app.cell
def _(TIME_COLUMN, VALUE_COLUMN, train_diffusion_from_parquet):
    bnb_12h_path = "/Users/vitya/Documents/MMML/DIffusionTImeSeriesDenoiser/data/bnbusdt_candles_2025_12h.parquet"

    model_12h, losses_12h = train_diffusion_from_parquet(
        bnb_12h_path,
        value_column=VALUE_COLUMN,
        time_column=TIME_COLUMN,
        window_size=60,
        step_size=20,
        batch_size=64,
        timesteps=1000,
        epochs=200,
        lr=1e-4,
        model_save_path = "models/bnb_12h_denoiser"
    )
    return


@app.cell
def _(TIME_COLUMN, VALUE_COLUMN, train_diffusion_from_parquet):
    bnb_1h_path = "/Users/vitya/Documents/MMML/DIffusionTImeSeriesDenoiser/data/bnbusdt_candles_2025_1h.parquet"

    model_1h, losses_1h = train_diffusion_from_parquet(
        bnb_1h_path,
        value_column=VALUE_COLUMN,
        time_column=TIME_COLUMN,
        window_size=60,
        step_size=20,
        batch_size=64,
        timesteps=1000,
        epochs=200,
        lr=1e-4,
        model_save_path = "models/bnb_1h_denoiser"
    )
    return


@app.cell
def _(TIME_COLUMN, VALUE_COLUMN, train_diffusion_from_parquet):
    bnb_15m_path = "/Users/vitya/Documents/MMML/DIffusionTImeSeriesDenoiser/data/bnbusdt_candles_2025_15m.parquet"

    model_15m, losses_15m = train_diffusion_from_parquet(
        bnb_15m_path,
        value_column=VALUE_COLUMN,
        time_column=TIME_COLUMN,
        window_size=60,
        step_size=20,
        batch_size=64,
        timesteps=1000,
        epochs=200,
        lr=1e-4,
        model_save_path = "models/bnb_15m_denoiser"
    )
    return


@app.cell
def _(TIME_COLUMN, VALUE_COLUMN, train_diffusion_from_parquet):
    bnb_1m_path = "/Users/vitya/Documents/MMML/DIffusionTImeSeriesDenoiser/data/bnbusdt_candles_2025_1m.parquet"

    model_1m, losses_1m = train_diffusion_from_parquet(
        bnb_1m_path,
        value_column=VALUE_COLUMN,
        time_column=TIME_COLUMN,
        window_size=60,
        step_size=20,
        batch_size=64,
        timesteps=1000,
        epochs=200,
        lr=1e-4,
        model_save_path = "models/bnb_1m_denoiser"
    )
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
