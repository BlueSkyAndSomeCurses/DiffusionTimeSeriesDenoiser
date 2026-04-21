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
    from diffusion_model.data_loader import create_candle_dataloader

    return DiffusionModel, Path, create_candle_dataloader, torch


@app.cell
def _():
    PARQUET_PATH = "/Users/vitya/Documents/MMML/DIffusionTImeSeriesDenoiser/data/bnbusdt_candles_2025_1h.parquet"
    TIME_COLUMN = "OpenTime"
    VALUE_COLUMN = "Close"

    return PARQUET_PATH, TIME_COLUMN, VALUE_COLUMN


@app.cell
def _(DiffusionModel, Path, create_candle_dataloader, torch):
    def train_diffusion_from_parquet(
        parquet_path: str | Path,
        *,
        value_column: str = "Close",
        time_column: str = "OpenTime",
        window_size: int = 60,
        step_size: int = 20,
        batch_size: int = 64,
        timesteps: int = 200,
        epochs: int = 2,
        lr: float = 1e-4,
        num_workers: int = 0,
    ) -> tuple[DiffusionModel, list[float]]:
        dataloader = create_candle_dataloader(
            parquet_path=parquet_path,
            feature_columns=[value_column],
            window_size=window_size,
            step_size=step_size,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            return_dict=True,
            include_condition=True,
            time_column=time_column,
            normalize_time=True,
            dtype=torch.float32,
        )

        device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.mps.is_available()
            else "cpu"
        )
        model = DiffusionModel(timesteps=timesteps).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        losses = model.train_ddpm(
            dataloader,
            epochs=epochs,
            optimizer=optimizer,
            device=device,
        )
        return model, losses


    return (train_diffusion_from_parquet,)


@app.cell
def _(PARQUET_PATH, TIME_COLUMN, VALUE_COLUMN, train_diffusion_from_parquet):
    model, losses = train_diffusion_from_parquet(
        PARQUET_PATH,
        value_column=VALUE_COLUMN,
        time_column=TIME_COLUMN,
        window_size=60,
        step_size=20,
        batch_size=64,
        timesteps=200,
        epochs=50,
        lr=1e-4,
    )

    {"final_loss": losses[-1], "all_losses": losses}
    return


if __name__ == "__main__":
    app.run()
