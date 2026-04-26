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

    return DiffusionModel, create_candle_train_val_dataloaders, pl, torch


@app.cell
def _():
    TIME_COLUMN = "OpenTime"
    VALUE_COLUMN = "Close"
    return TIME_COLUMN, VALUE_COLUMN


@app.cell
def _(DiffusionModel, create_candle_train_val_dataloaders, pl, torch):
    def train_diffusion_from_parquet(
        fin_time_series: pl.DataFrame,
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
            fin_time_series,
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

        device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu")
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
def _(TIME_COLUMN, VALUE_COLUMN, pl, train_diffusion_from_parquet):
    eth_12h = pl.read_parquet(
        "/Users/vitya/Documents/MMML/DIffusionTImeSeriesDenoiser/data/ethusdt_candles_2008_12h.parquet"
    )


    model_12h_eth, losses_12h_eth = train_diffusion_from_parquet(
        eth_12h,
        value_column=VALUE_COLUMN,
        time_column=TIME_COLUMN,
        window_size=60,
        step_size=20,
        batch_size=64,
        timesteps=1000,
        epochs=200,
        lr=1e-4,
        model_save_path="models/eth_12h_denoiser",
    )
    return


@app.cell
def _(TIME_COLUMN, VALUE_COLUMN, pl, train_diffusion_from_parquet):
    eth_1h = pl.read_parquet("/Users/vitya/Documents/MMML/DIffusionTImeSeriesDenoiser/data/ethusdt_candles_2020_1h.parquet")


    model_1h_eth, losses_1h_eth = train_diffusion_from_parquet(
        eth_1h,
        value_column=VALUE_COLUMN,
        time_column=TIME_COLUMN,
        window_size=60,
        step_size=20,
        batch_size=64,
        timesteps=1000,
        epochs=200,
        lr=1e-4,
        model_save_path="models/eth_1h_denoiser",
    )
    return


@app.cell
def _(TIME_COLUMN, VALUE_COLUMN, pl, train_diffusion_from_parquet):
    eth_15m = pl.read_parquet(
        "/Users/vitya/Documents/MMML/DIffusionTImeSeriesDenoiser/data/ethusdt_candles_2022_15m.parquet"
    )

    model_15m_eth, losses_15m_eth = train_diffusion_from_parquet(
        eth_15m,
        value_column=VALUE_COLUMN,
        time_column=TIME_COLUMN,
        window_size=60,
        step_size=20,
        batch_size=64,
        timesteps=1000,
        epochs=200,
        lr=1e-4,
        model_save_path="models/eth_15m_denoiser",
    )
    return


@app.cell
def _(TIME_COLUMN, VALUE_COLUMN, pl, train_diffusion_from_parquet):
    eth_1m = pl.read_parquet("/Users/vitya/Documents/MMML/DIffusionTImeSeriesDenoiser/data/ethusdt_candles_2025_1m.parquet")

    model_1m_eth, losses_1m_eth = train_diffusion_from_parquet(
        eth_1m,
        value_column=VALUE_COLUMN,
        time_column=TIME_COLUMN,
        window_size=60,
        step_size=20,
        batch_size=64,
        timesteps=1000,
        epochs=200,
        lr=1e-4,
        model_save_path="models/eth_1m_denoiser",
    )
    return


@app.cell
def _(TIME_COLUMN, VALUE_COLUMN, pl, train_diffusion_from_parquet):
    crypto_12h = pl.concat(
        [
            pl.read_parquet(
                "/Users/vitya/Documents/MMML/DIffusionTImeSeriesDenoiser/data/bnbusdt_candles_2008_12h.parquet"
            ),
            pl.read_parquet(
                "/Users/vitya/Documents/MMML/DIffusionTImeSeriesDenoiser/data/ethusdt_candles_2008_12h.parquet"
            ),
            pl.read_parquet(
                "/Users/vitya/Documents/MMML/DIffusionTImeSeriesDenoiser/data/xrpusdt_candles_2008_12h.parquet"
            ),
        ]
    )

    model_12h_general, losses_12h_general = train_diffusion_from_parquet(
        crypto_12h,
        value_column=VALUE_COLUMN,
        time_column=TIME_COLUMN,
        window_size=60,
        step_size=20,
        batch_size=64,
        timesteps=1000,
        epochs=200,
        lr=1e-4,
        model_save_path="models/general_12h",
    )
    return (crypto_12h,)


@app.cell
def _(TIME_COLUMN, VALUE_COLUMN, pl, train_diffusion_from_parquet):
    crypto_1h = pl.concat(
        [
            pl.read_parquet("/Users/vitya/Documents/MMML/DIffusionTImeSeriesDenoiser/data/bnbusdt_candles_2020_1h.parquet"),
            pl.read_parquet("/Users/vitya/Documents/MMML/DIffusionTImeSeriesDenoiser/data/ethusdt_candles_2020_1h.parquet"),
            pl.read_parquet("/Users/vitya/Documents/MMML/DIffusionTImeSeriesDenoiser/data/xrpusdt_candles_2020_1h.parquet"),
        ]
    )

    model_1h_general, losses_1h_general = train_diffusion_from_parquet(
        crypto_1h,
        value_column=VALUE_COLUMN,
        time_column=TIME_COLUMN,
        window_size=60,
        step_size=20,
        batch_size=64,
        timesteps=1000,
        epochs=200,
        lr=1e-4,
        model_save_path="models/general_1h",
    )
    return (crypto_1h,)


@app.cell
def _(TIME_COLUMN, VALUE_COLUMN, pl, train_diffusion_from_parquet):
    crypto_15m = pl.concat(
        [
            pl.read_parquet(
                "/Users/vitya/Documents/MMML/DIffusionTImeSeriesDenoiser/data/bnbusdt_candles_2022_15m.parquet"
            ),
            pl.read_parquet(
                "/Users/vitya/Documents/MMML/DIffusionTImeSeriesDenoiser/data/ethusdt_candles_2022_15m.parquet"
            ),
            pl.read_parquet(
                "/Users/vitya/Documents/MMML/DIffusionTImeSeriesDenoiser/data/xrpusdt_candles_2022_15m.parquet"
            ),
        ]
    )

    model_15m_general, losses_15m_general = train_diffusion_from_parquet(
        crypto_15m,
        value_column=VALUE_COLUMN,
        time_column=TIME_COLUMN,
        window_size=60,
        step_size=20,
        batch_size=64,
        timesteps=1000,
        epochs=200,
        lr=1e-4,
        model_save_path="models/general_15m",
    )
    return (crypto_15m,)


@app.cell
def _(TIME_COLUMN, VALUE_COLUMN, pl, train_diffusion_from_parquet):
    crypto_1m = pl.concat(
        [
            pl.read_parquet("/Users/vitya/Documents/MMML/DIffusionTImeSeriesDenoiser/data/bnbusdt_candles_2025_1m.parquet"),
            pl.read_parquet("/Users/vitya/Documents/MMML/DIffusionTImeSeriesDenoiser/data/ethusdt_candles_2025_1m.parquet"),
            pl.read_parquet("/Users/vitya/Documents/MMML/DIffusionTImeSeriesDenoiser/data/xrpusdt_candles_2025_1m.parquet"),
        ]
    )

    model_1m_general, losses_1m_general = train_diffusion_from_parquet(
        crypto_1m,
        value_column=VALUE_COLUMN,
        time_column=TIME_COLUMN,
        window_size=60,
        step_size=20,
        batch_size=64,
        timesteps=1000,
        epochs=200,
        lr=1e-4,
        model_save_path="models/general_1m",
    )
    return (crypto_1m,)


@app.cell
def _(
    TIME_COLUMN,
    VALUE_COLUMN,
    crypto_12h,
    crypto_15m,
    crypto_1h,
    crypto_1m,
    pl,
    train_diffusion_from_parquet,
):
    all_crypto_data = pl.concat([crypto_1m, crypto_15m, crypto_1h, crypto_12h])


    model_general, losses_general = train_diffusion_from_parquet(
        all_crypto_data,
        value_column=VALUE_COLUMN,
        time_column=TIME_COLUMN,
        window_size=60,
        step_size=20,
        batch_size=64,
        timesteps=1000,
        epochs=200,
        lr=1e-4,
        model_save_path="models/general_model",
    )
    return


if __name__ == "__main__":
    app.run()
