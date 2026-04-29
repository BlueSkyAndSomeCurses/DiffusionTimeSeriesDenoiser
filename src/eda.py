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
    from testing.diffusion_inference import run_validation_financial_inference
    import torch

    import random
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    return (
        DiffusionModel,
        create_candle_train_val_dataloaders,
        go,
        make_subplots,
        pl,
        plot_random_denoise_samples,
        random,
        run_validation_financial_inference,
        torch,
    )


@app.cell
def _(torch):
    TIMESTAMPS = 1000
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.mps.is_available()
        else "cpu"
    )
    print(device)
    return TIMESTAMPS, device


@app.cell
def _(DiffusionModel, TIMESTAMPS, device, torch):
    model_12h = DiffusionModel(timesteps=TIMESTAMPS).to(device)
    model_12h.load_state_dict(torch.load("models/bnb_12h_denoiser"))
    return (model_12h,)


@app.cell
def _(create_candle_train_val_dataloaders, pl, torch):
    train_dataloader_12h, val_dataloader_12h, _ = create_candle_train_val_dataloaders(
        pl.read_parquet("data/bnbusdt_candles_2025_12h.parquet"),
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
    pl,
    plot_random_denoise_samples,
    torch,
):
    model_1h = DiffusionModel(timesteps=TIMESTAMPS).to(device)
    model_1h.load_state_dict(torch.load("models/bnb_1h_denoiser"))

    train_dataloader_1h, val_dataloader_1h, _ = create_candle_train_val_dataloaders(
        pl.read_parquet("data/bnbusdt_candles_2025_1h.parquet"),
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
    pl,
    plot_random_denoise_samples,
    torch,
):
    model_15m = DiffusionModel(timesteps=TIMESTAMPS).to(device)
    model_15m.load_state_dict(torch.load("models/bnb_15m_denoiser"))

    train_dataloader_15m, val_dataloader_15m, _ = create_candle_train_val_dataloaders(
        pl.read_parquet("data/bnbusdt_candles_2025_15m.parquet"),
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
    pl,
    plot_random_denoise_samples,
    torch,
):
    model_1m = DiffusionModel(timesteps=TIMESTAMPS).to(device)
    model_1m.load_state_dict(torch.load("models/bnb_1m_denoiser"))

    train_dataloader_1m, val_dataloader_1m, _ = create_candle_train_val_dataloaders(
        pl.read_parquet("data/bnbusdt_candles_2025_1m.parquet"),
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
def _(
    go,
    make_subplots,
    pl,
    random,
    run_validation_financial_inference,
    torch,
):
    def plot_random_subsequences_plotly(klines: pl.DataFrame, model_params_path: str, T_prime: int, n: int, plot_name: str):
        inf_res = run_validation_financial_inference(
            model_params_path=model_params_path,
            fin_time_series=klines,
            feature_columns=["Close"],
            k_future_points=10,
            N=10,
            T_prime=T_prime,
            corrector_steps=10,
            f_cutoff=0.1,
            inference_batch_size=64,
            device=torch.device("mps")
        )

        original_subsequence = inf_res.validation_split.original_subsequence
        denoised_subsequence = inf_res.validation_split.denoised_subsequence

        num_samples = len(original_subsequence)
        random_indices = random.sample(range(num_samples), n)

        subplot_titles = [f"{plot_name} T' = {T_prime}: {idx}" for idx in random_indices]

        fig = make_subplots(rows=n, cols=1, subplot_titles=subplot_titles)

        for i, idx in enumerate(random_indices):
            show_legend = True if i == 0 else False

            fig.add_trace(
                go.Scatter(
                    y=original_subsequence[idx],
                    mode='lines',
                    name='Original',
                    line=dict(color='gray', width=2),
                    opacity=0.7,
                    legendgroup='original',
                    showlegend=show_legend
                ),
                row=i + 1, col=1
            )

            fig.add_trace(
                go.Scatter(
                    y=denoised_subsequence[idx],
                    mode='lines',
                    name='Denoised',
                    line=dict(color='blue', width=2, dash='dash'),
                    legendgroup='denoised',
                    showlegend=show_legend
                ),
                row=i + 1, col=1
            )

            fig.update_xaxes(title_text="Time Step", row=i + 1, col=1)
            fig.update_yaxes(title_text="Value", row=i + 1, col=1)

        fig.update_layout(
            height=300 * n,
            width=900,
            title_text=f"Comparison of {n} Random Original vs Denoised Subsequences",
            hovermode="x unified",
            template="plotly_white"
        )

        fig.show()

    return (plot_random_subsequences_plotly,)


@app.cell
def _(pl, plot_random_subsequences_plotly):
    plot_random_subsequences_plotly(
        klines = pl.read_parquet("data/bnbusdt_candles_2008_12h.parquet").filter(pl.col("OpenTime").ge(1711972800000)),
        model_params_path="models/bnb_12h_denoiser",
        n=1,
        T_prime=100,
        plot_name="BNB/USDT 12h klines"
    )
    return


@app.cell
def _(pl, plot_random_subsequences_plotly):
    plot_random_subsequences_plotly(
        klines = pl.read_parquet("data/bnbusdt_candles_2008_12h.parquet").filter(pl.col("OpenTime").ge(1711972800000)),
        model_params_path="models/bnb_12h_denoiser",
        n=1,
        T_prime=350,
        plot_name="BNB/USDT 12h klines"
    )
    return


@app.cell
def _(pl, plot_random_subsequences_plotly):
    plot_random_subsequences_plotly(
        klines = pl.read_parquet("data/bnbusdt_candles_2008_12h.parquet").filter(pl.col("OpenTime").ge(1711972800000)),
        model_params_path="models/bnb_12h_denoiser",
        n=1,
        T_prime=700,
        plot_name="BNB/USDT 12h klines"
    )
    return


@app.cell
def _(pl, plot_random_subsequences_plotly):
    plot_random_subsequences_plotly(
        klines = pl.read_parquet("data/ethusdt_candles_2022_15m.parquet").filter(pl.col("OpenTime").ge(1711972800000)),
        model_params_path="models/eth_15m_denoiser",
        n=1,
        T_prime=100,
        plot_name="ETH/USDT 15m klines"
    )
    return


@app.cell
def _(pl, plot_random_subsequences_plotly):
    plot_random_subsequences_plotly(
        klines = pl.read_parquet("data/ethusdt_candles_2022_15m.parquet").filter(pl.col("OpenTime").ge(1711972800000)),
        model_params_path="models/eth_15m_denoiser",
        n=1,
        T_prime=350,
        plot_name="ETH/USDT 15m klines"
    )
    return


@app.cell
def _(pl, plot_random_subsequences_plotly):
    plot_random_subsequences_plotly(
        klines = pl.read_parquet("data/xrpusdt_candles_2020_1h.parquet").filter(pl.col("OpenTime").ge(1711972800000)),
        model_params_path="models/general_model",
        n=1,
        T_prime=250,
        plot_name="XRP/USDT 1h klines"
    )

    return


@app.cell
def _(pl, plot_random_subsequences_plotly):
    plot_random_subsequences_plotly(
        klines = pl.read_parquet("data/xrpusdt_candles_2020_1h.parquet").filter(pl.col("OpenTime").ge(1711972800000)),
        model_params_path="models/general_model",
        n=1,
        T_prime=500,
        plot_name="XRP/USDT 1h klines"
    )
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
