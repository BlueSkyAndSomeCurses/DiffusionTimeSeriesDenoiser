import marimo

__generated_with = "0.23.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import polars as pl
    from testing.diffusion_inference import run_validation_financial_inference
    from testing.gbt_training import train_gbt_from_validation_result
    import torch

    return (
        pl,
        run_validation_financial_inference,
        torch,
        train_gbt_from_validation_result,
    )


@app.cell
def _(torch):
    device = torch.device("mps")
    return


@app.cell
def _(
    pl,
    run_validation_financial_inference,
    train_gbt_from_validation_result,
):
    def run_experiment(model_params_path: str, klines: pl.DataFrame) -> None:

        symbol = klines["Symbol"][0]
        diff = klines.select(
            pl.col("CloseTimeUtc").str.to_datetime(("%Y-%m-%d %H:%M:%S.%3f %Z")).diff().alias("CloseTimeDateTime")
        )["CloseTimeDateTime"].drop_nulls()[0]

        print(symbol, diff)
        for T_prime in [100, 200, 300, 400, 500, 750]:
            bnb_12h_validation = run_validation_financial_inference(
                model_params_path=model_params_path,
                fin_time_series=klines,
                feature_columns=["Close"],
                k_future_points=10,
                N=10,
                T_prime=T_prime,
                corrector_steps=10,
                f_cutoff=0.1,
                inference_batch_size=64,
            )
            for future_points in [0, 4, 9]:
                bnb_12h_tree_no_denoise = train_gbt_from_validation_result(
                    bnb_12h_validation, use_denoised_features=False, future_time_points_num=future_points
                )
                print(
                    f"{symbol} {diff} initial, T_prime={T_prime}, future points num = {future_points} | "
                    f"accuracy = {bnb_12h_tree_no_denoise.acc:.3f} | "
                    f"F1 = {bnb_12h_tree_no_denoise.f1:.3f} | "
                    f"MCC = {bnb_12h_tree_no_denoise.mcc:.3f} | "
                )

                bnb_12h_tree_denoise = train_gbt_from_validation_result(
                    bnb_12h_validation, use_denoised_features=True, future_time_points_num=future_points
                )
                print(
                    f"{symbol} {diff} denoised, T_prime={T_prime}, future points num = {future_points} | "
                    f"accuracy = {bnb_12h_tree_denoise.acc:.3f} | "
                    f"F1 = {bnb_12h_tree_denoise.f1:.3f} | "
                    f"MCC = {bnb_12h_tree_denoise.mcc:.3f} | "
                )

            del bnb_12h_validation

    return (run_experiment,)


@app.cell
def _(pl):
    bnb_12h_klines = pl.read_parquet("data/bnbusdt_candles_2008_12h.parquet").filter(pl.col("OpenTime").ge(1711972800000))
    bnb_1h_klines = pl.read_parquet("data/bnbusdt_candles_2020_1h.parquet").filter(pl.col("OpenTime").ge(1711972800000))
    bnb_15m_klines = pl.read_parquet("data/bnbusdt_candles_2024_15m.parquet").filter(pl.col("OpenTime").ge(1711972800000))
    bnb_1m_klines = pl.read_parquet("data/bnbusdt_candles_2025_1m.parquet")
    return bnb_12h_klines, bnb_15m_klines, bnb_1h_klines


@app.cell
def _(bnb_12h_klines, run_experiment):
    print("BNB 12h exclusive")
    run_experiment(klines=bnb_12h_klines, model_params_path="models/bnb_12h_denoiser")
    return


@app.cell
def _(bnb_1h_klines, pl, run_experiment):
    print("BNB 1h exclusive")
    run_experiment(
        klines=bnb_1h_klines.filter(pl.col("OpenTime").ge(1711972800000)), model_params_path="models/bnb_1h_denoiser"
    )
    return


@app.cell
def _(bnb_15m_klines, run_experiment):
    print("BNB 15m exclusive")
    run_experiment(klines=bnb_15m_klines, model_params_path="models/bnb_15m_denoiser")
    return


@app.cell
def _(pl):
    eth_12h_klines = pl.read_parquet("data/ethusdt_candles_2008_12h.parquet").filter(pl.col("OpenTime").ge(1711972800000))
    eth_1h_klines = pl.read_parquet("data/ethusdt_candles_2020_1h.parquet").filter(pl.col("OpenTime").ge(1711972800000))
    eth_15m_klines = pl.read_parquet("data/ethusdt_candles_2022_15m.parquet").filter(pl.col("OpenTime").ge(1711972800000))
    return eth_12h_klines, eth_15m_klines, eth_1h_klines


@app.cell
def _(eth_12h_klines, run_experiment):
    run_experiment(klines=eth_12h_klines, model_params_path="models/eth_12h_denoiser")
    return


@app.cell
def _(eth_1h_klines, run_experiment):
    run_experiment(klines=eth_1h_klines, model_params_path="models/eth_1h_denoiser")
    return


@app.cell
def _(eth_15m_klines, run_experiment):
    run_experiment(klines=eth_15m_klines, model_params_path="models/eth_15m_denoiser")
    return


@app.cell
def _(pl):
    xrp_12h_klines = pl.read_parquet("data/xrpusdt_candles_2008_12h.parquet").filter(pl.col("OpenTime").ge(1711972800000))
    xrp_1h_klines = pl.read_parquet("data/xrpusdt_candles_2020_1h.parquet").filter(pl.col("OpenTime").ge(1711972800000))
    xrp_15m_klines = pl.read_parquet("data/xrpusdt_candles_2022_15m.parquet").filter(pl.col("OpenTime").ge(1711972800000))
    return xrp_12h_klines, xrp_15m_klines, xrp_1h_klines


@app.cell
def _(run_experiment, xrp_12h_klines):
    run_experiment(klines=xrp_12h_klines, model_params_path="models/xrp_12h_denoiser")
    return


@app.cell
def _(run_experiment, xrp_1h_klines):
    run_experiment(klines=xrp_1h_klines, model_params_path="models/xrp_1h_denoiser")
    return


@app.cell
def _(run_experiment, xrp_15m_klines):
    run_experiment(klines=xrp_15m_klines, model_params_path="models/xrp_15m_denoiser")
    return


@app.cell
def _(bnb_12h_klines, eth_12h_klines, run_experiment, xrp_12h_klines):
    run_experiment(klines=bnb_12h_klines, model_params_path="models/general_12h")
    run_experiment(klines=eth_12h_klines, model_params_path="models/general_12h")
    run_experiment(klines=xrp_12h_klines, model_params_path="models/general_12h")
    return


@app.cell
def _(bnb_1h_klines, eth_1h_klines, run_experiment, xrp_1h_klines):
    run_experiment(klines=bnb_1h_klines, model_params_path="models/general_1h")
    run_experiment(klines=eth_1h_klines, model_params_path="models/general_1h")
    run_experiment(klines=xrp_1h_klines, model_params_path="models/general_1h")
    return


@app.cell
def _(bnb_12h_klines, eth_12h_klines, run_experiment, xrp_12h_klines):
    run_experiment(klines=bnb_12h_klines, model_params_path="models/general_model")
    run_experiment(klines=eth_12h_klines, model_params_path="models/general_model")
    run_experiment(klines=xrp_12h_klines, model_params_path="models/general_model")
    return


@app.cell
def _(bnb_1h_klines, eth_1h_klines, run_experiment, xrp_1h_klines):
    run_experiment(klines=bnb_1h_klines, model_params_path="models/general_model")
    run_experiment(klines=eth_1h_klines, model_params_path="models/general_model")
    run_experiment(klines=xrp_1h_klines, model_params_path="models/general_model")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
