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
def _(pl):
    bnb_12h_klines = pl.read_parquet("data/bnbusdt_candles_2008_12h.parquet").filter(pl.col("OpenTime").ge(1704067200000))
    return (bnb_12h_klines,)


@app.cell
def _(
    bnb_12h_klines,
    run_validation_financial_inference,
    train_gbt_from_validation_result,
):
    for T_prime in range(100, 1000, 200):
        bnb_12h_validation = run_validation_financial_inference(
            model_params_path="models/bnb_12h_denoiser",
            fin_time_series=bnb_12h_klines,
            feature_columns=["Close"],
            k_future_points=10,
            N=10,
            T_prime=T_prime,
            corrector_steps=10,
            f_cutoff=0.1,
            inference_batch_size=64,
        )
        for future_points in [0,4,9]:
            bnb_12h_tree_no_denoise = train_gbt_from_validation_result(
                bnb_12h_validation,
                use_denoised_features=False,
                future_time_points_num=future_points
            )
            print(
                f"BNB/USDT 12h initial, T_prime={T_prime}, future points num = {future_points} | " 
                f"accuracy = {bnb_12h_tree_no_denoise.acc} | "
                f"F1 = {bnb_12h_tree_no_denoise.f1} | "
                f"MCC = {bnb_12h_tree_no_denoise.mcc} | "
            )

            bnb_12h_tree_denoise = train_gbt_from_validation_result(
                bnb_12h_validation,
                use_denoised_features=True,
                future_time_points_num=future_points
            )
            print(
                f"BNB/USDT 12h denoised, T_prime={T_prime}, future points num = {future_points} | " 
                f"accuracy = {bnb_12h_tree_denoise.acc} | "
                f"F1 = {bnb_12h_tree_denoise.f1} | "
                f"MCC = {bnb_12h_tree_denoise.mcc} | "
            )

        del bnb_12h_validation
    return


@app.cell
def _(bnb_12h_klines, run_validation_financial_inference):
    tmp0 = run_validation_financial_inference(
            model_params_path="models/bnb_12h_denoiser",
            fin_time_series=bnb_12h_klines,
            feature_columns=["Close"],
            k_future_points=10,
            N=10,
            T_prime=100,
            corrector_steps=10,
            f_cutoff=0.1,
            inference_batch_size=64,
        )
    return (tmp0,)


@app.cell
def _(tmp0):
    tmp0.train_split.original_subsequence - tmp0.train_split.denoised_subsequence
    return


@app.cell
def _(tmp0, train_gbt_from_validation_result):
    tmp1 = train_gbt_from_validation_result(
                tmp0,
                use_denoised_features=False,
                future_time_points_num=1
            )
    return (tmp1,)


@app.cell
def _(tmp1):
    tmp1
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
