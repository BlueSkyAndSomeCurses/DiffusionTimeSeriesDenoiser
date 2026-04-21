import marimo

__generated_with = "0.22.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import polars as pl

    return (pl,)


@app.cell
def _(pl):
    bnb_data = pl.read_parquet("data/bnbusdt_candles_2025_12h.parquet")
    return (bnb_data,)


@app.cell
def _(bnb_data):
    bnb_data
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
