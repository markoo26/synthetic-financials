"""
learn_ydata_synthetic.py
========================
Explore ydata-synthetic using real market data fetched from Yahoo Finance.

Usage examples
--------------
# Tabular synthesis with CTGAN on S&P 500 daily OHLCV
uv run learn_ydata_synthetic.py --model ctgan --ticker ^GSPC

# Faster tabular run with Gaussian Mixture, custom date range
uv run learn_ydata_synthetic.py --model gmm --ticker AAPL --start 2020-01-01 --end 2024-01-01

# Time-series synthesis with TimeGAN (slow; lower epochs for exploration)
uv run learn_ydata_synthetic.py --model timegan --ticker ^GSPC --epochs 200 --samples 30

# Save synthetic data to CSV
uv run learn_ydata_synthetic.py --model ctgan --ticker ^GSPC --output synth_sp500.csv
"""

import sys

import click
import numpy as np
import pandas as pd
import yfinance as yf


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def banner(text: str) -> None:
    click.echo(f"\n{'─' * 60}")
    click.echo(f"  {text}")
    click.echo(f"{'─' * 60}")


def compare_stats(real: pd.DataFrame, synth: pd.DataFrame) -> None:
    """Print side-by-side descriptive statistics for real vs synthetic data."""
    real_stats  = real.describe().T[["mean", "std", "min", "max"]]
    synth_stats = synth.describe().T[["mean", "std", "min", "max"]]

    combined = real_stats.join(synth_stats, lsuffix="_real", rsuffix="_synth")
    click.echo("\n  Column statistics (real vs synthetic):")
    click.echo(combined.to_string())


# ---------------------------------------------------------------------------
# Data fetching
# ---------------------------------------------------------------------------

def fetch_ohlcv(ticker: str, start: str, end: str) -> pd.DataFrame:
    """
    Download daily OHLCV data from Yahoo Finance and return a clean DataFrame.
    Derived features added:
      - return_pct   : daily close-to-close return (%)
      - hl_spread_pct: high-low spread as % of close (volatility proxy)
    """
    click.echo(f"\n  Downloading {ticker}  [{start} → {end}] ...")
    raw = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)

    if raw.empty:
        click.echo(f"  ERROR: No data returned for '{ticker}'. Check the ticker symbol.", err=True)
        sys.exit(1)

    # Flatten MultiIndex columns that yfinance sometimes produces
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    df = raw[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.columns = ["open", "high", "low", "close", "volume"]
    df["return_pct"]    = df["close"].pct_change() * 100
    df["hl_spread_pct"] = (df["high"] - df["low"]) / df["close"] * 100
    df = df.dropna().reset_index(drop=True)

    click.echo(f"  Downloaded {len(df)} rows, columns: {list(df.columns)}")
    click.echo(f"\n  Real data head:")
    click.echo(df.head().to_string())
    return df


# ---------------------------------------------------------------------------
# Synthesizers
# ---------------------------------------------------------------------------

TABULAR_COLS = ["open", "high", "low", "close", "volume", "return_pct", "hl_spread_pct"]


def run_ctgan(df: pd.DataFrame, epochs: int, samples: int) -> pd.DataFrame:
    from ydata_synthetic.synthesizers import ModelParameters, TrainParameters
    from ydata_synthetic.synthesizers.regular import RegularSynthesizer

    banner("CTGAN – Conditional Tabular GAN")
    click.echo(f"  Treats each row as an independent observation (tabular mode).")
    click.echo(f"  epochs={epochs}  samples to generate={samples}")

    # ModelParameters controls the neural-network architecture and optimizer
    model_args = ModelParameters(batch_size=500, lr=2e-4, betas=(0.5, 0.9))

    # TrainParameters controls the training loop
    train_args = TrainParameters(epochs=epochs)

    synth = RegularSynthesizer(modelname="ctgan", model_parameters=model_args)

    click.echo("\n  Training ...")
    synth.fit(
        data=df[TABULAR_COLS],
        train_arguments=train_args,
        num_cols=TABULAR_COLS,   # all columns are numeric here
        cat_cols=[],
    )

    click.echo(f"\n  Generating {samples} synthetic rows ...")
    synth_df = synth.sample(samples)

    compare_stats(df[TABULAR_COLS], synth_df)
    return synth_df


def run_gmm(df: pd.DataFrame, epochs: int, samples: int) -> pd.DataFrame:
    from ydata_synthetic.synthesizers import ModelParameters, TrainParameters
    from ydata_synthetic.synthesizers.regular import RegularSynthesizer

    banner("Fast/GMM – Gaussian Mixture Model (fast baseline)")
    click.echo(f"  No GPU required. Good for quick exploration and sanity-checks.")
    click.echo(f"  epochs={epochs}  samples to generate={samples}")

    model_args = ModelParameters(batch_size=500, lr=2e-4, betas=(0.5, 0.9))
    train_args = TrainParameters(epochs=epochs)

    # In ydata-synthetic >=1.4 the Gaussian Mixture model was renamed "fast"
    synth = RegularSynthesizer(modelname="fast", model_parameters=model_args)

    click.echo("\n  Training ...")
    synth.fit(
        data=df[TABULAR_COLS],
        train_arguments=train_args,
        num_cols=TABULAR_COLS,
        cat_cols=[],
    )

    click.echo(f"\n  Generating {samples} synthetic rows ...")
    synth_df = synth.sample(samples)

    compare_stats(df[TABULAR_COLS], synth_df)
    return synth_df


def run_timegan(df: pd.DataFrame, epochs: int, samples: int,
                sequence_length: int) -> np.ndarray:
    from ydata_synthetic.synthesizers import ModelParameters, TrainParameters
    from ydata_synthetic.synthesizers.timeseries import TimeSeriesSynthesizer

    banner("TimeGAN – Time-Series GAN")
    click.echo(f"  Treats data as ordered sequences (preserves temporal dependencies).")
    click.echo(f"  epochs={epochs}  sequence_length={sequence_length}  samples={samples}")
    click.echo("  NOTE: 10 000–50 000 epochs needed for production quality.")

    cols = TABULAR_COLS  # TimeGAN requires all-numeric columns

    # Key hyperparameters:
    #   noise_dim   – dimension of random noise fed to the generator
    #   latent_dim  – size of the latent (embedding) space
    #   layers_dim  – width of each LSTM layer
    #   gamma       – weight of the supervised (moment-matching) loss
    model_args = ModelParameters(
        batch_size=128,
        lr=5e-4,
        noise_dim=32,
        layers_dim=128,
        latent_dim=24,
        gamma=1,
    )

    # sequence_length  – number of consecutive time steps per training window
    # number_sequences – how many windows to sample per gradient step
    train_args = TrainParameters(
        epochs=epochs,
        sequence_length=sequence_length,
        number_sequences=6,
    )

    synth = TimeSeriesSynthesizer(modelname="timegan", model_parameters=model_args)

    click.echo("\n  Training ...")
    synth.fit(df[cols], train_args, num_cols=cols)

    click.echo(f"\n  Generating {samples} synthetic sequences ...")
    synth_arr = synth.sample(n_samples=samples)
    # Output shape: (n_samples, sequence_length, n_features)

    click.echo(f"\n  Output shape: {synth_arr.shape}")
    click.echo(f"  Interpretation: {synth_arr.shape[0]} sequences × "
               f"{synth_arr.shape[1]} time steps × "
               f"{synth_arr.shape[2]} features")

    # Preview first sequence as a DataFrame
    sample_df = pd.DataFrame(synth_arr[0], columns=cols)
    click.echo(f"\n  Synthetic sequence #0 (head):")
    click.echo(sample_df.head().to_string())

    # Compare real vs synthetic statistics (flatten all sequences for comparison)
    flat = synth_arr.reshape(-1, synth_arr.shape[2])
    flat_df = pd.DataFrame(flat, columns=cols)
    compare_stats(df[cols], flat_df)

    return synth_arr


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

MODEL_CHOICES = click.Choice(["ctgan", "gmm", "timegan"], case_sensitive=False)

DEFAULT_EPOCHS = {"ctgan": 300, "gmm": 300, "timegan": 200}

# ydata-synthetic >=1.4 renamed "gmm" to "fast" internally
_MODEL_NAME_MAP = {"gmm": "fast", "ctgan": "ctgan", "timegan": "timegan"}


@click.command()
@click.option(
    "--ticker", default="SPY", show_default=True,
    help="Yahoo Finance ticker symbol.  Examples: SPY, AAPL, BTC-USD, ^GSPC",
)
@click.option(
    "--start", default="2015-01-01", show_default=True,
    help="Start date for historical data (YYYY-MM-DD).",
)
@click.option(
    "--end", default="2024-12-31", show_default=True,
    help="End date for historical data (YYYY-MM-DD).",
)
@click.option(
    "--model", default="ctgan", show_default=True, type=MODEL_CHOICES,
    help="Synthesizer to use: ctgan | gmm | timegan",
)
@click.option(
    "--epochs", default=None, type=int,
    help=(
        "Training epochs. "
        f"Defaults: ctgan={DEFAULT_EPOCHS['ctgan']}, "
        f"gmm={DEFAULT_EPOCHS['gmm']}, "
        f"timegan={DEFAULT_EPOCHS['timegan']}."
    ),
)
@click.option(
    "--samples", default=500, show_default=True,
    help="Number of synthetic rows / sequences to generate.",
)
@click.option(
    "--sequence-length", default=24, show_default=True,
    help="[TimeGAN only] Number of time steps per training window.",
)
@click.option(
    "--output", default=None,
    help="Optional path to save synthetic data as CSV (tabular) or NPY (timegan).",
)
def main(ticker, start, end, model, epochs, samples, sequence_length, output):
    """
    Learn ydata-synthetic using real market data from Yahoo Finance.

    Fetches OHLCV data for TICKER, trains the chosen synthesizer, generates
    SAMPLES synthetic observations, and prints comparative statistics.
    """
    click.echo("=" * 60)
    click.echo("  ydata-synthetic  ×  Yahoo Finance")
    click.echo("=" * 60)
    click.echo(f"  ticker={ticker}  model={model}  start={start}  end={end}")

    effective_epochs = epochs if epochs is not None else DEFAULT_EPOCHS[model.lower()]

    # 1. Fetch real data
    df = fetch_ohlcv(ticker, start, end)

    # 2. Train and generate
    model_lower = model.lower()

    if model_lower == "ctgan":
        result = run_ctgan(df, effective_epochs, samples)
    elif model_lower == "gmm":
        result = run_gmm(df, effective_epochs, samples)
    else:
        result = run_timegan(df, effective_epochs, samples, sequence_length)

    # 3. Optionally save output
    if output:
        if model_lower == "timegan":
            np.save(output, result)
            click.echo(f"\n  Saved numpy array → {output}")
        else:
            result.to_csv(output, index=False)
            click.echo(f"\n  Saved CSV → {output}")

    click.echo("\n\nDone.")


if __name__ == "__main__":
    main()
