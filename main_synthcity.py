
import click
import pandas as pd

from utils import (
    banner, compare_stats, fetch_ohlcv,
    store_artifacts, plot_training_losses,
)

from synthcity.plugins import Plugins
from synthcity.plugins.core.dataloader import GenericDataLoader, TimeSeriesDataLoader

# Tabular columns for DDPM (row-independent)
TABULAR_COLS = ["open", "high", "low", "close", "volume", "return_pct", "hl_spread_pct"]

# Time-series columns for TimeVAE / FFlows (6 features, hl_spread_pct excluded — derivable)
TS_COLS = ["open", "high", "low", "close", "volume", "return_pct"]


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _build_ts_loader(df: pd.DataFrame, cols: list[str],
                     sequence_length: int) -> TimeSeriesDataLoader:
    """
    Slice df into overlapping windows of length sequence_length and wrap
    the result in a TimeSeriesDataLoader.

    Each window becomes one training sequence.  The next-step return_pct is
    used as the outcome so the model has a meaningful target to condition on.
    """
    temporal_data:     list[pd.DataFrame] = []
    observation_times: list[list[int]]    = []
    outcomes:          list[dict]         = []

    steps = list(range(sequence_length))

    for i in range(len(df) - sequence_length):
        window = df[cols].iloc[i : i + sequence_length].reset_index(drop=True)
        temporal_data.append(window)
        observation_times.append(steps)
        outcomes.append({"next_return": float(df["return_pct"].iloc[i + sequence_length])})

    outcome_df = pd.DataFrame(outcomes)

    return TimeSeriesDataLoader(
        temporal_data=temporal_data,
        observation_times=observation_times,
        outcome=outcome_df,
    )


def _flatten_ts_result(result_loader: TimeSeriesDataLoader,
                       cols: list[str]) -> pd.DataFrame:
    """
    Extract one representative row per generated sequence (the first timestep).
    Returns a flat DataFrame with columns == cols.
    """
    _, temporal, _, _ = result_loader.unpack()
    return pd.DataFrame(
        [seq.iloc[0].values for seq in temporal],
        columns=cols,
    )


# ---------------------------------------------------------------------------
# Model runners
# ---------------------------------------------------------------------------

def run_ddpm(df: pd.DataFrame, epochs: int, samples: int) -> pd.DataFrame:

    banner("DDPM – Denoising Diffusion Probabilistic Model (tabular)")
    click.echo("  Row-independent tabular synthesis via diffusion process.")
    click.echo(f"  epochs={epochs}  samples={samples}")

    loader = GenericDataLoader(df[TABULAR_COLS])
    plugin = Plugins().get("ddpm", n_iter=epochs, batch_size=512)

    click.echo("\n  Training ...")
    plugin.fit(loader)

    # ── Loss logging ──────────────────────────────────────────────────────
    loss_history = getattr(plugin, "loss_history", None)
    if loss_history is not None and len(loss_history) > 0:
        loss_df = pd.DataFrame({"step": range(len(loss_history)),
                                "loss": loss_history})
        loss_csv = "outputs/ddpm_training_log.csv"
        loss_df.to_csv(loss_csv, index=False)
        click.echo(f"  Training log → {loss_csv}  ({len(loss_df)} steps recorded)")
        plot_training_losses(loss_df, "ddpm", loss_cols=["loss"])
    else:
        click.echo("  (no loss history exposed by plugin)")

    click.echo(f"\n  Generating {samples} synthetic rows ...")
    synth_df = plugin.generate(count=samples).dataframe()

    store_artifacts(df, synth_df, "ddpm", TABULAR_COLS)
    compare_stats(df[TABULAR_COLS], synth_df)
    return synth_df


def run_timevae(df: pd.DataFrame, epochs: int, samples: int,
                sequence_length: int) -> pd.DataFrame:

    banner("TimeVAE – Variational AutoEncoder for Time Series")
    click.echo("  Learns temporal dynamics via a VAE over sliding windows.")
    click.echo(f"  epochs={epochs}  sequence_length={sequence_length}  samples={samples}")

    loader = _build_ts_loader(df, TS_COLS, sequence_length)
    click.echo(f"  Training sequences: {len(loader)}")

    plugin = Plugins().get("timevae", n_iter=epochs, batch_size=128)

    click.echo("\n  Training ...")
    plugin.fit(loader)

    click.echo(f"\n  Generating {samples} synthetic sequences ...")
    result_loader = plugin.generate(count=samples)
    flat_df = _flatten_ts_result(result_loader, TS_COLS)

    click.echo(f"  Generated {len(flat_df)} sequences × {len(TS_COLS)} features")
    click.echo(f"\n  Synthetic data head:")
    click.echo(flat_df.head().to_string())

    store_artifacts(df, flat_df, "timevae", TS_COLS)
    compare_stats(df[TS_COLS], flat_df)
    return flat_df


def run_fflows(df: pd.DataFrame, epochs: int, samples: int,
               sequence_length: int) -> pd.DataFrame:

    banner("FFlows – Fourier Flows for Time Series")
    click.echo("  Models temporal structure in frequency space via normalising flows.")
    click.echo(f"  epochs={epochs}  sequence_length={sequence_length}  samples={samples}")

    loader = _build_ts_loader(df, TS_COLS, sequence_length)
    click.echo(f"  Training sequences: {len(loader)}")

    plugin = Plugins().get("fflows", n_iter=epochs, batch_size=128)

    click.echo("\n  Training ...")
    plugin.fit(loader)

    click.echo(f"\n  Generating {samples} synthetic sequences ...")
    result_loader = plugin.generate(count=samples)
    flat_df = _flatten_ts_result(result_loader, TS_COLS)

    click.echo(f"  Generated {len(flat_df)} sequences × {len(TS_COLS)} features")
    click.echo(f"\n  Synthetic data head:")
    click.echo(flat_df.head().to_string())

    store_artifacts(df, flat_df, "fflows", TS_COLS)
    compare_stats(df[TS_COLS], flat_df)
    return flat_df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

MODEL_CHOICES = click.Choice(["ddpm", "timevae", "fflows"], case_sensitive=False)

DEFAULT_EPOCHS = {"ddpm": 1000, "timevae": 1000, "fflows": 500}


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
    "--model", default="ddpm", show_default=True, type=MODEL_CHOICES,
    help="Synthesizer to use: ddpm | timevae | fflows",
)
@click.option(
    "--epochs", default=None, type=int,
    help="Training epochs / iterations. Defaults: ddpm=1000, timevae=1000, fflows=500.",
)
@click.option(
    "--samples", default=500, show_default=True,
    help="Number of synthetic rows / sequences to generate.",
)
@click.option(
    "--sequence-length", default=24, show_default=True,
    help="[timevae / fflows] Number of time steps per training window.",
)
@click.option(
    "--output", default=None,
    help="Optional path to save synthetic data as CSV.",
)
def main(ticker, start, end, model, epochs, samples, sequence_length, output):
    """
    Synthetic financial data generation using the synthcity library.

    Fetches OHLCV data for TICKER, trains the chosen synthesizer, generates
    SAMPLES synthetic observations, and prints comparative statistics.

    NOTE: This script requires a separate venv from main.py (synthcity uses
    PyTorch; ydata-synthetic uses TensorFlow).  Install with:
        uv pip install -e ".[synthcity]"
    """
    click.echo("=" * 60)
    click.echo("  synthcity  ×  Yahoo Finance")
    click.echo("=" * 60)
    click.echo(f"  ticker={ticker}  model={model}  start={start}  end={end}")

    effective_epochs = epochs if epochs is not None else DEFAULT_EPOCHS[model.lower()]

    df = fetch_ohlcv(ticker, start, end)

    model_lower = model.lower()

    if model_lower == "ddpm":
        result = run_ddpm(df, effective_epochs, samples)
    elif model_lower == "timevae":
        result = run_timevae(df, effective_epochs, samples, sequence_length)
    else:
        result = run_fflows(df, effective_epochs, samples, sequence_length)

    if output:
        result.to_csv(output, index=False)
        click.echo(f"\n  Saved CSV → {output}")

    click.echo("\n\nDone.")


if __name__ == "__main__":
    main()
