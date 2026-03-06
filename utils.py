import io
import os
import re
import sys

import click
import matplotlib
matplotlib.use("Agg")  # headless — no GUI needed
import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf


# ---------------------------------------------------------------------------
# Console helpers
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
    click.echo(f"\n  yfinance version : {yf.__version__}")

    # Quick single-day probe to confirm the ticker is recognised before the
    # full range download.  fast_info is lightweight (no full history fetch).
    t = yf.Ticker(ticker)
    try:
        info = t.fast_info
        click.echo(f"  Ticker probe     : {ticker}")
        click.echo(f"    exchange       : {getattr(info, 'exchange', 'N/A')}")
        click.echo(f"    currency       : {getattr(info, 'currency', 'N/A')}")
        click.echo(f"    timezone       : {getattr(info, 'timezone', 'N/A')}")
        click.echo(f"    last price     : {getattr(info, 'last_price', 'N/A')}")
    except Exception as probe_err:
        click.echo(f"  WARNING: ticker probe failed ({probe_err}). Proceeding anyway.")

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
# Artifact helpers shared across all models
# ---------------------------------------------------------------------------

PLOTS_DIR = "outputs/plots"
os.makedirs("outputs", exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)


def store_artifacts(real_df: pd.DataFrame, synth_df: pd.DataFrame,
                    model_name: str, real_cols: list[str]) -> None:
    """Save real and synthetic data to CSV for external analysis."""
    ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    real_path  = f"outputs/real_data_{model_name}_{ts}.csv"
    synth_path = f"outputs/synthetic_data_{model_name}_{ts}.csv"
    real_df[real_cols].to_csv(real_path,  index=False)
    synth_df.to_csv(synth_path, index=False)
    click.echo(f"  Real data      → {real_path}")
    click.echo(f"  Synthetic data → {synth_path}")


def save_architecture_plots(model_pairs: list[tuple[str, object]]) -> None:
    """
    Save Keras model architecture diagrams as PNGs.

    Args:
        model_pairs: list of (filename_prefix, keras_model) — e.g.
                     [("dragan_generator", synth.generator), ...]
    """
    import tensorflow as tf
    for name, model in model_pairs:
        path = f"{PLOTS_DIR}/{name}_architecture.png"
        try:
            tf.keras.utils.plot_model(
                model, to_file=path, show_shapes=True,
                show_layer_names=True, expand_nested=True,
            )
            click.echo(f"  Architecture plot → {path}")
        except Exception as e:
            click.echo(f"  WARNING: could not plot {name} ({e}). "
                       "Install pydot + graphviz to enable architecture diagrams.")


def plot_training_losses(loss_df: pd.DataFrame, model_name: str,
                         loss_cols: list[str] | None = None) -> None:
    """
    Save a training-loss curve PNG using matplotlib.

    Args:
        loss_df:    DataFrame with an 'epoch' (or 'step') column and one or
                    more loss columns.
        model_name: Used as the filename prefix.
        loss_cols:  Which columns to plot. Defaults to all non-index columns
                    except 'epoch'/'step'.
    """
    if loss_df.empty:
        click.echo("  WARNING: no loss records captured — skipping loss plot.")
        return

    x_col = "epoch" if "epoch" in loss_df.columns else "step"
    if loss_cols is None:
        loss_cols = [c for c in loss_df.columns if c != x_col]

    fig, ax = plt.subplots(figsize=(10, 4))
    for col in loss_cols:
        ax.plot(loss_df[x_col], loss_df[col], label=col)
    ax.set_xlabel(x_col.capitalize())
    ax.set_ylabel("Loss")
    ax.set_title(f"{model_name.upper()} – Training Losses")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    path = f"{PLOTS_DIR}/{model_name}_training_losses.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    click.echo(f"  Loss plot         → {path}")


# ---------------------------------------------------------------------------
# Stdout tee — captures printed epoch losses (DRAGAN / CTGAN style)
# ---------------------------------------------------------------------------

class LossCaptureTee:
    """
    Wraps sys.stdout during training:
      - Accumulates all output in a buffer for post-training parsing.
      - Forwards non-epoch lines immediately; epoch lines only every
        `interval` epochs (to avoid console spam).

    Usage::

        tee = LossCaptureTee(sys.stdout, pattern=..., col_names=..., interval=20)
        sys.stdout = tee
        try:
            synth.fit(...)
        finally:
            sys.stdout = tee._real
        loss_df = tee.to_dataframe()
    """

    def __init__(self, real_stdout, pattern: str,
                 col_names: list[str], interval: int = 20):
        """
        Args:
            real_stdout: the original sys.stdout.
            pattern:     regex with one group per col_name (first group = epoch).
            col_names:   column names for the captured groups, e.g.
                         ["epoch", "disc_loss", "gen_loss"].
            interval:    print epoch lines to console every N epochs.
        """
        self._real     = real_stdout
        self._re       = re.compile(pattern)
        self._cols     = col_names
        self._interval = interval
        self._buf      = io.StringIO()
        self._pending  = ""

    def write(self, text: str):
        self._buf.write(text)
        self._pending += text
        if "\n" in self._pending:
            *complete, self._pending = self._pending.split("\n")
            for line in complete:
                self._route(line)

    def _route(self, line: str):
        m = self._re.search(line)
        if m:
            epoch = int(m.group(1))
            if epoch % self._interval == 0:
                self._real.write(line + "\n")
                self._real.flush()
        else:
            self._real.write(line + "\n")
            self._real.flush()

    def flush(self):
        self._real.flush()

    def to_dataframe(self) -> pd.DataFrame:
        """Parse the captured buffer and return one row per matched epoch."""
        records = []
        for line in self._buf.getvalue().splitlines():
            m = self._re.search(line)
            if m:
                row = {}
                for i, col in enumerate(self._cols):
                    raw = m.group(i + 1)
                    row[col] = int(raw) if col == "epoch" else float(raw)
                records.append(row)
        return pd.DataFrame(records)
