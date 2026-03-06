
import click
import pandas as pd

from utils import (
    banner, compare_stats, fetch_ohlcv,
    store_artifacts, save_architecture_plots,
    plot_training_losses, LossCaptureTee,
)

from ydata_synthetic.synthesizers import ModelParameters, TrainParameters
from ydata_synthetic.synthesizers.regular import RegularSynthesizer
from ydata_synthetic.synthesizers.timeseries import TimeSeriesSynthesizer

TABULAR_COLS = ["open", "high", "low", "close", "volume", "return_pct", "hl_spread_pct"]
TIMEGAN_COLS  = ["open", "high", "low", "close", "volume", "return_pct"]  # 6 features (hl_spread_pct excluded — derived from others)


def run_ctgan(df: pd.DataFrame, epochs: int, samples: int) -> pd.DataFrame:

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


def run_dragan(df: pd.DataFrame, epochs: int, samples: int) -> pd.DataFrame:

    banner("DRAGAN – Deep Regret Analytic GAN")
    click.echo("  Gradient-penalty GAN variant; stable training without mode collapse.")
    click.echo(f"  epochs={epochs}  samples to generate={samples}")

    model_args = ModelParameters(batch_size=500, lr=2e-4, betas=(0.5, 0.9),
                                 noise_dim=32, layers_dim=128)
    train_args = TrainParameters(epochs=epochs, sample_interval=epochs + 1)  # disable checkpointing

    # n_discriminator: number of discriminator updates per generator step (paper recommends 3)
    synth = RegularSynthesizer(modelname="dragan", model_parameters=model_args,
                               n_discriminator=3)

    # DRAGAN prints "Epoch: X | disc_loss: Y | gen_loss: Z" to stdout each epoch.
    # Capture it so we can parse losses and throttle console output.
    _DRAGAN_RE = (
        r"Epoch:\s*(\d+)"
        r".*?disc_loss:\s*([-+]?\d[\d.eE+-]*)"
        r".*?gen_loss:\s*([-+]?\d[\d.eE+-]*)"
    )
    tee = LossCaptureTee(
        sys.stdout,
        pattern=_DRAGAN_RE,
        col_names=["epoch", "disc_loss", "gen_loss"],
        interval=20,
    )
    click.echo("\n  Training ...")
    sys.stdout = tee
    try:
        synth.fit(data=df[TABULAR_COLS], train_arguments=train_args,
                  num_cols=TABULAR_COLS, cat_cols=[])
    finally:
        sys.stdout = tee._real

    # ── Training losses ───────────────────────────────────────────────────
    loss_df = tee.to_dataframe()
    loss_csv = "outputs/dragan_training_log.csv"
    loss_df.to_csv(loss_csv, index=False)
    click.echo(f"\n  Training log → {loss_csv}  ({len(loss_df)} epochs recorded)")
    plot_training_losses(loss_df, "dragan", loss_cols=["disc_loss", "gen_loss"])

    # ── Architecture plots ────────────────────────────────────────────────
    save_architecture_plots([
        ("dragan_generator",     synth.generator),
        ("dragan_discriminator", synth.discriminator),
    ])

    click.echo(f"\n  Generating {samples} synthetic rows ...")
    synth_df = synth.sample(samples)

    store_artifacts(df, synth_df, "dragan", TABULAR_COLS)
    compare_stats(df[TABULAR_COLS], synth_df)
    return synth_df


def run_timegan(df: pd.DataFrame, epochs: int, samples: int,
                sequence_length: int) -> list:

    banner("TimeGAN – Time-Series GAN")
    click.echo(f"  Treats data as ordered sequences (preserves temporal dependencies).")
    click.echo(f"  epochs={epochs}  sequence_length={sequence_length}  samples={samples}")
    click.echo("  NOTE: 10 000–50 000 epochs needed for production quality.")

    cols = TIMEGAN_COLS  # 6 features — TimeGAN architecture is built from this count

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

    # ── Loss logging via instance-level monkey-patch ──────────────────────
    # TimeGAN's train() calls self.train_generator / self.train_discriminator.
    # Python resolves instance __dict__ before class-level descriptors, so
    # patching the instance attribute is enough to intercept every call.
    # We only cover the joint-training phase — that's where both losses exist.
    joint_loss_records: list[dict] = []
    _orig_train_generator    = synth.train_generator
    _orig_train_discriminator = synth.train_discriminator

    def _logging_train_generator(x, z, opt):
        g_loss_u, g_loss_s, g_loss_v = _orig_train_generator(x, z, opt)
        joint_loss_records.append({
            "g_loss_unsupervised": float(g_loss_u),
            "g_loss_supervised":   float(g_loss_s),
            "g_loss_moments":      float(g_loss_v),
        })
        return g_loss_u, g_loss_s, g_loss_v

    def _logging_train_discriminator(x, z, opt):
        d_loss = _orig_train_discriminator(x, z, opt)
        if joint_loss_records:
            joint_loss_records[-1]["d_loss"] = float(d_loss)
        return d_loss

    synth.train_generator     = _logging_train_generator
    synth.train_discriminator = _logging_train_discriminator

    click.echo("\n  Training ...")
    synth.fit(df[cols], train_args, num_cols=cols)

    # ── Training losses ───────────────────────────────────────────────────
    if joint_loss_records:
        loss_df = pd.DataFrame(joint_loss_records)
        loss_df.index.name = "step"
        loss_df = loss_df.reset_index()
        loss_csv = "outputs/timegan_training_log.csv"
        loss_df.to_csv(loss_csv, index=False)
        click.echo(f"\n  Training log → {loss_csv}  ({len(loss_df)} steps recorded)")
        plot_training_losses(
            loss_df, "timegan",
            loss_cols=["g_loss_unsupervised", "g_loss_supervised",
                       "g_loss_moments", "d_loss"],
        )
    else:
        click.echo("\n  WARNING: no joint-training loss records captured "
                   "(monkey-patch may not have intercepted train_generator).")

    click.echo(f"\n  Generating {samples} synthetic sequences ...")
    # sample() returns a list of DataFrames, one per sequence
    synth_sequences = synth.sample(n_samples=samples)

    click.echo(f"\n  Generated {len(synth_sequences)} sequences, "
               f"each {synth_sequences[0].shape[0]} steps × "
               f"{synth_sequences[0].shape[1]} features")

    # Preview first sequence
    click.echo(f"\n  Synthetic sequence #0 (head):")
    click.echo(synth_sequences[0].head().to_string())

    # One representative row per sequence → same sample count as tabular models.
    flat_df = pd.DataFrame(
        [seq.iloc[0].values for seq in synth_sequences],
        columns=cols,
    )
    store_artifacts(df, flat_df, "timegan", cols)
    compare_stats(df[cols], flat_df)

    return synth_sequences


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

MODEL_CHOICES = click.Choice(["ctgan", "gmm", "timegan", "dragan"], case_sensitive=False)

DEFAULT_EPOCHS = {"ctgan": 300, "gmm": 300, "timegan": 200, "dragan": 300}


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
    help="Synthesizer to use: ctgan | gmm | timegan | dragan",
)
@click.option(
    "--epochs", default=None, type=int,
    help=("Training epochs. Applicable and required for all models except GMM (fast), which ignores this parameter.")
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
    elif model_lower == "dragan":
        result = run_dragan(df, effective_epochs, samples)
    else:
        result = run_timegan(df, effective_epochs, samples, sequence_length)

    # 3. Optionally save output
    if output:
        if model_lower == "timegan":
            # result is a list of DataFrames; save the flattened first-step version
            flat = pd.DataFrame(
                [seq.iloc[0].values for seq in result], columns=TIMEGAN_COLS
            )
            flat.to_csv(output, index=False)
            click.echo(f"\n  Saved CSV → {output}")
        else:
            result.to_csv(output, index=False)
            click.echo(f"\n  Saved CSV → {output}")

    click.echo("\n\nDone.")


if __name__ == "__main__":
    main()
