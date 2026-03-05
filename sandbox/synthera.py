"""
Synthera Python Client - Basic Exploration Script

Synthera is an AI-powered synthetic financial data API focused on fixed income
(yield curves). This script demonstrates the core workflow:
  1. Connect to the API
  2. Discover available models
  3. Inspect model metadata
  4. Simulate synthetic yield curves
  5. Access and visualize the results
"""

import os
from sandbox.synthera import SyntheraClient


def main():
    # -------------------------------------------------------------------------
    # 1. Create the client
    #    API key can be passed directly or set via SYNTHERA_API_KEY env var.
    # -------------------------------------------------------------------------
    api_key = os.getenv("SYNTHERA_API_KEY", "YOUR_API_KEY_HERE")
    client = SyntheraClient(api_key=api_key)

    print(f"Synthera SDK version: {client.version}")

    # -------------------------------------------------------------------------
    # 2. Health check
    # -------------------------------------------------------------------------
    is_healthy = client.healthy()
    print(f"API healthy: {is_healthy}")
    if not is_healthy:
        print("API is not reachable. Check your API key and network connection.")
        return

    # -------------------------------------------------------------------------
    # 3. List available fixed-income models
    # -------------------------------------------------------------------------
    fi = client.fixed_income
    model_labels = fi.get_model_labels()
    print(f"\nAvailable models ({len(model_labels)}):")
    for label in model_labels:
        print(f"  - {label}")

    if not model_labels:
        print("No models available.")
        return

    # -------------------------------------------------------------------------
    # 4. Inspect metadata for the first model
    # -------------------------------------------------------------------------
    first_label = model_labels[0]
    meta = fi.get_model_metadata(first_label)
    print(f"\nMetadata for '{first_label}':")
    print(f"  Dataset:          {meta.dataset}")
    print(f"  Universe:         {meta.universe}")
    print(f"  Training period:  {meta.start_date_training} → {meta.end_date_training}")
    print(f"  Max simulate days:{meta.max_simulate_days}")
    print(f"  Conditional days: {meta.conditional_days}")
    print(f"  Curve labels:     {meta.curve_labels}")
    print(f"  Maturities:       {meta.maturities}")

    # -------------------------------------------------------------------------
    # 5. Simulate synthetic yield curves
    #    Adjust num_samples and num_days to explore the output shape.
    # -------------------------------------------------------------------------
    NUM_SAMPLES = 5
    NUM_DAYS = 30

    print(f"\nSimulating {NUM_SAMPLES} samples over {NUM_DAYS} days ...")
    results = fi.simulate({
        "model_label": first_label,
        "num_samples": NUM_SAMPLES,
        "num_days": NUM_DAYS,
    })

    # -------------------------------------------------------------------------
    # 6. Explore the SimulateResults object
    # -------------------------------------------------------------------------
    print(f"\nSimulation complete.")
    print(f"  Countries / curves: {results.names}")
    print(f"  Column names:       {results.column_names}")
    print(f"  ndarray shape:      {results.ndarray.shape}")
    #   Shape is (num_samples, num_days, num_countries, num_maturities)

    dates = results.get_dates()
    print(f"  Date range:         {dates[0].date()} → {dates[-1].date()}  ({len(dates)} days)")

    yc_indices = results.get_yc_indices()
    print(f"  Yield-curve indices (business days): {yc_indices[:5]} ...")

    # -------------------------------------------------------------------------
    # 7. Access data for a specific country / sample
    # -------------------------------------------------------------------------
    first_country = results.names[0]

    # 2-D array: (num_days, num_maturities) for sample 0
    yc_sample = results.get_yc_sample(first_country, sample_num=0)
    print(f"\n  '{first_country}' sample 0 shape: {yc_sample.shape}")
    print(f"  First row (t=0):  {yc_sample[0]}")
    print(f"  Last  row (t=-1): {yc_sample[-1]}")

    # 1-D array: yield curve snapshot at t=0 for sample 0
    snapshot = results.get_country_sample_at_t(first_country, time_idx=0, sample_num=0)
    print(f"\n  Yield curve snapshot at t=0, sample 0: {snapshot}")

    # 3-D array: all samples for a country (num_samples, num_days, num_maturities)
    all_samples = results.get_country_yc_samples(first_country)
    print(f"  All samples for '{first_country}' shape: {all_samples.shape}")

    # -------------------------------------------------------------------------
    # 8. DataFrames — one per country, indexed by date
    # -------------------------------------------------------------------------
    print(f"\n  Available dataframes: {list(results.dataframes.keys())}")
    df = results.dataframes[first_country]
    print(f"\n  DataFrame for '{first_country}' (first sample, shape {df.shape}):")
    print(df.head())

    # -------------------------------------------------------------------------
    # 9. Plotting (saves figures rather than showing interactively)
    # -------------------------------------------------------------------------
    print("\nGenerating plots ...")

    # Yield curve for one sample over time (animated-style line plot)
    fig1 = results.plot_country_sample_yield_curve_over_time(
        first_country, sample_num=0, show_plot=False
    )
    fig1.savefig("yc_over_time.png", dpi=100)
    print("  Saved: yc_over_time.png")

    # All samples at t=0 (fan chart)
    fig2 = results.plot_country_all_samples_at_time(
        first_country, time_idx=0, show_plot=False
    )
    fig2.savefig("yc_all_samples_t0.png", dpi=100)
    print("  Saved: yc_all_samples_t0.png")

    # Single sample at t=0
    fig3 = results.plot_country_sample_at_time(
        first_country, time_idx=0, sample_num=0, show_plot=False
    )
    fig3.savefig("yc_sample0_t0.png", dpi=100)
    print("  Saved: yc_sample0_t0.png")

    print("\nDone.")


if __name__ == "__main__":
    main()
