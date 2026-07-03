"""Run a Kaplan-Meier baseline on the synthetic survival dataset."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "synthetic_asset_survival.csv"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"


def main() -> None:
    df = pd.read_csv(DATA_PATH)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    kmf = KaplanMeierFitter()
    kmf.fit(df["duration"], event_observed=df["event_observed"], label="All assets")

    survival_at_60 = float(kmf.predict(60))
    median_survival = kmf.median_survival_time_

    print(f"Survival probability at 60 cycles: {survival_at_60:.1%}")
    print(f"Median survival time: {median_survival:.2f} cycles")

    fig, ax = plt.subplots(figsize=(9, 5))
    kmf.plot_survival_function(ax=ax, ci_show=True)
    ax.set_title("Kaplan-Meier Survival Curve - Synthetic Asset Failure")
    ax.set_xlabel("Operational time")
    ax.set_ylabel("Survival probability")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "kaplan_meier_all_assets.png", dpi=160)

    fig, ax = plt.subplots(figsize=(9, 5))
    for group, group_df in df.groupby("maintenance_quality"):
        kmf.fit(
            group_df["duration"],
            event_observed=group_df["event_observed"],
            label=f"Maintenance quality: {group}",
        )
        kmf.plot_survival_function(ax=ax, ci_show=False)

    ax.set_title("Survival Curves by Maintenance Quality")
    ax.set_xlabel("Operational time")
    ax.set_ylabel("Survival probability")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "kaplan_meier_by_maintenance_quality.png", dpi=160)

    low = df[df["maintenance_quality"] == "low"]
    high = df[df["maintenance_quality"] == "high"]
    result = logrank_test(
        low["duration"],
        high["duration"],
        event_observed_A=low["event_observed"],
        event_observed_B=high["event_observed"],
    )
    print(f"Log-rank test low vs high maintenance p-value: {result.p_value:.6f}")


if __name__ == "__main__":
    main()
