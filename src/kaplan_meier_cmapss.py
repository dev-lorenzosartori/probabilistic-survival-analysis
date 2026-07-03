"""Kaplan-Meier baseline on the real C-MAPSS landmark-30 modeling table.

Reproduces the Phase 3 analysis: overall survival curve, censoring summary,
covariate screening by Spearman correlation (train split only), tercile risk
groups on the top-ranked covariate, and a log-rank test between the extreme
groups.

Uses `src/km_utils.py`, a from-scratch Kaplan-Meier / log-rank implementation
validated against hand-computed toy examples (see km_utils.py __main__).
Swap in `lifelines` (already in requirements.txt) once you have a network
connection -- it is the recommended library of record for the published
notebook; km_utils.py exists only to make this analysis reproducible offline.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.km_utils import (
    kaplan_meier,
    km_median_survival,
    km_survival_at,
    logrank_test_2groups,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELING_TABLE_PATH = (
    PROJECT_ROOT / "data" / "processed" / "cmapss_fd001_landmark30_modeling_table.csv"
)
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"

LOW_VARIANCE_COVARIATES = [
    "mean_op_setting_1_first_30", "slope_op_setting_1_first_30",
    "mean_op_setting_2_first_30", "slope_op_setting_2_first_30",
    "mean_op_setting_3_first_30", "slope_op_setting_3_first_30",
    "mean_sensor_1_first_30", "slope_sensor_1_first_30",
    "mean_sensor_5_first_30", "slope_sensor_5_first_30",
    "mean_sensor_6_first_30", "slope_sensor_6_first_30",
    "slope_sensor_8_first_30", "mean_sensor_10_first_30",
    "slope_sensor_10_first_30", "slope_sensor_13_first_30",
    "slope_sensor_15_first_30", "mean_sensor_16_first_30",
    "slope_sensor_16_first_30", "mean_sensor_18_first_30",
    "slope_sensor_18_first_30", "mean_sensor_19_first_30",
    "slope_sensor_19_first_30",
]


def screen_covariates(df: pd.DataFrame) -> pd.DataFrame:
    """Rank early-window covariates by |Spearman rho| with time_after_landmark.

    Restricted to the training split, where every unit has a real (uncensored)
    failure time -- this keeps screening simple and separate from the
    censoring-aware modeling that belongs in the Phase 4 Cox model.
    """
    candidate_cols = [
        c for c in df.columns
        if c.startswith(("mean_", "slope_")) and c not in LOW_VARIANCE_COVARIATES
    ]
    train = df[df["split"] == "train"]

    rows = []
    for col in candidate_cols:
        rho, p_value = spearmanr(train[col], train["time_after_landmark"])
        rows.append({"covariate": col, "spearman_rho": rho, "p_value": p_value})

    return (
        pd.DataFrame(rows)
        .sort_values("spearman_rho", key=lambda s: s.abs(), ascending=False)
        .reset_index(drop=True)
    )


def plot_overall_km(df: pd.DataFrame) -> None:
    km_all = kaplan_meier(df["time_after_landmark"], df["event_observed"])

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.step(km_all["time"], km_all["survival"], where="post", linewidth=2, color="#2b6cb0", label="All engines (landmark=30)")
    ax.fill_between(km_all["time"], km_all["ci_lower"], km_all["ci_upper"], step="post", alpha=0.2, color="#2b6cb0")
    ax.set_title("Kaplan-Meier: Residual Survival After Cycle-30 Landmark (C-MAPSS FD001)")
    ax.set_xlabel("Cycles after landmark (cycle 30)")
    ax.set_ylabel("Survival probability")
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "cmapss_km_overall_landmark30.png", dpi=160)
    plt.close(fig)

    print(f"Survival at +60 cycles:  {km_survival_at(km_all, 60):.1%}")
    print(f"Survival at +100 cycles: {km_survival_at(km_all, 100):.1%}")
    print(f"Median residual survival: {km_median_survival(km_all):.1f} cycles after landmark")


def plot_group_km(df: pd.DataFrame, covariate: str) -> None:
    df = df.copy()
    df["risk_group"] = pd.qcut(
        df[covariate], q=3,
        labels=[f"Low {covariate} (Q1)", f"Mid {covariate} (Q2)", f"High {covariate} (Q3)"],
    )

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ["#2f855a", "#d69e2e", "#c53030"]
    for color, (group, gdf) in zip(colors, df.groupby("risk_group", observed=True)):
        km_g = kaplan_meier(gdf["time_after_landmark"], gdf["event_observed"])
        ax.step(km_g["time"], km_g["survival"], where="post", linewidth=2, label=group, color=color)

    ax.set_title(f"Survival by Early {covariate} Tercile (Landmark = Cycle 30)")
    ax.set_xlabel("Cycles after landmark (cycle 30)")
    ax.set_ylabel("Survival probability")
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"cmapss_km_by_{covariate}_tercile.png", dpi=160)
    plt.close(fig)

    low = df[df["risk_group"] == f"Low {covariate} (Q1)"]
    high = df[df["risk_group"] == f"High {covariate} (Q3)"]
    chi2_stat, p_value = logrank_test_2groups(
        low["time_after_landmark"], low["event_observed"],
        high["time_after_landmark"], high["event_observed"],
    )
    print(f"Log-rank test, Low vs High {covariate} tercile: chi2={chi2_stat:.3f}, p={p_value:.5f}")


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(MODELING_TABLE_PATH)

    print(f"Modeling table: {len(df)} units | event rate: {df['event_observed'].mean():.1%}\n")

    plot_overall_km(df)

    print("\nCovariate screening (train split, Spearman vs time_after_landmark):")
    ranking = screen_covariates(df)
    print(ranking.head(10).to_string(index=False))

    top_covariate = ranking.iloc[0]["covariate"]
    print(f"\nUsing top-ranked covariate for exploratory risk groups: {top_covariate}")
    plot_group_km(df, top_covariate)


if __name__ == "__main__":
    main()
