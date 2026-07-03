"""Generate a synthetic predictive-maintenance survival dataset.

This script is only a starter dataset for validating the project pipeline before
the real benchmark dataset is integrated.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "synthetic_asset_survival.csv"


def generate_dataset(n_assets: int = 1200, random_state: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)

    asset_age = rng.gamma(shape=4.0, scale=8.0, size=n_assets)
    avg_temperature = rng.normal(loc=72, scale=9, size=n_assets)
    vibration_score = rng.lognormal(mean=0.2, sigma=0.45, size=n_assets)
    load_factor = rng.beta(a=5, b=2, size=n_assets)
    maintenance_quality = rng.choice(
        ["low", "medium", "high"], size=n_assets, p=[0.22, 0.52, 0.26]
    )

    quality_effect = pd.Series(maintenance_quality).map(
        {"low": 0.45, "medium": 0.0, "high": -0.35}
    ).to_numpy()

    linear_risk = (
        0.018 * asset_age
        + 0.035 * (avg_temperature - 72)
        + 0.55 * (vibration_score - vibration_score.mean())
        + 1.10 * (load_factor - load_factor.mean())
        + quality_effect
    )

    baseline_scale = 95
    weibull_shape = 1.65
    individual_scale = baseline_scale * np.exp(-linear_risk / weibull_shape)
    event_time = individual_scale * rng.weibull(weibull_shape, size=n_assets)

    censoring_time = rng.uniform(40, 180, size=n_assets)
    duration = np.minimum(event_time, censoring_time)
    event_observed = (event_time <= censoring_time).astype(int)

    df = pd.DataFrame(
        {
            "asset_id": [f"asset_{i:04d}" for i in range(n_assets)],
            "duration": np.round(duration, 2),
            "event_observed": event_observed,
            "asset_age": np.round(asset_age, 2),
            "avg_temperature": np.round(avg_temperature, 2),
            "vibration_score": np.round(vibration_score, 3),
            "load_factor": np.round(load_factor, 3),
            "maintenance_quality": maintenance_quality,
        }
    )

    return df


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df = generate_dataset()
    df.to_csv(OUTPUT_PATH, index=False)
    event_rate = df["event_observed"].mean()
    print(f"Saved {len(df):,} rows to {OUTPUT_PATH}")
    print(f"Observed event rate: {event_rate:.1%}")


if __name__ == "__main__":
    main()
