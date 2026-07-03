"""Transform raw NASA C-MAPSS files into survival-analysis modeling tables.

Expected standard C-MAPSS FD00x layout (space-delimited, no header):

    unit_number, time_cycles, op_setting_1..3, sensor_1..21

Survival framing:
    train_FDxxx.txt -> engines run until failure -> event_observed = 1
    test_FDxxx.txt  -> engines are truncated before failure -> event_observed = 0
    RUL_FDxxx.txt   -> true remaining useful life for test units
                       kept only for later validation, never used for model fitting

The script writes two tables:
    1. Full audit table with every unit, including incomplete early windows.
    2. Landmark modeling table with only units observed for the full early window.

The landmark modeling table answers this question:

    Given the first N operating cycles, what is the probability of surviving
    beyond future cycle horizons?
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "raw" / "cmapss"
DEFAULT_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

SENSOR_COLS = [f"sensor_{i}" for i in range(1, 22)]
SETTING_COLS = [f"op_setting_{i}" for i in range(1, 4)]
ALL_COVARIATE_SOURCE_COLS = SETTING_COLS + SENSOR_COLS
COLUMN_NAMES = ["unit_number", "time_cycles"] + ALL_COVARIATE_SOURCE_COLS


def load_cmapss_file(path: Path) -> pd.DataFrame:
    """Load a train_ or test_ C-MAPSS file, tolerant of trailing whitespace."""
    raw = pd.read_csv(path, sep=r"\s+", header=None, engine="python")
    raw = raw.iloc[:, : len(COLUMN_NAMES)]
    raw.columns = COLUMN_NAMES
    raw["unit_number"] = raw["unit_number"].astype(int)
    raw["time_cycles"] = raw["time_cycles"].astype(int)
    return raw


def load_rul_file(path: Path) -> pd.DataFrame:
    """Load RUL_FDxxx.txt. Row order corresponds to unit_number 1..N in test."""
    rul = pd.read_csv(path, sep=r"\s+", header=None, engine="python").iloc[:, :1]
    rul.columns = ["true_rul"]
    rul["unit_number"] = np.arange(1, len(rul) + 1)
    return rul[["unit_number", "true_rul"]]


def engineer_early_window_covariates(df: pd.DataFrame, window: int = 30) -> pd.DataFrame:
    """Build mean/slope covariates from the first `window` cycles of each unit.

    Units with fewer than `window` observed cycles are not removed here.
    They are flagged with `has_full_window = False` so the modeling decision
    remains explicit and auditable.
    """
    records: list[dict[str, float | int | bool]] = []

    for unit_id, group in df.groupby("unit_number"):
        group = group.sort_values("time_cycles")
        total_cycles = int(group["time_cycles"].max())
        window_df = group[group["time_cycles"] <= window]

        row: dict[str, float | int | bool] = {
            "unit_number": int(unit_id),
            "total_cycles_available": total_cycles,
            "has_full_window": len(window_df) >= window,
        }

        cycles = window_df["time_cycles"].to_numpy(dtype=float)
        for col in ALL_COVARIATE_SOURCE_COLS:
            values = window_df[col].to_numpy(dtype=float)
            row[f"mean_{col}_first_{window}"] = float(np.mean(values)) if len(values) else np.nan
            row[f"slope_{col}_first_{window}"] = (
                float(np.polyfit(cycles, values, 1)[0]) if len(values) >= 2 else np.nan
            )

        records.append(row)

    return pd.DataFrame(records)


def build_duration_table(df: pd.DataFrame, event_observed: int, split: str) -> pd.DataFrame:
    """Build one row per unit with observed duration and event indicator."""
    duration = df.groupby("unit_number")["time_cycles"].max().rename("duration").reset_index()
    duration["event_observed"] = int(event_observed)
    duration["split"] = split
    return duration


def validate_rul_alignment(test_duration: pd.DataFrame, rul: pd.DataFrame) -> None:
    """Validate that the RUL file has one row for every test unit."""
    test_units = set(test_duration["unit_number"])
    rul_units = set(rul["unit_number"])
    if test_units != rul_units:
        missing_in_rul = sorted(test_units - rul_units)
        extra_in_rul = sorted(rul_units - test_units)
        raise ValueError(
            "RUL/test unit mismatch. "
            f"Missing in RUL: {missing_in_rul[:10]}; extra in RUL: {extra_in_rul[:10]}"
        )


def flag_low_variance_covariates(
    df: pd.DataFrame,
    covariate_cols: list[str],
    threshold: float = 1e-6,
) -> list[str]:
    """Return numeric covariates with near-zero variance."""
    variances = df[covariate_cols].var(numeric_only=True)
    return variances[variances < threshold].index.tolist()


def add_landmark_columns(table: pd.DataFrame, window: int) -> pd.DataFrame:
    """Add landmark-analysis columns for a fixed early observation window."""
    out = table.copy()
    out["landmark_cycle"] = window
    out["time_after_landmark"] = out["duration"] - window
    out["is_model_eligible"] = out["has_full_window"] & (out["time_after_landmark"] >= 0)
    return out


def build_survival_table(
    train_path: Path,
    test_path: Path,
    rul_path: Path,
    window: int = 30,
) -> pd.DataFrame:
    """Build the full audit survival table from raw C-MAPSS files."""
    train_raw = load_cmapss_file(train_path)
    test_raw = load_cmapss_file(test_path)
    rul = load_rul_file(rul_path)

    train_duration = build_duration_table(train_raw, event_observed=1, split="train")
    test_duration = build_duration_table(test_raw, event_observed=0, split="test")
    validate_rul_alignment(test_duration, rul)

    test_duration = test_duration.merge(rul, on="unit_number", how="left")
    test_duration["true_event_time"] = test_duration["duration"] + test_duration["true_rul"]
    train_duration["true_event_time"] = np.nan
    train_duration["true_rul"] = np.nan

    train_cov = engineer_early_window_covariates(train_raw, window=window)
    test_cov = engineer_early_window_covariates(test_raw, window=window)

    train_table = train_duration.merge(train_cov, on="unit_number", how="left")
    test_table = test_duration.merge(test_cov, on="unit_number", how="left")

    combined = pd.concat([train_table, test_table], ignore_index=True, sort=False)
    combined["entity_id"] = combined["split"] + "_" + combined["unit_number"].astype(str)
    combined = add_landmark_columns(combined, window=window)

    # Stable column order for readability.
    front_cols = [
        "entity_id",
        "unit_number",
        "split",
        "duration",
        "event_observed",
        "landmark_cycle",
        "time_after_landmark",
        "is_model_eligible",
        "true_rul",
        "true_event_time",
        "total_cycles_available",
        "has_full_window",
    ]
    other_cols = [c for c in combined.columns if c not in front_cols]
    return combined[front_cols + other_cols]


def build_modeling_table(full_table: pd.DataFrame) -> pd.DataFrame:
    """Return units eligible for the cycle-N landmark model.

    Units with incomplete early windows are excluded from the baseline model because
    the model is explicitly defined at the landmark time. They remain available in
    the full audit table for transparency and possible sensitivity analyses.
    """
    return full_table.loc[full_table["is_model_eligible"]].copy()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build C-MAPSS survival-analysis tables.")
    parser.add_argument("--subset", default="FD001", help="C-MAPSS subset, e.g. FD001.")
    parser.add_argument("--window", type=int, default=30, help="Early observation window in cycles.")
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR, help="Directory with raw C-MAPSS files.")
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=DEFAULT_PROCESSED_DIR,
        help="Directory where processed tables will be saved.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    subset = args.subset.upper()

    train_path = args.raw_dir / f"train_{subset}.txt"
    test_path = args.raw_dir / f"test_{subset}.txt"
    rul_path = args.raw_dir / f"RUL_{subset}.txt"

    for path in (train_path, test_path, rul_path):
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {path}. Download the NASA C-MAPSS {subset} files and place them "
                f"in {args.raw_dir}."
            )

    full_table = build_survival_table(train_path, test_path, rul_path, window=args.window)
    modeling_table = build_modeling_table(full_table)

    covariate_cols = [c for c in full_table.columns if c.startswith(("mean_", "slope_"))]
    low_variance = flag_low_variance_covariates(
        modeling_table[modeling_table["split"] == "train"],
        covariate_cols,
    )

    incomplete_window = int((~full_table["has_full_window"]).sum())

    args.processed_dir.mkdir(parents=True, exist_ok=True)
    full_output = args.processed_dir / f"cmapss_{subset.lower()}_survival_table_full.csv"
    modeling_output = args.processed_dir / f"cmapss_{subset.lower()}_landmark{args.window}_modeling_table.csv"

    full_table.to_csv(full_output, index=False)
    modeling_table.to_csv(modeling_output, index=False)

    print(f"Saved full audit table: {len(full_table):,} rows -> {full_output}")
    print(f"Saved modeling table: {len(modeling_table):,} rows -> {modeling_output}")
    print(full_table["split"].value_counts().to_string())
    print(f"Overall event rate, full table: {full_table['event_observed'].mean():.1%}")
    print(f"Units with incomplete first-window (<{args.window} cycles): {incomplete_window}")
    print(f"Low-variance covariates flagged (train/modeling, var < 1e-6): {low_variance}")


if __name__ == "__main__":
    main()
