"""Smoke test for the C-MAPSS survival table transformation.

Creates tiny synthetic files with the same column layout as C-MAPSS and verifies:
- train units become observed events;
- test units become censored observations;
- true_event_time is duration + RUL only for test units;
- units with incomplete early windows are flagged and excluded from the modeling table.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from src.build_cmapss_survival_table import build_modeling_table, build_survival_table


def _row(unit: int, cycle: int) -> str:
    settings = [0.0, 0.0, 100.0]
    sensors = [float(unit + cycle + i / 100) for i in range(1, 22)]
    values = [unit, cycle] + settings + sensors
    return " ".join(str(v) for v in values)


def _write_cmapss_file(path: Path, unit_cycles: dict[int, int]) -> None:
    lines = []
    for unit, max_cycle in unit_cycles.items():
        for cycle in range(1, max_cycle + 1):
            lines.append(_row(unit, cycle))
    path.write_text("\n".join(lines) + "\n")


def test_cmapss_transformation_smoke() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        train_path = tmpdir / "train_FD001.txt"
        test_path = tmpdir / "test_FD001.txt"
        rul_path = tmpdir / "RUL_FD001.txt"

        _write_cmapss_file(train_path, {1: 35, 2: 20})
        _write_cmapss_file(test_path, {1: 40})
        rul_path.write_text("15\n")

        full = build_survival_table(train_path, test_path, rul_path, window=30)
        modeling = build_modeling_table(full)

        assert len(full) == 3
        assert len(modeling) == 2
        assert set(full["event_observed"]) == {0, 1}
        assert full.loc[full["entity_id"] == "test_1", "true_event_time"].iloc[0] == 55
        assert full.loc[full["entity_id"] == "train_2", "has_full_window"].iloc[0] == np.False_
        assert "train_2" not in set(modeling["entity_id"])


if __name__ == "__main__":
    test_cmapss_transformation_smoke()
    print("Smoke test passed.")
