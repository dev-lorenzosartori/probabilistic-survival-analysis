"""Evaluation helpers with survival-metric conventions made explicit."""

from __future__ import annotations

from collections.abc import Sequence
import numpy as np
import pandas as pd
from sksurv.metrics import brier_score, concordance_index_censored, cumulative_dynamic_auc, integrated_brier_score


def make_survival_target(event: Sequence[bool], time: Sequence[float]) -> np.ndarray:
    event_array = np.asarray(event, dtype=bool)
    time_array = np.asarray(time, dtype=float)
    if event_array.shape != time_array.shape:
        raise ValueError("event and time must have identical shapes")
    if np.any(time_array <= 0):
        raise ValueError("survival times must be strictly positive")
    return np.array(list(zip(event_array, time_array, strict=True)),
                    dtype=[("event", "?"), ("time", "<f8")])


def harrell_c_index(y: np.ndarray, risk_score: Sequence[float]) -> float:
    """Harrell's C where larger scores mean higher risk / earlier failure."""
    return float(concordance_index_censored(y["event"], y["time"], np.asarray(risk_score))[0])


def bootstrap_c_index(y, risk_score, *, n_bootstrap=1_000, random_state=42):
    rng = np.random.default_rng(random_state)
    risk = np.asarray(risk_score, dtype=float)
    estimates = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, len(y), size=len(y))
        try:
            estimates.append(harrell_c_index(y[idx], risk[idx]))
        except ValueError:
            continue
    if not estimates:
        raise ValueError("No valid bootstrap samples")
    return tuple(np.quantile(estimates, [0.025, 0.975]))


def survival_probability_matrix(model, X, horizons):
    times = np.asarray(horizons, dtype=float)
    return np.vstack([fn(times) for fn in model.predict_survival_function(X)])


def time_dependent_auc_table(y_train, y_test, risk_score, horizons):
    times = np.asarray(horizons, dtype=float)
    auc, mean_auc = cumulative_dynamic_auc(y_train, y_test, np.asarray(risk_score), times)
    return pd.DataFrame({"horizon": times, "auc": auc}), float(mean_auc)


def brier_metrics(y_train, y_test, survival_probabilities, horizons):
    times = np.asarray(horizons, dtype=float)
    _, scores = brier_score(y_train, y_test, survival_probabilities, times)
    ibs = integrated_brier_score(y_train, y_test, survival_probabilities, times)
    return pd.DataFrame({"horizon": times, "brier_score": scores}), float(ibs)


def calibration_table(y, predicted_survival, *, horizon, n_bins=4):
    if not bool(np.all(y["event"])):
        raise ValueError("Calibration requires fully observed event times")
    frame = pd.DataFrame({
        "predicted_survival": np.asarray(predicted_survival, dtype=float),
        "observed_survival": (y["time"] > horizon).astype(float),
    })
    frame["bin"] = pd.qcut(frame["predicted_survival"], q=n_bins, labels=False, duplicates="drop")
    result = frame.groupby("bin", as_index=False, observed=True).agg(
        n=("observed_survival", "size"),
        predicted_survival=("predicted_survival", "mean"),
        observed_survival=("observed_survival", "mean"),
    ).sort_values("predicted_survival")
    result["absolute_gap"] = (result["predicted_survival"] - result["observed_survival"]).abs()
    return result

