"""Minimal, dependency-free Kaplan-Meier + log-rank implementation.

No internet access is available in this environment to install `lifelines`,
so this implements the product-limit estimator (Kaplan-Meier), Greenwood's
variance formula for confidence intervals, and the Mantel-Haenszel log-rank
test from first principles. Validated against hand-computed toy examples
below before being used on real data.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import chi2 as chi2_dist


def kaplan_meier(duration: np.ndarray, event: np.ndarray) -> pd.DataFrame:
    duration = np.asarray(duration, dtype=float)
    event = np.asarray(event, dtype=int)
    event_times = np.sort(np.unique(duration[event == 1]))

    times = [0.0]
    survival = [1.0]
    n_at_risk = [len(duration)]
    n_events = [0]
    greenwood_terms = 0.0
    variances = [0.0]
    surv = 1.0

    for t in event_times:
        n_i = int((duration >= t).sum())
        d_i = int(((duration == t) & (event == 1)).sum())
        if n_i == 0:
            continue
        surv *= (1 - d_i / n_i)
        if n_i > d_i:
            greenwood_terms += d_i / (n_i * (n_i - d_i))
        times.append(float(t))
        survival.append(surv)
        n_at_risk.append(n_i)
        n_events.append(d_i)
        variances.append((surv ** 2) * greenwood_terms)

    out = pd.DataFrame({
        "time": times, "survival": survival,
        "n_at_risk": n_at_risk, "n_events": n_events, "greenwood_var": variances,
    })
    out["se"] = np.sqrt(out["greenwood_var"])
    out["ci_lower"] = np.clip(out["survival"] - 1.96 * out["se"], 0, 1)
    out["ci_upper"] = np.clip(out["survival"] + 1.96 * out["se"], 0, 1)
    return out


def km_survival_at(km_df: pd.DataFrame, t: float) -> float:
    sub = km_df[km_df["time"] <= t]
    return float(sub["survival"].iloc[-1])


def km_median_survival(km_df: pd.DataFrame) -> float:
    below = km_df[km_df["survival"] <= 0.5]
    return float(below["time"].iloc[0]) if len(below) else float("inf")


def logrank_test_2groups(dur_a, ev_a, dur_b, ev_b):
    dur_a, ev_a = np.asarray(dur_a, float), np.asarray(ev_a, int)
    dur_b, ev_b = np.asarray(dur_b, float), np.asarray(ev_b, int)
    all_event_times = np.sort(np.unique(np.concatenate([dur_a[ev_a == 1], dur_b[ev_b == 1]])))

    O_A = E_A = V = 0.0
    for t in all_event_times:
        n_a = int((dur_a >= t).sum())
        n_b = int((dur_b >= t).sum())
        n = n_a + n_b
        d_a = int(((dur_a == t) & (ev_a == 1)).sum())
        d_b = int(((dur_b == t) & (ev_b == 1)).sum())
        d = d_a + d_b
        if n <= 1 or d == 0:
            continue
        e_a = d * n_a / n
        v = d * (n_a / n) * (n_b / n) * (n - d) / (n - 1) if n > 1 else 0.0
        O_A += d_a
        E_A += e_a
        V += v

    chi2_stat = (O_A - E_A) ** 2 / V if V > 0 else 0.0
    p_value = 1 - chi2_dist.cdf(chi2_stat, df=1)
    return chi2_stat, p_value


# ---- validation against hand-computed toy examples ----
if __name__ == "__main__":
    # Example 1: no censoring, 5 units, durations 1..5, all events.
    km1 = kaplan_meier([1, 2, 3, 4, 5], [1, 1, 1, 1, 1])
    expected1 = [1.0, 0.8, 0.6, 0.4, 0.2, 0.0]
    assert np.allclose(km1["survival"].to_numpy(), expected1), km1

    # Example 2: with censoring, hand-computed above.
    km2 = kaplan_meier([1, 2, 2, 3, 4], [1, 0, 1, 1, 1])
    expected2 = [1.0, 0.8, 0.6, 0.3, 0.0]
    assert np.allclose(km2["survival"].to_numpy(), expected2), km2

    # Example 3: log-rank on two identical groups -> p close to 1.
    rng = np.random.default_rng(0)
    d = rng.exponential(10, size=200)
    e = np.ones(200, dtype=int)
    chi2_same, p_same = logrank_test_2groups(d, e, d, e)
    assert p_same > 0.99, (chi2_same, p_same)

    # Example 4: log-rank on clearly different groups -> small p.
    d_low = rng.exponential(5, size=200)
    d_high = rng.exponential(20, size=200)
    e2 = np.ones(200, dtype=int)
    chi2_diff, p_diff = logrank_test_2groups(d_low, e2, d_high, e2)
    assert p_diff < 0.001, (chi2_diff, p_diff)

    print("All KM / log-rank validation checks passed.")
    print(f"  Same-group log-rank: chi2={chi2_same:.4f}, p={p_same:.4f}")
    print(f"  Different-group log-rank: chi2={chi2_diff:.2f}, p={p_diff:.2e}")
