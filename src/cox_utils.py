"""Auditable Cox proportional hazards regression (Efron partial likelihood).

Validated on synthetic known-beta and PH-violation scenarios. Final FD001
coefficients are regression-tested against ``lifelines``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import optimize, stats


def _neg_log_partial_likelihood(beta: np.ndarray, X: np.ndarray, durations: np.ndarray, events: np.ndarray) -> float:
    """Efron-approximation negative log partial likelihood."""
    event_times = np.unique(durations[events == 1])
    ll = 0.0
    xb = X @ beta
    exp_xb = np.exp(xb)

    for t in event_times:
        risk_mask = durations >= t
        death_mask = (durations == t) & (events == 1)
        d = int(death_mask.sum())

        s_beta = xb[death_mask].sum()
        risk_sum = exp_xb[risk_mask].sum()
        death_sum = exp_xb[death_mask].sum()

        ll += s_beta
        for l in range(d):
            ll -= np.log(risk_sum - (l / d) * death_sum)

    return -ll


def _hessian_fd(func, x: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    """Central-difference Hessian of a scalar function."""
    n = len(x)
    H = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            xpp, xpm, xmp, xmm = x.copy(), x.copy(), x.copy(), x.copy()
            xpp[i] += eps; xpp[j] += eps
            xpm[i] += eps; xpm[j] -= eps
            xmp[i] -= eps; xmp[j] += eps
            xmm[i] -= eps; xmm[j] -= eps
            H[i, j] = H[j, i] = (func(xpp) - func(xpm) - func(xmp) + func(xmm)) / (4 * eps ** 2)
    return H


def fit_cox_ph(X: np.ndarray, durations: np.ndarray, events: np.ndarray, covariate_names: list[str]) -> dict:
    """Fit a Cox PH model. Returns beta, se, covariance matrix, and a summary table."""
    n_cov = X.shape[1]
    x0 = np.zeros(n_cov)

    objective = lambda b: _neg_log_partial_likelihood(b, X, durations, events)
    result = optimize.minimize(objective, x0, method="BFGS", options={"gtol": 1e-8, "maxiter": 2000})
    if not result.success:
        # retry from the BFGS endpoint with Nelder-Mead as a robustness fallback
        result = optimize.minimize(objective, result.x, method="Nelder-Mead",
                                    options={"xatol": 1e-8, "fatol": 1e-8, "maxiter": 5000})

    beta_hat = result.x
    hessian = _hessian_fd(objective, beta_hat)

    try:
        cov = np.linalg.inv(hessian)
        se = np.sqrt(np.diag(cov))
        condition_number = np.linalg.cond(hessian)
    except np.linalg.LinAlgError:
        cov = np.full((n_cov, n_cov), np.nan)
        se = np.full(n_cov, np.nan)
        condition_number = np.inf

    z = beta_hat / se
    p_value = 2 * (1 - stats.norm.cdf(np.abs(z)))

    summary = pd.DataFrame({
        "covariate": covariate_names,
        "beta": beta_hat,
        "se": se,
        "hazard_ratio": np.exp(beta_hat),
        "hr_ci_lower": np.exp(beta_hat - 1.96 * se),
        "hr_ci_upper": np.exp(beta_hat + 1.96 * se),
        "z": z,
        "p_value": p_value,
    })

    return {
        "beta": beta_hat,
        "se": se,
        "cov": cov,
        "summary": summary,
        "converged": result.success,
        "hessian_condition_number": condition_number,
        "log_likelihood": -result.fun,
    }


def schoenfeld_residuals(beta: np.ndarray, X: np.ndarray, durations: np.ndarray, events: np.ndarray) -> pd.DataFrame:
    """Unscaled Schoenfeld residuals, one row per event (ties handled individually)."""
    xb = X @ beta
    exp_xb = np.exp(xb)
    rows = []

    event_idx = np.where(events == 1)[0]
    for i in event_idx:
        t = durations[i]
        risk_mask = durations >= t
        w = exp_xb[risk_mask]
        xr = X[risk_mask]
        x_bar = (w[:, None] * xr).sum(axis=0) / w.sum()
        residual = X[i] - x_bar
        rows.append({"time": t, **{f"r_{k}": residual[k] for k in range(X.shape[1])}})

    return pd.DataFrame(rows)


def ph_assumption_test(beta: np.ndarray, X: np.ndarray, durations: np.ndarray, events: np.ndarray, covariate_names: list[str]) -> pd.DataFrame:
    """Approximate proportional-hazards check: correlation of (unscaled) Schoenfeld
    residuals with event-time rank. A significant correlation suggests the covariate's
    effect changes over time (PH violated).

    This is a simplified stand-in for the scaled-residual chi-square test used by
    `cox.zph` in R / `check_assumptions` in lifelines. Re-run with lifelines for the
    published notebook to get the full Grambsch-Therneau test.
    """
    resid = schoenfeld_residuals(beta, X, durations, events)
    resid["time_rank"] = resid["time"].rank()

    rows = []
    for k, name in enumerate(covariate_names):
        rho, p_value = stats.pearsonr(resid["time_rank"], resid[f"r_{k}"])
        rows.append({"covariate": name, "corr_with_time": rho, "p_value": p_value})

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Validation against synthetic data with known ground truth.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    rng = np.random.default_rng(42)

    # --- Check 1: large-sample recovery of a known beta ---
    n = 3000
    X1 = rng.normal(0, 1, n)
    X2 = rng.normal(0, 1, n)
    true_beta = np.array([0.70, -0.40])
    X = np.column_stack([X1, X2])

    baseline_rate = 0.05
    U = rng.uniform(size=n)
    event_time = -np.log(U) / (baseline_rate * np.exp(X @ true_beta))
    censor_time = rng.exponential(scale=1 / (baseline_rate * 0.6), size=n)
    durations = np.minimum(event_time, censor_time)
    events = (event_time <= censor_time).astype(int)

    fit = fit_cox_ph(X, durations, events, ["x1", "x2"])
    recovered = fit["beta"]
    print("Check 1: recovered beta vs true beta")
    print(f"  true={true_beta}, recovered={np.round(recovered, 3)}, converged={fit['converged']}")
    assert np.allclose(recovered, true_beta, atol=0.10), "Cox beta recovery failed"

    # --- Check 2: PH holds -> Schoenfeld test should NOT reject for most covariates ---
    ph_check = ph_assumption_test(fit["beta"], X, durations, events, ["x1", "x2"])
    print("\nCheck 2: PH holds by construction -> expect large p-values")
    print(ph_check.to_string(index=False))
    assert (ph_check["p_value"] > 0.01).all(), "False positive PH violation on PH-holding data"

    # --- Check 3: PH violated by construction (piecewise time-varying effect for x1) ---
    # Genuine non-proportional hazards: the hazard ratio for x1 flips from exp(+1.2) to
    # exp(-0.8) at tau, while x2 keeps a constant, PH-respecting effect throughout.
    beta1_early, beta1_late = 1.2, -0.8
    beta2_const = -0.4
    tau = 8.0

    U2 = rng.uniform(size=n)
    neg_log_u = -np.log(U2)
    rate_early = baseline_rate * np.exp(beta1_early * X1 + beta2_const * X2)
    rate_late = baseline_rate * np.exp(beta1_late * X1 + beta2_const * X2)
    cum_haz_at_tau = rate_early * tau

    event_time_tv = np.where(
        neg_log_u <= cum_haz_at_tau,
        neg_log_u / rate_early,
        tau + (neg_log_u - cum_haz_at_tau) / rate_late,
    )
    censor_time_tv = rng.exponential(scale=30.0, size=n)
    durations_tv = np.minimum(event_time_tv, censor_time_tv)
    events_tv = (event_time_tv <= censor_time_tv).astype(int)

    fit_tv = fit_cox_ph(np.column_stack([X1, X2]), durations_tv, events_tv, ["x1", "x2"])
    ph_check_tv = ph_assumption_test(fit_tv["beta"], np.column_stack([X1, X2]), durations_tv, events_tv, ["x1", "x2"])
    print("\nCheck 3: PH violated by construction for x1 (constant for x2) -> expect small p-value for x1, large for x2")
    print(ph_check_tv.to_string(index=False))
    assert ph_check_tv.loc[ph_check_tv.covariate == "x1", "p_value"].iloc[0] < 0.01, "Failed to detect true PH violation"
    assert ph_check_tv.loc[ph_check_tv.covariate == "x2", "p_value"].iloc[0] > 0.05, "False positive PH violation on constant-effect covariate"

    print("\nAll Cox / Schoenfeld validation checks passed.")
