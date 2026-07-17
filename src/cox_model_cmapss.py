"""Phase 4: Cox proportional hazards model on the C-MAPSS landmark-30 table.

Workflow (each step is deliberate, not just "throw covariates at a model"):

1. Load the 25 non-low-variance covariates and check events-per-variable (EPV),
   VIF, and pairwise correlation -- BEFORE fitting anything.
2. Fit the naive full-covariate Cox model anyway, as a documented diagnostic:
   with ~100 events and 25 covariates (EPV ~= 4, well under the ~10 rule of
   thumb), we expect instability. This is recorded, not hidden.
3. Screen covariates univariately (Spearman vs. time_after_landmark, train
   split only). The significant ones turn out to be highly inter-correlated
   (PC1 explains >95% of their shared variance) -- i.e. ~1 latent "early
   degradation level" factor, not 12 independent signals.
4. Build a composite degradation index (PC1) from those correlated sensors,
   and look for a second, genuinely uncorrelated covariate (low |corr| with
   PC1) among the slope_ (early trend) features.
5. Fit the final, well-conditioned 2-covariate Cox model and formally check
   the proportional-hazards assumption via Schoenfeld residuals.

Uses `src/cox_utils.py` (from-scratch Cox + Schoenfeld implementation,
validated against synthetic ground truth -- see cox_utils.py __main__).
Swap in `lifelines` for the published notebook once you have network access.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression

from src.cox_utils import fit_cox_ph, ph_assumption_test

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELING_TABLE_PATH = PROJECT_ROOT / "data" / "processed" / "cmapss_fd001_landmark30_modeling_table.csv"
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


def compute_vif(Xz: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in Xz.columns:
        y = Xz[col].to_numpy()
        Xo = Xz.drop(columns=[col]).to_numpy()
        r2 = LinearRegression().fit(Xo, y).score(Xo, y)
        vif = 1 / (1 - r2) if r2 < 0.9999 else np.inf
        rows.append({"covariate": col, "VIF": vif})
    return pd.DataFrame(rows).sort_values("VIF", ascending=False)


def screen_univariate(df: pd.DataFrame, candidate_cols: list[str]) -> pd.DataFrame:
    train = df[df["split"] == "train"]
    rows = []
    for col in candidate_cols:
        rho, p_value = spearmanr(train[col], train["time_after_landmark"])
        rows.append({"covariate": col, "rho": rho, "p_value": p_value})
    return pd.DataFrame(rows).sort_values("rho", key=lambda s: s.abs(), ascending=False)


def build_degradation_index(df: pd.DataFrame, sensor_cols: list[str]) -> tuple[pd.Series, float]:
    """First principal component of the given (standardized) sensor columns,
    oriented so higher = more degraded (shorter residual survival)."""
    X = df[sensor_cols]
    Xz = ((X - X.mean()) / X.std()).to_numpy()
    U, S, _ = np.linalg.svd(Xz, full_matrices=False)
    pc1 = U[:, 0] * S[0]
    if np.corrcoef(pc1, Xz[:, 0])[0, 1] < 0:
        pc1 = -pc1
    explained = (S ** 2 / (S ** 2).sum())[0]
    return pd.Series(pc1, index=df.index, name="pc1_degradation_index"), explained


def main() -> None:
    df = pd.read_csv(MODELING_TABLE_PATH)
    durations = df["time_after_landmark"].to_numpy(dtype=float)
    events = df["event_observed"].to_numpy(dtype=int)

    candidate_cols = [
        c for c in df.columns
        if c.startswith(("mean_", "slope_")) and c not in LOW_VARIANCE_COVARIATES
    ]
    n_events = int(events.sum())
    print(f"Step 1: {len(candidate_cols)} candidate covariates, {n_events} events "
          f"-> EPV = {n_events/len(candidate_cols):.1f} (rule of thumb: >=10 recommended)")

    Xz_full = (df[candidate_cols] - df[candidate_cols].mean()) / df[candidate_cols].std()
    vif = compute_vif(Xz_full)
    print(f"Top VIF: {vif.iloc[0]['covariate']} = {vif.iloc[0]['VIF']:.1f} (>10 = severe multicollinearity)")

    print("\nStep 2: naive full-covariate Cox fit (documented diagnostic, not the final model)")
    fit_naive = fit_cox_ph(Xz_full.to_numpy(), durations, events, candidate_cols)
    print(f"  Converged: {fit_naive['converged']} | Hessian condition number: {fit_naive['hessian_condition_number']:.2e}")
    widest_ci = fit_naive["summary"].assign(
        ci_width=lambda d: d["hr_ci_upper"] - d["hr_ci_lower"]
    ).sort_values("ci_width", ascending=False).iloc[0]
    print(f"  Widest 95% HR CI: {widest_ci['covariate']} = [{widest_ci['hr_ci_lower']:.2f}, {widest_ci['hr_ci_upper']:.2f}]")

    print("\nStep 3: univariate screening (train split) + correlation structure")
    screen = screen_univariate(df, candidate_cols)
    significant = screen[screen["p_value"] < 0.05]
    print(f"  {len(significant)} covariates significant at p<0.05")

    sig_cols = significant["covariate"].tolist()
    Xz_sig = (df[sig_cols] - df[sig_cols].mean()) / df[sig_cols].std()
    eigvals = np.linalg.eigvalsh(Xz_sig.corr().to_numpy())[::-1]
    print(f"  PC1 of the {len(sig_cols)} significant covariates explains {eigvals[0]/len(sig_cols):.1%} of their variance")

    print("\nStep 4: build composite degradation index + find an uncorrelated second axis")
    pc1, explained = build_degradation_index(df, sig_cols)
    df["pc1_degradation_index"] = pc1

    slope_candidates = [c for c in df.columns if c.startswith("slope_") and c not in LOW_VARIANCE_COVARIATES]
    corr_with_pc1 = {c: abs(np.corrcoef(df[c], pc1)[0, 1]) for c in slope_candidates}
    second_axis = min(corr_with_pc1, key=corr_with_pc1.get)
    print(f"  Least correlated slope covariate with PC1: {second_axis} (|corr|={corr_with_pc1[second_axis]:.3f})")

    final_cols = ["pc1_degradation_index", second_axis]
    X_final = df[final_cols].to_numpy(dtype=float)
    Xz_final = (X_final - X_final.mean(axis=0)) / X_final.std(axis=0)

    print(f"\nStep 5: final model -- {final_cols}")
    fit_final = fit_cox_ph(Xz_final, durations, events, final_cols)
    print(f"  Converged: {fit_final['converged']} | Hessian condition number: {fit_final['hessian_condition_number']:.2f}")
    print(fit_final["summary"].to_string(index=False))

    ph = ph_assumption_test(fit_final["beta"], Xz_final, durations, events, final_cols)
    print("\n  Proportional-hazards check (Schoenfeld residual correlation with time):")
    print(ph.to_string(index=False))

    # forest plot
    s = fit_final["summary"]
    fig, ax = plt.subplots(figsize=(8, 3.5))
    y_pos = np.arange(len(s))[::-1]
    ax.errorbar(s["hazard_ratio"], y_pos,
                xerr=[s["hazard_ratio"] - s["hr_ci_lower"], s["hr_ci_upper"] - s["hazard_ratio"]],
                fmt="o", color="#2b6cb0", ecolor="#2b6cb0", capsize=4, markersize=8)
    ax.axvline(1.0, color="gray", linestyle="--", linewidth=1)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(final_cols)
    ax.set_xlabel("Hazard ratio (per 1-SD increase, 95% CI)")
    ax.set_title("Cox Model: Hazard Ratios, Landmark-30 C-MAPSS FD001")
    ax.grid(alpha=0.25, axis="x")
    fig.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / "cmapss_cox_hazard_ratios.png", dpi=160)
    plt.close(fig)

    df.to_csv(MODELING_TABLE_PATH, index=False)
    print(f"\nSaved updated modeling table (with pc1_degradation_index) to {MODELING_TABLE_PATH}")


if __name__ == "__main__":
    main()
