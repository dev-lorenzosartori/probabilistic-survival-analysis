"""Leakage-safe external evaluation on NASA C-MAPSS FD001."""

from __future__ import annotations
import argparse
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sksurv.ensemble import GradientBoostingSurvivalAnalysis, RandomSurvivalForest
from sksurv.linear_model import CoxPHSurvivalAnalysis
from src.evaluation_utils import (bootstrap_c_index, brier_metrics, calibration_table,
    harrell_c_index, make_survival_target, survival_probability_matrix, time_dependent_auc_table)
from src.feature_engineering import LandmarkFeatureTransformer

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/processed/cmapss_fd001_landmark30_modeling_table.csv"
REPORTS, FIGURES, MODELS = ROOT / "reports", ROOT / "reports/figures", ROOT / "models"
HORIZONS = np.asarray([120., 150., 180., 210., 240.])
CAL_HORIZON = 180.


def load_external_evaluation_data():
    df = pd.read_csv(DATA)
    df = df[df["is_model_eligible"].astype(bool)].copy()
    train, test = df[df.split == "train"].copy(), df[df.split == "test"].copy()
    if len(train) != 100 or len(test) != 100:
        raise ValueError("FD001 requires 100 train and 100 test units")
    if not train.event_observed.eq(1).all() or not test.event_observed.eq(0).all():
        raise ValueError("Unexpected FD001 event/censoring contract")
    if test.true_event_time.isna().any():
        raise ValueError("NASA RUL labels are required for external evaluation")
    y_train = make_survival_target(np.ones(len(train), bool), train.time_after_landmark)
    y_test = make_survival_target(np.ones(len(test), bool),
                                  test.true_event_time - test.landmark_cycle)
    return train, y_train, test, y_test


def build_models():
    estimators = {
        "Cox PH": CoxPHSurvivalAnalysis(alpha=.01),
        "Random Survival Forest": RandomSurvivalForest(n_estimators=500,
            min_samples_split=10, min_samples_leaf=5, max_features="sqrt",
            n_jobs=-1, random_state=42),
        "Gradient Boosting Survival": GradientBoostingSurvivalAnalysis(loss="coxph",
            n_estimators=250, learning_rate=.05, max_depth=2, min_samples_leaf=8,
            random_state=42),
    }
    return {name: Pipeline([("features", LandmarkFeatureTransformer()), ("model", model)])
            for name, model in estimators.items()}


def evaluate(n_bootstrap=1_000):
    train, y_train, test, y_test = load_external_evaluation_data()
    FIGURES.mkdir(parents=True, exist_ok=True); MODELS.mkdir(parents=True, exist_ok=True)
    rows, aucs, briers, calibrations = [], [], [], []
    for name, pipe in build_models().items():
        pipe.fit(train, y_train)
        risk = pipe.predict(test)
        surv = survival_probability_matrix(pipe, test, HORIZONS)
        c = harrell_c_index(y_test, risk)
        low, high = bootstrap_c_index(y_test, risk, n_bootstrap=n_bootstrap)
        auc, mean_auc = time_dependent_auc_table(y_train, y_test, risk, HORIZONS)
        brier, ibs = brier_metrics(y_train, y_test, surv, HORIZONS)
        idx = int(np.where(HORIZONS == CAL_HORIZON)[0][0])
        cal = calibration_table(y_test, surv[:, idx], horizon=CAL_HORIZON)
        rows.append({"model": name, "c_index": c, "c_index_ci_low": low,
            "c_index_ci_high": high, "mean_time_dependent_auc": mean_auc,
            "integrated_brier_score": ibs,
            "calibration_mae_180": float(cal.absolute_gap.mean())})
        aucs.append(auc.assign(model=name)); briers.append(brier.assign(model=name))
        calibrations.append(cal.assign(model=name))
        joblib.dump(pipe, MODELS / (name.lower().replace(" ", "_") + "_landmark30.joblib"))
    comparison = pd.DataFrame(rows).sort_values(["c_index", "integrated_brier_score"],
                                                ascending=[False, True])
    auc_results, brier_results = pd.concat(aucs), pd.concat(briers)
    cal_results = pd.concat(calibrations)
    comparison.to_csv(REPORTS / "phase5_model_comparison.csv", index=False)
    auc_results.to_csv(REPORTS / "phase5_time_dependent_auc.csv", index=False)
    brier_results.to_csv(REPORTS / "phase5_brier_scores.csv", index=False)
    cal_results.to_csv(REPORTS / "phase5_calibration_180_cycles.csv", index=False)
    render_figures(comparison, auc_results, cal_results)
    write_note(comparison, n_bootstrap)
    return comparison


def render_figures(comparison, aucs, calibration):
    colors = {"Cox PH":"#2563eb", "Random Survival Forest":"#7c3aed",
              "Gradient Boosting Survival":"#059669"}
    ordered = comparison.sort_values("c_index")
    fig, ax = plt.subplots(figsize=(8.5,4.2)); y = np.arange(len(ordered))
    ax.errorbar(ordered.c_index, y,
        xerr=np.vstack([ordered.c_index-ordered.c_index_ci_low,
                        ordered.c_index_ci_high-ordered.c_index]), fmt="none",
        ecolor="#94a3b8", capsize=4, linewidth=2)
    for pos, row in zip(y, ordered.itertuples(), strict=True):
        ax.scatter(row.c_index, pos, s=90, color=colors[row.model], zorder=3)
        ax.text(row.c_index+.008, pos, f"{row.c_index:.3f}", va="center")
    ax.axvline(.5, color="#64748b", ls="--", lw=1)
    ax.set_yticks(y, ordered.model); ax.set_xlim(.45,.75)
    ax.set_xlabel("Harrell's C-index on held-out NASA test set (95% bootstrap CI)")
    ax.set_title("External discrimination is modest and uncertainty is visible")
    ax.grid(axis="x", alpha=.2); fig.tight_layout()
    fig.savefig(FIGURES/"phase5_model_comparison.png", dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5,4.8))
    for name, group in aucs.groupby("model"):
        ax.plot(group.horizon, group.auc, marker="o", lw=2, label=name, color=colors[name])
    ax.axhline(.5,color="#64748b",ls="--",lw=1); ax.set_ylim(.4,.8)
    ax.set_xlabel("Cycles after cycle-30 landmark"); ax.set_ylabel("Cumulative/dynamic AUC")
    ax.set_title("Discrimination by decision horizon"); ax.legend(frameon=False); ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(FIGURES/"phase5_time_dependent_auc.png",dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.3,6)); ax.plot([0,1],[0,1],ls="--",color="#64748b",label="Perfect calibration")
    for name, group in calibration.groupby("model"):
        ax.plot(group.predicted_survival,group.observed_survival,marker="o",lw=2,label=name,color=colors[name])
    ax.set(xlim=(0,1),ylim=(0,1),xlabel="Mean predicted probability of surviving 180 cycles",
           ylabel="Observed survival proportion",title="Quartile calibration at 180 cycles after landmark")
    ax.legend(frameon=False,fontsize=9); ax.grid(alpha=.2); fig.tight_layout()
    fig.savefig(FIGURES/"phase5_calibration_180_cycles.png",dpi=180); plt.close(fig)


def write_note(comparison, n_bootstrap):
    lines = [f"| {r.model} | {r.c_index:.3f} [{r.c_index_ci_low:.3f}, {r.c_index_ci_high:.3f}] | {r.mean_time_dependent_auc:.3f} | {r.integrated_brier_score:.3f} | {r.calibration_mae_180:.3f} |" for r in comparison.itertuples()]
    text = f"""# Technical Note — Phase 5: External Model Evaluation

## Decision question
Using only the first 30 operating cycles, how well can the model rank engines by future failure risk and estimate survival probabilities?

## Evaluation design
- Development: 100 official FD001 training engines.
- Held-out evaluation: 100 official FD001 test engines.
- NASA RUL labels reveal test event times only after fitting.
- PCA, scaling and fitting use development units exclusively.
- All models receive the same two Phase-4 features.
- Horizons: 120, 150, 180, 210 and 240 cycles after landmark.

## Results
| Model | Harrell C (95% bootstrap CI) | Mean time-dependent AUC | Integrated Brier | Calibration MAE at 180 |
|---|---:|---:|---:|---:|
{chr(10).join(lines)}

Higher is better for C-index/AUC; lower is better for Brier/calibration. Intervals use {n_bootstrap:,} bootstrap resamples.

## Interpretation
Differences are modest and confidence intervals overlap. Cox remains the most defensible interpretable baseline when probability accuracy and calibration matter.

## Caveats
FD001 is a benchmark simulation, not a live fleet. Business thresholds require real costs, lead times, interventions and validation under the fleet's censoring policy. Hyperparameters were fixed before held-out evaluation; no test-set tuning was performed.

## Reproduce
```bash
make evaluate
```
"""
    (REPORTS/"technical_note_phase_5_model_evaluation.md").write_text(text)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--bootstrap",type=int,default=1_000)
    args = parser.parse_args(); print(evaluate(args.bootstrap).to_string(index=False,float_format=lambda x:f"{x:.3f}"))
