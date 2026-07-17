# Technical Note - Phase 3: Kaplan-Meier Baseline on Real C-MAPSS Data

## Objective

Validate the landmark-30 survival framing on the real NASA C-MAPSS FD001 dataset and
establish the first statistical baseline: overall survival curve, censoring structure,
and an exploratory risk-group comparison.

## Data

- Source: NASA C-MAPSS FD001 (single operating condition, single fault mode).
- 100 training units (run to failure -> `event_observed = 1`).
- 100 test units (truncated before failure -> `event_observed = 0`, censored).
- Landmark = cycle 30. All 200 units had a complete first-30-cycle window
  (0 excluded), because the shortest trajectory in FD001 is well above 30 cycles.
- Overall censoring rate in the landmark modeling table: 50.0% (by construction of the
  train/test split, not a natural failure base rate -- this should be stated explicitly
  wherever this number is reported, since a reader could otherwise mistake it for a
  real-world failure prevalence).

## Baseline Survival Curve (Landmark-Conditioned)

The Kaplan-Meier estimator was applied to `time_after_landmark` /
`event_observed`, i.e. survival time measured from the cycle-30 landmark onward.

| Horizon after landmark | Survival probability | Units still at risk |
|---|---|---|
| +60 cycles  | 100.0% | 200 |
| +100 cycles | 99.4%  | 156 |
| +150 cycles | 77.4%  | 90  |
| +200 cycles | 30.9%  | 29  |
| +250 cycles | 11.4%  | 10  |
| +300 cycles | 4.8%   | 4   |

Median residual survival time after the landmark: **172 cycles**.
Earliest real failure observed after the landmark: **+98 cycles** (unit `train_39`).

**Reliability note:** the risk set shrinks sharply beyond +200 cycles (29 units, then 10,
then 4). Estimates in that region carry wide uncertainty and should be presented with
their confidence band, not as point estimates, in any published version of this curve.

## Covariate Screening

Before building the Cox model (Phase 4), each early-window covariate
(`mean_<sensor>_first_30`, `slope_<sensor>_first_30`) was screened for association with
`time_after_landmark`, using Spearman correlation restricted to the training split (all
units there have a real, uncensored failure time, which keeps this screening step simple
and avoids conflating it with censoring adjustments that belong in the Cox model itself).

Top associated covariates (all p < 0.05):

| Covariate | Spearman rho | p-value |
|---|---|---|
| mean_sensor_3_first_30  | -0.271 | 0.0064 |
| mean_sensor_4_first_30  | -0.261 | 0.0088 |
| mean_sensor_15_first_30 | -0.247 | 0.0133 |
| mean_sensor_7_first_30  |  0.245 | 0.0142 |
| mean_sensor_11_first_30 | -0.243 | 0.0147 |
| mean_sensor_21_first_30 |  0.242 | 0.0151 |
| mean_sensor_2_first_30  | -0.241 | 0.0156 |

No single sensor dominates -- correlations sit in a modest 0.20-0.27 band. This is
consistent with the degradation signal in FD001 being distributed across several sensors
rather than concentrated in one, and should temper any claim of a single "smoking gun"
variable. This diffuse-signal pattern is itself a legitimate, reportable finding, not a
weakness of the screening: it is exactly why a multivariate Cox model, not a single
threshold rule, is the right next step.

23 engineered covariates (mostly the three operating-condition settings and sensors
1, 5, 6, 10, 16, 18, 19) were flagged as near-zero variance in FD001 and excluded from
this screening and from the downstream Cox model.

## Exploratory Risk Groups

Units were split into terciles of `mean_sensor_3_first_30` (the top-ranked covariate)
purely for visualization -- this is descriptive, not the final model.

Log-rank test, low vs. high tercile: **chi2 = 6.08, p = 0.0137**.

The difference in residual survival between engines with low vs. high early sensor-3
readings is statistically significant at the 5% level. This is a first, honest piece of
evidence that the first 30 cycles carry real prognostic information -- exactly the
question the landmark framing was designed to answer.

## Scope of This Baseline

- This phase does not adjust for multiple covariates simultaneously; Phase 4 adds the Cox model.
- The proportional-hazards diagnostics are reported in Phase 4.
- Formal discrimination, calibration, and model comparisons are reported in Phase 5.
- The tercile split is exploratory, not a proposed decision rule.

## Subsequent Work

Phase 4 fit and diagnosed the Cox proportional hazards model. Phase 5 then evaluated
Cox, Random Survival Forest, and gradient boosting survival models on the held-out NASA
test engines.

## Implementation Note

Kaplan-Meier and the log-rank test are implemented from first principles in
`src/km_utils.py` (product-limit estimator with Greenwood's variance and a
Mantel-Haenszel log-rank test). The implementation is covered by no-censoring,
censoring, identical-group, and clearly-separated-group test cases before being applied
to FD001. The repository keeps this implementation to make the statistical mechanics
auditable; `lifelines` remains available as the reference library for extensions.
