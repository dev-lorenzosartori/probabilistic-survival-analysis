# Technical Note - Phase 4: Cox Proportional Hazards Model

## Objective

Fit a Cox PH model on the landmark-30 modeling table using the covariates that survived
the low-variance filter, with a formal check of the proportional-hazards assumption and
hazard ratios with confidence intervals -- as specified in the roadmap.

## Step 1: Check Before Fitting, Not After

25 covariates remained after removing the 23 near-zero-variance ones. With 100 observed
events, that is **4.0 events per variable (EPV)** -- well under the ~10 EPV rule of thumb
generally used as a floor for stable Cox estimation. Before fitting anything, VIF
(variance inflation factor) was computed for all 25: the worst offender
(`mean_sensor_8_first_30`) had **VIF = 83**, and 67 covariate pairs had |correlation| > 0.85.
This was a predictable consequence of building 25 covariates from 21 sensors that all
respond, to varying degrees, to the same underlying engine-degradation process.

## Step 2: The Naive Full Model, Documented as a Diagnostic

The 25-covariate Cox model was fit anyway, deliberately, to have empirical (not just
theoretical) evidence of the instability. It converged, but:

- Hessian condition number: **2.86e+03** (a well-conditioned model is close to 1).
- Widest 95% hazard-ratio confidence interval: `mean_sensor_8_first_30`,
  **HR = 5.62, 95% CI [0.89, 35.7]** -- a range spanning nearly two orders of magnitude,
  uninformative for any real decision.
- `mean_sensor_3_first_30`, the single strongest predictor in the Phase 3 univariate
  screening (Spearman rho = -0.27, p = 0.006), became statistically insignificant in the
  multivariate fit (p = 0.87). This is a textbook multicollinearity artifact: its
  predictive signal did not disappear, it was absorbed and diluted by ten other sensors
  carrying nearly the same information.

This model is not used for interpretation. It is kept in the repository as evidence for
*why* the final model looks the way it does.

## Step 3: The Covariates Are Not 12 Independent Signals

Univariate screening (Spearman, train split) found 12 covariates significant at p < 0.05,
all of them `mean_<sensor>_first_30` features. Their pairwise correlations are almost
all above 0.90 in absolute value, and a principal component analysis of just these 12
covariates shows the **first component explains 95.4% of their combined variance**.

In other words: the "12 significant sensors" are, statistically, close to a single
latent factor -- most plausibly a shared early-life engine health signal that many
sensors register simultaneously, not 12 independent risk drivers. Feeding all 12 into a
regression does not add 12 covariates' worth of information; it adds one signal measured
12 times with noise, which is exactly what produced the instability in Step 2.

## Step 4: A Defensible, Low-Dimensional Model

Rather than arbitrarily keeping one sensor and discarding eleven, the 12 correlated
covariates were combined into a single composite **early-degradation index** via PCA
(first principal component, oriented so higher values indicate more degradation /
shorter expected residual life).

A second, genuinely independent axis was then sought among the early-window `slope_`
(trend) covariates, screening for the one least correlated with the new index:
`slope_sensor_21_first_30` (|correlation with the index| = 0.011 -- effectively
orthogonal). This covariate did not reach p < 0.05 on its own in Phase 3 (p = 0.081), but
with the multicollinearity problem resolved there is ample room (EPV = 50 for a
2-covariate model) to let the multivariate fit judge its contribution properly, rather
than discarding it on a marginal univariate screen alone.

## Final Model

| Covariate | Hazard ratio (per 1-SD) | 95% CI | p-value |
|---|---|---|---|
| Early degradation index (PC1 of 12 sensors) | 1.32 | [1.09, 1.59] | 0.0046 |
| Early trend, sensor 21 (slope, first 30 cycles) | 0.79 | [0.63, 0.99] | 0.0420 |

Hessian condition number: **1.55** (versus 2,860 for the naive model) -- a well-behaved,
stable fit.

**Interpretation:**

- A one-standard-deviation increase in the early degradation index is associated with a
  32% higher hazard of failure at any given time after the cycle-30 landmark, holding
  the trend covariate fixed.
- A one-standard-deviation increase in the sensor-21 early trend is associated with a
  21% *lower* hazard -- i.e. engines whose sensor 21 rises faster in the first 30 cycles
  tend to fail later, holding the degradation index fixed. This is a directional,
  statistical finding; the physical interpretation should be checked against the
  official C-MAPSS sensor-description table before it appears in any public write-up,
  since this note does not assert what sensor 21 physically measures.
- The sensor-21 effect is only marginally significant (p = 0.042, CI upper bound 0.99)
  and should be described with that caveat, not overstated, in the published version.

## Proportional Hazards Assumption

Checked via (unscaled) Schoenfeld residual correlation with event-time rank -- see the
Implementation Note below for what this test does and does not cover.

| Covariate | Correlation with time | p-value |
|---|---|---|
| Early degradation index | -0.111 | 0.273 |
| Early trend, sensor 21 | 0.046 | 0.653 |

Neither covariate shows a significant time trend in its residuals. The proportional
hazards assumption is not rejected for either covariate in this model.

## Scope of This Model

- Concordance, calibration, and comparison against Random Survival Forest and gradient
  boosting are reported in the completed Phase 5 evaluation.
- A physically grounded name for the composite index beyond "PC1 of correlated sensors"
  -- worth revisiting once sensor physical descriptions are cross-checked.

## Implementation Note

Cox regression (Efron partial likelihood) and the Schoenfeld-residual PH check are
implemented in `src/cox_utils.py`. Before use on real data, the implementation was
validated against:

- Synthetic Cox PH data with a known true beta (large-sample recovery within ~0.07 of
  the true coefficients).
- Synthetic data engineered to satisfy PH by construction (test correctly does not
  reject).
- Synthetic data engineered to violate PH by construction via a genuine piecewise
  time-varying effect (test correctly rejects, p < 1e-100, while correctly not flagging
  the covariate whose effect was held constant).

The fitted coefficients also match the `lifelines` Efron reference within **2e-7** in
the automated test suite.

The PH check here is a simplified, unscaled version of the Schoenfeld residual test.
The standard implementation (Grambsch & Therneau, used by `cox.zph` in R and
`check_assumptions` in lifelines) scales residuals by their estimated variance-covariance
matrix and produces a chi-square test per covariate. That implementation is appropriate
for any expanded confirmatory analysis.

## Subsequent Work

Phase 5 completed the held-out comparison of this Cox baseline with Random Survival
Forest and gradient boosting survival models, including concordance, time-dependent
AUC, Brier score, and calibration.
