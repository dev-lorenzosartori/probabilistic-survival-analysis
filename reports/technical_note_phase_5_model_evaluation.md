# Technical Note — Phase 5: External Model Evaluation

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
| Gradient Boosting Survival | 0.553 [0.484, 0.617] | 0.577 | 0.183 | 0.124 |
| Cox PH | 0.551 [0.478, 0.619] | 0.553 | 0.176 | 0.062 |
| Random Survival Forest | 0.546 [0.478, 0.607] | 0.551 | 0.182 | 0.092 |

Higher is better for C-index/AUC; lower is better for Brier/calibration. Intervals use 1,000 bootstrap resamples.

## Interpretation
Differences are modest and confidence intervals overlap. Cox remains the most defensible interpretable baseline when probability accuracy and calibration matter.

## Caveats
FD001 is a benchmark simulation, not a live fleet. Business thresholds require real costs, lead times, interventions and validation under the fleet's censoring policy. Hyperparameters were fixed before held-out evaluation; no test-set tuning was performed.

## Reproduce
```bash
make evaluate
```
