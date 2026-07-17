# Technical Note - Phase 1 Baseline

## Objective

The first baseline validates the survival-analysis workflow using a synthetic
predictive-maintenance dataset with censored observations.

The goal is not to claim final business findings yet. The goal is to prove that
the project structure can estimate survival probabilities, compare risk groups,
and generate interpretable statistical outputs.

## Dataset

The starter dataset contains 1,200 synthetic industrial assets.

Each row represents one asset and includes:

- `duration`: observed operational time until failure or censoring
- `event_observed`: 1 if failure was observed, 0 if the observation was censored
- `asset_age`
- `avg_temperature`
- `vibration_score`
- `load_factor`
- `maintenance_quality`

Observed event rate: 78.2%.

## Baseline Method

The Kaplan-Meier estimator was used to estimate the survival function.

The survival function answers:

> What is the probability that an asset remains operational beyond time t?

## Initial Results

- Estimated survival probability at 60 operational cycles: 42.2%
- Median survival time: 53.13 operational cycles
- Log-rank test comparing low vs high maintenance quality: p-value below 0.000001

## Interpretation

The baseline result shows a clear separation between assets with low, medium,
and high maintenance quality.

Assets in the low-maintenance group lose survival probability faster, while
high-maintenance assets remain operational for longer periods.

This is exactly the type of analysis that a binary classifier would not express
as clearly. A classifier could estimate whether failure is likely, but the
survival model shows how risk evolves over time.

## Subsequent Work

The subsequent phases replaced the synthetic proof of concept with NASA C-MAPSS FD001
and defined a defensible transformation into survival format:

- entity identifier
- duration
- event indicator
- covariates
- censoring logic

The completed case now includes a landmark Kaplan-Meier baseline, a diagnosed Cox
model, and held-out comparison with Random Survival Forest and gradient boosting.
