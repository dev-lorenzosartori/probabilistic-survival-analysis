# Dataset Plan

## Recommended Primary Dataset

NASA C-MAPSS Turbofan Engine Degradation dataset.

## Why This Dataset

The dataset is a recognized benchmark for prognostics and health management. It contains
engine-level operational cycles, sensor readings, operating settings, and failure-related
trajectories. This makes it a strong candidate for a survival-analysis portfolio project
because it connects engineering, time-to-event modeling, predictive maintenance, and
production-style model monitoring.

## Survival Analysis Framing

Each engine is treated as an entity observed over time.

The survival-analysis table should contain:

- `engine_id`: unique engine identifier
- `duration`: observed number of cycles
- `event_observed`: 1 if failure is observed, 0 if censored
- engineered covariates from sensors and operating settings

## Training Data Logic

In C-MAPSS training data, engines usually run until failure. Therefore:

- duration = last observed cycle for each engine
- event_observed = 1

Covariates can be engineered from early-life or recent-window sensor summaries, such as:

- mean sensor value in first N cycles
- slope/trend over the first N cycles
- rolling-window statistics
- degradation indicators
- operating condition summaries

## Test Data Logic

In C-MAPSS test data, engines are stopped before failure and the true remaining useful life
is provided separately. For survival framing:

- duration = last observed cycle in the test sequence
- event_observed = 0 if treated as censored at observation cutoff
- optional true event time = observed cycles + true RUL

This creates a useful comparison between censored survival modeling and remaining useful
life prediction.

## First Modeling Table

The first real modeling table should be asset-level, not cycle-level.

Example columns:

- engine_id
- duration
- event_observed
- mean_sensor_2_first_30
- slope_sensor_2_first_30
- mean_sensor_3_first_30
- slope_sensor_3_first_30
- mean_sensor_4_first_30
- slope_sensor_4_first_30
- mean_operational_setting_1
- mean_operational_setting_2
- mean_operational_setting_3

## Why Asset-Level First

Starting at asset level keeps the Cox model interpretable. Later phases can introduce
time-varying covariates or sequence models.

## Key Risks

- Some sensors may have low variance and should be removed.
- Cox proportional hazards assumptions may not hold for all covariates.
- Feature engineering must avoid leakage from cycles too close to failure.
- The portfolio should clearly separate baseline statistical modeling from advanced
  sequence modeling.

## First Decision

Use the first 30 operational cycles to create early-warning covariates.

Reason:

This simulates a realistic question:

> Based on early operational behavior, which assets are likely to fail earlier?

This framing avoids using information from the end of life that would not be available
when making preventive decisions.


## Landmark-Window Decision

The transformation now keeps two layers:

- a full audit survival table, preserving every unit and marking `has_full_window`;
- a landmark-30 modeling table, excluding units without a complete first-30-cycle observation window.

This is treated as a cycle-30 landmark model. The business question is not “can we predict
from any amount of history?”, but rather:

> Given that an engine has reached cycle 30, and given what was observed in its first 30
> cycles, what is its future survival profile?

This keeps the statistical framing clean and avoids mixing covariates computed over different
observation horizons.
