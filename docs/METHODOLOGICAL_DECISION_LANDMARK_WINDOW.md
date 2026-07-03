# Methodological Decision: Early-Window Landmark Modeling

## Decision

The baseline C-MAPSS survival model will use a fixed early observation window of 30 cycles.
Units with fewer than 30 observed cycles will be kept in the full audit table, but excluded
from the first baseline modeling table.

## Rationale

The model is framed as a landmark analysis:

> Given the first 30 operating cycles, estimate the probability that an engine survives
> beyond future cycle horizons.

Under this framing, a unit that failed before cycle 30 was never eligible for a prediction
at the cycle-30 landmark. Keeping it in the model with a shorter covariate window would mix
different information horizons and make coefficients harder to interpret.

## Implementation

The transformation script creates two outputs:

1. `cmapss_fd001_survival_table_full.csv`
   - Includes all train and test units.
   - Keeps `has_full_window` and `is_model_eligible` for auditability.

2. `cmapss_fd001_landmark30_modeling_table.csv`
   - Includes only units with a complete first-30-cycle window.
   - Adds `time_after_landmark = duration - 30`.
   - Uses the same event indicator: train units are observed failures, test units are right-censored.

## Why Not Silently Keep Partial Windows?

Keeping partial windows would increase sample size, but it would introduce inconsistent
feature definitions. For example, a mean sensor value over 12 cycles is not directly
comparable to a mean sensor value over 30 cycles when the model is supposed to represent a
cycle-30 decision point.

## Sensitivity Analysis Later

A later project phase may compare this baseline with shorter landmark windows, such as 10
or 20 cycles. That would allow early failures to be modeled under an explicitly different
business question.
