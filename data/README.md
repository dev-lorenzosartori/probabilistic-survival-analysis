# Data

This folder is reserved for local datasets.

Raw datasets should not be committed if they are large or license-restricted.

Planned structure:

- `raw/`: original downloaded files
- `interim/`: intermediate transformed files
- `processed/`: final modeling tables

The main modeling table should contain at least:

- `entity_id`: asset or customer identifier
- `duration`: observed time until event or censoring
- `event_observed`: 1 if event occurred, 0 if censored
- covariates used by the survival model
