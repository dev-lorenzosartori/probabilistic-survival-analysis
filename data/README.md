# Data contract

The project uses NASA's C-MAPSS FD001 turbofan benchmark. Raw source files are excluded;
compact derived tables needed to reproduce the published analysis are committed.

Place `train_FD001.txt`, `test_FD001.txt`, and `RUL_FD001.txt` under
`data/raw/cmapss/`, then run `make cmapss` to rebuild the processed tables.

- `cmapss_fd001_survival_table_full.csv`: audit layer with censoring metadata.
- `cmapss_fd001_landmark30_modeling_table.csv`: one eligible engine per row at cycle 30.
- `synthetic_asset_survival.csv`: synthetic Phase-1 proof of concept.

`true_rul` and `true_event_time` are never used for fitting. They are revealed only for
external evaluation of the untouched official NASA test units.
