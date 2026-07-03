PYTHON ?= python
MPLCONFIGDIR ?= /tmp/matplotlib-cache

.PHONY: install synthetic km-baseline cmapss km-cmapss smoke-test

install:
	$(PYTHON) -m pip install -r requirements.txt

synthetic:
	$(PYTHON) src/generate_synthetic_survival_data.py

km-baseline:
	MPLCONFIGDIR=$(MPLCONFIGDIR) $(PYTHON) src/kaplan_meier_baseline.py


cmapss:
	$(PYTHON) src/build_cmapss_survival_table.py --subset FD001 --window 30

km-cmapss:
	MPLCONFIGDIR=$(MPLCONFIGDIR) PYTHONPATH=. $(PYTHON) src/kaplan_meier_cmapss.py

smoke-test:
	PYTHONPATH=. $(PYTHON) tests/smoke_test_cmapss_transformation.py
