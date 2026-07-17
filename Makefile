PYTHON ?= python3
MPLCONFIGDIR ?= /tmp/matplotlib-cache
PYTHONPATH_VALUE ?= .

.PHONY: install synthetic km-baseline cmapss km-cmapss cox-cmapss evaluate smoke-test unit-test test
install:
	$(PYTHON) -m pip install -r requirements.txt
synthetic:
	PYTHONPATH=$(PYTHONPATH_VALUE) $(PYTHON) src/generate_synthetic_survival_data.py
km-baseline:
	MPLCONFIGDIR=$(MPLCONFIGDIR) PYTHONPATH=$(PYTHONPATH_VALUE) $(PYTHON) src/kaplan_meier_baseline.py
cmapss:
	PYTHONPATH=$(PYTHONPATH_VALUE) $(PYTHON) src/build_cmapss_survival_table.py --subset FD001 --window 30
km-cmapss:
	MPLCONFIGDIR=$(MPLCONFIGDIR) PYTHONPATH=$(PYTHONPATH_VALUE) $(PYTHON) src/kaplan_meier_cmapss.py
cox-cmapss:
	MPLCONFIGDIR=$(MPLCONFIGDIR) PYTHONPATH=$(PYTHONPATH_VALUE) $(PYTHON) src/cox_model_cmapss.py
evaluate:
	MPLCONFIGDIR=$(MPLCONFIGDIR) PYTHONPATH=$(PYTHONPATH_VALUE) $(PYTHON) src/evaluate_models_cmapss.py
smoke-test:
	PYTHONPATH=$(PYTHONPATH_VALUE) $(PYTHON) tests/smoke_test_cmapss_transformation.py
unit-test:
	PYTHONPATH=$(PYTHONPATH_VALUE) $(PYTHON) -m unittest discover -s tests -p 'test_*.py'
test: smoke-test unit-test
