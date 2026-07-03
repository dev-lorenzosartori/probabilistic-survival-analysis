<<<<<<< HEAD
# Probabilistic Survival Analysis for Predictive Maintenance

This project applies survival analysis to estimate time-to-failure risk in industrial assets.
Instead of asking only whether an asset will fail, the project answers a more useful business
question:

> What is the probability that an asset remains operational after a given time, and which
> factors increase its failure risk?

The project is designed as a professional portfolio case study for Data Science, Analytics
Engineering, AI Engineering, and MLOps-oriented roles.

## Why This Project Matters

Many machine learning projects reduce real-world problems to binary classification:
churn or no churn, failure or no failure, default or no default. This is often too limited
for decision-making.

Survival analysis handles time-to-event problems and censored data, making it more useful
for questions such as:

- When is an asset likely to fail?
- How does risk evolve over time?
- Which covariates increase or reduce the hazard rate?
- When should preventive action be taken?
- How can model behavior be monitored over time?

## Core Statistical Concepts

The project will cover:

- Survival function
- Hazard function
- Censored observations
- Kaplan-Meier estimator
- Cox proportional hazards model
- Hazard ratios
- Model evaluation with concordance index and time-dependent metrics
- Drift monitoring for production-style model reliability

## Planned Technical Stack

- Python
- pandas and NumPy
- lifelines and/or scikit-survival
- SQL for analytical data preparation
- Streamlit for an interactive demo
- Docker for reproducible execution
- Optional: MLflow for experiment tracking

## Main Deliverables

- Technical notebook
- Production-style Python pipeline
- Streamlit dashboard
- Executive README
- Technical report
- LinkedIn article
- Kaggle notebook version

## Status

Project initialized. Current phase: C-MAPSS transformation and landmark survival-table construction.


## Reproducible Commands

Generate the synthetic proof of concept:

```bash
make synthetic
make km-baseline
```

After placing the NASA C-MAPSS FD001 files in `data/raw/cmapss/`, build the real survival tables:

```bash
make cmapss
```

Run the transformation smoke test:

```bash
make smoke-test
```
=======
# probabilistic-survival-analysis
Probabilistic Survival Analysis for Predictive Maintenance
>>>>>>> 62a930eb98a34c23d2492713f189cb5ae0b7fb64
