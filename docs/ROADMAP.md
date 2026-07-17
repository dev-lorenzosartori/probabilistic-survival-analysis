# Roadmap

- ✅ Phases 1–5 complete and validated.
- 🟡 Phase 6 is the next planned increment.
- ⚪ Phases 7–8 remain future expansion; no production-monitoring claim is made.

## Phase 1 - Problem Framing — Complete

Goal: define the business problem, statistical framing, project scope, and final artifacts.

Deliverables:

- Project charter
- Initial README
- Dataset decision
- Clear target variables: duration and event observed

Acceptance criteria:

- A recruiter can understand the project objective in less than one minute.
- A technical reviewer can see that this is a time-to-event problem, not a simple classifier.

## Phase 2 - Dataset Validation — Complete

Goal: validate the selected dataset and define how it will be transformed for survival analysis.

Deliverables:

- Data dictionary
- Raw-to-processed data plan
- Definition of censoring logic
- Initial exploratory analysis notebook

Acceptance criteria:

- The project has a defensible survival-analysis table.
- Every row has a duration and an event indicator.

## Phase 3 - Statistical Baseline — Complete

Goal: implement the core survival analysis baseline.

Deliverables:

- Kaplan-Meier curves
- Risk group comparison
- Log-rank test
- Initial written interpretation

Acceptance criteria:

- The analysis explains how survival probability changes over time.
- The visuals are suitable for LinkedIn and portfolio publication.

## Phase 4 - Cox Model and Interpretation — Complete

Goal: estimate covariate effects on failure risk.

Deliverables:

- Cox proportional hazards model
- Hazard ratio table
- Assumption checks
- Business interpretation of key variables

Acceptance criteria:

- The model explains which factors increase or reduce hazard.
- The findings are written in business language, not only statistical language.

## Phase 5 - Advanced Modeling — Complete

Goal: compare classical survival models with more flexible approaches.

Candidate methods:

- Weibull or other parametric survival model
- Random Survival Forest
- Gradient boosting survival model

Deliverables:

- Model comparison table
- Concordance index
- Calibration-oriented discussion

Acceptance criteria:

- The project demonstrates judgment, not only library usage.
- The final model choice is justified.

## Phase 6 - Decision Layer — Planned

Goal: convert survival predictions into decision support.

Deliverables:

- Risk bands
- Preventive maintenance recommendation logic
- Scenario analysis
- Executive summary

Acceptance criteria:

- The model output supports a concrete business decision.

## Phase 7 - MLOps Layer — Future

Goal: show that the model can be monitored after deployment.

Deliverables:

- Reproducible pipeline
- Drift checks
- Model monitoring notebook or script
- Optional API/demo app

Acceptance criteria:

- The project connects statistical modeling with production reliability.

## Phase 8 - Publication — In progress

Goal: package the project for the market.

Deliverables:

- Final GitHub README
- Streamlit dashboard
- Technical article
- LinkedIn post
- Kaggle notebook

Acceptance criteria:

- The project is easy to inspect.
- The business value and statistical depth are clear.
