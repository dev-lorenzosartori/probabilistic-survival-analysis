# Project Charter

## Working Title

Probabilistic Survival Analysis for Predictive Maintenance

## Positioning

This is a portfolio-grade data science project focused on probability, statistics,
industrial analytics, and model reliability. The goal is to show the ability to move
from a real business problem to a statistically rigorous and technically reproducible
solution.

## Target Audience

- Data Science recruiters
- AI/ML hiring managers
- Analytics Engineering teams
- MLOps teams
- Industrial analytics and predictive maintenance teams
- Financial, insurance, telecom, and operations teams interested in time-to-event modeling

## Business Problem

Organizations often need to know not only whether an event will happen, but when it is
likely to happen. In industrial contexts, this can mean estimating the time until equipment
failure. In customer analytics, it can mean estimating time until churn.

Binary classification does not fully answer this problem because it ignores the temporal
dimension and often mishandles censored observations.

## Main Question

What is the probability that an asset survives beyond a given time, and which factors
increase its risk of failure?

## Secondary Questions

- How does survival probability change over operational time?
- Which variables are associated with higher failure risk?
- How can risk groups be compared visually and statistically?
- How can the model support preventive maintenance decisions?
- How can model reliability be monitored when new data arrives?

## Proposed Primary Dataset

NASA C-MAPSS Turbofan Engine Degradation dataset.

Reason:

- It is a recognized benchmark for prognostics and health management.
- It has an industrial/engineering narrative.
- It supports time-to-event and remaining useful life reasoning.
- It connects naturally to predictive maintenance and operational risk.

## Optional Secondary Dataset

Telco Customer Churn dataset.

Reason:

- It can demonstrate that the same survival analysis framework applies to customer retention.
- It is easy for non-technical business readers to understand.
- It helps connect the project to commercial analytics roles.

## Statistical Methods

- Kaplan-Meier estimator
- Log-rank test
- Cox proportional hazards model
- Parametric survival models, such as Weibull or Exponential models
- Optional: Random Survival Forest or gradient boosting survival models

## Evaluation Approach

- Concordance index
- Calibration-oriented analysis
- Survival curve inspection
- Hazard ratio interpretation
- Time-based validation
- Drift checks on input distributions and predicted risk scores

## Final Narrative

The project should communicate that reliable analytics is not only about prediction,
but also about uncertainty, time, decision thresholds, and model behavior after deployment.
