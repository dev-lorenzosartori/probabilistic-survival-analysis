"""Leakage-safe feature engineering for the C-MAPSS landmark model."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

DEGRADATION_SIGNAL_COLUMNS = [
    "mean_sensor_3_first_30", "mean_sensor_4_first_30", "mean_sensor_15_first_30",
    "mean_sensor_7_first_30", "mean_sensor_11_first_30", "mean_sensor_21_first_30",
    "mean_sensor_2_first_30", "mean_sensor_17_first_30", "mean_sensor_12_first_30",
    "mean_sensor_8_first_30", "mean_sensor_20_first_30", "mean_sensor_13_first_30",
]
INDEPENDENT_TREND_COLUMN = "slope_sensor_21_first_30"


class LandmarkFeatureTransformer(BaseEstimator, TransformerMixin):
    """Reduce correlated early-life signals to two stable model inputs.

    PCA and scaling are fitted only on the development data. The first component
    is oriented using the development target so larger values mean shorter survival.
    """

    def __init__(self, signal_columns=tuple(DEGRADATION_SIGNAL_COLUMNS),
                 trend_column=INDEPENDENT_TREND_COLUMN):
        self.signal_columns = signal_columns
        self.trend_column = trend_column

    def fit(self, X: pd.DataFrame, y: np.ndarray | None = None):
        self._check_columns(X)
        self.signal_scaler_ = StandardScaler()
        signal_scaled = self.signal_scaler_.fit_transform(X[list(self.signal_columns)])
        self.pca_ = PCA(n_components=1, random_state=42)
        component = self.pca_.fit_transform(signal_scaled).ravel()
        self.orientation_ = 1.0
        if y is not None:
            target_time = y["time"] if getattr(y.dtype, "names", None) else np.asarray(y)
            correlation = spearmanr(component, target_time).statistic
            if np.isfinite(correlation) and correlation > 0:
                self.orientation_ = -1.0
        self.trend_scaler_ = StandardScaler().fit(X[[self.trend_column]])
        self.explained_variance_ratio_ = float(self.pca_.explained_variance_ratio_[0])
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        self._check_columns(X)
        signal_scaled = self.signal_scaler_.transform(X[list(self.signal_columns)])
        degradation = self.orientation_ * self.pca_.transform(signal_scaled).ravel()
        trend = self.trend_scaler_.transform(X[[self.trend_column]]).ravel()
        return np.column_stack([degradation, trend])

    def get_feature_names_out(self, input_features=None):
        return np.asarray(["early_degradation_index", "sensor_21_early_trend"], dtype=object)

    def _check_columns(self, X: pd.DataFrame) -> None:
        missing = [c for c in [*self.signal_columns, self.trend_column] if c not in X.columns]
        if missing:
            raise ValueError(f"Missing landmark feature columns: {missing}")

