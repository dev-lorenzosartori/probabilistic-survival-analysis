from __future__ import annotations
import unittest
import numpy as np
import pandas as pd
from src.evaluation_utils import calibration_table, harrell_c_index, make_survival_target
from src.feature_engineering import DEGRADATION_SIGNAL_COLUMNS, INDEPENDENT_TREND_COLUMN, LandmarkFeatureTransformer

class SurvivalMetricTests(unittest.TestCase):
    def test_concordance_sign(self):
        y=make_survival_target([True,True,True],[1.,2.,3.])
        self.assertAlmostEqual(harrell_c_index(y,[3.,2.,1.]),1.)
    def test_calibration_requires_observed_times(self):
        y=make_survival_target([True,False],[1.,2.])
        with self.assertRaises(ValueError): calibration_table(y,[.2,.8],horizon=1.5,n_bins=2)
    def test_feature_shape(self):
        n=12; data={c:np.linspace(i,i+1,n) for i,c in enumerate(DEGRADATION_SIGNAL_COLUMNS)}
        data[INDEPENDENT_TREND_COLUMN]=np.linspace(-1,1,n); frame=pd.DataFrame(data)
        y=make_survival_target(np.ones(n,bool),np.arange(1,n+1))
        transformed=LandmarkFeatureTransformer().fit_transform(frame,y)
        self.assertEqual(transformed.shape,(n,2)); self.assertTrue(np.isfinite(transformed).all())

