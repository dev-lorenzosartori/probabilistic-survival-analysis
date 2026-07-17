from __future__ import annotations
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from src.cox_model_cmapss import LOW_VARIANCE_COVARIATES, build_degradation_index, screen_univariate
from src.cox_utils import fit_cox_ph
ROOT=Path(__file__).resolve().parents[1]

class CoxReferenceTests(unittest.TestCase):
    def test_coefficients_match_lifelines(self):
        df=pd.read_csv(ROOT/"data/processed/cmapss_fd001_landmark30_modeling_table.csv")
        candidates=[c for c in df if c.startswith(("mean_","slope_")) and c not in LOW_VARIANCE_COVARIATES]
        significant=screen_univariate(df,candidates).query("p_value < 0.05").covariate.tolist()
        pc1,_=build_degradation_index(df,significant); X=np.column_stack([pc1,df.slope_sensor_21_first_30])
        Xz=(X-X.mean(0))/X.std(0)
        custom=fit_cox_ph(Xz,df.time_after_landmark.to_numpy(float),df.event_observed.to_numpy(int),["pc1","slope"])
        ref=pd.DataFrame(Xz,columns=["pc1","slope"]).assign(time=df.time_after_landmark,event=df.event_observed)
        model=CoxPHFitter().fit(ref,"time","event")
        np.testing.assert_allclose(custom["beta"],model.params_.to_numpy(),rtol=0,atol=1e-5)
