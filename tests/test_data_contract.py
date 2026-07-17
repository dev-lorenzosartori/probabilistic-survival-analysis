from __future__ import annotations
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]

class LandmarkDataContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.df=pd.read_csv(ROOT/"data/processed/cmapss_fd001_landmark30_modeling_table.csv")
    def test_grain_split(self):
        self.assertEqual(len(self.df),200); self.assertFalse(self.df.entity_id.duplicated().any())
        self.assertEqual(self.df.split.value_counts().to_dict(),{"train":100,"test":100})
    def test_event_time_contract(self):
        train=self.df[self.df.split=="train"]; test=self.df[self.df.split=="test"]
        self.assertTrue(train.event_observed.eq(1).all()); self.assertTrue(test.event_observed.eq(0).all())
        self.assertTrue(self.df.time_after_landmark.eq(self.df.duration-self.df.landmark_cycle).all())
        self.assertTrue(test.true_event_time.eq(test.duration+test.true_rul).all())
    def test_features_complete(self):
        cols=[c for c in self.df if c.startswith(("mean_","slope_"))]
        self.assertEqual(len(cols),48); self.assertFalse(self.df[cols].isna().any().any())
        self.assertTrue(np.isfinite(self.df[cols].to_numpy()).all())

