"""
Unit tests for Streamflow inference engine and flood risk categorization.
"""

import unittest
from pathlib import Path
import pandas as pd
import numpy as np

from models.predictor import StreamflowPredictor, FloodRiskLevel, PredictionResult


class TestStreamflowPredictor(unittest.TestCase):
    def setUp(self):
        self.predictor = StreamflowPredictor()

    def test_classify_risk_thresholds(self):
        """Risk classification must strictly adhere to hydrological thresholds."""
        self.assertEqual(self.predictor.classify_risk(50.0), FloodRiskLevel.LOW)
        self.assertEqual(self.predictor.classify_risk(99.9), FloodRiskLevel.LOW)

        self.assertEqual(self.predictor.classify_risk(100.0), FloodRiskLevel.MODERATE)
        self.assertEqual(self.predictor.classify_risk(499.0), FloodRiskLevel.MODERATE)

        self.assertEqual(self.predictor.classify_risk(500.0), FloodRiskLevel.HIGH)
        self.assertEqual(self.predictor.classify_risk(1499.0), FloodRiskLevel.HIGH)

        self.assertEqual(self.predictor.classify_risk(1500.0), FloodRiskLevel.CRITICAL)
        self.assertEqual(self.predictor.classify_risk(4500.0), FloodRiskLevel.CRITICAL)

    def test_predict_single_structure(self):
        """Single prediction must return a valid PredictionResult with non-negative discharge."""
        sample = {
            "streamflow_today_cumecs": 280.0,
            "flow_rate_of_change": 5.0,
            "antecedent_rain_3d_sum": 12.0,
            "soil_saturation_score": 0.4,
            "day_of_year": 180,
        }

        result = self.predictor.predict_single(sample)

        self.assertIsInstance(result, PredictionResult)
        self.assertGreaterEqual(result.predicted_streamflow, 0.0, "Discharge must satisfy physical constraint Q >= 0")
        self.assertAlmostEqual(result.current_streamflow, 280.0)
        self.assertIn(result.risk_level, [r.label for r in FloodRiskLevel])
        self.assertEqual(len(result.feature_vector), 36)

    def test_predict_batch_shape_and_columns(self):
        """Batch prediction must return the original DataFrame augmented with metrics."""
        batch_df = pd.DataFrame([
            {"station_name": "Station A", "streamflow_today_cumecs": 50.0, "antecedent_rain_3d_sum": 0.0},
            {"station_name": "Station B", "streamflow_today_cumecs": 750.0, "antecedent_rain_3d_sum": 30.0},
            {"station_name": "Station C", "streamflow_today_cumecs": 2200.0, "antecedent_rain_3d_sum": 80.0},
        ])

        output_df = self.predictor.predict_batch(batch_df)

        self.assertEqual(len(output_df), 3)
        self.assertIn("predicted_streamflow_cumecs", output_df.columns)
        self.assertIn("delta_cumecs", output_df.columns)
        self.assertIn("percent_change", output_df.columns)
        self.assertIn("flood_risk_level", output_df.columns)
        self.assertIn("risk_color", output_df.columns)

        # Check that negative streamflow is clipped
        self.assertTrue((output_df["predicted_streamflow_cumecs"] >= 0.0).all())

    def test_sample_stations_file(self):
        """Predictor must cleanly process data/sample_stations.csv."""
        csv_path = Path(__file__).resolve().parent.parent / "data" / "sample_stations.csv"
        self.assertTrue(csv_path.exists(), "Sample stations CSV must exist")

        stations_df = pd.read_csv(csv_path)
        output_df = self.predictor.predict_batch(stations_df)

        self.assertEqual(len(output_df), len(stations_df))
        self.assertFalse(output_df["predicted_streamflow_cumecs"].isnull().any())


if __name__ == "__main__":
    unittest.main()
