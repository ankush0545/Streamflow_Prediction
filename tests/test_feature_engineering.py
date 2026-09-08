"""
Unit tests for hydrological feature engineering pipeline.
"""

import unittest
import pandas as pd
import numpy as np

from preprocessing.feature_engineering import StreamflowFeaturePipeline, feature


class TestStreamflowFeaturePipeline(unittest.TestCase):
    def setUp(self):
        self.pipeline = StreamflowFeaturePipeline()

    def test_expected_features_length(self):
        """Pipeline must strictly produce exactly 36 canonical features."""
        self.assertEqual(len(self.pipeline.EXPECTED_FEATURES), 36)

    def test_empty_dataframe(self):
        """Empty input must yield an empty DataFrame with the exact 36 columns."""
        empty_df = pd.DataFrame()
        result = self.pipeline.transform(empty_df)
        self.assertEqual(list(result.columns), self.pipeline.EXPECTED_FEATURES)
        self.assertEqual(len(result), 0)

    def test_single_row_inference(self):
        """Single-row input must successfully synthesize lags without producing NaNs."""
        sample_input = pd.DataFrame([{
            "day_of_year": 210,
            "streamflow_today_cumecs": 450.0,
            "flow_rate_of_change": 15.0,
            "antecedent_rain_3d_sum": 25.0,
            "soil_saturation_score": 0.65,
        }])

        result = self.pipeline.transform(sample_input)

        self.assertEqual(len(result), 1)
        self.assertEqual(list(result.columns), self.pipeline.EXPECTED_FEATURES)
        self.assertFalse(result.isnull().any().any(), "Transformed output must contain no NaNs")
        self.assertAlmostEqual(result["streamflow_today_cumecs"].iloc[0], 450.0)
        self.assertGreater(result["flow_lag_1"].iloc[0], 0.0)

    def test_batch_rolling_windows(self):
        """Batch input must correctly calculate rolling statistics and temporal lags."""
        batch_input = pd.DataFrame({
            "streamflow_today_cumecs": [100.0, 150.0, 200.0, 250.0, 300.0, 350.0, 400.0, 450.0],
            "day_of_year": [10, 11, 12, 13, 14, 15, 16, 17],
            "antecedent_rain_3d_sum": [0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0],
        })

        result = self.pipeline.transform(batch_input)

        self.assertEqual(len(result), len(batch_input))
        self.assertEqual(list(result.columns), self.pipeline.EXPECTED_FEATURES)
        self.assertFalse(result.isnull().any().any(), "Batch output must contain no NaNs")
        # 7-day rolling mean on row 6 (index 6, value 400) should be around 250
        self.assertAlmostEqual(result["rolling_mean_7"].iloc[6], 250.0, places=1)

    def test_physics_interaction_terms(self):
        """Interaction terms (rain x soil, rain x urban, rain x slope) must be computed."""
        data = pd.DataFrame([{
            "streamflow_today_cumecs": 500.0,
            "antecedent_rain_3d_sum": 10.0,
            "soil_saturation_score": 0.5,
            "urban_cover_scaled": 0.8,
            "slope_scaled": 0.3,
        }])

        result = self.pipeline.transform(data)

        self.assertAlmostEqual(result["rain_soilmoisture_interaction"].iloc[0], 5.0)
        self.assertAlmostEqual(result["rain_urban_interaction"].iloc[0], 8.0)
        self.assertAlmostEqual(result["rain_slope_interaction"].iloc[0], 3.0)

    def test_backward_compatible_function(self):
        """The legacy feature() function must produce the same result as the class method."""
        data = pd.DataFrame([{"streamflow_today_cumecs": 120.0}])
        res_class = self.pipeline.transform(data)
        res_func = feature(data)

        pd.testing.assert_frame_equal(res_class, res_func)


if __name__ == "__main__":
    unittest.main()
