"""
Hydrological Feature Engineering & Data Preprocessing Pipeline.

Provides production-grade temporal lag calculations, rolling window hydraulics,
physics-informed hydrological interactions, and robust alignment with trained models.
"""

from typing import List, Optional
import numpy as np
import pandas as pd


class StreamflowFeaturePipeline:
    """
    End-to-end feature engineering pipeline for streamflow and flood forecasting.
    Transforms raw station readings into the exact 36-feature schema expected by
    the trained deep neural network and scaler.
    """

    # Canonical 36 features in exact sequence expected by model and StandardScaler
    EXPECTED_FEATURES: List[str] = [
        "day_of_year",
        "streamflow_today_cumecs",
        "streamflow_anomaly_zscore",
        "flow_rate_of_change",
        "flow_velocity_km_per_day",
        "antecedent_rain_3d_sum",
        "antecedent_rain_7d_sum",
        "antecedent_rain_15d_sum",
        "antecedent_rain_ewm",
        "rainfall_anomaly_zscore",
        "upstream_rain_mean_scaled",
        "upstream_rain_weighted_scaled",
        "upstream_rain_lagged_dist_sink",
        "soil_saturation_score",
        "antecedent_saturation_interaction",
        "is_post_monsoon_saturated",
        "monsoon_intensity",
        "monsoon_cumulative_rain",
        "dist_to_outlet_scaled",
        "upstream_area_scaled",
        "slope_scaled",
        "slope_uav_scaled",
        "forest_cover_scaled",
        "urban_cover_scaled",
        "rain_soilmoisture_interaction",
        "rain_urban_interaction",
        "rain_slope_interaction",
        "flow_lag_1",
        "flow_lag_3",
        "flow_lag_7",
        "lag_14",
        "lag_30",
        "rolling_mean_7",
        "rolling_std_7",
        "diff_1",
        "ema_7",
    ]

    # Redundant, non-predictive, or high-collinearity columns removed during training
    COLUMNS_TO_DROP: List[str] = [
        "row_id",
        "station_id",
        "station_name",
        "basin_region",
        "month",
        "antecedent_rain_30d_sum",
        "antecedent_rain_60d",
        "rolling_max_7",
        "rain_basinsize_interaction",
        "uparea_rain_interaction",
        "ratio_7",
        "streamflow_tomorrow_cumecs",
    ]

    # Baseline hydrological fallback defaults for missing attributes
    DEFAULT_FEATURE_VALUES = {
        "day_of_year": 180.0,
        "streamflow_today_cumecs": 250.0,
        "streamflow_anomaly_zscore": 0.0,
        "flow_rate_of_change": 0.0,
        "flow_velocity_km_per_day": 0.5,
        "antecedent_rain_3d_sum": 0.0,
        "antecedent_rain_7d_sum": 0.0,
        "antecedent_rain_15d_sum": 0.0,
        "antecedent_rain_ewm": 0.0,
        "rainfall_anomaly_zscore": 0.0,
        "upstream_rain_mean_scaled": -0.46,
        "upstream_rain_weighted_scaled": -0.46,
        "upstream_rain_lagged_dist_sink": -0.23,
        "soil_saturation_score": 0.20,
        "antecedent_saturation_interaction": 0.0,
        "is_post_monsoon_saturated": 0.0,
        "monsoon_intensity": 0.0,
        "monsoon_cumulative_rain": 0.0,
        "dist_to_outlet_scaled": 0.0,
        "upstream_area_scaled": 0.0,
        "slope_scaled": 0.0,
        "slope_uav_scaled": 0.0,
        "forest_cover_scaled": 0.0,
        "urban_cover_scaled": 0.0,
        "rain_soilmoisture_interaction": -0.51,
        "rain_urban_interaction": -0.51,
        "rain_slope_interaction": -0.51,
    }

    def __init__(self, imputation_strategy: str = "median") -> None:
        self.imputation_strategy = imputation_strategy

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Processes an input DataFrame and returns a clean DataFrame strictly
        conforming to EXPECTED_FEATURES.

        Args:
            data: Raw DataFrame containing station readings.

        Returns:
            Cleaned and transformed DataFrame with exact 36 model features.
        """
        if data.empty:
            return pd.DataFrame(columns=self.EXPECTED_FEATURES)

        df = data.copy()

        # Handle temporal lag features
        is_single_sample = len(df) == 1
        has_lags = all(col in df.columns for col in ["flow_lag_1", "rolling_mean_7", "diff_1"])

        if not has_lags:
            if is_single_sample:
                # Approximate dynamic lags for instantaneous scenario forecasting
                flow = float(df["streamflow_today_cumecs"].iloc[0]) if "streamflow_today_cumecs" in df else 250.0
                rate = float(df["flow_rate_of_change"].iloc[0]) if "flow_rate_of_change" in df else 0.0

                df["flow_lag_1"] = max(0.0, flow - rate)
                df["flow_lag_3"] = max(0.0, flow - (3 * rate))
                df["flow_lag_7"] = max(0.0, flow - (7 * rate * 0.5))
                df["lag_14"] = flow
                df["lag_30"] = flow
                df["rolling_mean_7"] = (flow + df["flow_lag_1"].iloc[0] + df["flow_lag_3"].iloc[0]) / 3.0
                df["rolling_std_7"] = abs(rate)
                df["diff_1"] = rate
                df["ema_7"] = 0.7 * flow + 0.3 * df["flow_lag_1"].iloc[0]
            else:
                # Time-series batch window calculations
                flow = df["streamflow_today_cumecs"] if "streamflow_today_cumecs" in df else pd.Series([250.0] * len(df))

                df["flow_lag_1"] = flow.shift(1).bfill()
                df["flow_lag_3"] = flow.shift(3).bfill()
                df["flow_lag_7"] = flow.shift(7).bfill()
                df["lag_14"] = flow.shift(14).bfill()
                df["lag_30"] = flow.shift(30).bfill()

                df["rolling_mean_7"] = flow.rolling(7, min_periods=1).mean()
                df["rolling_std_7"] = flow.rolling(7, min_periods=1).std().fillna(0.0)
                df["diff_1"] = flow - df["flow_lag_1"]
                df["ema_7"] = flow.ewm(span=7).mean()

        # Compute physical interaction terms if missing
        if "rain_soilmoisture_interaction" not in df.columns:
            rain = df.get("antecedent_rain_3d_sum", 0.0)
            soil = df.get("soil_saturation_score", 0.2)
            df["rain_soilmoisture_interaction"] = rain * soil

        if "rain_urban_interaction" not in df.columns:
            rain = df.get("antecedent_rain_3d_sum", 0.0)
            urban = df.get("urban_cover_scaled", 0.0)
            df["rain_urban_interaction"] = rain * urban

        if "rain_slope_interaction" not in df.columns:
            rain = df.get("antecedent_rain_3d_sum", 0.0)
            slope = df.get("slope_scaled", 0.0)
            df["rain_slope_interaction"] = rain * slope

        # Fill any missing required feature with domain defaults
        for col in self.EXPECTED_FEATURES:
            if col not in df.columns:
                df[col] = self.DEFAULT_FEATURE_VALUES.get(col, 0.0)
            else:
                # Impute remaining NaNs with column median or default
                if df[col].isnull().any():
                    median_val = df[col].median()
                    if pd.isna(median_val):
                        median_val = self.DEFAULT_FEATURE_VALUES.get(col, 0.0)
                    df[col] = df[col].fillna(median_val)

        # Strictly select and order the 36 features
        ordered_df = df[self.EXPECTED_FEATURES].astype(np.float64)
        return ordered_df


def feature(train: pd.DataFrame) -> pd.DataFrame:
    """
    Backwards-compatible convenience function for feature engineering.
    """
    pipeline = StreamflowFeaturePipeline()
    return pipeline.transform(train)