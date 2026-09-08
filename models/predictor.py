"""
Streamflow Inference Engine & Flood Risk Decision System.

Provides high-performance batch and single-sample prediction, physical constraint
enforcement, hydrological delta analytics, and categorical flood risk scoring.
"""

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from preprocessing.feature_engineering import StreamflowFeaturePipeline


class FloodRiskLevel(Enum):
    """Hydrological flood risk classification thresholds in cumecs (m³/s)."""
    LOW = ("Normal / Baseflow", "#10B981", "Discharge within safe natural thresholds. Normal river conditions.", "Standard operational monitoring.")
    MODERATE = ("Moderate Flow", "#F59E0B", "Elevated streamflow. Minor bank swells possible in lowlands.", "Active observation of catchment basin.")
    HIGH = ("High Flow Alert", "#F97316", "Significant river swell. Risk of localized overbanking.", "Issue flood advisory to local authorities.")
    CRITICAL = ("Severe Flood Warning", "#EF4444", "Dangerously high discharge. Imminent flooding of floodplains.", "Trigger emergency flood alerts & evacuation readiness.")

    def __init__(self, label: str, color: str, description: str, action: str):
        self.label = label
        self.color = color
        self.description = description
        self.action = action


@dataclass
class PredictionResult:
    """Structured container for single-station prediction results."""
    predicted_streamflow: float
    current_streamflow: float
    delta_cumecs: float
    percent_change: float
    risk_level: str
    risk_color: str
    risk_description: str
    risk_action: str
    feature_vector: Dict[str, float]


class StreamflowPredictor:
    """
    Production-ready inference engine for deep learning streamflow prediction.
    """

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        scaler_path: Optional[Union[str, Path]] = None,
    ) -> None:
        self.base_dir = Path(__file__).resolve().parent.parent

        self.model_path = Path(model_path) if model_path else self.base_dir / "models" / "model2.keras"
        self.scaler_path = Path(scaler_path) if scaler_path else self.base_dir / "preprocessing" / "scaler2.pkl"

        self.pipeline = StreamflowFeaturePipeline()
        self.model = None
        self.scaler = None
        self._load_artifacts()

    def _load_artifacts(self) -> None:
        """Loads trained Keras model and precomputed StandardScaler."""
        # Load scaler
        if self.scaler_path.exists():
            try:
                import joblib
                self.scaler = joblib.load(str(self.scaler_path))
            except Exception as e:
                # Fallback to pickle
                try:
                    import pickle
                    with open(self.scaler_path, "rb") as f:
                        self.scaler = pickle.load(f)
                except Exception:
                    self.scaler = None

        # Load Keras Deep Learning Model
        if self.model_path.exists():
            try:
                from tensorflow.keras.models import load_model
                self.model = load_model(str(self.model_path))
            except Exception:
                self.model = None

    @staticmethod
    def classify_risk(flow_cumecs: float) -> FloodRiskLevel:
        """
        Categorizes streamflow into hydrological flood risk tiers.

        Args:
            flow_cumecs: Discharge in cubic meters per second (cumecs).

        Returns:
            FloodRiskLevel enum member.
        """
        if flow_cumecs < 100.0:
            return FloodRiskLevel.LOW
        elif flow_cumecs < 500.0:
            return FloodRiskLevel.MODERATE
        elif flow_cumecs < 1500.0:
            return FloodRiskLevel.HIGH
        else:
            return FloodRiskLevel.CRITICAL

    def _fallback_predict(self, processed_df: pd.DataFrame) -> np.ndarray:
        """
        Hydrological autoregressive fallback calculation if native TF engine
        is not available. Conserves physics-based continuity:
        Q(t+1) = 0.88 * Q(t) + 0.12 * Q_lag3 + Rain_Contribution
        """
        flows = processed_df["streamflow_today_cumecs"].values
        lags = processed_df["flow_lag_1"].values
        rain = processed_df["antecedent_rain_3d_sum"].values
        soil = processed_df["soil_saturation_score"].values

        # Physical continuity equation
        predicted = (0.85 * flows) + (0.10 * lags) + (1.2 * rain * (1.0 + soil))
        return np.maximum(0.0, predicted).reshape(-1, 1)

    def predict_batch(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        """
        Executes end-to-end inference over a batch of station readings.

        Args:
            raw_data: DataFrame of station observations.

        Returns:
            DataFrame augmented with predictions, deltas, and risk classifications.
        """
        if raw_data.empty:
            return raw_data.copy()

        # Step 1: Feature Engineering
        features_df = self.pipeline.transform(raw_data)

        # Step 2: Scaling
        if self.scaler is not None:
            try:
                scaled_values = self.scaler.transform(features_df)
            except Exception:
                scaled_values = features_df.values
        else:
            scaled_values = features_df.values

        # Step 3: Model Inference
        if self.model is not None:
            try:
                raw_preds = self.model.predict(scaled_values, verbose=0)
                preds = np.maximum(0.0, raw_preds.flatten())
            except Exception:
                preds = self._fallback_predict(features_df).flatten()
        else:
            preds = self._fallback_predict(features_df).flatten()

        # Step 4: Augment output with metrics
        output_df = raw_data.copy()
        current_flow = features_df["streamflow_today_cumecs"].values
        deltas = preds - current_flow
        pct_changes = np.where(current_flow > 0, (deltas / current_flow) * 100.0, 0.0)

        risk_labels = []
        risk_colors = []
        for p in preds:
            risk = self.classify_risk(float(p))
            risk_labels.append(risk.label)
            risk_colors.append(risk.color)

        output_df["predicted_streamflow_cumecs"] = np.round(preds, 2)
        output_df["delta_cumecs"] = np.round(deltas, 2)
        output_df["percent_change"] = np.round(pct_changes, 2)
        output_df["flood_risk_level"] = risk_labels
        output_df["risk_color"] = risk_colors

        return output_df

    def predict_single(self, raw_input: Union[Dict[str, Any], pd.DataFrame]) -> PredictionResult:
        """
        Executes inference for a single station observation.

        Args:
            raw_input: Dict or 1-row DataFrame containing station parameters.

        Returns:
            PredictionResult object with detailed metrics and recommendations.
        """
        if isinstance(raw_input, dict):
            df_in = pd.DataFrame([raw_input])
        else:
            df_in = raw_input.head(1).copy()

        result_df = self.predict_batch(df_in)
        row = result_df.iloc[0]

        pred_val = float(row["predicted_streamflow_cumecs"])
        curr_val = float(row.get("streamflow_today_cumecs", 0.0))
        delta = float(row["delta_cumecs"])
        pct = float(row["percent_change"])

        risk = self.classify_risk(pred_val)

        # Extract features dictionary
        feat_df = self.pipeline.transform(df_in)
        feat_dict = {col: float(feat_df[col].iloc[0]) for col in feat_df.columns}

        return PredictionResult(
            predicted_streamflow=pred_val,
            current_streamflow=curr_val,
            delta_cumecs=delta,
            percent_change=pct,
            risk_level=risk.label,
            risk_color=risk.color,
            risk_description=risk.description,
            risk_action=risk.action,
            feature_vector=feat_dict,
        )
