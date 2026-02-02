"""Model service for loading and using ML models."""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal
import pickle
import joblib

import numpy as np
import pandas as pd
import streamlit as st


@dataclass
class ForecastResult:
    """Container for forecast results."""
    
    timestamps: list[datetime] = field(default_factory=list)
    predictions: np.ndarray = field(default_factory=lambda: np.array([]))
    lower_bound: np.ndarray = field(default_factory=lambda: np.array([]))
    upper_bound: np.ndarray = field(default_factory=lambda: np.array([]))
    model_type: str = ""
    confidence_level: float = 0.95
    metrics: dict = field(default_factory=dict)
    
    @property
    def horizon(self) -> int:
        return len(self.predictions)
    
    @property
    def mean_prediction(self) -> float:
        return float(np.mean(self.predictions)) if len(self.predictions) > 0 else 0.0
    
    @property
    def max_prediction(self) -> float:
        return float(np.max(self.predictions)) if len(self.predictions) > 0 else 0.0


class ModelService:
    """Service for ML model management and predictions."""
    
    SUPPORTED_MODELS = ["lgbm", "prophet", "sarima", "ensemble"]
    
    def __init__(self, models_dir: Path):
        self.models_dir = models_dir
        self._loaded_models: dict = {}
        self._feature_scaler = None
        self._feature_names: list[str] = []
    
    def get_available_models(self) -> dict[str, bool]:
        """Check which models are available."""
        available = {}
        for model_type in self.SUPPORTED_MODELS:
            if model_type == "ensemble":
                # Ensemble requires at least 2 other models
                count = sum(1 for m in ["lgbm", "prophet", "sarima"] if self._model_exists(m))
                available[model_type] = count >= 2
            else:
                available[model_type] = self._model_exists(model_type)
        return available
    
    def _model_exists(self, model_type: str) -> bool:
        """Check if model file exists."""
        model_file = self.models_dir / f"{model_type}_5m.pkl"
        return model_file.exists()

    def _load_pickle_or_joblib(self, file_path: Path):
        """Robust loader that supports both joblib and pickle files."""
        try:
            return joblib.load(file_path)
        except ModuleNotFoundError as missing_dep:
            # Common when prophet/statsmodels/lightgbm not installed in current env
            raise RuntimeError(
                f"Missing dependency while loading {file_path.name}: {missing_dep}. "
                "Please ensure all requirements are installed in the same environment as Streamlit."
            ) from missing_dep
        except Exception as joblib_err:
            try:
                with open(file_path, 'rb') as f:
                    return pickle.load(f)
            except Exception as pickle_err:
                raise RuntimeError(
                    f"Joblib load failed ({joblib_err}); pickle load failed ({pickle_err})"
                ) from pickle_err
    
    def load_model(self, model_type: str) -> bool:
        """Load a pre-trained model."""
        if model_type in self._loaded_models:
            return True
        
        model_file = self.models_dir / f"{model_type}_5m.pkl"
        
        if not model_file.exists():
            st.error(f"Model file not found: {model_file}")
            return False
        
        try:
            loaded = self._load_pickle_or_joblib(model_file)

            # Some persisted artifacts are stored as dicts (e.g., Prophet/SARIMA bundles)
            if isinstance(loaded, dict):
                model_obj = (
                    loaded.get("model")
                    or loaded.get("fitted_model")
                    or loaded
                )
            else:
                model_obj = loaded

            self._loaded_models[model_type] = model_obj
            
            # Load feature scaler if exists
            scaler_file = self.models_dir / "feature_scaler.pkl"
            if scaler_file.exists() and self._feature_scaler is None:
                try:
                    self._feature_scaler = self._load_pickle_or_joblib(scaler_file)
                except Exception as e:
                    st.warning(f"Feature scaler load failed: {e}")
            
            # Load feature names if exists
            feature_file = self.models_dir / "feature_names.json"
            if feature_file.exists() and not self._feature_names:
                import json
                with open(feature_file, 'r') as f:
                    self._feature_names = json.load(f)
            
            st.success(f"Loaded {model_type.upper()} model")
            return True
            
        except Exception as e:
            # Show full stack to help users debug missing deps (e.g., prophet)
            st.exception(e)
            st.error(f"Error loading model: {e}")
            return False
    
    def forecast(
        self,
        historical_data: np.ndarray,
        horizon: int,
        model_type: str = "lgbm",
        confidence_level: float = 0.95,
    ) -> ForecastResult | None:
        """Generate forecast using specified model."""
        
        if model_type == "ensemble":
            return self._ensemble_forecast(historical_data, horizon, confidence_level)
        
        if model_type not in self._loaded_models:
            if not self.load_model(model_type):
                return None
        
        model = self._loaded_models[model_type]
        
        try:
            with st.spinner(f"Generating {horizon}-period forecast with {model_type.upper()}..."):
                if model_type == "lgbm":
                    return self._lgbm_forecast(model, historical_data, horizon, confidence_level)
                elif model_type == "prophet":
                    return self._prophet_forecast(model, historical_data, horizon, confidence_level)
                elif model_type == "sarima":
                    return self._sarima_forecast(model, historical_data, horizon, confidence_level)
                else:
                    st.error(f"Unknown model type: {model_type}")
                    return None
        except Exception as e:
            st.error(f"Forecast error: {e}")
            return None
    
    def _lgbm_forecast(
        self,
        model,
        historical_data: np.ndarray,
        horizon: int,
        confidence_level: float,
    ) -> ForecastResult:
        """Generate forecast using LightGBM model.
        
        Note: Model was trained with 91 complex features. Since we only have
        raw request counts, we use simple seasonal forecast instead.
        """
        # Model needs 91 features but we only have request_count time series
        # Fallback to seasonal forecast which works well for traffic patterns
        st.warning(
            "LightGBM model requires 91 engineered features. "
            "Using seasonal pattern forecast instead."
        )
        
        predictions = self._simple_seasonal_forecast(historical_data, horizon)
        
        # Estimate prediction intervals
        std_estimate = np.std(historical_data[-100:]) if len(historical_data) >= 100 else np.std(historical_data)
        z_score = 1.96 if confidence_level == 0.95 else 1.28
        
        lower = predictions - z_score * std_estimate
        upper = predictions + z_score * std_estimate
        
        # Generate timestamps
        timestamps = self._generate_timestamps(horizon)
        
        return ForecastResult(
            timestamps=timestamps,
            predictions=predictions,
            lower_bound=np.maximum(lower, 0),
            upper_bound=upper,
            model_type="lgbm (seasonal fallback)",
            confidence_level=confidence_level,
        )
    
    def _prophet_forecast(
        self,
        model,
        historical_data: np.ndarray,
        horizon: int,
        confidence_level: float,
    ) -> ForecastResult:
        """Generate forecast using Prophet model."""
        if hasattr(model, 'predict') and hasattr(model, 'make_future_dataframe'):
            # Actual Prophet model
            future = model.make_future_dataframe(periods=horizon, freq='5T')
            forecast = model.predict(future)
            
            predictions = forecast['yhat'].iloc[-horizon:].values
            lower = forecast['yhat_lower'].iloc[-horizon:].values
            upper = forecast['yhat_upper'].iloc[-horizon:].values
            timestamps = forecast['ds'].iloc[-horizon:].tolist()
        else:
            # Fallback: simple seasonal decomposition
            predictions = self._simple_seasonal_forecast(historical_data, horizon)
            std_estimate = np.std(historical_data[-288:]) if len(historical_data) >= 288 else np.std(historical_data)
            z_score = 1.96 if confidence_level == 0.95 else 1.28
            lower = predictions - z_score * std_estimate
            upper = predictions + z_score * std_estimate
            timestamps = self._generate_timestamps(horizon)
        
        return ForecastResult(
            timestamps=timestamps,
            predictions=predictions,
            lower_bound=np.maximum(lower, 0),
            upper_bound=upper,
            model_type="prophet",
            confidence_level=confidence_level,
        )
    
    def _sarima_forecast(
        self,
        model,
        historical_data: np.ndarray,
        horizon: int,
        confidence_level: float,
    ) -> ForecastResult:
        """Generate forecast using SARIMA model."""
        if hasattr(model, 'forecast'):
            try:
                forecast_result = model.forecast(steps=horizon)
                if isinstance(forecast_result, tuple):
                    predictions = forecast_result[0]
                    conf_int = forecast_result[1] if len(forecast_result) > 1 else None
                else:
                    predictions = forecast_result
                    conf_int = None
                
                if conf_int is not None:
                    lower = conf_int[:, 0]
                    upper = conf_int[:, 1]
                else:
                    std_estimate = np.std(historical_data[-100:])
                    z_score = 1.96 if confidence_level == 0.95 else 1.28
                    lower = predictions - z_score * std_estimate
                    upper = predictions + z_score * std_estimate
            except Exception:
                predictions = self._simple_seasonal_forecast(historical_data, horizon)
                std_estimate = np.std(historical_data[-100:])
                z_score = 1.96
                lower = predictions - z_score * std_estimate
                upper = predictions + z_score * std_estimate
        else:
            predictions = self._simple_seasonal_forecast(historical_data, horizon)
            std_estimate = np.std(historical_data[-100:])
            z_score = 1.96
            lower = predictions - z_score * std_estimate
            upper = predictions + z_score * std_estimate
        
        timestamps = self._generate_timestamps(horizon)
        
        return ForecastResult(
            timestamps=timestamps,
            predictions=np.array(predictions),
            lower_bound=np.maximum(np.array(lower), 0),
            upper_bound=np.array(upper),
            model_type="sarima",
            confidence_level=confidence_level,
        )
    
    def _ensemble_forecast(
        self,
        historical_data: np.ndarray,
        horizon: int,
        confidence_level: float,
    ) -> ForecastResult | None:
        """Generate ensemble forecast by averaging multiple models."""
        forecasts = []
        model_types = []
        
        for model_type in ["lgbm", "prophet", "sarima"]:
            if self._model_exists(model_type):
                result = self.forecast(historical_data, horizon, model_type, confidence_level)
                if result is not None:
                    forecasts.append(result.predictions)
                    model_types.append(model_type)
        
        if len(forecasts) < 2:
            st.error("Ensemble requires at least 2 models")
            return None
        
        # Average predictions
        predictions = np.mean(forecasts, axis=0)
        
        # Ensemble uncertainty (wider bounds)
        std_across_models = np.std(forecasts, axis=0)
        std_historical = np.std(historical_data[-100:])
        combined_std = np.sqrt(std_across_models**2 + std_historical**2)
        
        z_score = 1.96 if confidence_level == 0.95 else 1.28
        lower = predictions - z_score * combined_std
        upper = predictions + z_score * combined_std
        
        timestamps = self._generate_timestamps(horizon)
        
        return ForecastResult(
            timestamps=timestamps,
            predictions=predictions,
            lower_bound=np.maximum(lower, 0),
            upper_bound=upper,
            model_type=f"ensemble({','.join(model_types)})",
            confidence_level=confidence_level,
            metrics={"models_used": model_types},
        )
    
    def _create_features(self, data: np.ndarray) -> np.ndarray:
        """Create features from historical data for prediction."""
        # Simple lag features
        features = []
        
        # Last N values as features
        for lag in [1, 2, 3, 6, 12, 24, 48, 288]:
            if len(data) > lag:
                features.append(data[-lag])
            else:
                features.append(data[-1])
        
        # Rolling statistics
        for window in [6, 12, 24]:
            if len(data) >= window:
                features.append(np.mean(data[-window:]))
                features.append(np.std(data[-window:]))
            else:
                features.append(np.mean(data))
                features.append(np.std(data))
        
        return np.array(features)
    
    def _simple_seasonal_forecast(self, data: np.ndarray, horizon: int) -> np.ndarray:
        """Simple seasonal forecast using historical patterns."""
        # Assume daily seasonality (288 periods for 5-min data)
        seasonal_period = 288
        
        if len(data) >= seasonal_period:
            # Use average of same periods from past days
            predictions = []
            for i in range(horizon):
                idx = i % seasonal_period
                # Get same time from previous days
                past_values = data[idx::seasonal_period]
                predictions.append(np.mean(past_values[-7:]))  # Last 7 days
            return np.array(predictions)
        else:
            # Just use mean
            return np.full(horizon, np.mean(data))
    
    def _generate_timestamps(self, horizon: int) -> list[datetime]:
        """Generate future timestamps."""
        from datetime import timedelta
        now = datetime.now()
        return [now + timedelta(minutes=5*i) for i in range(1, horizon + 1)]
    
    def get_model_metrics(self, model_type: str) -> dict | None:
        """Get metrics for a loaded model."""
        metrics_file = self.models_dir / "all_model_results.json"
        if metrics_file.exists():
            import json
            with open(metrics_file, 'r') as f:
                all_results = json.load(f)
            return all_results.get(model_type, {})
        return None
