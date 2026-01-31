"""Prophet model wrapper for time series forecasting."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import joblib

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    Prophet = None  # Type hint placeholder
    PROPHET_AVAILABLE = False


@dataclass
class ProphetConfig:
    """Configuration for Prophet model."""

    # Growth model
    growth: Literal["linear", "logistic", "flat"] = "linear"
    # Seasonality settings
    yearly_seasonality: bool | int = False  # Only ~2 months of data
    weekly_seasonality: bool | int = True
    daily_seasonality: bool | int = True
    # Seasonality mode
    seasonality_mode: Literal["additive", "multiplicative"] = "additive"
    # Changepoint settings
    n_changepoints: int = 25
    changepoint_prior_scale: float = 0.05
    # Seasonality prior scale
    seasonality_prior_scale: float = 10.0
    # Holiday settings
    holidays_prior_scale: float = 10.0
    # Uncertainty
    interval_width: float = 0.95
    # Custom seasonalities
    custom_seasonalities: list = field(default_factory=list)


class ProphetModel:
    """Prophet model wrapper for traffic prediction.

    Facebook Prophet model for time series forecasting with
    automatic seasonality detection and holiday effects.
    """

    def __init__(self, config: ProphetConfig | None = None):
        """Initialize Prophet model.

        Args:
            config: Model configuration
        """
        if not PROPHET_AVAILABLE:
            raise ImportError("Prophet is not installed. Install with: pip install prophet")

        self.config = config or ProphetConfig()
        self.model = None
        self.train_df = None

    def _create_model(self):
        """Create Prophet model with configuration.

        Returns:
            Configured Prophet model
        """
        model = Prophet(
            growth=self.config.growth,
            yearly_seasonality=self.config.yearly_seasonality,
            weekly_seasonality=self.config.weekly_seasonality,
            daily_seasonality=self.config.daily_seasonality,
            seasonality_mode=self.config.seasonality_mode,
            n_changepoints=self.config.n_changepoints,
            changepoint_prior_scale=self.config.changepoint_prior_scale,
            seasonality_prior_scale=self.config.seasonality_prior_scale,
            holidays_prior_scale=self.config.holidays_prior_scale,
            interval_width=self.config.interval_width,
        )

        # Add custom seasonalities
        for seasonality in self.config.custom_seasonalities:
            model.add_seasonality(**seasonality)

        return model

    def fit(
        self,
        df: pd.DataFrame,
        y_col: str = "y",
        ds_col: str = "ds",
        verbose: bool = True,
    ) -> "ProphetModel":
        """Fit Prophet model.

        Args:
            df: DataFrame with timestamp and target columns
            y_col: Name of target column
            ds_col: Name of timestamp column
            verbose: Whether to print progress

        Returns:
            Self for method chaining
        """
        # Prepare data in Prophet format
        train_df = df[[ds_col, y_col]].copy()
        train_df.columns = ["ds", "y"]

        # Ensure datetime
        train_df["ds"] = pd.to_datetime(train_df["ds"])

        self.train_df = train_df
        self.model = self._create_model()

        if verbose:
            print("Fitting Prophet model...")

        self.model.fit(train_df)

        if verbose:
            print("Model fitted successfully")

        return self

    def predict(
        self,
        periods: int,
        freq: str = "5min",
        include_history: bool = False,
    ) -> pd.DataFrame:
        """Generate forecasts.

        Args:
            periods: Number of periods to forecast
            freq: Frequency of predictions
            include_history: Whether to include historical predictions

        Returns:
            DataFrame with predictions
        """
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")

        # Create future dataframe
        future = self.model.make_future_dataframe(
            periods=periods,
            freq=freq,
            include_history=include_history,
        )

        # Generate predictions
        forecast = self.model.predict(future)

        return forecast

    def forecast(
        self,
        future_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Generate forecasts for specific dates.

        Args:
            future_df: DataFrame with 'ds' column

        Returns:
            DataFrame with predictions
        """
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")

        return self.model.predict(future_df)

    def get_components(self) -> dict:
        """Get seasonality components.

        Returns:
            Dictionary of component DataFrames
        """
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")

        components = {}

        # Get component names
        if self.model.weekly_seasonality:
            components["weekly"] = self._get_weekly_component()
        if self.model.daily_seasonality:
            components["daily"] = self._get_daily_component()

        return components

    def _get_weekly_component(self) -> pd.DataFrame:
        """Get weekly seasonality component.

        Returns:
            DataFrame with weekly component
        """
        days = pd.date_range(start="2017-01-01", periods=7, freq="D")
        df = pd.DataFrame({"ds": days})
        df["dow"] = df["ds"].dt.day_name()
        forecast = self.model.predict(df)
        return df.assign(weekly=forecast["weekly"])

    def _get_daily_component(self) -> pd.DataFrame:
        """Get daily seasonality component.

        Returns:
            DataFrame with daily component
        """
        hours = pd.date_range(start="2017-01-01", periods=24 * 12, freq="5min")
        df = pd.DataFrame({"ds": hours})
        df["hour"] = df["ds"].dt.hour + df["ds"].dt.minute / 60
        forecast = self.model.predict(df)
        return df.assign(daily=forecast["daily"])

    def plot_components(self, figsize: tuple = (12, 10)):
        """Plot model components.

        Args:
            figsize: Figure size
        """
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")

        # Create forecast for plotting
        future = self.model.make_future_dataframe(periods=0, include_history=True)
        forecast = self.model.predict(future)

        return self.model.plot_components(forecast, figsize=figsize)

    def plot_forecast(self, forecast: pd.DataFrame, figsize: tuple = (14, 6)):
        """Plot forecast with uncertainty.

        Args:
            forecast: Forecast DataFrame from predict()
            figsize: Figure size
        """
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")

        return self.model.plot(forecast, figsize=figsize)

    def cross_validate(
        self,
        initial: str = "30 days",
        period: str = "7 days",
        horizon: str = "1 days",
    ) -> pd.DataFrame:
        """Perform cross-validation.

        Args:
            initial: Initial training period
            period: Period between cutoff dates
            horizon: Forecast horizon

        Returns:
            Cross-validation results DataFrame
        """
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")

        from prophet.diagnostics import cross_validation

        return cross_validation(
            self.model,
            initial=initial,
            period=period,
            horizon=horizon,
        )

    def get_cv_metrics(self, cv_results: pd.DataFrame) -> pd.DataFrame:
        """Calculate cross-validation metrics.

        Args:
            cv_results: Results from cross_validate()

        Returns:
            Metrics DataFrame
        """
        from prophet.diagnostics import performance_metrics

        return performance_metrics(cv_results)

    def add_regressor(self, name: str, **kwargs) -> "ProphetModel":
        """Add external regressor.

        Args:
            name: Regressor name
            **kwargs: Additional arguments for add_regressor

        Returns:
            Self for method chaining
        """
        if self.model is None:
            self.model = self._create_model()

        self.model.add_regressor(name, **kwargs)
        return self

    def add_holidays(self, holidays_df: pd.DataFrame) -> "ProphetModel":
        """Add custom holidays.

        Args:
            holidays_df: DataFrame with 'holiday', 'ds' columns

        Returns:
            Self for method chaining
        """
        if self.model is None:
            self.model = self._create_model()

        self.model.holidays = holidays_df
        return self

    def save(self, filepath: str | Path) -> None:
        """Save model to file.

        Args:
            filepath: Path to save model
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        model_data = {
            "config": self.config,
            "model": self.model,
            "train_df": self.train_df,
        }
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")

    @classmethod
    def load(cls, filepath: str | Path) -> "ProphetModel":
        """Load model from file.

        Args:
            filepath: Path to saved model

        Returns:
            Loaded model
        """
        model_data = joblib.load(filepath)
        instance = cls(config=model_data["config"])
        instance.model = model_data["model"]
        instance.train_df = model_data["train_df"]
        return instance

    @staticmethod
    def prepare_dataframe(
        df: pd.DataFrame,
        timestamp_col: str = "timestamp",
        target_col: str = "request_count",
    ) -> pd.DataFrame:
        """Prepare DataFrame for Prophet.

        Args:
            df: Input DataFrame
            timestamp_col: Name of timestamp column
            target_col: Name of target column

        Returns:
            DataFrame in Prophet format (ds, y)
        """
        prophet_df = df[[timestamp_col, target_col]].copy()
        prophet_df.columns = ["ds", "y"]
        prophet_df["ds"] = pd.to_datetime(prophet_df["ds"])
        return prophet_df
