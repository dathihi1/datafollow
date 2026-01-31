"""SARIMA model wrapper for time series forecasting."""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import joblib
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller


@dataclass
class SARIMAConfig:
    """Configuration for SARIMA model."""

    # ARIMA order (p, d, q)
    order: tuple[int, int, int] = (1, 1, 1)
    # Seasonal order (P, D, Q, m)
    seasonal_order: tuple[int, int, int, int] = (1, 1, 1, 12)
    # Trend component
    trend: Literal["n", "c", "t", "ct"] | None = None
    # Enforce stationarity
    enforce_stationarity: bool = True
    # Enforce invertibility
    enforce_invertibility: bool = True


class SARIMAModel:
    """SARIMA model wrapper for traffic prediction.

    Seasonal AutoRegressive Integrated Moving Average model
    for time series forecasting with seasonality.
    """

    def __init__(self, config: SARIMAConfig | None = None):
        """Initialize SARIMA model.

        Args:
            config: Model configuration
        """
        self.config = config or SARIMAConfig()
        self.model = None
        self.fitted_model = None
        self.train_index = None

    def fit(
        self,
        y: pd.Series,
        exog: pd.DataFrame | None = None,
        verbose: bool = True,
    ) -> "SARIMAModel":
        """Fit SARIMA model.

        Args:
            y: Target time series (should have datetime index)
            exog: Exogenous variables (optional)
            verbose: Whether to print fitting progress

        Returns:
            Self for method chaining
        """
        self.train_index = y.index

        if verbose:
            print(f"Fitting SARIMA{self.config.order}x{self.config.seasonal_order}...")

        self.model = SARIMAX(
            y,
            exog=exog,
            order=self.config.order,
            seasonal_order=self.config.seasonal_order,
            trend=self.config.trend,
            enforce_stationarity=self.config.enforce_stationarity,
            enforce_invertibility=self.config.enforce_invertibility,
        )

        self.fitted_model = self.model.fit(disp=verbose)

        if verbose:
            print(f"AIC: {self.fitted_model.aic:.2f}")
            print(f"BIC: {self.fitted_model.bic:.2f}")

        return self

    def predict(
        self,
        steps: int,
        exog: pd.DataFrame | None = None,
        return_conf_int: bool = False,
        alpha: float = 0.05,
    ) -> pd.Series | tuple[pd.Series, pd.DataFrame]:
        """Generate forecasts.

        Args:
            steps: Number of steps to forecast
            exog: Exogenous variables for forecast period
            return_conf_int: Whether to return confidence intervals
            alpha: Significance level for confidence intervals

        Returns:
            Forecast series, optionally with confidence intervals
        """
        if self.fitted_model is None:
            raise ValueError("Model not fitted. Call fit() first.")

        forecast = self.fitted_model.get_forecast(steps=steps, exog=exog)
        predictions = forecast.predicted_mean

        if return_conf_int:
            conf_int = forecast.conf_int(alpha=alpha)
            return predictions, conf_int

        return predictions

    def forecast(
        self,
        start: int | str | pd.Timestamp,
        end: int | str | pd.Timestamp,
        exog: pd.DataFrame | None = None,
    ) -> pd.Series:
        """Generate forecasts for a specific range.

        Args:
            start: Start of forecast period
            end: End of forecast period
            exog: Exogenous variables

        Returns:
            Forecast series
        """
        if self.fitted_model is None:
            raise ValueError("Model not fitted. Call fit() first.")

        return self.fitted_model.predict(start=start, end=end, exog=exog)

    def get_fitted_values(self) -> pd.Series:
        """Get in-sample fitted values.

        Returns:
            Fitted values series
        """
        if self.fitted_model is None:
            raise ValueError("Model not fitted. Call fit() first.")

        return self.fitted_model.fittedvalues

    def get_residuals(self) -> pd.Series:
        """Get model residuals.

        Returns:
            Residuals series
        """
        if self.fitted_model is None:
            raise ValueError("Model not fitted. Call fit() first.")

        return self.fitted_model.resid

    def summary(self) -> str:
        """Get model summary.

        Returns:
            Summary string
        """
        if self.fitted_model is None:
            raise ValueError("Model not fitted. Call fit() first.")

        return str(self.fitted_model.summary())

    def get_metrics(self) -> dict:
        """Get model metrics.

        Returns:
            Dictionary of metrics
        """
        if self.fitted_model is None:
            raise ValueError("Model not fitted. Call fit() first.")

        return {
            "aic": self.fitted_model.aic,
            "bic": self.fitted_model.bic,
            "hqic": self.fitted_model.hqic,
            "llf": self.fitted_model.llf,
        }

    def save(self, filepath: str | Path) -> None:
        """Save model to file.

        Args:
            filepath: Path to save model
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        model_data = {
            "config": self.config,
            "fitted_model": self.fitted_model,
            "train_index": self.train_index,
        }
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")

    @classmethod
    def load(cls, filepath: str | Path) -> "SARIMAModel":
        """Load model from file.

        Args:
            filepath: Path to saved model

        Returns:
            Loaded model
        """
        model_data = joblib.load(filepath)
        model = cls(config=model_data["config"])
        model.fitted_model = model_data["fitted_model"]
        model.train_index = model_data["train_index"]
        return model

    @staticmethod
    def test_stationarity(series: pd.Series, verbose: bool = True) -> dict:
        """Test series stationarity using ADF test.

        Args:
            series: Time series to test
            verbose: Whether to print results

        Returns:
            Test results dictionary
        """
        result = adfuller(series.dropna(), autolag="AIC")

        output = {
            "adf_statistic": result[0],
            "p_value": result[1],
            "lags_used": result[2],
            "n_observations": result[3],
            "critical_values": result[4],
            "is_stationary": result[1] < 0.05,
        }

        if verbose:
            print("ADF Stationarity Test:")
            print(f"  ADF Statistic: {output['adf_statistic']:.4f}")
            print(f"  p-value: {output['p_value']:.4f}")
            print(f"  Critical Values:")
            for key, value in output["critical_values"].items():
                print(f"    {key}: {value:.4f}")
            print(f"  Stationary: {output['is_stationary']}")

        return output

    @staticmethod
    def grid_search(
        y: pd.Series,
        p_range: range = range(0, 3),
        d_range: range = range(0, 2),
        q_range: range = range(0, 3),
        P_range: range = range(0, 2),
        D_range: range = range(0, 2),
        Q_range: range = range(0, 2),
        m: int = 12,
        criterion: str = "aic",
        verbose: bool = True,
    ) -> tuple[tuple, float]:
        """Grid search for optimal SARIMA parameters.

        Args:
            y: Target time series
            p_range, d_range, q_range: ARIMA order ranges
            P_range, D_range, Q_range: Seasonal order ranges
            m: Seasonal period
            criterion: Selection criterion ('aic' or 'bic')
            verbose: Whether to print progress

        Returns:
            Tuple of (best_order, best_seasonal_order, best_score)
        """
        best_score = np.inf
        best_order = None
        best_seasonal_order = None

        total_combinations = (
            len(p_range) * len(d_range) * len(q_range) *
            len(P_range) * len(D_range) * len(Q_range)
        )
        tested = 0

        for p in p_range:
            for d in d_range:
                for q in q_range:
                    for P in P_range:
                        for D in D_range:
                            for Q in Q_range:
                                tested += 1
                                order = (p, d, q)
                                seasonal_order = (P, D, Q, m)

                                try:
                                    model = SARIMAX(
                                        y,
                                        order=order,
                                        seasonal_order=seasonal_order,
                                        enforce_stationarity=False,
                                        enforce_invertibility=False,
                                    )
                                    fitted = model.fit(disp=False)
                                    score = fitted.aic if criterion == "aic" else fitted.bic

                                    if score < best_score:
                                        best_score = score
                                        best_order = order
                                        best_seasonal_order = seasonal_order

                                        if verbose:
                                            print(f"[{tested}/{total_combinations}] "
                                                  f"SARIMA{order}x{seasonal_order}: {criterion.upper()}={score:.2f}")

                                except Exception:
                                    continue

        if verbose:
            print(f"\nBest model: SARIMA{best_order}x{best_seasonal_order}")
            print(f"Best {criterion.upper()}: {best_score:.2f}")

        return best_order, best_seasonal_order, best_score
