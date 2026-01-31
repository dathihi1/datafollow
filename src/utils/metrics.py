"""Evaluation metrics for time series forecasting."""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Root Mean Squared Error.

    Args:
        y_true: Actual values
        y_pred: Predicted values

    Returns:
        RMSE value
    """
    return np.sqrt(mean_squared_error(y_true, y_pred))


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Mean Squared Error.

    Args:
        y_true: Actual values
        y_pred: Predicted values

    Returns:
        MSE value
    """
    return mean_squared_error(y_true, y_pred)


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Mean Absolute Error.

    Args:
        y_true: Actual values
        y_pred: Predicted values

    Returns:
        MAE value
    """
    return mean_absolute_error(y_true, y_pred)


def mape(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-10) -> float:
    """Calculate Mean Absolute Percentage Error.

    Args:
        y_true: Actual values
        y_pred: Predicted values
        epsilon: Small value to avoid division by zero

    Returns:
        MAPE value (as percentage, e.g., 15.5 means 15.5%)
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Avoid division by zero
    mask = np.abs(y_true) > epsilon
    if not np.any(mask):
        return np.nan

    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def smape(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-10) -> float:
    """Calculate Symmetric Mean Absolute Percentage Error.

    More robust than MAPE when dealing with values close to zero.

    Args:
        y_true: Actual values
        y_pred: Predicted values
        epsilon: Small value to avoid division by zero

    Returns:
        sMAPE value (as percentage)
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2 + epsilon
    return np.mean(np.abs(y_true - y_pred) / denominator) * 100


def evaluate_forecast(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    name: str = "Model",
) -> dict:
    """Evaluate forecast with multiple metrics.

    Args:
        y_true: Actual values
        y_pred: Predicted values
        name: Model name for reporting

    Returns:
        Dictionary of metrics
    """
    return {
        "model": name,
        "rmse": rmse(y_true, y_pred),
        "mse": mse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "mape": mape(y_true, y_pred),
        "smape": smape(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
    }


def compare_models(results: list[dict]) -> pd.DataFrame:
    """Compare multiple model results.

    Args:
        results: List of evaluation dictionaries

    Returns:
        DataFrame with comparison
    """
    df = pd.DataFrame(results)
    df = df.set_index("model")
    return df.round(4)


def print_metrics(metrics: dict) -> None:
    """Print metrics in a formatted way.

    Args:
        metrics: Dictionary of metrics
    """
    print(f"\nModel: {metrics.get('model', 'Unknown')}")
    print("-" * 40)
    print(f"  RMSE:  {metrics['rmse']:.4f}")
    print(f"  MSE:   {metrics['mse']:.4f}")
    print(f"  MAE:   {metrics['mae']:.4f}")
    print(f"  MAPE:  {metrics['mape']:.2f}%")
    print(f"  sMAPE: {metrics['smape']:.2f}%")
    print(f"  R2:    {metrics['r2']:.4f}")
