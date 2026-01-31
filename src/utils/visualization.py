"""Visualization utilities for time series analysis."""

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Set style
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("husl")


def plot_time_series(
    df: pd.DataFrame,
    value_col: str = "request_count",
    timestamp_col: str = "timestamp",
    title: str = "Time Series",
    figsize: tuple = (14, 5),
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """Plot time series data.

    Args:
        df: DataFrame with time series data
        value_col: Column to plot
        timestamp_col: Timestamp column
        title: Plot title
        figsize: Figure size
        ax: Existing axes to plot on

    Returns:
        Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    ax.plot(df[timestamp_col], df[value_col], linewidth=0.8)
    ax.set_xlabel("Time")
    ax.set_ylabel(value_col.replace("_", " ").title())
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=45)

    plt.tight_layout()
    return ax


def plot_hourly_pattern(
    df: pd.DataFrame,
    value_col: str = "request_count",
    figsize: tuple = (12, 5),
) -> plt.Figure:
    """Plot hourly traffic pattern.

    Args:
        df: DataFrame with hour column
        value_col: Value column to aggregate
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Box plot
    hourly_data = df.groupby("hour")[value_col].apply(list)
    ax1 = axes[0]
    ax1.boxplot([hourly_data.get(h, []) for h in range(24)], positions=range(24))
    ax1.set_xlabel("Hour of Day")
    ax1.set_ylabel(value_col.replace("_", " ").title())
    ax1.set_title("Hourly Distribution")
    ax1.set_xticks(range(0, 24, 2))

    # Mean line plot
    ax2 = axes[1]
    hourly_mean = df.groupby("hour")[value_col].mean()
    ax2.plot(hourly_mean.index, hourly_mean.values, marker="o", linewidth=2)
    ax2.fill_between(hourly_mean.index, hourly_mean.values, alpha=0.3)
    ax2.set_xlabel("Hour of Day")
    ax2.set_ylabel(f"Mean {value_col.replace('_', ' ').title()}")
    ax2.set_title("Average Hourly Pattern")
    ax2.set_xticks(range(0, 24, 2))

    plt.tight_layout()
    return fig


def plot_weekly_pattern(
    df: pd.DataFrame,
    value_col: str = "request_count",
    figsize: tuple = (10, 5),
) -> plt.Figure:
    """Plot weekly traffic pattern.

    Args:
        df: DataFrame with day_of_week column
        value_col: Value column to aggregate
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    daily_mean = df.groupby("day_of_week")[value_col].mean()

    bars = ax.bar(range(7), [daily_mean.get(i, 0) for i in range(7)])

    # Color weekends differently
    bars[5].set_color("lightcoral")
    bars[6].set_color("lightcoral")

    ax.set_xticks(range(7))
    ax.set_xticklabels(days)
    ax.set_xlabel("Day of Week")
    ax.set_ylabel(f"Mean {value_col.replace('_', ' ').title()}")
    ax.set_title("Weekly Pattern")

    plt.tight_layout()
    return fig


def plot_heatmap(
    df: pd.DataFrame,
    value_col: str = "request_count",
    figsize: tuple = (12, 8),
) -> plt.Figure:
    """Plot heatmap of hour vs day of week.

    Args:
        df: DataFrame with hour and day_of_week columns
        value_col: Value column to aggregate
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    pivot = df.pivot_table(
        values=value_col,
        index="day_of_week",
        columns="hour",
        aggfunc="mean",
    )

    days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

    sns.heatmap(
        pivot,
        ax=ax,
        cmap="YlOrRd",
        annot=False,
        fmt=".0f",
        yticklabels=days,
        cbar_kws={"label": value_col.replace("_", " ").title()},
    )

    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Day of Week")
    ax.set_title(f"Traffic Heatmap: Hour vs Day of Week")

    plt.tight_layout()
    return fig


def plot_forecast_comparison(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    timestamps: Optional[pd.Series] = None,
    title: str = "Forecast vs Actual",
    figsize: tuple = (14, 5),
) -> plt.Figure:
    """Plot forecast comparison with actual values.

    Args:
        y_true: Actual values
        y_pred: Predicted values
        timestamps: Optional timestamps for x-axis
        title: Plot title
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    x = timestamps if timestamps is not None else range(len(y_true))

    # Time series comparison
    ax1 = axes[0]
    ax1.plot(x, y_true, label="Actual", linewidth=1.5)
    ax1.plot(x, y_pred, label="Predicted", linewidth=1.5, linestyle="--")
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Value")
    ax1.set_title(title)
    ax1.legend()
    if timestamps is not None:
        ax1.tick_params(axis="x", rotation=45)

    # Scatter plot
    ax2 = axes[1]
    ax2.scatter(y_true, y_pred, alpha=0.5, s=10)

    # Add diagonal line
    min_val = min(min(y_true), min(y_pred))
    max_val = max(max(y_true), max(y_pred))
    ax2.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2)

    ax2.set_xlabel("Actual")
    ax2.set_ylabel("Predicted")
    ax2.set_title("Prediction vs Actual Scatter")

    plt.tight_layout()
    return fig


def plot_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    figsize: tuple = (14, 4),
) -> plt.Figure:
    """Plot residual analysis.

    Args:
        y_true: Actual values
        y_pred: Predicted values
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    residuals = np.array(y_true) - np.array(y_pred)

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # Residuals over time
    ax1 = axes[0]
    ax1.plot(residuals, linewidth=0.5)
    ax1.axhline(y=0, color="r", linestyle="--")
    ax1.set_xlabel("Index")
    ax1.set_ylabel("Residual")
    ax1.set_title("Residuals Over Time")

    # Residual histogram
    ax2 = axes[1]
    ax2.hist(residuals, bins=50, edgecolor="black", alpha=0.7)
    ax2.axvline(x=0, color="r", linestyle="--")
    ax2.set_xlabel("Residual")
    ax2.set_ylabel("Frequency")
    ax2.set_title("Residual Distribution")

    # Q-Q plot (simplified)
    ax3 = axes[2]
    sorted_residuals = np.sort(residuals)
    theoretical_quantiles = np.linspace(0.01, 0.99, len(residuals))
    expected_values = np.percentile(residuals, theoretical_quantiles * 100)
    ax3.scatter(expected_values, sorted_residuals, alpha=0.5, s=5)
    ax3.plot([min(residuals), max(residuals)], [min(residuals), max(residuals)], "r--")
    ax3.set_xlabel("Theoretical Quantiles")
    ax3.set_ylabel("Sample Quantiles")
    ax3.set_title("Q-Q Plot")

    plt.tight_layout()
    return fig


def plot_scaling_timeline(
    df: pd.DataFrame,
    figsize: tuple = (14, 8),
) -> plt.Figure:
    """Plot autoscaling timeline with traffic and server count.

    Args:
        df: DataFrame with timestamp, request_count, and servers columns
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)

    # Traffic
    ax1 = axes[0]
    ax1.plot(df["timestamp"], df["request_count"], linewidth=0.8)
    ax1.set_ylabel("Requests")
    ax1.set_title("Traffic Volume")

    # Server count
    ax2 = axes[1]
    ax2.step(df["timestamp"], df["servers"], where="post", linewidth=1.5)
    ax2.set_ylabel("Servers")
    ax2.set_title("Active Servers")

    # Utilization
    ax3 = axes[2]
    if "utilization" in df.columns:
        ax3.plot(df["timestamp"], df["utilization"] * 100, linewidth=0.8)
        ax3.axhline(y=80, color="r", linestyle="--", label="Scale-out threshold")
        ax3.axhline(y=30, color="g", linestyle="--", label="Scale-in threshold")
        ax3.set_ylabel("Utilization (%)")
        ax3.set_title("Server Utilization")
        ax3.legend(loc="upper right")

    ax3.set_xlabel("Time")
    ax3.tick_params(axis="x", rotation=45)

    plt.tight_layout()
    return fig
