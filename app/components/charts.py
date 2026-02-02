"""Shared chart components."""

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.scaling.config import ScalingConfig
from app.services.model_service import ForecastResult


def create_traffic_chart(
    loads: np.ndarray,
    servers: list[int],
    config: ScalingConfig,
    title: str = "Traffic Load vs Capacity",
) -> go.Figure:
    """Create traffic load vs capacity chart."""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        y=loads,
        mode='lines',
        name='Request Load',
        line=dict(color='#1f77b4', width=1),
    ))
    
    capacity = np.array(servers) * config.requests_per_server
    fig.add_trace(go.Scatter(
        y=capacity,
        mode='lines',
        name='Capacity',
        line=dict(color='#2ca02c', width=2, dash='dash'),
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Time Period (5-min intervals)",
        yaxis_title="Requests",
        height=400,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
    )
    
    return fig


def create_scaling_chart(
    loads: np.ndarray,
    servers: list[int],
    utilizations: list[float],
    config: ScalingConfig,
) -> go.Figure:
    """Create multi-panel scaling behavior chart."""
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=("Request Load", "Active Servers", "Utilization"),
    )
    
    fig.add_trace(
        go.Scatter(y=loads, mode='lines', name='Load', line=dict(color='#1f77b4')),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(y=servers, mode='lines', name='Servers', line=dict(color='#ff7f0e')),
        row=2, col=1,
    )
    fig.add_trace(
        go.Scatter(y=utilizations, mode='lines', name='Utilization', line=dict(color='#2ca02c')),
        row=3, col=1,
    )
    
    # Threshold lines
    fig.add_hline(
        y=config.scale_out_threshold, line_dash="dash", line_color="red",
        annotation_text="Scale Out", row=3, col=1,
    )
    fig.add_hline(
        y=config.scale_in_threshold, line_dash="dash", line_color="blue",
        annotation_text="Scale In", row=3, col=1,
    )
    
    fig.update_layout(height=600, showlegend=False)
    fig.update_yaxes(title_text="Requests", row=1, col=1)
    fig.update_yaxes(title_text="Servers", row=2, col=1)
    fig.update_yaxes(title_text="Utilization", row=3, col=1)
    fig.update_xaxes(title_text="Time Period", row=3, col=1)
    
    return fig


def create_cost_comparison_chart(
    comparison_data: list[dict],
    metric: str = "Cost",
) -> go.Figure:
    """Create cost comparison bar chart."""
    import pandas as pd
    df = pd.DataFrame(comparison_data)
    
    fig = px.bar(
        df,
        x="Strategy",
        y=metric,
        color="Strategy",
        title=f"{metric} Comparison",
    )
    fig.update_layout(height=400, showlegend=False)
    
    return fig


def create_cumulative_cost_chart(
    costs_autoscale: list[float],
    costs_fixed_max: list[float],
    costs_fixed_optimal: list[float],
    labels: tuple[str, str, str] = ("Autoscaling", "Fixed Max", "Fixed Optimal"),
) -> go.Figure:
    """Create cumulative cost comparison chart."""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        y=np.cumsum(costs_autoscale),
        mode='lines',
        name=labels[0],
        line=dict(color='#1f77b4', width=2),
    ))
    
    fig.add_trace(go.Scatter(
        y=np.cumsum(costs_fixed_max),
        mode='lines',
        name=labels[1],
        line=dict(color='#ff7f0e', width=2, dash='dash'),
    ))
    
    fig.add_trace(go.Scatter(
        y=np.cumsum(costs_fixed_optimal),
        mode='lines',
        name=labels[2],
        line=dict(color='#2ca02c', width=2, dash='dot'),
    ))
    
    fig.update_layout(
        title="Cumulative Cost Over Time",
        xaxis_title="Time Period",
        yaxis_title="Cumulative Cost ($)",
        height=400,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
    )
    
    return fig


def create_forecast_chart(
    historical: np.ndarray,
    forecast: ForecastResult,
    title: str = "Traffic Forecast",
) -> go.Figure:
    """Create forecast visualization with confidence intervals."""
    fig = go.Figure()
    
    # Historical data
    n_hist = len(historical)
    fig.add_trace(go.Scatter(
        x=list(range(n_hist)),
        y=historical,
        mode='lines',
        name='Historical',
        line=dict(color='#636363', width=1),
    ))
    
    # Forecast
    forecast_x = list(range(n_hist, n_hist + len(forecast.predictions)))
    
    # Confidence interval (filled area)
    fig.add_trace(go.Scatter(
        x=forecast_x + forecast_x[::-1],
        y=list(forecast.upper_bound) + list(forecast.lower_bound[::-1]),
        fill='toself',
        fillcolor='rgba(31, 119, 180, 0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        name=f'{int(forecast.confidence_level*100)}% CI',
        showlegend=True,
    ))
    
    # Forecast line
    fig.add_trace(go.Scatter(
        x=forecast_x,
        y=forecast.predictions,
        mode='lines',
        name='Forecast',
        line=dict(color='#1f77b4', width=2),
    ))
    
    # Vertical line at forecast start
    fig.add_vline(x=n_hist, line_dash="dash", line_color="gray", opacity=0.5)
    
    fig.update_layout(
        title=title,
        xaxis_title="Time Period",
        yaxis_title="Request Load",
        height=450,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
    )
    
    return fig


def create_comparison_matrix_chart(
    matrix_data: list[dict],
) -> go.Figure:
    """Create heatmap for config/policy comparison matrix."""
    import pandas as pd
    
    df = pd.DataFrame(matrix_data)
    
    # Pivot for heatmap
    pivot = df.pivot(index="Config", columns="Policy", values="Cost")
    
    fig = go.Figure(data=go.Heatmap(
        z=pivot.values,
        x=pivot.columns,
        y=pivot.index,
        colorscale="RdYlGn_r",
        text=[[f"${v:.2f}" for v in row] for row in pivot.values],
        texttemplate="%{text}",
        textfont={"size": 12},
        hovertemplate="Config: %{y}<br>Policy: %{x}<br>Cost: %{text}<extra></extra>",
    ))
    
    fig.update_layout(
        title="Cost Matrix: Config × Policy",
        xaxis_title="Policy",
        yaxis_title="Configuration",
        height=300,
    )
    
    return fig


def create_what_if_chart(
    base_costs: list[float],
    scenario_costs: list[float],
    scenario_name: str = "Scenario",
) -> go.Figure:
    """Create what-if scenario comparison chart."""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        y=np.cumsum(base_costs),
        mode='lines',
        name='Base Case',
        line=dict(color='#1f77b4', width=2),
    ))
    
    fig.add_trace(go.Scatter(
        y=np.cumsum(scenario_costs),
        mode='lines',
        name=scenario_name,
        line=dict(color='#ff7f0e', width=2, dash='dash'),
    ))
    
    fig.update_layout(
        title="What-If Scenario Comparison",
        xaxis_title="Time Period",
        yaxis_title="Cumulative Cost ($)",
        height=400,
    )
    
    return fig


def downsample_for_viz(
    data: np.ndarray,
    max_points: int = 10000,
) -> np.ndarray:
    """Downsample data for visualization using LTTB-like algorithm."""
    if len(data) <= max_points:
        return data
    
    # Simple averaging downsample (preserves patterns better than skip)
    factor = len(data) // max_points
    n_full = (len(data) // factor) * factor
    
    downsampled = data[:n_full].reshape(-1, factor).mean(axis=1)
    
    # Add remaining points
    if n_full < len(data):
        downsampled = np.append(downsampled, data[n_full:].mean())
    
    return downsampled
