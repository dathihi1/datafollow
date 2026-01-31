"""Streamlit dashboard for NASA traffic autoscaling analysis."""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.scaling.config import ScalingConfig, BALANCED_CONFIG, CONSERVATIVE_CONFIG, AGGRESSIVE_CONFIG
from src.scaling.policy import ScalingPolicy, PredictivePolicy, ReactivePolicy
from src.scaling.simulator import CostSimulator

logger = logging.getLogger(__name__)

# Constants
DAILY_PERIODS_5MIN = 288
MIN_LOAD_VALUE = 10
BASE_DAILY_LOAD = 50
DAILY_AMPLITUDE = 80
NOISE_STD = 10
SPIKE_COUNT = 10
SPIKE_MIN = 50
SPIKE_MAX = 150
MAX_CSV_SIZE_MB = 50
MAX_CSV_ROWS = 1_000_000

# Page configuration
st.set_page_config(
    page_title="NASA Traffic Autoscaling Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)


def get_config(preset: str) -> ScalingConfig:
    """Get scaling configuration by preset name."""
    configs = {
        "conservative": CONSERVATIVE_CONFIG,
        "balanced": BALANCED_CONFIG,
        "aggressive": AGGRESSIVE_CONFIG,
    }
    return configs.get(preset, BALANCED_CONFIG)


def generate_sample_load(n_periods: int = DAILY_PERIODS_5MIN, seed: int = 42) -> np.ndarray:
    """Generate sample load data with realistic patterns."""
    rng = np.random.default_rng(seed)

    hours = np.arange(n_periods) % DAILY_PERIODS_5MIN
    hour_of_day = hours / 12

    daily_pattern = BASE_DAILY_LOAD + DAILY_AMPLITUDE * np.sin(np.pi * (hour_of_day - 6) / 12)
    daily_pattern = np.maximum(daily_pattern, 20)

    spikes = np.zeros(n_periods)
    spike_indices = rng.choice(n_periods, size=SPIKE_COUNT, replace=False)
    spikes[spike_indices] = rng.uniform(SPIKE_MIN, SPIKE_MAX, size=SPIKE_COUNT)

    noise = rng.normal(0, NOISE_STD, n_periods)

    load = daily_pattern + spikes + noise
    return np.maximum(load, MIN_LOAD_VALUE)


def run_simulation(
    loads: np.ndarray,
    config: ScalingConfig,
    policy_type: str = "balanced",
) -> tuple[dict, list, list, list] | None:
    """Run simulation and return metrics. Returns None on error."""
    try:
        simulator = CostSimulator(config)

        if policy_type == "reactive":
            policy = ReactivePolicy(config)
        elif policy_type == "predictive":
            policy = PredictivePolicy(config)
        else:
            policy = ScalingPolicy(config)

        metrics = simulator.simulate(loads, policy)

        return (
            metrics.__dict__,
            metrics.servers_over_time,
            metrics.utilization_over_time,
            metrics.cost_over_time,
        )
    except Exception as e:
        logger.error("Simulation failed: %s", e)
        st.error(f"Simulation failed: {e}")
        return None


def _render_overview_tab(
    metrics: dict,
    config_preset: str,
    policy_type: str,
    config: ScalingConfig,
) -> None:
    """Render performance overview tab."""
    st.header("Performance Overview")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Total Cost",
            f"${metrics['total_cost']:.2f}",
            help="Total cost over the simulation period",
        )
    with col2:
        st.metric(
            "Avg Servers",
            f"{metrics['avg_servers']:.1f}",
            help="Average number of servers used",
        )
    with col3:
        st.metric(
            "Avg Utilization",
            f"{metrics['avg_utilization']:.1%}",
            help="Average server utilization",
        )
    with col4:
        st.metric(
            "SLA Violations",
            f"{metrics['sla_violations']} ({metrics['sla_violation_rate']:.1%})",
            help="Number of periods with overload",
        )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Scaling Events",
            f"{metrics['scaling_events']}",
            help=f"Out: {metrics['scale_out_events']}, In: {metrics['scale_in_events']}",
        )
    with col2:
        st.metric(
            "Server Range",
            f"{metrics['min_servers']} - {metrics['max_servers']}",
            help="Min and max servers during simulation",
        )
    with col3:
        st.metric(
            "Cost per Hour",
            f"${metrics['avg_cost_per_hour']:.4f}",
            help="Average cost per hour",
        )
    with col4:
        st.metric(
            "Wasted Capacity",
            f"{metrics['wasted_capacity_periods']}",
            help="Periods with < 30% utilization",
        )

    st.subheader("Current Configuration")
    config_df = pd.DataFrame([{
        "Preset": config_preset,
        "Policy": policy_type,
        "Min Servers": config.min_servers,
        "Max Servers": config.max_servers,
        "Scale Out Threshold": f"{config.scale_out_threshold:.0%}",
        "Scale In Threshold": f"{config.scale_in_threshold:.0%}",
        "Requests/Server": config.requests_per_server,
    }])
    st.dataframe(config_df, hide_index=True, use_container_width=True)


def _render_traffic_tab(
    loads: np.ndarray,
    servers: list,
    config: ScalingConfig,
) -> None:
    """Render traffic analysis tab."""
    st.header("Traffic Analysis")

    fig_traffic = go.Figure()
    fig_traffic.add_trace(go.Scatter(
        y=loads,
        mode='lines',
        name='Request Load',
        line=dict(color='#1f77b4', width=1),
    ))

    capacity = np.array(servers) * config.requests_per_server
    fig_traffic.add_trace(go.Scatter(
        y=capacity,
        mode='lines',
        name='Capacity',
        line=dict(color='#2ca02c', width=2, dash='dash'),
    ))

    fig_traffic.update_layout(
        title="Traffic Load vs Capacity",
        xaxis_title="Time Period (5-min intervals)",
        yaxis_title="Requests",
        height=400,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
    )
    st.plotly_chart(fig_traffic, use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        fig_hist = px.histogram(
            x=loads,
            nbins=50,
            title="Load Distribution",
            labels={"x": "Requests", "y": "Frequency"},
        )
        fig_hist.update_layout(height=300)
        st.plotly_chart(fig_hist, use_container_width=True)

    with col2:
        st.subheader("Load Statistics")
        stats_df = pd.DataFrame([
            {"Metric": "Mean", "Value": f"{np.mean(loads):.1f}"},
            {"Metric": "Std Dev", "Value": f"{np.std(loads):.1f}"},
            {"Metric": "Min", "Value": f"{np.min(loads):.1f}"},
            {"Metric": "Max", "Value": f"{np.max(loads):.1f}"},
            {"Metric": "Median", "Value": f"{np.median(loads):.1f}"},
            {"Metric": "95th Percentile", "Value": f"{np.percentile(loads, 95):.1f}"},
        ])
        st.dataframe(stats_df, hide_index=True, use_container_width=True)


def _render_scaling_tab(
    loads: np.ndarray,
    servers: list,
    utilizations: list,
    metrics: dict,
    config: ScalingConfig,
) -> None:
    """Render scaling behavior tab."""
    st.header("Scaling Behavior")

    fig_scaling = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=("Request Load", "Active Servers", "Utilization"),
    )

    fig_scaling.add_trace(
        go.Scatter(y=loads, mode='lines', name='Load', line=dict(color='#1f77b4')),
        row=1, col=1,
    )
    fig_scaling.add_trace(
        go.Scatter(y=servers, mode='lines', name='Servers', line=dict(color='#ff7f0e')),
        row=2, col=1,
    )
    fig_scaling.add_trace(
        go.Scatter(y=utilizations, mode='lines', name='Utilization', line=dict(color='#2ca02c')),
        row=3, col=1,
    )

    fig_scaling.add_hline(
        y=config.scale_out_threshold, line_dash="dash", line_color="red",
        annotation_text="Scale Out", row=3, col=1,
    )
    fig_scaling.add_hline(
        y=config.scale_in_threshold, line_dash="dash", line_color="blue",
        annotation_text="Scale In", row=3, col=1,
    )

    fig_scaling.update_layout(height=600, showlegend=False)
    fig_scaling.update_yaxes(title_text="Requests", row=1, col=1)
    fig_scaling.update_yaxes(title_text="Servers", row=2, col=1)
    fig_scaling.update_yaxes(title_text="Utilization", row=3, col=1)
    fig_scaling.update_xaxes(title_text="Time Period", row=3, col=1)

    st.plotly_chart(fig_scaling, use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Scaling Summary")
        scaling_df = pd.DataFrame([
            {"Metric": "Total Scaling Events", "Value": metrics['scaling_events']},
            {"Metric": "Scale Out Events", "Value": metrics['scale_out_events']},
            {"Metric": "Scale In Events", "Value": metrics['scale_in_events']},
            {"Metric": "Min Servers Used", "Value": metrics['min_servers']},
            {"Metric": "Max Servers Used", "Value": metrics['max_servers']},
        ])
        st.dataframe(scaling_df, hide_index=True, use_container_width=True)

    with col2:
        fig_util = px.histogram(
            x=utilizations, nbins=30,
            title="Utilization Distribution",
            labels={"x": "Utilization", "y": "Frequency"},
        )
        fig_util.add_vline(x=config.scale_out_threshold, line_dash="dash", line_color="red")
        fig_util.add_vline(x=config.scale_in_threshold, line_dash="dash", line_color="blue")
        fig_util.update_layout(height=300)
        st.plotly_chart(fig_util, use_container_width=True)


def _render_cost_tab(
    loads: np.ndarray,
    metrics: dict,
    costs: list,
    config: ScalingConfig,
) -> None:
    """Render cost analysis tab."""
    st.header("Cost Analysis")

    simulator = CostSimulator(config)

    fixed_min = simulator.simulate_fixed(loads, config.min_servers)
    fixed_max = simulator.simulate_fixed(loads, config.max_servers)

    avg_load = np.mean(loads)
    optimal_fixed = config.get_required_servers(avg_load, target_utilization=0.7)
    fixed_optimal = simulator.simulate_fixed(loads, optimal_fixed)

    comparison_data = pd.DataFrame([
        {"Strategy": "Autoscaling", "Cost": metrics['total_cost'], "SLA Violations": metrics['sla_violations']},
        {"Strategy": f"Fixed Min ({config.min_servers})", "Cost": fixed_min.total_cost, "SLA Violations": fixed_min.sla_violations},
        {"Strategy": f"Fixed Max ({config.max_servers})", "Cost": fixed_max.total_cost, "SLA Violations": fixed_max.sla_violations},
        {"Strategy": f"Fixed Optimal ({optimal_fixed})", "Cost": fixed_optimal.total_cost, "SLA Violations": fixed_optimal.sla_violations},
    ])

    col1, col2 = st.columns(2)

    with col1:
        fig_cost = px.bar(
            comparison_data, x="Strategy", y="Cost",
            title="Cost Comparison", color="Strategy",
        )
        fig_cost.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig_cost, use_container_width=True)

    with col2:
        fig_sla = px.bar(
            comparison_data, x="Strategy", y="SLA Violations",
            title="SLA Violations Comparison", color="Strategy",
        )
        fig_sla.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig_sla, use_container_width=True)

    _render_cost_savings(metrics, fixed_max, fixed_optimal)
    _render_cumulative_cost(loads, costs, config, optimal_fixed)

    st.subheader("Detailed Strategy Comparison")
    st.dataframe(comparison_data, hide_index=True, use_container_width=True)


def _render_cost_savings(
    metrics: dict,
    fixed_max,
    fixed_optimal,
) -> None:
    """Render cost savings section."""
    st.subheader("Cost Savings from Autoscaling")

    savings_vs_max = fixed_max.total_cost - metrics['total_cost']
    savings_pct = (savings_vs_max / fixed_max.total_cost * 100) if fixed_max.total_cost > 0 else 0

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Savings vs Fixed Max",
            f"${savings_vs_max:.2f}",
            f"{savings_pct:.1f}%",
        )

    with col2:
        savings_vs_opt = fixed_optimal.total_cost - metrics['total_cost']
        savings_pct_opt = (savings_vs_opt / fixed_optimal.total_cost * 100) if fixed_optimal.total_cost > 0 else 0
        st.metric(
            "Savings vs Fixed Optimal",
            f"${savings_vs_opt:.2f}",
            f"{savings_pct_opt:.1f}%",
        )

    with col3:
        sla_diff = fixed_optimal.sla_violations - metrics['sla_violations']
        st.metric(
            "SLA Improvement vs Optimal",
            f"{abs(sla_diff)} periods",
            "better" if sla_diff > 0 else ("worse" if sla_diff < 0 else "same"),
        )


def _render_cumulative_cost(
    loads: np.ndarray,
    costs: list,
    config: ScalingConfig,
    optimal_fixed: int,
) -> None:
    """Render cumulative cost chart."""
    st.subheader("Cumulative Cost Over Time")

    cumulative_autoscale = np.cumsum(costs)
    cumulative_fixed_max = np.cumsum([config.get_cost_per_period(config.max_servers)] * len(loads))
    cumulative_fixed_opt = np.cumsum([config.get_cost_per_period(optimal_fixed)] * len(loads))

    fig_cumcost = go.Figure()
    fig_cumcost.add_trace(go.Scatter(
        y=cumulative_autoscale, mode='lines', name='Autoscaling',
        line=dict(color='#1f77b4', width=2),
    ))
    fig_cumcost.add_trace(go.Scatter(
        y=cumulative_fixed_max, mode='lines', name=f'Fixed Max ({config.max_servers})',
        line=dict(color='#ff7f0e', width=2, dash='dash'),
    ))
    fig_cumcost.add_trace(go.Scatter(
        y=cumulative_fixed_opt, mode='lines', name=f'Fixed Optimal ({optimal_fixed})',
        line=dict(color='#2ca02c', width=2, dash='dot'),
    ))

    fig_cumcost.update_layout(
        xaxis_title="Time Period", yaxis_title="Cumulative Cost ($)",
        height=400,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
    )
    st.plotly_chart(fig_cumcost, use_container_width=True)


def _load_csv_data(uploaded_file) -> np.ndarray | None:
    """Load and validate CSV data from upload."""
    if uploaded_file.size > MAX_CSV_SIZE_MB * 1024 * 1024:
        st.error(f"File too large. Maximum size is {MAX_CSV_SIZE_MB}MB.")
        return None

    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Error reading CSV: {e}")
        return None

    if len(df) > MAX_CSV_ROWS:
        st.error(f"CSV has too many rows ({len(df):,}). Maximum is {MAX_CSV_ROWS:,}.")
        return None

    if "load" in df.columns:
        return df["load"].values
    if "request_count" in df.columns:
        return df["request_count"].values

    st.error("CSV must contain 'load' or 'request_count' column")
    return None


def main():
    """Main dashboard application."""
    st.title("NASA Traffic Autoscaling Dashboard")
    st.markdown("Analyze traffic patterns and optimize autoscaling decisions for NASA web server logs.")

    # Sidebar configuration
    st.sidebar.header("Configuration")

    data_source = st.sidebar.radio(
        "Data Source",
        ["Sample Data", "Upload CSV", "Manual Input"],
    )
    config_preset = st.sidebar.selectbox(
        "Scaling Configuration",
        ["balanced", "conservative", "aggressive"],
        index=0,
    )
    policy_type = st.sidebar.selectbox(
        "Scaling Policy",
        ["balanced", "reactive", "predictive"],
        index=0,
    )

    config = get_config(config_preset)

    with st.sidebar.expander("Advanced Settings"):
        min_servers = st.number_input("Min Servers", 1, 50, config.min_servers)
        max_servers = st.number_input("Max Servers", 1, 50, config.max_servers)
        requests_per_server = st.number_input("Requests per Server", 10, 500, config.requests_per_server)
        scale_out_threshold = st.slider("Scale Out Threshold", 0.5, 1.0, config.scale_out_threshold)
        scale_in_threshold = st.slider("Scale In Threshold", 0.1, 0.5, config.scale_in_threshold)

        config = ScalingConfig(
            min_servers=min_servers,
            max_servers=max_servers,
            requests_per_server=requests_per_server,
            scale_out_threshold=scale_out_threshold,
            scale_in_threshold=scale_in_threshold,
            scale_out_consecutive=config.scale_out_consecutive,
            scale_in_consecutive=config.scale_in_consecutive,
            cooldown_minutes=config.cooldown_minutes,
            scale_out_increment=config.scale_out_increment,
            scale_in_decrement=config.scale_in_decrement,
            cost_per_server_per_hour=config.cost_per_server_per_hour,
        )

    # Load data
    loads = None

    if data_source == "Sample Data":
        n_periods = st.sidebar.slider("Number of Periods (5-min intervals)", 48, 576, DAILY_PERIODS_5MIN)
        seed = st.sidebar.number_input("Random Seed", 0, 1000, 42)
        loads = generate_sample_load(n_periods, seed)

    elif data_source == "Upload CSV":
        uploaded_file = st.sidebar.file_uploader("Upload CSV with 'load' column", type="csv")
        if uploaded_file is not None:
            loads = _load_csv_data(uploaded_file)

    elif data_source == "Manual Input":
        load_text = st.sidebar.text_area(
            "Enter loads (comma-separated)",
            "100, 120, 150, 180, 200, 180, 150, 120, 100, 80",
        )
        try:
            loads = np.array([float(x.strip()) for x in load_text.split(",")])
        except ValueError:
            st.error("Invalid input format. Please enter comma-separated numbers.")

    if loads is None or len(loads) == 0:
        st.warning("Please provide load data to analyze.")
        return

    # Run simulation
    result = run_simulation(loads, config, policy_type)
    if result is None:
        return

    metrics, servers, utilizations, costs = result

    # Render tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "Overview", "Traffic Analysis", "Scaling Behavior", "Cost Analysis",
    ])

    with tab1:
        _render_overview_tab(metrics, config_preset, policy_type, config)

    with tab2:
        _render_traffic_tab(loads, servers, config)

    with tab3:
        _render_scaling_tab(loads, servers, utilizations, metrics, config)

    with tab4:
        _render_cost_tab(loads, metrics, costs, config)


if __name__ == "__main__":
    main()
