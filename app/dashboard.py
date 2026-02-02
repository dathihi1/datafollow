"""Streamlit dashboard for NASA traffic autoscaling analysis.

Optimized version with:
- LTTB downsampling for large datasets (preserves visual patterns)
- Disk caching for data loading and simulations
- Database persistence for simulation history
- Interactive visualizations with export
"""

import logging
import sys
from pathlib import Path
from datetime import datetime

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
MAX_CSV_SIZE_MB = 500
MAX_CSV_ROWS = 10_000_000
VIZ_MAX_POINTS = 10000  # Maximum points for visualization
UPLOADS_DIR = Path(__file__).parent.parent / "DATA" / "uploads"

# Page configuration
st.set_page_config(
    page_title="NASA Traffic Autoscaling Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize session state
if 'loaded_data' not in st.session_state:
    st.session_state.loaded_data = None
if 'loaded_data_original' not in st.session_state:
    st.session_state.loaded_data_original = None  # Full original data
if 'data_source_type' not in st.session_state:
    st.session_state.data_source_type = None
if 'uploaded_file_id' not in st.session_state:
    st.session_state.uploaded_file_id = None
if 'uploaded_file_path' not in st.session_state:
    st.session_state.uploaded_file_path = None
if 'simulation_cache' not in st.session_state:
    st.session_state.simulation_cache = {}

# Create uploads directory if it doesn't exist
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)


def save_uploaded_file(uploaded_file, file_type: str) -> Path | None:
    """Save uploaded file to uploads directory."""
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{uploaded_file.name}"
        filepath = UPLOADS_DIR / filename
        
        uploaded_file.seek(0)
        with open(filepath, 'wb') as f:
            f.write(uploaded_file.read())
        
        uploaded_file.seek(0)  # Reset for further processing
        return filepath
    except Exception as e:
        st.warning(f"Could not save file: {e}")
        return None


def lttb_downsample(data: np.ndarray, target_points: int) -> tuple[np.ndarray, np.ndarray]:
    """Downsample using Largest-Triangle-Three-Buckets algorithm.

    LTTB preserves visual characteristics better than averaging by
    selecting points that maximize the triangle area with neighbors.
    This keeps peaks and valleys visible in the downsampled data.

    Args:
        data: 1D array of values
        target_points: Target number of points

    Returns:
        Tuple of (indices, downsampled values)
    """
    n = len(data)
    if n <= target_points:
        return np.arange(n), data.copy()

    # Always keep first and last points
    indices = [0]
    sampled = [data[0]]

    bucket_size = (n - 2) / (target_points - 2)
    a = 0  # Previous selected point index

    for i in range(target_points - 2):
        # Calculate bucket range
        bucket_start = int((i + 1) * bucket_size) + 1
        bucket_end = int((i + 2) * bucket_size) + 1
        bucket_end = min(bucket_end, n - 1)

        # Calculate average of next bucket
        next_bucket_start = int((i + 2) * bucket_size) + 1
        next_bucket_end = int((i + 3) * bucket_size) + 1
        next_bucket_end = min(next_bucket_end, n)

        if next_bucket_start < next_bucket_end:
            avg_next = np.mean(data[next_bucket_start:next_bucket_end])
        else:
            avg_next = data[-1]

        # Find point with largest triangle area
        max_area = -1
        max_idx = bucket_start

        for j in range(bucket_start, bucket_end):
            # Simplified triangle area calculation
            area = abs(
                (j - a) * (avg_next - data[a]) -
                (bucket_end - a) * (data[j] - data[a])
            )
            if area > max_area:
                max_area = area
                max_idx = j

        indices.append(max_idx)
        sampled.append(data[max_idx])
        a = max_idx

    indices.append(n - 1)
    sampled.append(data[-1])

    return np.array(indices), np.array(sampled)


def downsample_data(loads: np.ndarray, max_points: int = 10000) -> np.ndarray:
    """Downsample data using LTTB algorithm for better visual fidelity."""
    if len(loads) <= max_points:
        return loads
    _, downsampled = lttb_downsample(loads, max_points)
    return downsampled


def downsample_multiple(*arrays: np.ndarray, max_points: int = 10000) -> list[np.ndarray]:
    """Downsample multiple arrays consistently using LTTB on first array.

    All arrays are downsampled using the same indices from LTTB on the first array,
    ensuring time-aligned visualization.
    """
    if len(arrays) == 0:
        return []

    n = len(arrays[0])
    if n <= max_points:
        return list(arrays)

    # Use LTTB on first array to get indices
    indices, _ = lttb_downsample(arrays[0], max_points)

    # Apply same indices to all arrays
    return [arr[indices] for arr in arrays]


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
    """Render traffic analysis tab with enhanced interactivity."""
    st.header("Traffic Analysis")

    fig_traffic = go.Figure()
    fig_traffic.add_trace(go.Scatter(
        y=loads,
        mode='lines',
        name='Request Load',
        line=dict(color='#1f77b4', width=1),
        hovertemplate="Period: %{x}<br>Load: %{y:.0f} requests<extra></extra>",
    ))

    capacity = np.array(servers) * config.requests_per_server
    fig_traffic.add_trace(go.Scatter(
        y=capacity,
        mode='lines',
        name='Capacity',
        line=dict(color='#2ca02c', width=2, dash='dash'),
        hovertemplate="Period: %{x}<br>Capacity: %{y:.0f} requests<extra></extra>",
    ))

    fig_traffic.update_layout(
        title="Traffic Load vs Capacity",
        xaxis=dict(
            title="Time Period (5-min intervals)",
            rangeslider=dict(visible=True, thickness=0.05),
        ),
        yaxis_title="Requests",
        height=500,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        hovermode="x unified",
    )
    st.plotly_chart(fig_traffic, use_container_width=True)

    # Export button
    col1, col2, col3 = st.columns([1, 1, 4])
    with col1:
        traffic_df = pd.DataFrame({
            "period": range(len(loads)),
            "load": loads,
            "capacity": capacity,
        })
        st.download_button(
            "Download Data (CSV)",
            traffic_df.to_csv(index=False),
            f"traffic_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            "text/csv",
            key="export_traffic_csv",
        )

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


def _load_txt_data(uploaded_file) -> np.ndarray | None:
    """Load and validate TXT data from upload."""
    # Check file size
    try:
        file_size_mb = uploaded_file.size / (1024 * 1024)
        if file_size_mb > MAX_CSV_SIZE_MB:
            st.error(f"File too large ({file_size_mb:.1f}MB). Maximum size is {MAX_CSV_SIZE_MB}MB.")
            return None
        
        if file_size_mb > 50:
            st.info(f"Loading large file ({file_size_mb:.1f}MB)... This may take a moment.")
    except AttributeError:
        pass

    try:
        uploaded_file.seek(0)
        content = uploaded_file.read().decode('utf-8')
        
        # Try different parsing strategies
        loads = []
        
        # Strategy 1: One number per line
        lines = content.strip().split('\n')
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):  # Skip empty lines and comments
                continue
            try:
                # Try to parse as single number
                loads.append(float(line))
            except ValueError:
                # Strategy 2: Space/tab/comma separated values in line
                parts = line.replace(',', ' ').replace('\t', ' ').split()
                for part in parts:
                    try:
                        loads.append(float(part))
                    except ValueError:
                        continue
        
        if not loads:
            st.error("Could not parse any numeric values from TXT file.")
            return None
        
        loads = np.array(loads)
        
        if len(loads) > MAX_CSV_ROWS:
            st.error(f"TXT has too many values ({len(loads):,}). Maximum is {MAX_CSV_ROWS:,}.")
            return None
        
        # Validate data
        if np.any(np.isnan(loads)):
            st.warning("TXT contains NaN values. They will be replaced with 0.")
            loads = np.nan_to_num(loads, nan=0.0)
        
        if np.any(loads < 0):
            st.warning("TXT contains negative values. They will be set to 0.")
            loads = np.maximum(loads, 0)
        
        st.success(f"Successfully loaded {len(loads):,} data points from TXT.")
        return loads
        
    except UnicodeDecodeError:
        st.error("Could not decode TXT file. Please ensure it's a text file with UTF-8 encoding.")
        return None
    except Exception as e:
        st.error(f"Error reading TXT: {type(e).__name__}: {e}")
        return None


def _load_csv_data(uploaded_file) -> np.ndarray | None:
    """Load and validate CSV data from upload."""
    # Check file size
    try:
        file_size_mb = uploaded_file.size / (1024 * 1024)
        if file_size_mb > MAX_CSV_SIZE_MB:
            st.error(f"File too large ({file_size_mb:.1f}MB). Maximum size is {MAX_CSV_SIZE_MB}MB.")
            return None
        
        if file_size_mb > 50:
            st.info(f"Loading large file ({file_size_mb:.1f}MB)... This may take a moment.")
    except AttributeError:
        # Some file objects may not have size attribute
        pass

    try:
        # For large files, use chunking to avoid memory issues
        uploaded_file.seek(0)  # Reset file pointer
        
        # Try to read file in chunks for large files
        if hasattr(uploaded_file, 'size') and uploaded_file.size > 50 * 1024 * 1024:
            chunks = []
            chunk_size = 100000  # Read 100k rows at a time
            
            with st.spinner("Reading large CSV file in chunks..."):
                for chunk in pd.read_csv(uploaded_file, chunksize=chunk_size):
                    chunks.append(chunk)
                    if sum(len(c) for c in chunks) > MAX_CSV_ROWS:
                        st.error(f"CSV has too many rows. Maximum is {MAX_CSV_ROWS:,}.")
                        return None
            
            df = pd.concat(chunks, ignore_index=True)
        else:
            df = pd.read_csv(uploaded_file)
            
    except pd.errors.EmptyDataError:
        st.error("CSV file is empty.")
        return None
    except pd.errors.ParserError as e:
        st.error(f"Error parsing CSV: {e}")
        return None
    except Exception as e:
        st.error(f"Error reading CSV: {type(e).__name__}: {e}")
        return None

    if len(df) > MAX_CSV_ROWS:
        st.error(f"CSV has too many rows ({len(df):,}). Maximum is {MAX_CSV_ROWS:,}.")
        return None
    
    if len(df) == 0:
        st.error("CSV file contains no data rows.")
        return None

    # Check for required columns
    if "load" in df.columns:
        loads = df["load"].values
    elif "request_count" in df.columns:
        loads = df["request_count"].values
    else:
        available_cols = ", ".join(df.columns.tolist())
        st.error(f"CSV must contain 'load' or 'request_count' column. Available columns: {available_cols}")
        return None
    
    # Validate data
    if np.any(np.isnan(loads)):
        st.warning("CSV contains NaN values. They will be replaced with 0.")
        loads = np.nan_to_num(loads, nan=0.0)
    
    if np.any(loads < 0):
        st.warning("CSV contains negative values. They will be set to 0.")
        loads = np.maximum(loads, 0)
    
    st.success(f"Successfully loaded {len(loads):,} data points from CSV.")
    return loads


def main():
    """Main dashboard application."""
    st.title("NASA Traffic Autoscaling Dashboard")
    st.markdown("Analyze traffic patterns and optimize autoscaling decisions for NASA web server logs.")

    # Sidebar configuration
    st.sidebar.header("Configuration")

    data_source = st.sidebar.radio(
        "Data Source",
        ["Sample Data", "Upload CSV", "Upload TXT", "Manual Input"],
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
    
    # Visualization settings
    with st.sidebar.expander("⚙️ Visualization Settings"):
        enable_downsample = st.checkbox(
            "Auto-downsample for visualization",
            value=True,
            help=f"Downsample to {VIZ_MAX_POINTS:,} points for faster charts. Simulation still uses full data."
        )
        if st.session_state.uploaded_file_path:
            st.info(f"💾 Saved: {st.session_state.uploaded_file_path.name}")
            if st.button("Clear saved file"):
                st.session_state.uploaded_file_path = None
                st.rerun()

    # Load data with caching
    loads = None
    data_changed = False

    if data_source == "Sample Data":
        n_periods = st.sidebar.slider("Number of Periods (5-min intervals)", 48, 576, DAILY_PERIODS_5MIN)
        seed = st.sidebar.number_input("Random Seed", 0, 1000, 42)
        
        data_key = f"sample_{n_periods}_{seed}"
        if st.session_state.data_source_type != data_key:
            loads = generate_sample_load(n_periods, seed)
            st.session_state.loaded_data = loads
            st.session_state.data_source_type = data_key
            data_changed = True
        else:
            loads = st.session_state.loaded_data

    elif data_source == "Upload CSV":
        st.sidebar.info(f"Maximum file size: {MAX_CSV_SIZE_MB}MB")
        uploaded_file = st.sidebar.file_uploader(
            "Upload CSV with 'load' or 'request_count' column",
            type="csv",
            help="CSV file should contain a column named 'load' or 'request_count' with numeric values",
            key="csv_uploader"
        )
        if uploaded_file is not None:
            if hasattr(uploaded_file, 'size'):
                st.sidebar.text(f"File size: {uploaded_file.size / (1024 * 1024):.2f}MB")
            
            file_id = f"csv_{uploaded_file.name}_{uploaded_file.size}"
            if st.session_state.uploaded_file_id != file_id:
                # Save uploaded file
                saved_path = save_uploaded_file(uploaded_file, "csv")
                if saved_path:
                    st.session_state.uploaded_file_path = saved_path
                    st.sidebar.success(f"✔ Saved to: DATA/uploads/")
                
                loads = _load_csv_data(uploaded_file)
                if loads is not None:
                    st.session_state.loaded_data_original = loads  # Store full data
                    st.session_state.loaded_data = loads
                    st.session_state.data_source_type = "csv"
                    st.session_state.uploaded_file_id = file_id
                    data_changed = True
            else:
                loads = st.session_state.loaded_data

    elif data_source == "Upload TXT":
        st.sidebar.info(f"Maximum file size: {MAX_CSV_SIZE_MB}MB")
        uploaded_file = st.sidebar.file_uploader(
            "Upload TXT file with numeric values",
            type="txt",
            help="TXT file with one number per line, or space/comma-separated values. Lines starting with # are ignored.",
            key="txt_uploader"
        )
        if uploaded_file is not None:
            if hasattr(uploaded_file, 'size'):
                st.sidebar.text(f"File size: {uploaded_file.size / (1024 * 1024):.2f}MB")
            
            file_id = f"txt_{uploaded_file.name}_{uploaded_file.size}"
            if st.session_state.uploaded_file_id != file_id:
                # Save uploaded file
                saved_path = save_uploaded_file(uploaded_file, "txt")
                if saved_path:
                    st.session_state.uploaded_file_path = saved_path
                    st.sidebar.success(f"✔ Saved to: DATA/uploads/")
                
                loads = _load_txt_data(uploaded_file)
                if loads is not None:
                    st.session_state.loaded_data_original = loads  # Store full data
                    st.session_state.loaded_data = loads
                    st.session_state.data_source_type = "txt"
                    st.session_state.uploaded_file_id = file_id
                    data_changed = True
            else:
                loads = st.session_state.loaded_data

    elif data_source == "Manual Input":
        load_text = st.sidebar.text_area(
            "Enter loads (comma-separated)",
            "100, 120, 150, 180, 200, 180, 150, 120, 100, 80",
        )
        try:
            data_key = f"manual_{hash(load_text)}"
            if st.session_state.data_source_type != data_key:
                loads = np.array([float(x.strip()) for x in load_text.split(",")])
                st.session_state.loaded_data = loads
                st.session_state.data_source_type = data_key
                data_changed = True
            else:
                loads = st.session_state.loaded_data
        except ValueError:
            st.error("Invalid input format. Please enter comma-separated numbers.")

    if loads is None or len(loads) == 0:
        st.warning("Please provide load data to analyze.")
        return

    # Display data info
    data_info_col1, data_info_col2, data_info_col3 = st.columns(3)
    with data_info_col1:
        st.metric("📊 Data Points", f"{len(loads):,}")
    with data_info_col2:
        if st.session_state.loaded_data_original is not None:
            st.metric("💾 Original Data", f"{len(st.session_state.loaded_data_original):,}")
        else:
            st.metric("💾 Original Data", f"{len(loads):,}")
    with data_info_col3:
        st.metric("⏱️ Time Range", f"{len(loads) * 5} min" if len(loads) < 1000 else f"{len(loads) * 5 / 60:.1f} hrs")
    
    # Create cache key for simulation (using full data)
    config_key = f"{config.min_servers}_{config.max_servers}_{config.requests_per_server}_{config.scale_out_threshold}_{config.scale_in_threshold}"
    cache_key = f"{st.session_state.data_source_type}_{config_key}_{policy_type}"
    
    # IMPORTANT: Run simulation on FULL data (loads contains all data)
    if cache_key not in st.session_state.simulation_cache or data_changed:
        with st.spinner(f"Running simulation on {len(loads):,} data points..."):
            result = run_simulation(loads, config, policy_type)
            if result is None:
                return
            st.session_state.simulation_cache[cache_key] = result
        st.success(f"✅ Simulation completed on full dataset ({len(loads):,} points)")
    else:
        result = st.session_state.simulation_cache[cache_key]
        st.info(f"♻️ Using cached simulation results ({len(loads):,} points)")
    
    metrics, servers, utilizations, costs = result
    
    # Prepare visualization data (downsample if needed and enabled)
    loads_viz = loads
    servers_viz = servers
    utilizations_viz = utilizations
    costs_viz = costs
    
    if enable_downsample and len(loads) > VIZ_MAX_POINTS:
        st.info(f"📈 Downsampling visualization from {len(loads):,} to {VIZ_MAX_POINTS:,} points for performance. Simulation used full data.")
        loads_viz = downsample_data(loads, VIZ_MAX_POINTS)
        servers_viz = downsample_data(np.array(servers), VIZ_MAX_POINTS).tolist()
        utilizations_viz = downsample_data(np.array(utilizations), VIZ_MAX_POINTS).tolist()
        costs_viz = downsample_data(np.array(costs), VIZ_MAX_POINTS).tolist()

    # Save to history option in sidebar
    with st.sidebar.expander("Save to History"):
        try:
            from app.db.service import SimulationService
            sim_service = SimulationService()

            save_name = st.text_input("Result Name", key="save_name")
            save_desc = st.text_area("Description", key="save_desc", height=68)

            if st.button("Save Result", key="save_result"):
                data_source_type = st.session_state.data_source_type or "unknown"
                if data_source_type.startswith("sample_"):
                    data_source_type = "sample"
                elif data_source_type.startswith("manual_"):
                    data_source_type = "manual"

                result_id = sim_service.save_result(
                    metrics=metrics,
                    servers=servers,
                    utilizations=utilizations,
                    costs=costs,
                    data_source_type=data_source_type,
                    data_points_count=len(loads),
                    config_preset=config_preset,
                    policy_type=policy_type,
                    config_dict=config.to_dict(),
                    name=save_name or None,
                    description=save_desc or None,
                    loads=loads,
                )
                st.success(f"Saved! ID: {result_id}")
        except ImportError:
            st.info("Install SQLAlchemy: `pip install sqlalchemy`")
        except Exception as e:
            st.error(f"Error: {e}")

    # Cache management
    with st.sidebar.expander("Cache Management"):
        try:
            from app.cache import clear_cache, get_cache_stats
            stats = get_cache_stats()
            st.write(f"Cached items: {stats['count']}")
            st.write(f"Cache size: {stats['total_size_mb']:.2f} MB")
            if st.button("Clear Cache"):
                count = clear_cache()
                st.session_state.simulation_cache = {}
                st.success(f"Cleared {count} disk cache items")
        except ImportError:
            if st.button("Clear Session Cache"):
                st.session_state.simulation_cache = {}
                st.success("Session cache cleared")

    # Render tabs (using visualization data)
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Overview", "Traffic Analysis", "Scaling Behavior", "Cost Analysis", "History",
    ])

    with tab1:
        _render_overview_tab(metrics, config_preset, policy_type, config)

    with tab2:
        _render_traffic_tab(loads_viz, servers_viz, config)

    with tab3:
        _render_scaling_tab(loads_viz, servers_viz, utilizations_viz, metrics, config)

    with tab4:
        _render_cost_tab(loads_viz, metrics, costs_viz, config)

    with tab5:
        _render_history_tab()


def _render_history_tab() -> None:
    """Render simulation history tab with database integration."""
    st.header("Simulation History")

    try:
        from app.db.service import SimulationService
        sim_service = SimulationService()
    except ImportError:
        st.error("Database not available. Install SQLAlchemy: `pip install sqlalchemy`")
        return
    except Exception as e:
        st.error(f"Database error: {e}")
        return

    # Statistics
    try:
        stats = sim_service.get_statistics()

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Saved", stats.get("total_results", 0))
        with col2:
            avg_cost = stats.get("avg_cost")
            st.metric("Avg Cost", f"${avg_cost:.2f}" if avg_cost else "N/A")
        with col3:
            avg_sla = stats.get("avg_sla_violation_rate")
            st.metric("Avg SLA Rate", f"{avg_sla:.1%}" if avg_sla else "N/A")
        with col4:
            if st.button("Refresh", key="refresh_history"):
                st.rerun()

        st.markdown("---")

        # Filters
        col1, col2, col3 = st.columns(3)
        with col1:
            filter_preset = st.selectbox(
                "Filter by Preset",
                ["All", "balanced", "conservative", "aggressive"],
                key="filter_preset",
            )
        with col2:
            filter_policy = st.selectbox(
                "Filter by Policy",
                ["All", "balanced", "reactive", "predictive"],
                key="filter_policy",
            )
        with col3:
            search = st.text_input("Search", key="search_history")

        # List results
        results = sim_service.list_results(
            limit=20,
            config_preset=filter_preset if filter_preset != "All" else None,
            policy_type=filter_policy if filter_policy != "All" else None,
            search=search if search else None,
        )

        if results:
            for res in results:
                with st.expander(f"{res['name']} - ${res['total_cost']:.2f}", expanded=False):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.write(f"**Created:** {res['created_at'][:16] if res['created_at'] else 'N/A'}")
                        st.write(f"**Preset:** {res['config_preset']}")
                        st.write(f"**Policy:** {res['policy_type']}")
                    with col2:
                        st.write(f"**Data Points:** {res['data_points_count']:,}")
                        st.write(f"**Avg Servers:** {res['avg_servers']:.1f}")
                        st.write(f"**SLA Violations:** {res['sla_violations']}")
                    with col3:
                        if st.button("Delete", key=f"del_{res['id']}"):
                            sim_service.delete_result(res['id'])
                            st.success("Deleted!")
                            st.rerun()
        else:
            st.info("No saved simulations yet. Run a simulation and save it from the sidebar.")

    except Exception as e:
        st.error(f"Error loading history: {e}")


if __name__ == "__main__":
    main()
