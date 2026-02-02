"""Historical analysis mode tabs."""

import numpy as np
import pandas as pd
import streamlit as st

from src.scaling.config import ScalingConfig
from app.services.simulator_service import SimulatorService, SimulationResult
from app.components.charts import (
    create_traffic_chart,
    create_scaling_chart,
    create_cost_comparison_chart,
    create_cumulative_cost_chart,
    downsample_for_viz,
)


def render_historical_mode(
    loads: np.ndarray,
    result: SimulationResult,
    config: ScalingConfig,
    config_preset: str,
    policy_type: str,
    enable_downsample: bool = True,
    viz_max_points: int = 10000,
):
    """Render all tabs for historical analysis mode."""
    
    # Prepare visualization data
    if enable_downsample and len(loads) > viz_max_points:
        loads_viz = downsample_for_viz(loads, viz_max_points)
        servers_viz = downsample_for_viz(np.array(result.servers_over_time), viz_max_points).tolist()
        utilizations_viz = downsample_for_viz(np.array(result.utilization_over_time), viz_max_points).tolist()
        costs_viz = downsample_for_viz(np.array(result.cost_over_time), viz_max_points).tolist()
        st.info(f"📈 Visualization downsampled from {len(loads):,} to {len(loads_viz):,} points. Simulation used full data.")
    else:
        loads_viz = loads
        servers_viz = result.servers_over_time
        utilizations_viz = result.utilization_over_time
        costs_viz = result.cost_over_time
    
    # Create tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Overview",
        "📈 Traffic Analysis",
        "🔄 Scaling Behavior",
        "💰 Cost Analysis",
        "📥 Export Report",
    ])
    
    with tab1:
        _render_overview(result.metrics, config_preset, policy_type, config)
    
    with tab2:
        _render_traffic_analysis(loads_viz, servers_viz, config)
    
    with tab3:
        _render_scaling_behavior(loads_viz, servers_viz, utilizations_viz, result.metrics, config)
    
    with tab4:
        _render_cost_analysis(loads, result.metrics, costs_viz, config)
    
    with tab5:
        _render_export(result, config_preset, policy_type, config)


def _render_overview(
    metrics: dict,
    config_preset: str,
    policy_type: str,
    config: ScalingConfig,
):
    """Render performance overview."""
    st.header("Performance Overview")
    
    # Key metrics row 1
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "💰 Total Cost",
            f"${metrics['total_cost']:.2f}",
            help="Total cost over simulation period",
        )
    with col2:
        st.metric(
            "🖥️ Avg Servers",
            f"{metrics['avg_servers']:.1f}",
            help="Average servers used",
        )
    with col3:
        st.metric(
            "📊 Avg Utilization",
            f"{metrics['avg_utilization']:.1%}",
            help="Average server utilization",
        )
    with col4:
        st.metric(
            "⚠️ SLA Violations",
            f"{metrics['sla_violations']} ({metrics['sla_violation_rate']:.1%})",
            help="Periods with overload",
        )
    
    # Key metrics row 2
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "🔄 Scaling Events",
            f"{metrics['scaling_events']}",
            help=f"Out: {metrics['scale_out_events']}, In: {metrics['scale_in_events']}",
        )
    with col2:
        st.metric(
            "📉 Server Range",
            f"{metrics['min_servers']} - {metrics['max_servers']}",
            help="Min and max servers during simulation",
        )
    with col3:
        st.metric(
            "💵 Cost/Hour",
            f"${metrics['avg_cost_per_hour']:.4f}",
            help="Average cost per hour",
        )
    with col4:
        st.metric(
            "🗑️ Wasted Capacity",
            f"{metrics['wasted_capacity_periods']}",
            help="Periods with < 30% utilization",
        )
    
    # Configuration info
    st.subheader("Current Configuration")
    config_df = pd.DataFrame([{
        "Preset": config_preset.capitalize(),
        "Policy": policy_type.capitalize(),
        "Min Servers": config.min_servers,
        "Max Servers": config.max_servers,
        "Scale Out": f"{config.scale_out_threshold:.0%}",
        "Scale In": f"{config.scale_in_threshold:.0%}",
        "Requests/Server": config.requests_per_server,
    }])
    st.dataframe(config_df, hide_index=True, use_container_width=True)


def _render_traffic_analysis(
    loads: np.ndarray,
    servers: list,
    config: ScalingConfig,
):
    """Render traffic analysis tab."""
    st.header("Traffic Analysis")
    
    # Traffic chart
    fig = create_traffic_chart(loads, servers, config)
    st.plotly_chart(fig, use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Load distribution histogram
        import plotly.express as px
        fig_hist = px.histogram(
            x=loads,
            nbins=50,
            title="Load Distribution",
            labels={"x": "Requests", "y": "Frequency"},
        )
        fig_hist.update_layout(height=300)
        st.plotly_chart(fig_hist, use_container_width=True)
    
    with col2:
        st.subheader("📊 Load Statistics")
        stats_df = pd.DataFrame([
            {"Metric": "Mean", "Value": f"{np.mean(loads):.1f}"},
            {"Metric": "Std Dev", "Value": f"{np.std(loads):.1f}"},
            {"Metric": "Min", "Value": f"{np.min(loads):.1f}"},
            {"Metric": "Max", "Value": f"{np.max(loads):.1f}"},
            {"Metric": "Median", "Value": f"{np.median(loads):.1f}"},
            {"Metric": "95th Percentile", "Value": f"{np.percentile(loads, 95):.1f}"},
        ])
        st.dataframe(stats_df, hide_index=True, use_container_width=True)


def _render_scaling_behavior(
    loads: np.ndarray,
    servers: list,
    utilizations: list,
    metrics: dict,
    config: ScalingConfig,
):
    """Render scaling behavior tab."""
    st.header("Scaling Behavior")
    
    # Multi-panel chart
    fig = create_scaling_chart(loads, servers, utilizations, config)
    st.plotly_chart(fig, use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🔄 Scaling Summary")
        scaling_df = pd.DataFrame([
            {"Metric": "Total Scaling Events", "Value": metrics['scaling_events']},
            {"Metric": "Scale Out Events", "Value": metrics['scale_out_events']},
            {"Metric": "Scale In Events", "Value": metrics['scale_in_events']},
            {"Metric": "Min Servers Used", "Value": metrics['min_servers']},
            {"Metric": "Max Servers Used", "Value": metrics['max_servers']},
        ])
        st.dataframe(scaling_df, hide_index=True, use_container_width=True)
    
    with col2:
        import plotly.express as px
        fig_util = px.histogram(
            x=utilizations,
            nbins=30,
            title="Utilization Distribution",
            labels={"x": "Utilization", "y": "Frequency"},
        )
        fig_util.add_vline(x=config.scale_out_threshold, line_dash="dash", line_color="red")
        fig_util.add_vline(x=config.scale_in_threshold, line_dash="dash", line_color="blue")
        fig_util.update_layout(height=300)
        st.plotly_chart(fig_util, use_container_width=True)


def _render_cost_analysis(
    loads: np.ndarray,
    metrics: dict,
    costs: list,
    config: ScalingConfig,
):
    """Render cost analysis tab."""
    st.header("Cost Analysis")
    
    # Initialize simulator for comparisons
    simulator = SimulatorService()
    fixed_results = simulator.compare_with_fixed(loads, config)
    
    # Build comparison data
    comparison_data = [
        {"Strategy": "Autoscaling", "Cost": metrics['total_cost'], "SLA Violations": metrics['sla_violations']},
        {"Strategy": fixed_results["fixed_min"].config_name, "Cost": fixed_results["fixed_min"].total_cost, "SLA Violations": fixed_results["fixed_min"].sla_violations},
        {"Strategy": fixed_results["fixed_max"].config_name, "Cost": fixed_results["fixed_max"].total_cost, "SLA Violations": fixed_results["fixed_max"].sla_violations},
        {"Strategy": fixed_results["fixed_optimal"].config_name, "Cost": fixed_results["fixed_optimal"].total_cost, "SLA Violations": fixed_results["fixed_optimal"].sla_violations},
    ]
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig_cost = create_cost_comparison_chart(comparison_data, "Cost")
        st.plotly_chart(fig_cost, use_container_width=True)
    
    with col2:
        fig_sla = create_cost_comparison_chart(comparison_data, "SLA Violations")
        st.plotly_chart(fig_sla, use_container_width=True)
    
    # Cost savings
    st.subheader("💸 Cost Savings from Autoscaling")
    
    fixed_max_cost = fixed_results["fixed_max"].total_cost
    fixed_optimal_cost = fixed_results["fixed_optimal"].total_cost
    
    savings_vs_max = fixed_max_cost - metrics['total_cost']
    savings_pct_max = (savings_vs_max / fixed_max_cost * 100) if fixed_max_cost > 0 else 0
    
    savings_vs_opt = fixed_optimal_cost - metrics['total_cost']
    savings_pct_opt = (savings_vs_opt / fixed_optimal_cost * 100) if fixed_optimal_cost > 0 else 0
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "vs Fixed Max",
            f"${savings_vs_max:.2f}",
            f"{savings_pct_max:.1f}%",
        )
    with col2:
        st.metric(
            "vs Fixed Optimal",
            f"${savings_vs_opt:.2f}",
            f"{savings_pct_opt:.1f}%",
        )
    with col3:
        sla_diff = fixed_results["fixed_optimal"].sla_violations - metrics['sla_violations']
        st.metric(
            "SLA vs Optimal",
            f"{abs(sla_diff)} periods",
            "better" if sla_diff > 0 else ("worse" if sla_diff < 0 else "same"),
        )
    
    # Cumulative cost chart
    st.subheader("📈 Cumulative Cost Over Time")
    
    fig_cumcost = create_cumulative_cost_chart(
        costs_autoscale=costs,
        costs_fixed_max=fixed_results["fixed_max"].cost_over_time,
        costs_fixed_optimal=fixed_results["fixed_optimal"].cost_over_time,
        labels=(
            "Autoscaling",
            fixed_results["fixed_max"].config_name,
            fixed_results["fixed_optimal"].config_name,
        ),
    )
    st.plotly_chart(fig_cumcost, use_container_width=True)
    
    # Detailed comparison table
    st.subheader("📋 Detailed Comparison")
    st.dataframe(pd.DataFrame(comparison_data), hide_index=True, use_container_width=True)


def _render_export(
    result: SimulationResult,
    config_preset: str,
    policy_type: str,
    config: ScalingConfig,
):
    """Render export options."""
    st.header("📥 Export Report")
    
    st.write("Download simulation results in various formats.")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # CSV export
        metrics_df = pd.DataFrame([result.metrics])
        csv_data = metrics_df.to_csv(index=False)
        st.download_button(
            label="📊 Download Metrics (CSV)",
            data=csv_data,
            file_name="simulation_metrics.csv",
            mime="text/csv",
        )
    
    with col2:
        # JSON export
        import json
        json_data = json.dumps({
            "config_preset": config_preset,
            "policy_type": policy_type,
            "config": {
                "min_servers": config.min_servers,
                "max_servers": config.max_servers,
                "scale_out_threshold": config.scale_out_threshold,
                "scale_in_threshold": config.scale_in_threshold,
            },
            "metrics": result.metrics,
        }, indent=2)
        st.download_button(
            label="📄 Download Report (JSON)",
            data=json_data,
            file_name="simulation_report.json",
            mime="application/json",
        )
    
    with col3:
        # Time series export
        ts_df = pd.DataFrame({
            "servers": result.servers_over_time,
            "utilization": result.utilization_over_time,
            "cost": result.cost_over_time,
        })
        ts_csv = ts_df.to_csv(index=False)
        st.download_button(
            label="📈 Download Time Series (CSV)",
            data=ts_csv,
            file_name="simulation_timeseries.csv",
            mime="text/csv",
        )
    
    # Report preview
    st.subheader("📋 Report Preview")
    
    report_text = f"""
AUTOSCALING SIMULATION REPORT
=============================

Configuration: {config_preset.upper()}
Policy: {policy_type.upper()}

PERFORMANCE METRICS
-------------------
Total Cost: ${result.metrics['total_cost']:.2f}
Average Servers: {result.metrics['avg_servers']:.1f}
SLA Violations: {result.metrics['sla_violations']} ({result.metrics['sla_violation_rate']:.2%})
Scaling Events: {result.metrics['scaling_events']}

CONFIGURATION SETTINGS
----------------------
Min Servers: {config.min_servers}
Max Servers: {config.max_servers}
Scale Out Threshold: {config.scale_out_threshold:.0%}
Scale In Threshold: {config.scale_in_threshold:.0%}
Requests per Server: {config.requests_per_server}
    """
    
    st.text_area("Report", report_text, height=350, disabled=True)
