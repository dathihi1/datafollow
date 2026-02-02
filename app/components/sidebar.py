"""Sidebar component for dashboard configuration."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import streamlit as st
import numpy as np

from src.scaling.config import ScalingConfig, BALANCED_CONFIG, CONSERVATIVE_CONFIG, AGGRESSIVE_CONFIG


@dataclass
class SidebarConfig:
    """Configuration from sidebar inputs."""
    
    # Mode
    mode: Literal["historical", "predictive"] = "historical"
    
    # Data source
    data_source: Literal["sample", "upload_csv", "upload_txt", "saved", "manual"] = "sample"
    
    # Scaling
    config_preset: Literal["conservative", "balanced", "aggressive"] = "balanced"
    policy_type: Literal["balanced", "reactive", "predictive"] = "balanced"
    
    # Custom config
    config: ScalingConfig = field(default_factory=lambda: BALANCED_CONFIG)
    
    # Sample data settings
    n_periods: int = 288
    seed: int = 42
    
    # Visualization
    enable_downsample: bool = True
    viz_max_points: int = 10000
    
    # Predictive mode settings
    model_type: Literal["lgbm", "prophet", "sarima", "ensemble"] = "lgbm"
    forecast_horizon_days: int = 7
    confidence_level: float = 0.95
    
    # Priority for recommendations
    optimization_priority: Literal["cost", "sla", "balanced"] = "balanced"


def get_config(preset: str) -> ScalingConfig:
    """Get scaling configuration by preset name."""
    configs = {
        "conservative": CONSERVATIVE_CONFIG,
        "balanced": BALANCED_CONFIG,
        "aggressive": AGGRESSIVE_CONFIG,
    }
    return configs.get(preset, BALANCED_CONFIG)


def render_sidebar(uploads_dir: Path) -> SidebarConfig:
    """Render sidebar and return configuration."""
    
    st.sidebar.title("⚙️ Configuration")
    
    # Mode Selection
    st.sidebar.header("🔄 Mode")
    mode = st.sidebar.radio(
        "Analysis Mode",
        ["Historical Analysis", "Predictive Planning"],
        index=0,
        help="Historical: Analyze past data. Predictive: Forecast future traffic."
    )
    mode_key = "historical" if mode == "Historical Analysis" else "predictive"
    
    st.sidebar.divider()
    
    # Data Source
    st.sidebar.header("📊 Data Source")
    
    if mode_key == "historical":
        data_options = ["Sample Data", "Upload CSV", "Upload TXT", "Saved Files", "Manual Input"]
    else:
        data_options = ["Upload CSV", "Upload TXT", "Saved Files", "Sample Data"]
    
    data_source = st.sidebar.radio("Select Data Source", data_options)
    
    data_source_map = {
        "Sample Data": "sample",
        "Upload CSV": "upload_csv",
        "Upload TXT": "upload_txt",
        "Saved Files": "saved",
        "Manual Input": "manual",
    }
    data_source_key = data_source_map.get(data_source, "sample")
    
    # Sample data settings
    n_periods = 288
    seed = 42
    if data_source_key == "sample":
        n_periods = st.sidebar.slider("Periods (5-min intervals)", 48, 576, 288)
        seed = st.sidebar.number_input("Random Seed", 0, 1000, 42)
    
    st.sidebar.divider()
    
    # Scaling Configuration
    st.sidebar.header("🎛️ Scaling Configuration")
    
    config_preset = st.sidebar.selectbox(
        "Configuration Preset",
        ["balanced", "conservative", "aggressive"],
        index=0,
        format_func=str.capitalize,
    )
    
    policy_type = st.sidebar.selectbox(
        "Scaling Policy",
        ["balanced", "reactive", "predictive"],
        index=0,
        format_func=str.capitalize,
    )
    
    # Get base config
    config = get_config(config_preset)
    
    # Advanced Settings
    with st.sidebar.expander("🔧 Advanced Settings"):
        min_servers = st.number_input("Min Servers", 1, 50, config.min_servers)
        max_servers = st.number_input("Max Servers", 1, 50, config.max_servers)
        requests_per_server = st.number_input("Requests/Server", 10, 500, config.requests_per_server)
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
    
    # Visualization Settings
    with st.sidebar.expander("📈 Visualization"):
        enable_downsample = st.checkbox(
            "Auto-downsample charts",
            value=True,
            help="Downsample for faster rendering. Simulation uses full data."
        )
        viz_max_points = 10000
        if enable_downsample:
            viz_max_points = st.slider("Max chart points", 1000, 50000, 10000)
    
    # Predictive Mode Settings
    model_type = "lgbm"
    forecast_horizon_days = 7
    confidence_level = 0.95
    optimization_priority = "balanced"
    
    if mode_key == "predictive":
        st.sidebar.divider()
        st.sidebar.header("🔮 Forecast Settings")
        
        model_type = st.sidebar.selectbox(
            "Prediction Model",
            ["lgbm", "prophet", "sarima", "ensemble"],
            index=0,
            format_func=lambda x: {
                "lgbm": "LightGBM (Fast)",
                "prophet": "Prophet (Seasonal)",
                "sarima": "SARIMA (Statistical)",
                "ensemble": "Ensemble (All)",
            }.get(x, x),
        )
        
        forecast_horizon_days = st.sidebar.selectbox(
            "Forecast Horizon",
            [7, 14, 30],
            index=0,
            format_func=lambda x: f"{x} days",
        )
        
        confidence_level = st.sidebar.select_slider(
            "Confidence Level",
            options=[0.80, 0.90, 0.95, 0.99],
            value=0.95,
            format_func=lambda x: f"{x:.0%}",
        )
        
        optimization_priority = st.sidebar.radio(
            "Optimization Priority",
            ["cost", "sla", "balanced"],
            index=2,
            format_func=lambda x: {
                "cost": "💰 Minimize Cost",
                "sla": "🛡️ Minimize SLA Violations",
                "balanced": "⚖️ Balanced",
            }.get(x, x),
        )
    
    # Show saved files info
    saved_files = list(uploads_dir.glob("*"))
    if saved_files:
        st.sidebar.divider()
        st.sidebar.caption(f"📁 {len(saved_files)} saved files in DATA/uploads/")
    
    return SidebarConfig(
        mode=mode_key,
        data_source=data_source_key,
        config_preset=config_preset,
        policy_type=policy_type,
        config=config,
        n_periods=n_periods,
        seed=seed,
        enable_downsample=enable_downsample,
        viz_max_points=viz_max_points,
        model_type=model_type,
        forecast_horizon_days=forecast_horizon_days,
        confidence_level=confidence_level,
        optimization_priority=optimization_priority,
    )
