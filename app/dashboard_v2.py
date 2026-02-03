"""
Streamlit Dashboard for NASA Traffic Autoscaling Analysis.

Hybrid dashboard with two modes:
1. Historical Analysis: Analyze past traffic data and simulate scaling strategies
2. Predictive Planning: Forecast future traffic and recommend optimal configurations

Usage:
    streamlit run app/dashboard_v2.py
"""

import sys
from pathlib import Path

import numpy as np
import streamlit as st

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Local imports
from app.components.sidebar import render_sidebar, SidebarConfig
from app.components.historical import render_historical_mode
from app.components.predictive import render_predictive_mode
from app.services.data_loader import DataLoader, LoadedData
from app.services.simulator_service import SimulatorService

# Constants
UPLOADS_DIR = PROJECT_ROOT / "DATA" / "uploads"
MODELS_DIR = PROJECT_ROOT / "models"

# Page configuration
st.set_page_config(
    page_title="NASA Traffic Autoscaling Dashboard",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)


def init_session_state():
    """Initialize session state variables."""
    defaults = {
        'loaded_data': None,
        'data_file_id': None,
        'simulation_result': None,
        'forecast_result': None,
        'comparison_matrix': None,
        'recommendation': None,
    }
    for key, default in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default


def load_data(
    config: SidebarConfig,
    data_loader: DataLoader,
) -> LoadedData | None:
    """Load data based on sidebar configuration.

    Maintains uploaded file across mode changes by caching in session state.
    """

    if config.data_source == "sample":
        return data_loader.generate_sample(config.n_periods, config.seed)
    
    elif config.data_source == "upload_csv":
        uploaded_file = st.sidebar.file_uploader(
            "Upload CSV",
            type=["csv"],
            help="CSV with numeric column for load/traffic data",
            key="csv_uploader",
        )
        if uploaded_file:
            # Get available columns for selection
            numeric_cols, all_cols = data_loader.get_csv_columns(uploaded_file)

            if not numeric_cols:
                st.sidebar.error("No numeric columns found in CSV")
                return None

            # Default to common column names if present
            default_idx = 0
            for i, col in enumerate(numeric_cols):
                if col.lower() in ["load", "request_count", "target", "packets/time"]:
                    default_idx = i
                    break

            selected_column = st.sidebar.selectbox(
                "Select load column",
                options=numeric_cols,
                index=default_idx,
                help="Choose which column contains the traffic/load data",
            )

            # Timestamp column selection (optional) for auto-calculating interval
            timestamp_options = ["(None - use manual interval)"] + all_cols
            default_ts_idx = 0
            for i, col in enumerate(all_cols):
                if col.lower() in ["timestamp", "time", "datetime", "date"]:
                    default_ts_idx = i + 1  # +1 because of "(None)" option
                    break

            selected_timestamp = st.sidebar.selectbox(
                "Timestamp column (optional)",
                options=timestamp_options,
                index=default_ts_idx,
                help="Select timestamp column to auto-calculate time interval",
            )

            # If no timestamp selected, show manual interval selector
            timestamp_col = None if selected_timestamp.startswith("(None") else selected_timestamp
            time_interval = None

            if timestamp_col is None:
                time_interval = st.sidebar.selectbox(
                    "Time interval per row",
                    options=[1, 5, 15, 30, 60],
                    index=1,
                    format_func=lambda x: f"{x} minute{'s' if x > 1 else ''}",
                    help="Time interval each data row represents",
                )

            file_id = f"csv_{uploaded_file.name}_{uploaded_file.size}_{selected_column}_{timestamp_col}_{time_interval}"
            if st.session_state.data_file_id != file_id:
                result = data_loader.load_csv(
                    uploaded_file,
                    column_name=selected_column,
                    timestamp_column=timestamp_col,
                    time_interval_minutes=float(time_interval) if time_interval else None,
                )
                if result:
                    st.session_state.loaded_data = result
                    st.session_state.data_file_id = file_id
                return result
            return st.session_state.loaded_data
    
    elif config.data_source == "upload_txt":
        uploaded_file = st.sidebar.file_uploader(
            "Upload TXT",
            type=["txt"],
            help="NASA log format or numeric values (one per line/comma-separated)",
            key="txt_uploader",
        )
        if uploaded_file:
            # Aggregation interval selector for NASA logs (5min is optimal for ML)
            agg_interval = st.sidebar.selectbox(
                "Aggregation interval",
                options=[1, 5, 15],
                index=1,  # default 5min (optimal for ML)
                format_func=lambda x: f"{x} min" + (" (recommended)" if x == 5 else ""),
                help="5-minute interval achieves best model accuracy (LightGBM RMSE 3.59)",
                key="txt_agg_interval",
            )
            # Show model performance context based on benchmark results
            if agg_interval == 5:
                st.sidebar.caption("LightGBM RMSE: 3.59 (best)")
            elif agg_interval == 1:
                st.sidebar.caption("Prophet RMSE: 21.45 (noisier)")
            else:  # 15
                st.sidebar.caption("Prophet RMSE: 221.87 (coarser)")

            file_id = f"txt_{uploaded_file.name}_{uploaded_file.size}_{agg_interval}"
            if st.session_state.data_file_id != file_id:
                result = data_loader.load_txt(
                    uploaded_file,
                    aggregation_interval_minutes=agg_interval,
                )
                if result:
                    st.session_state.loaded_data = result
                    st.session_state.data_file_id = file_id
                return result
            return st.session_state.loaded_data
    
    elif config.data_source == "saved":
        saved_files = data_loader.list_saved_files()
        if saved_files:
            selected = st.sidebar.selectbox(
                "📂 Select Saved File",
                saved_files,
                format_func=lambda p: p.name,
            )
            if selected:
                file_id = f"saved_{selected.name}"
                if st.session_state.data_file_id != file_id:
                    result = data_loader.load_saved_file(selected)
                    if result:
                        st.session_state.loaded_data = result
                        st.session_state.data_file_id = file_id
                    return result
                return st.session_state.loaded_data
        else:
            st.sidebar.warning("No saved files found in DATA/uploads/")
    
    elif config.data_source == "manual":
        text = st.sidebar.text_area(
            "📝 Enter loads (comma-separated)",
            "100, 120, 150, 180, 200, 180, 150, 120, 100, 80",
        )
        return data_loader.parse_manual(text)
    
    return st.session_state.loaded_data


def main():
    """Main dashboard application."""
    
    # Initialize
    init_session_state()
    data_loader = DataLoader(UPLOADS_DIR)
    simulator = SimulatorService()
    
    # Title
    st.title("🚀 NASA Traffic Autoscaling Dashboard")
    
    # Render sidebar and get config
    config = render_sidebar(UPLOADS_DIR)
    
    # Load data
    loaded_data = load_data(config, data_loader)
    
    if loaded_data is None:
        st.warning("📊 Please select a data source and load data to begin.")
        
        # Show quick start guide
        with st.expander("📖 Quick Start Guide", expanded=True):
            st.markdown("""
            ### How to use this dashboard:
            
            1. **Select Mode** (sidebar):
               - **Historical Analysis**: Analyze past traffic and simulate scaling strategies
               - **Predictive Planning**: Forecast future traffic and get recommendations
            
            2. **Load Data**:
               - Upload a CSV/TXT file with traffic data
               - Use sample data for testing
               - Load previously saved files
            
            3. **Configure Scaling**:
               - Choose a preset (Conservative, Balanced, Aggressive)
               - Select a policy (Balanced, Reactive, Predictive)
               - Adjust advanced settings as needed
            
            4. **Analyze Results**:
               - View performance metrics
               - Compare cost strategies
               - Export reports
            
            ### File Format:
            - **CSV**: Any numeric column (select from dropdown after upload)
            - **TXT**: One number per line, or comma-separated values
            """)
        return
    
    # Data info header
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Data Points", f"{loaded_data.length:,}")
    with col2:
        time_span = loaded_data.time_range_days
        if time_span is not None:
            st.metric("Time Span", f"{time_span:.1f} days")
        else:
            st.metric("Max Load", f"{np.max(loaded_data.data):.1f}")
    with col3:
        interval = loaded_data.time_interval_minutes
        if interval is not None:
            st.metric("Interval", f"{int(interval)} min")
        else:
            st.metric("Mean Load", f"{np.mean(loaded_data.data):.1f}")
    with col4:
        st.metric("Source", loaded_data.source_type.upper())
    
    st.divider()
    
    # Route to appropriate mode
    if config.mode == "historical":
        # Update config with correct time interval from loaded data
        sim_config = config.config
        time_interval = loaded_data.time_interval_minutes
        if time_interval is not None and time_interval != sim_config.time_window_minutes:
            # Create new config with correct time window
            from dataclasses import replace
            sim_config = replace(sim_config, time_window_minutes=int(time_interval))

        # Run simulation if needed
        cache_key = f"{loaded_data.source_type}_{config.config_preset}_{config.policy_type}_{time_interval}"

        if st.session_state.simulation_result is None or \
           getattr(st.session_state, 'last_cache_key', None) != cache_key:

            with st.spinner("Running simulation..."):
                result = simulator.run_simulation(
                    loads=loaded_data.data,
                    config=sim_config,
                    policy_type=config.policy_type,
                    config_name=config.config_preset,
                )
                st.session_state.simulation_result = result
                st.session_state.last_cache_key = cache_key
        
        if st.session_state.simulation_result:
            render_historical_mode(
                loads=loaded_data.data,
                result=st.session_state.simulation_result,
                config=sim_config,
                config_preset=config.config_preset,
                policy_type=config.policy_type,
                enable_downsample=config.enable_downsample,
                viz_max_points=config.viz_max_points,
            )
    
    else:  # predictive mode
        render_predictive_mode(
            loads=loaded_data.data,
            models_dir=MODELS_DIR,
            config=config.config,
            model_type=config.model_type,
            forecast_horizon_days=config.forecast_horizon_days,
            confidence_level=config.confidence_level,
            optimization_priority=config.optimization_priority,
            enable_downsample=config.enable_downsample,
            viz_max_points=config.viz_max_points,
        )


if __name__ == "__main__":
    main()
