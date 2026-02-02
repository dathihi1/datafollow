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
    
    # If we have loaded data and source hasn't changed, return cached data
    if st.session_state.loaded_data is not None:
        # Return cached data for upload sources (persists across mode changes)
        if config.data_source in ["upload_csv", "upload_txt", "saved"]:
            return st.session_state.loaded_data
    
    if config.data_source == "sample":
        return data_loader.generate_sample(config.n_periods, config.seed)
    
    elif config.data_source == "upload_csv":
        uploaded_file = st.sidebar.file_uploader(
            "📁 Upload CSV",
            type=["csv"],
            help="CSV with 'load' or 'request_count' column",
            key="csv_uploader",
        )
        if uploaded_file:
            file_id = f"csv_{uploaded_file.name}_{uploaded_file.size}"
            if st.session_state.data_file_id != file_id:
                result = data_loader.load_csv(uploaded_file)
                if result:
                    st.session_state.loaded_data = result
                    st.session_state.data_file_id = file_id
                return result
            return st.session_state.loaded_data
    
    elif config.data_source == "upload_txt":
        uploaded_file = st.sidebar.file_uploader(
            "📁 Upload TXT",
            type=["txt"],
            help="TXT with one number per line or comma-separated",
            key="txt_uploader",
        )
        if uploaded_file:
            file_id = f"txt_{uploaded_file.name}_{uploaded_file.size}"
            if st.session_state.data_file_id != file_id:
                result = data_loader.load_txt(uploaded_file)
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
            - **CSV**: Must have column named `load` or `request_count`
            - **TXT**: One number per line, or comma-separated values
            """)
        return
    
    # Data info header
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📊 Data Points", f"{loaded_data.length:,}")
    with col2:
        st.metric("⏱️ Time Span", f"{loaded_data.time_range_days:.1f} days")
    with col3:
        st.metric("📈 Mean Load", f"{np.mean(loaded_data.data):.1f}")
    with col4:
        st.metric("📊 Source", loaded_data.source_type.upper())
    
    st.divider()
    
    # Route to appropriate mode
    if config.mode == "historical":
        # Run simulation if needed
        cache_key = f"{loaded_data.source_type}_{config.config_preset}_{config.policy_type}"
        
        if st.session_state.simulation_result is None or \
           getattr(st.session_state, 'last_cache_key', None) != cache_key:
            
            with st.spinner("Running simulation..."):
                result = simulator.run_simulation(
                    loads=loaded_data.data,
                    config=config.config,
                    policy_type=config.policy_type,
                    config_name=config.config_preset,
                )
                st.session_state.simulation_result = result
                st.session_state.last_cache_key = cache_key
        
        if st.session_state.simulation_result:
            render_historical_mode(
                loads=loaded_data.data,
                result=st.session_state.simulation_result,
                config=config.config,
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
