"""Predictive planning mode tabs."""

import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path

from src.scaling.config import ScalingConfig
from app.services.model_service import ModelService, ForecastResult
from app.services.simulator_service import SimulatorService, ComparisonMatrix
from app.services.recommendation_service import RecommendationService, Recommendation
from app.components.charts import (
    create_forecast_chart,
    create_comparison_matrix_chart,
    create_what_if_chart,
    downsample_for_viz,
)


def render_predictive_mode(
    loads: np.ndarray,
    models_dir: Path,
    config: ScalingConfig,
    model_type: str = "lgbm",
    forecast_horizon_days: int = 7,
    confidence_level: float = 0.95,
    optimization_priority: str = "balanced",
    enable_downsample: bool = True,
    viz_max_points: int = 10000,
):
    """Render all tabs for predictive planning mode."""
    
    # Initialize services
    model_service = ModelService(models_dir)
    simulator_service = SimulatorService()
    recommendation_service = RecommendationService(simulator_service)
    
    # Store in session state
    if 'forecast_result' not in st.session_state:
        st.session_state.forecast_result = None
    if 'comparison_matrix' not in st.session_state:
        st.session_state.comparison_matrix = None
    if 'recommendation' not in st.session_state:
        st.session_state.recommendation = None
    
    # Create tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Data & Model",
        "Forecast",
        "Simulation",
        "Recommendation",
        "What-If",
    ])
    
    with tab1:
        _render_data_model_tab(
            loads, model_service, model_type, enable_downsample, viz_max_points
        )
    
    with tab2:
        _render_forecast_tab(
            loads, model_service, model_type, forecast_horizon_days, confidence_level
        )
    
    with tab3:
        _render_simulation_tab(
            loads, simulator_service, st.session_state.forecast_result
        )
    
    with tab4:
        _render_recommendation_tab(
            loads, recommendation_service, optimization_priority,
            st.session_state.comparison_matrix
        )
    
    with tab5:
        _render_whatif_tab(
            loads, simulator_service, config, st.session_state.forecast_result
        )


def _render_data_model_tab(
    loads: np.ndarray,
    model_service: ModelService,
    model_type: str,
    enable_downsample: bool,
    viz_max_points: int,
):
    """Render data and model selection tab."""
    st.header(" Data & Model")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Historical Data Overview")
        
        # Data stats
        stats_col1, stats_col2, stats_col3, stats_col4 = st.columns(4)
        with stats_col1:
            st.metric("Data Points", f"{len(loads):,}")
        with stats_col2:
            st.metric("Max Load", f"{np.max(loads):.1f}")
        with stats_col3:
            st.metric("Mean Load", f"{np.mean(loads):.1f}")
        with stats_col4:
            st.metric("Std Dev", f"{np.std(loads):.1f}")
        
        # Preview chart
        if enable_downsample and len(loads) > viz_max_points:
            loads_viz = downsample_for_viz(loads, viz_max_points)
        else:
            loads_viz = loads
        
        import plotly.express as px
        fig = px.line(y=loads_viz, title="Historical Traffic Pattern")
        fig.update_layout(
            xaxis_title="Time Period",
            yaxis_title="Load",
            height=300,
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader(" Model Selection")
        
        # Check available models
        available = model_service.get_available_models()
        
        st.write("**Available Models:**")
        for name, is_available in available.items():
            icon = "[OK]" if is_available else "[X]"
            label = {
                "lgbm": "LightGBM",
                "prophet": "Prophet",
                "sarima": "SARIMA",
                "ensemble": "Ensemble",
            }.get(name, name)
            st.write(f"{icon} {label}")
        
        st.divider()
        
        # Load model button
        selected_model = st.selectbox(
            "Select Model",
            [m for m, avail in available.items() if avail],
            format_func=lambda x: {
                "lgbm": "LightGBM (Fast)",
                "prophet": "Prophet (Seasonal)",
                "sarima": "SARIMA (Statistical)",
                "ensemble": "Ensemble (All)",
            }.get(x, x),
        )
        
        if st.button(" Load Model", use_container_width=True):
            if model_service.load_model(selected_model):
                st.success(f"[OK] {selected_model.upper()} model loaded!")
            else:
                st.error("Failed to load model")
        
        # Model metrics if available
        metrics = model_service.get_model_metrics(selected_model)
        if metrics:
            st.divider()
            st.write("**Model Metrics:**")
            for key, value in list(metrics.items())[:5]:
                if isinstance(value, float):
                    st.write(f"• {key}: {value:.4f}")


def _render_forecast_tab(
    loads: np.ndarray,
    model_service: ModelService,
    model_type: str,
    horizon_days: int,
    confidence_level: float,
):
    """Render forecast tab."""
    st.header("Traffic Forecast")

    col1, col2, col3 = st.columns(3)

    with col1:
        horizon = st.selectbox(
            "Forecast Horizon",
            [7, 14, 30],
            index=[7, 14, 30].index(horizon_days),
            format_func=lambda x: f"{x} days ({x * 288} periods)",
        )

    with col2:
        confidence = st.select_slider(
            "Confidence Level",
            options=[0.80, 0.90, 0.95, 0.99],
            value=confidence_level,
            format_func=lambda x: f"{x:.0%}",
        )

    with col3:
        model = st.selectbox(
            "Model",
            ["lgbm", "prophet", "sarima", "ensemble"],
            format_func=lambda x: x.upper(),
        )

    # Cache key validation - invalidate when parameters change
    forecast_cache_key = f"forecast_{len(loads)}_{horizon}_{model}_{confidence}"
    if getattr(st.session_state, 'last_forecast_key', None) != forecast_cache_key:
        st.session_state.forecast_result = None
        st.session_state.last_forecast_key = forecast_cache_key

    # Generate forecast button
    if st.button("Generate Forecast", type="primary", use_container_width=True):
        horizon_periods = horizon * 288  # 5-min intervals
        
        # Get last timestamp from loaded data if available
        last_timestamp = None
        if hasattr(st.session_state, 'loaded_data') and st.session_state.loaded_data:
            last_timestamp = st.session_state.loaded_data.last_timestamp

        forecast = model_service.forecast(
            historical_data=loads,
            horizon=horizon_periods,
            model_type=model,
            confidence_level=confidence,
            last_timestamp=last_timestamp,
        )

        if forecast:
            st.session_state.forecast_result = forecast
            st.success(f"Generated {horizon}-day forecast with {model.upper()}")
    
    # Display forecast if available
    if st.session_state.forecast_result:
        forecast = st.session_state.forecast_result
        
        st.divider()
        
        # Forecast stats
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric(" Forecast Points", f"{forecast.horizon:,}")
        with col2:
            st.metric(" Mean Predicted", f"{forecast.mean_prediction:.1f}")
        with col3:
            st.metric(" Peak Predicted", f"{forecast.max_prediction:.1f}")
        with col4:
            st.metric(" Model", forecast.model_type.upper())
        
        # Forecast chart
        fig = create_forecast_chart(
            historical=loads[-min(len(loads), 2000):],  # Last N points
            forecast=forecast,
            title=f"Traffic Forecast ({forecast.model_type.upper()})",
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Cost projection based on forecasted traffic
        st.divider()
        st.subheader("💰 Cost Projection from Forecast")
        
        _render_cost_projection(forecast, horizon)
        
        # Forecast details
        with st.expander(" Forecast Details"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Prediction Summary:**")
                st.write(f"• Min predicted: {np.min(forecast.predictions):.1f}")
                st.write(f"• Max predicted: {np.max(forecast.predictions):.1f}")
                st.write(f"• Std dev: {np.std(forecast.predictions):.1f}")
            
            with col2:
                st.write("**Confidence Interval:**")
                st.write(f"• Level: {forecast.confidence_level:.0%}")
                st.write(f"• Lower bound range: {np.min(forecast.lower_bound):.1f} - {np.max(forecast.lower_bound):.1f}")
                st.write(f"• Upper bound range: {np.min(forecast.upper_bound):.1f} - {np.max(forecast.upper_bound):.1f}")
    else:
        st.info(" Click 'Generate Forecast' to create traffic predictions")


def _render_cost_projection(forecast: ForecastResult, horizon_days: int):
    """Calculate and display cost projection from forecast predictions.
    
    Uses formula from notebook 09_cost_simulation.ipynb:
    - requests_per_server = 100 (per 5-min window)
    - cost_per_server_per_hour = $0.85 (AWS t3.medium)
    - Required servers = ceil(load / (requests_per_server * target_utilization))
    """
    from src.scaling.config import BALANCED_CONFIG
    
    predictions = forecast.predictions
    config = BALANCED_CONFIG
    
    # Calculate required servers for each forecasted load
    # Using balanced target utilization (70%) from notebook
    required_servers = []
    costs_per_period = []
    
    for load in predictions:
        servers = config.get_required_servers(load, target_utilization=0.7)
        cost = config.get_cost_per_period(servers)
        required_servers.append(servers)
        costs_per_period.append(cost)
    
    costs_array = np.array(costs_per_period)
    
    # Calculate cumulative costs for different horizons
    periods_per_day = 288  # 5-minute intervals (1440 min/day ÷ 5 min/period)
    
    # IMPORTANT: Only calculate for the ACTUAL forecast length
    actual_days = len(predictions) / periods_per_day
    
    # Get cost for requested horizon and intermediate milestones
    milestones = []
    
    # Only show milestones that are within the forecast range
    if actual_days >= 7 and horizon_days >= 7:
        cost_7d = costs_array[:7*periods_per_day].sum()
        milestones.append(("7 Days", 7, cost_7d))
    
    if actual_days >= 14 and horizon_days >= 14:
        cost_14d = costs_array[:14*periods_per_day].sum()
        milestones.append(("14 Days", 14, cost_14d))
    
    if actual_days >= 30 and horizon_days >= 30:
        cost_30d = costs_array[:30*periods_per_day].sum()
        milestones.append(("30 Days", 30, cost_30d))
    
    # Always add the full forecasted horizon cost
    total_cost = costs_array.sum()
    forecast_label = f"{horizon_days} Days" if horizon_days == int(actual_days) else f"{actual_days:.1f} Days (Actual)"
    if not milestones or milestones[-1][1] != horizon_days:
        milestones.append((forecast_label, horizon_days, total_cost))
    
    # Display metrics
    cols = st.columns(len(milestones))
    for col, (label, days, cost) in zip(cols, milestones):
        with col:
            st.metric(
                label,
                f"${cost:,.2f}",
                help=f"Projected cost based on {days}-day forecast with balanced config (70% target utilization)"
            )
    
    # Additional details
    with st.expander("📊 Cost Calculation Details"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Traffic-Based Calculation:**")
            st.write(f"• Mean forecasted load: {np.mean(predictions):.1f} req/5min")
            st.write(f"• Peak forecasted load: {np.max(predictions):.1f} req/5min")
            st.write(f"• Avg servers needed: {np.mean(required_servers):.1f}")
            st.write(f"• Peak servers needed: {np.max(required_servers)}")
        
        with col2:
            st.write("**Cost Breakdown:**")
            st.write(f"• Cost per server/hour: ${config.cost_per_server_per_hour:.2f}")
            st.write(f"• Cost per period (5min): ${config.get_cost_per_period(1):.4f}")
            st.write(f"• Avg cost/hour: ${total_cost / (horizon_days * 24):.2f}")
            st.write(f"• Total forecast periods: {len(predictions):,}")


def _render_simulation_tab(
    loads: np.ndarray,
    simulator_service: SimulatorService,
    forecast: ForecastResult | None,
):
    """Render simulation tab."""
    st.header(" Simulation on Forecasted Data")
    
    # Choose data source
    use_forecast = st.checkbox(
        "Use forecasted data",
        value=forecast is not None,
        disabled=forecast is None,
    )
    
    if use_forecast and forecast:
        sim_data = forecast.predictions
        st.success(f"✅ Using {len(sim_data):,} forecasted data points ({len(sim_data)//288:.1f} days)")
        
        # Show forecast summary
        with st.expander("📊 Forecast Summary"):
            col1, col2, col3 = st.columns(3)
            col1.metric("Mean Load", f"{np.mean(sim_data):.1f}")
            col2.metric("Peak Load", f"{np.max(sim_data):.1f}")
            col3.metric("Min Load", f"{np.min(sim_data):.1f}")
    else:
        sim_data = loads
        st.info(f"Using {len(sim_data):,} historical data points ({len(sim_data)//288:.1f} days)")
    
    # Run all combinations
    if st.button(" Run All Simulations", type="primary", use_container_width=True):
        matrix = simulator_service.run_all_combinations(sim_data)
        st.session_state.comparison_matrix = matrix
        st.success("[OK] Completed 9 simulations (3 configs × 3 policies)")
    
    # Display results
    if st.session_state.comparison_matrix:
        matrix = st.session_state.comparison_matrix
        
        st.divider()
        
        # Winners
        col1, col2, col3 = st.columns(3)
        
        with col1:
            best_cost = matrix.best_cost
            if best_cost:
                st.success(f"$ **Lowest Cost**\n\n{best_cost.config_name} + {best_cost.policy_name}\n\n${best_cost.total_cost:.2f}")
        
        with col2:
            best_sla = matrix.best_sla
            if best_sla:
                st.success(f" **Best SLA**\n\n{best_sla.config_name} + {best_sla.policy_name}\n\n{best_sla.sla_violations} violations")
        
        with col3:
            best_balanced = matrix.best_balanced
            if best_balanced:
                st.success(f" **Best Balance**\n\n{best_balanced.config_name} + {best_balanced.policy_name}\n\n${best_balanced.total_cost:.2f} / {best_balanced.sla_violations} viol.")
        
        st.divider()
        
        # Results matrix
        st.subheader(" Results Matrix")
        
        # Create heatmap data
        matrix_data = []
        for (config_name, policy_name), result in matrix.results.items():
            matrix_data.append({
                "Config": config_name.capitalize(),
                "Policy": policy_name.capitalize(),
                "Cost": result.total_cost,
                "SLA": result.sla_violations,
            })
        
        # Heatmap
        fig = create_comparison_matrix_chart(matrix_data)
        st.plotly_chart(fig, use_container_width=True)
        
        # Detailed table
        st.subheader(" Detailed Results")
        df = matrix.to_dataframe()
        st.dataframe(df, hide_index=True, use_container_width=True)
    else:
        st.info(" Click 'Run All Simulations' to compare configurations")


def _render_recommendation_tab(
    loads: np.ndarray,
    recommendation_service: RecommendationService,
    priority: str,
    matrix: ComparisonMatrix | None,
):
    """Render recommendation tab."""
    st.header(" Configuration Recommendation")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader(" Optimization Settings")
        
        opt_priority = st.radio(
            "Optimization Priority",
            ["cost", "sla", "balanced"],
            index=["cost", "sla", "balanced"].index(priority),
            format_func=lambda x: {
                "cost": "$ Minimize Cost",
                "sla": " Minimize SLA Violations",
                "balanced": " Balanced",
            }.get(x, x),
        )
        
        max_sla_rate = st.slider(
            "Max SLA Violation Rate",
            0.01, 0.10, 0.05,
            format="%.2f",
            help="Maximum acceptable SLA violation rate",
        )
        
        if st.button(" Get Recommendation", type="primary", use_container_width=True):
            rec = recommendation_service.find_optimal(
                loads=loads,
                priority=opt_priority,
                max_sla_violation_rate=max_sla_rate,
            )
            st.session_state.recommendation = rec
    
    with col2:
        if st.session_state.recommendation:
            rec = st.session_state.recommendation
            
            # Main recommendation box
            st.success(f"""
            ###  RECOMMENDED CONFIGURATION
            
            **Configuration:** {rec.config_name.upper()}  
            **Policy:** {rec.policy_name.upper()}
            
            ---
            
            **Expected Performance:**
            - $ Cost: **${rec.expected_cost:.2f}**
            -  Avg Servers: **{rec.avg_servers:.1f}**
            -  SLA Violations: **{rec.expected_sla_violations}** ({rec.sla_violation_rate:.2%})
            -  Savings vs Fixed: **${rec.savings_vs_fixed_max:.2f}** ({rec.savings_vs_fixed_max_pct:.1f}%)
            """)
            
            # Settings to apply
            st.subheader(" Recommended Settings")
            
            settings_df = pd.DataFrame([{
                "Setting": "Min Servers",
                "Value": rec.min_servers,
            }, {
                "Setting": "Max Servers",
                "Value": rec.max_servers,
            }, {
                "Setting": "Scale Out Threshold",
                "Value": f"{rec.scale_out_threshold:.0%}",
            }, {
                "Setting": "Scale In Threshold",
                "Value": f"{rec.scale_in_threshold:.0%}",
            }])
            st.dataframe(settings_df, hide_index=True, use_container_width=True)
            
            # Risk analysis
            st.subheader(" Risk Analysis")
            
            risk_col1, risk_col2, risk_col3 = st.columns(3)
            
            with risk_col1:
                risk_color = {"low": "[LOW]", "medium": "[MED]", "high": "[HIGH]"}.get(rec.sla_risk_level, "[-]")
                st.metric("SLA Risk", f"{risk_color} {rec.sla_risk_level.upper()}")
            
            with risk_col2:
                peak_color = {"low": "[LOW]", "medium": "[MED]", "high": "[HIGH]"}.get(rec.peak_hour_risk, "[-]")
                st.metric("Peak Hour Risk", f"{peak_color} {rec.peak_hour_risk.upper()}")
            
            with risk_col3:
                st.metric("Cost Variance", f"±${rec.cost_variance:.2f}")
            
            # Reasoning
            st.info(f" **Reasoning:** {rec.reasoning}")
            
            # Export recommendation
            report = recommendation_service.generate_report(rec, matrix) if matrix else ""
            if report:
                st.download_button(
                    " Download Recommendation Report",
                    data=report,
                    file_name="recommendation_report.txt",
                    mime="text/plain",
                )
        else:
            st.info(" Click 'Get Recommendation' to generate optimal configuration")


def _render_whatif_tab(
    loads: np.ndarray,
    simulator_service: SimulatorService,
    config: ScalingConfig,
    forecast: ForecastResult | None,
):
    """Render what-if scenario tab."""
    st.header(" What-If Scenario Planning")
    
    st.write("Explore how different scenarios affect costs and performance.")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader(" Scenario Builder")
        
        # Traffic multiplier
        traffic_multiplier = st.slider(
            "Traffic Multiplier",
            0.5, 3.0, 1.0,
            step=0.1,
            help="Scale all traffic by this factor",
        )
        
        # Spike simulation
        add_spike = st.checkbox("Add Traffic Spike")
        spike_magnitude = 1.0
        spike_position = 0.5
        if add_spike:
            spike_magnitude = st.slider("Spike Magnitude", 1.5, 5.0, 2.0)
            spike_position = st.slider("Spike Position", 0.0, 1.0, 0.5)
        
        # Config override
        st.divider()
        st.write("**Configuration Override:**")
        
        scenario_config = st.selectbox(
            "Config for Scenario",
            ["conservative", "balanced", "aggressive"],
            index=1,
        )
        
        scenario_policy = st.selectbox(
            "Policy for Scenario",
            ["balanced", "reactive", "predictive"],
            index=0,
        )
    
    with col2:
        # Use forecast if available
        base_data = forecast.predictions if forecast else loads
        
        # Apply scenario modifications
        scenario_data = base_data.copy() * traffic_multiplier
        
        if add_spike:
            spike_idx = int(len(scenario_data) * spike_position)
            spike_width = max(1, len(scenario_data) // 20)
            start_idx = max(0, spike_idx - spike_width // 2)
            end_idx = min(len(scenario_data), spike_idx + spike_width // 2)
            scenario_data[start_idx:end_idx] *= spike_magnitude
        
        if st.button(" Run Scenario Comparison", type="primary"):
            # Run base simulation
            base_result = simulator_service.run_simulation(
                loads=base_data,
                config=config,
                policy_type="balanced",
                config_name="base",
            )
            
            # Run scenario simulation
            from src.scaling.config import CONSERVATIVE_CONFIG, BALANCED_CONFIG, AGGRESSIVE_CONFIG
            scenario_configs = {
                "conservative": CONSERVATIVE_CONFIG,
                "balanced": BALANCED_CONFIG,
                "aggressive": AGGRESSIVE_CONFIG,
            }
            
            scenario_result = simulator_service.run_simulation(
                loads=scenario_data,
                config=scenario_configs[scenario_config],
                policy_type=scenario_policy,
                config_name=f"scenario_{scenario_config}",
            )
            
            if base_result and scenario_result:
                # Comparison metrics
                st.subheader(" Scenario Comparison")
                
                comp_col1, comp_col2, comp_col3 = st.columns(3)
                
                with comp_col1:
                    cost_diff = scenario_result.total_cost - base_result.total_cost
                    cost_pct = (cost_diff / base_result.total_cost * 100) if base_result.total_cost > 0 else 0
                    st.metric(
                        "$ Cost Change",
                        f"${scenario_result.total_cost:.2f}",
                        f"{cost_diff:+.2f} ({cost_pct:+.1f}%)",
                    )
                
                with comp_col2:
                    sla_diff = scenario_result.sla_violations - base_result.sla_violations
                    st.metric(
                        " SLA Violations",
                        f"{scenario_result.sla_violations}",
                        f"{sla_diff:+d}",
                        delta_color="inverse",
                    )
                
                with comp_col3:
                    server_diff = scenario_result.avg_servers - base_result.avg_servers
                    st.metric(
                        " Avg Servers",
                        f"{scenario_result.avg_servers:.1f}",
                        f"{server_diff:+.1f}",
                    )
                
                # Comparison chart
                fig = create_what_if_chart(
                    base_costs=base_result.cost_over_time[:1000],  # Limit for perf
                    scenario_costs=scenario_result.cost_over_time[:1000],
                    scenario_name=f"{scenario_config.capitalize()} + {scenario_policy.capitalize()}",
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Summary
                st.info(f"""
                **Scenario Summary:**
                - Traffic multiplier: {traffic_multiplier}x
                - {'Spike added at ' + str(int(spike_position * 100)) + '%' if add_spike else 'No spike'}
                - Config: {scenario_config.capitalize()}, Policy: {scenario_policy.capitalize()}
                """)
        else:
            # Preview chart
            import plotly.graph_objects as go
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                y=base_data[:1000],
                mode='lines',
                name='Base',
                line=dict(color='#636363'),
            ))
            fig.add_trace(go.Scatter(
                y=scenario_data[:1000],
                mode='lines',
                name='Scenario',
                line=dict(color='#1f77b4', dash='dash'),
            ))
            fig.update_layout(
                title="Scenario Preview",
                xaxis_title="Time Period",
                yaxis_title="Load",
                height=350,
            )
            st.plotly_chart(fig, use_container_width=True)
            
            st.info(" Click 'Run Scenario Comparison' to see results")
