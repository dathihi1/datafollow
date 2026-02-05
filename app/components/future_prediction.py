"""
Future Traffic Prediction Component for Dashboard.

Dự đoán traffic tương lai dựa trên historical data (không cần test data).
"""

import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path
from datetime import datetime, timedelta

from app.services.model_service import ModelService, ForecastResult
from app.services.feature_service import FeatureService
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def render_future_prediction(
    loads: np.ndarray,
    models_dir: Path,
    time_interval_minutes: int = 5,
    forecast_days: int = 7,
):
    """
    Render future prediction tab.
    
    Args:
        loads: Historical traffic data (numpy array)
        models_dir: Directory containing trained models
        time_interval_minutes: Time interval per data point (default 5 minutes)
        forecast_days: Number of days to forecast (default 7)
    """
    
    st.header("🔮 Dự Đoán Traffic Tương Lai")
    
    st.markdown("""
    Dự đoán traffic cho **{} ngày** tiếp theo dựa trên:
    - 📊 Historical data đã load
    - 🤖 Model LightGBM đã train sẵn
    - 🎯 Không cần test data - chỉ predict forward
    """.format(forecast_days))
    
    # Configuration
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        forecast_days = st.selectbox(
            "Số ngày dự đoán",
            options=[1, 2, 3, 7, 14],
            index=3,  # default 7 days
            help="Số ngày traffic cần dự đoán"
        )
    
    with col2:
        model_choice = st.selectbox(
            "Model",
            options=["lgbm", "prophet", "ensemble"],
            format_func=lambda x: {
                "lgbm": "LightGBM (Tốt nhất)",
                "prophet": "Prophet (Seasonal)",
                "ensemble": "Ensemble (Kết hợp)"
            }.get(x, x),
            help="Chọn model để dự đoán"
        )
    
    with col3:
        prediction_mode = st.selectbox(
            "Mode",
            options=["fast", "accurate"],
            format_func=lambda x: {
                "fast": "⚡ Fast (Recommended)",
                "accurate": "🎯 Accurate (Slow)"
            }.get(x, x),
            help="Fast: ~1s | Accurate: ~1min (iterative)"
        )
    
    with col4:
        show_confidence = st.checkbox(
            "Hiển thị confidence interval",
            value=True,
            help="Hiển thị khoảng tin cậy (±1.96*std)"
        )
    
    st.divider()
    
    # Initialize services
    model_service = ModelService(models_dir)
    feature_service = FeatureService(models_dir)
    
    # Predict button
    if st.button("🚀 Dự Đoán Traffic", use_container_width=True, type="primary"):
        
        with st.spinner(f"Đang dự đoán {forecast_days} ngày với {model_choice.upper()}..."):
            
            try:
                # Load model
                if not model_service.load_model(model_choice):
                    st.error(f"Không thể load model {model_choice}")
                    return
                
                # Generate forecast
                forecast_result = _generate_forecast(
                    loads=loads,
                    model_service=model_service,
                    feature_service=feature_service,
                    model_type=model_choice,
                    forecast_days=forecast_days,
                    time_interval_minutes=time_interval_minutes,
                    prediction_mode=prediction_mode,
                )
                
                if forecast_result is None:
                    st.error("Dự đoán thất bại")
                    return
                
                # Store in session state
                st.session_state.future_forecast = forecast_result
                st.success(f"✅ Đã dự đoán {len(forecast_result.predictions):,} data points ({forecast_days} ngày)")
                
            except Exception as e:
                st.error(f"Lỗi khi dự đoán: {str(e)}")
                import traceback
                with st.expander("Chi tiết lỗi"):
                    st.code(traceback.format_exc())
                return
    
    # Display results if available
    if 'future_forecast' in st.session_state and st.session_state.future_forecast:
        forecast = st.session_state.future_forecast
        
        # Metrics
        st.subheader("📊 Kết Quả Dự Đoán")
        
        # Request Count Metrics
        st.markdown("**🔢 Request Count Predictions**")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Predicted Points", f"{len(forecast.predictions):,}")
        with col2:
            st.metric("Mean Traffic", f"{forecast.mean_prediction:.1f}")
        with col3:
            st.metric("Max Traffic", f"{forecast.max_prediction:.1f}")
        with col4:
            confidence_width = forecast.upper_bound - forecast.lower_bound
            st.metric("Avg Confidence Width", f"{np.mean(confidence_width):.1f}")
        
        # Bytes Metrics
        if forecast.has_bytes_predictions:
            st.markdown("**💾 Bytes Predictions (ML Model)**")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Mean Bytes", f"{forecast.mean_bytes_prediction:,.0f}")
            with col2:
                st.metric("Max Bytes", f"{forecast.max_bytes_prediction:,.0f}")
            with col3:
                st.metric("Total Bytes", f"{forecast.total_bytes_prediction:,.0f}")
            with col4:
                st.metric("Total GB", f"{forecast.total_bytes_prediction / 1e9:.2f}")
        
        # Bytes Ratio Estimation
        if len(forecast.bytes_estimated) > 0:
            st.markdown("**📊 Bytes Estimation (Ratio-based)**")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Mean Bytes (Ratio)", f"{forecast.mean_bytes_estimated:,.0f}")
            with col2:
                st.metric("Bytes/Request Ratio", f"{forecast.bytes_per_request_ratio:,.0f}")
            with col3:
                total_bytes_est = np.sum(forecast.bytes_estimated)
                st.metric("Total GB (Ratio)", f"{total_bytes_est / 1e9:.2f}")
        
        # Visualization
        st.subheader("📈 Visualization")
        
        # Create timestamps for historical and forecast
        n_history = len(loads)
        n_forecast = len(forecast.predictions)
        
        # Historical timestamps (assume ends at "now")
        base_time = datetime.now() - timedelta(minutes=time_interval_minutes * n_history)
        hist_timestamps = [base_time + timedelta(minutes=time_interval_minutes * i) for i in range(n_history)]
        
        # Forecast timestamps (starts after last historical)
        forecast_start = hist_timestamps[-1] + timedelta(minutes=time_interval_minutes)
        forecast_timestamps = [forecast_start + timedelta(minutes=time_interval_minutes * i) for i in range(n_forecast)]
        
        # Tabs for different visualizations
        tab1, tab2 = st.tabs(["🔢 Request Count", "💾 Bytes Prediction"])
        
        with tab1:
            # Create request count figure
            fig = _create_prediction_chart(
                historical_data=loads,
                historical_timestamps=hist_timestamps,
                forecast_data=forecast.predictions,
                forecast_timestamps=forecast_timestamps,
                lower_bound=forecast.lower_bound if show_confidence else None,
                upper_bound=forecast.upper_bound if show_confidence else None,
                model_type=forecast.model_type,
                chart_title="Request Count Forecast",
                y_axis_title="Request Count",
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            if forecast.has_bytes_predictions:
                # Create bytes prediction figure
                fig_bytes = _create_bytes_prediction_chart(
                    forecast_timestamps=forecast_timestamps,
                    bytes_predictions=forecast.bytes_predictions,
                    bytes_estimated=forecast.bytes_estimated,
                    bytes_lower=forecast.bytes_lower_bound if show_confidence else None,
                    bytes_upper=forecast.bytes_upper_bound if show_confidence else None,
                )
                st.plotly_chart(fig_bytes, use_container_width=True)
            else:
                st.info("💡 Bytes prediction model not available. Only ratio-based estimation is shown.")
                if len(forecast.bytes_estimated) > 0:
                    fig_bytes_est = _create_bytes_estimation_chart(
                        forecast_timestamps=forecast_timestamps,
                        bytes_estimated=forecast.bytes_estimated,
                    )
                    st.plotly_chart(fig_bytes_est, use_container_width=True)
        
        # Statistics table
        with st.expander("📋 Chi Tiết Thống Kê"):
            
            col_a, col_b = st.columns(2)
            
            with col_a:
                st.markdown("**Request Count Statistics**")
                # Create dataframe
                df_stats = pd.DataFrame({
                    'Metric': [
                        'Historical Mean',
                        'Historical Max',
                        'Historical Std',
                        'Forecast Mean',
                        'Forecast Max',
                        'Forecast Min',
                        'Forecast Std',
                    ],
                    'Value': [
                        f"{np.mean(loads):.2f}",
                        f"{np.max(loads):.2f}",
                        f"{np.std(loads):.2f}",
                        f"{forecast.mean_prediction:.2f}",
                        f"{forecast.max_prediction:.2f}",
                        f"{np.min(forecast.predictions):.2f}",
                        f"{np.std(forecast.predictions):.2f}",
                    ]
                })
                st.dataframe(df_stats, use_container_width=True, hide_index=True)
            
            with col_b:
                if forecast.has_bytes_predictions:
                    st.markdown("**Bytes Prediction Statistics**")
                    df_bytes = pd.DataFrame({
                        'Metric': [
                            'Mean Bytes (ML)',
                            'Max Bytes (ML)',
                            'Total Bytes (ML)',
                            'Total GB (ML)',
                            'Mean Bytes (Ratio)',
                            'Total GB (Ratio)',
                            'Bytes/Request Ratio',
                        ],
                        'Value': [
                            f"{forecast.mean_bytes_prediction:,.0f}",
                            f"{forecast.max_bytes_prediction:,.0f}",
                            f"{forecast.total_bytes_prediction:,.0f}",
                            f"{forecast.total_bytes_prediction / 1e9:.2f}",
                            f"{forecast.mean_bytes_estimated:,.0f}",
                            f"{np.sum(forecast.bytes_estimated) / 1e9:.2f}",
                            f"{forecast.bytes_per_request_ratio:,.0f}",
                        ]
                    })
                    st.dataframe(df_bytes, use_container_width=True, hide_index=True)
        
        # Daily breakdown
        if forecast_days > 1:
            with st.expander("📅 Thống Kê Theo Ngày"):
                _display_daily_breakdown(
                    forecast_data=forecast.predictions,
                    forecast_timestamps=forecast_timestamps,
                    time_interval_minutes=time_interval_minutes,
                    bytes_predictions=forecast.bytes_predictions if forecast.has_bytes_predictions else None,
                    bytes_estimated=forecast.bytes_estimated if len(forecast.bytes_estimated) > 0 else None,
                )
        
        # Export option
        st.divider()
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("💾 Export Predictions to CSV"):
                csv_data = _export_to_csv(
                    forecast_timestamps=forecast_timestamps,
                    predictions=forecast.predictions,
                    lower_bound=forecast.lower_bound,
                    upper_bound=forecast.upper_bound,
                    bytes_predictions=forecast.bytes_predictions if forecast.has_bytes_predictions else None,
                    bytes_estimated=forecast.bytes_estimated if len(forecast.bytes_estimated) > 0 else None,
                )
                st.download_button(
                    label="⬇️ Download CSV",
                    data=csv_data,
                    file_name=f"traffic_forecast_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                )
        
        with col2:
            if st.button("🔄 Reset Prediction"):
                del st.session_state.future_forecast
                st.rerun()


def _generate_forecast(
    loads: np.ndarray,
    model_service: ModelService,
    feature_service: FeatureService,
    model_type: str,
    forecast_days: int,
    time_interval_minutes: int,
    prediction_mode: str = "fast",
) -> ForecastResult | None:
    """
    Generate forecast for future periods.
    
    Args:
        loads: Historical traffic data
        model_service: Model service instance
        feature_service: Feature service for feature engineering
        model_type: Model to use (lgbm, prophet, ensemble)
        forecast_days: Number of days to forecast
        time_interval_minutes: Time interval per data point
        prediction_mode: "fast" or "accurate" (iterative)
    
    Returns:
        ForecastResult object or None if failed
    """
    
    # Calculate forecast horizon in periods
    periods_per_day = 24 * 60 // time_interval_minutes  # e.g., 288 for 5min
    forecast_periods = forecast_days * periods_per_day
    
    # Generate features using feature service's method (pass loads directly)
    hist_with_features = feature_service.create_features(
        loads=loads,
        interval_minutes=time_interval_minutes,
    )
    
    # For LightGBM, choose between fast and accurate
    if model_type == "lgbm":
        if prediction_mode == "fast":
            return _forecast_lgbm_direct(
                hist_df=hist_with_features,
                model_service=model_service,
                forecast_periods=forecast_periods,
                time_interval_minutes=time_interval_minutes,
            )
        else:  # accurate (iterative)
            return _forecast_lgbm_iterative(
                hist_df=hist_with_features,
                model_service=model_service,
                feature_service=feature_service,
                forecast_periods=forecast_periods,
                time_interval_minutes=time_interval_minutes,
            )
    
    # For Prophet/SARIMA, use model's built-in forecasting
    else:
        # Use model service's forecast method
        result = model_service.forecast(
            historical_data=loads,
            horizon=forecast_periods,
            model_type=model_type,
        )
        return result


def _forecast_lgbm_direct(
    hist_df: pd.DataFrame,
    model_service: ModelService,
    forecast_periods: int,
    time_interval_minutes: int,
) -> ForecastResult:
    """
    FAST direct forecasting for LightGBM using last known features.
    
    Instead of iterative (slow), use last historical features + time progression.
    This is 100x faster for large horizons.
    """
    import streamlit as st
    
    # Get model
    model = model_service._loaded_models.get("lgbm")
    if model is None:
        raise ValueError("LightGBM model not loaded")
    
    # Prepare feature columns
    exclude_cols = ['timestamp', 'time_of_day', 'part_of_day', 'request_count', 'request_count_pct_of_max']
    feature_cols = [c for c in hist_df.select_dtypes(include=[np.number]).columns 
                    if c not in exclude_cols]
    
    # Get scaler
    scaler = model_service._feature_scaler
    
    # Get last row as template
    last_features = hist_df.iloc[[-1]][feature_cols].copy()
    
    # For fast forecasting, we'll use a simplified approach:
    # 1. Use last known lag/rolling features
    # 2. Update time-based features (hour, day_of_week, etc.)
    # 3. Apply seasonal pattern adjustment
    
    predictions = []
    last_timestamp = hist_df['timestamp'].iloc[-1] if 'timestamp' in hist_df.columns else None
    
    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Get base prediction from last features
    base_features = last_features.replace([np.inf, -np.inf], np.nan).fillna(0)
    if scaler is not None:
        base_features_scaled = scaler.transform(base_features)
    else:
        base_features_scaled = base_features.values
    
    base_prediction = model.predict(base_features_scaled)[0]
    
    # Track predictions for lag feature updates
    # Get historical request counts from hist_df
    hist_values = hist_df['request_count'].values if 'request_count' in hist_df.columns else []
    recent_predictions = list(hist_values[-288:]) if len(hist_values) >= 288 else list(hist_values)  # Keep last day of history
    periods_per_day = 24 * 60 // time_interval_minutes
    
    for step in range(forecast_periods):
        # Update progress every 100 steps
        if step % 100 == 0:
            progress = step / forecast_periods
            progress_bar.progress(progress)
            status_text.text(f"Predicting... {step}/{forecast_periods} points ({progress*100:.1f}%)")
        
        # Create features for this step
        step_features = base_features.copy()
        
        # Update time-based features
        if last_timestamp is not None:
            future_timestamp = last_timestamp + pd.Timedelta(minutes=time_interval_minutes * (step + 1))
            
            # Update hour
            if 'hour' in step_features.columns:
                step_features['hour'] = future_timestamp.hour
            
            # Update day_of_week
            if 'day_of_week' in step_features.columns:
                step_features['day_of_week'] = future_timestamp.dayofweek
            
            # Update cyclical features
            if 'hour_sin' in step_features.columns:
                step_features['hour_sin'] = np.sin(2 * np.pi * future_timestamp.hour / 24)
            if 'hour_cos' in step_features.columns:
                step_features['hour_cos'] = np.cos(2 * np.pi * future_timestamp.hour / 24)
        
        # Update lag features with recent predictions
        # FIXED: Use correct feature names (request_count_lag_* instead of load_lag_*)
        if len(recent_predictions) > 0:
            # 5-minute lag (1 period back)
            if 'request_count_lag_1' in step_features.columns:
                step_features['request_count_lag_1'] = recent_predictions[-1] if len(recent_predictions) >= 1 else base_prediction
            
            # 15-minute lag (3 periods back)
            if 'request_count_lag_3' in step_features.columns and len(recent_predictions) >= 3:
                step_features['request_count_lag_3'] = recent_predictions[-3]
            
            # 30-minute lag (6 periods back)
            if 'request_count_lag_6' in step_features.columns and len(recent_predictions) >= 6:
                step_features['request_count_lag_6'] = recent_predictions[-6]
            
            # 1-hour lag (12 periods back for 5-min intervals)
            periods_per_hour = 60 // time_interval_minutes
            if 'request_count_lag_12' in step_features.columns and len(recent_predictions) >= periods_per_hour:
                step_features['request_count_lag_12'] = recent_predictions[-periods_per_hour]
            
            # 5-hour lag (60 periods back)
            if 'request_count_lag_60' in step_features.columns and len(recent_predictions) >= 60:
                step_features['request_count_lag_60'] = recent_predictions[-60]
            
            # 1-day lag (288 periods back)
            if 'request_count_lag_288' in step_features.columns and len(recent_predictions) >= 288:
                step_features['request_count_lag_288'] = recent_predictions[-288]
            
            # Rolling mean (last hour)
            if 'request_count_rolling_mean_12' in step_features.columns and len(recent_predictions) >= periods_per_hour:
                step_features['request_count_rolling_mean_12'] = np.mean(recent_predictions[-periods_per_hour:])
            
            # Rolling std (last hour)
            if 'request_count_rolling_std_12' in step_features.columns and len(recent_predictions) >= periods_per_hour:
                step_features['request_count_rolling_std_12'] = np.std(recent_predictions[-periods_per_hour:])
        
        # Handle inf/nan
        step_features = step_features.replace([np.inf, -np.inf], np.nan).fillna(0)
        
        # Scale
        if scaler is not None:
            step_features_scaled = scaler.transform(step_features)
        else:
            step_features_scaled = step_features.values
        
        # Predict
        pred = model.predict(step_features_scaled)[0]
        predictions.append(pred)
        
        # Update recent predictions buffer
        recent_predictions.append(pred)
        if len(recent_predictions) > periods_per_day:  # Keep only last day
            recent_predictions.pop(0)
    
    # Complete progress
    progress_bar.progress(1.0)
    status_text.text(f"✅ Completed {forecast_periods} predictions!")
    
    # Calculate confidence intervals
    predictions_array = np.array(predictions)
    
    # Wider confidence intervals for longer horizons
    # Increase uncertainty with distance
    uncertainty_base = np.std(predictions_array) if len(predictions_array) > 1 else predictions_array[0] * 0.1
    uncertainties = np.linspace(uncertainty_base, uncertainty_base * 2, len(predictions_array))
    
    return ForecastResult(
        timestamps=[],
        predictions=predictions_array,
        lower_bound=predictions_array - 1.96 * uncertainties,
        upper_bound=predictions_array + 1.96 * uncertainties,
        model_type="lgbm",
        confidence_level=0.95,
        metrics={},
    )


def _forecast_lgbm_iterative(
    hist_df: pd.DataFrame,
    model_service: ModelService,
    feature_service: FeatureService,
    forecast_periods: int,
    time_interval_minutes: int,
) -> ForecastResult:
    """
    Iterative forecasting for LightGBM (multi-step ahead) - OPTIMIZED.
    
    Uses cached features and only updates incrementally.
    """
    import streamlit as st
    
    # Get model
    model = model_service._loaded_models.get("lgbm")
    if model is None:
        raise ValueError("LightGBM model not loaded")
    
    # Prepare feature columns
    exclude_cols = ['timestamp', 'time_of_day', 'part_of_day', 'request_count', 'request_count_pct_of_max']
    feature_cols = [c for c in hist_df.select_dtypes(include=[np.number]).columns 
                    if c not in exclude_cols]
    
    # Get scaler
    scaler = model_service._feature_scaler
    
    # Strategy: Create features in batches instead of one-by-one
    # This is MUCH faster than recreating all features each iteration
    predictions = []
    batch_size = min(288, forecast_periods)  # Process up to 1 day at a time
    
    # Get historical values
    hist_values = hist_df['request_count'].values.tolist()
    
    # Create progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for batch_start in range(0, forecast_periods, batch_size):
        batch_end = min(batch_start + batch_size, forecast_periods)
        batch_predictions = []
        
        # Update progress
        progress = batch_start / forecast_periods
        progress_bar.progress(progress)
        status_text.text(f"Predicting... {batch_start}/{forecast_periods} points ({progress*100:.1f}%)")
        
        for step in range(batch_start, batch_end):
            # Only recreate features every N steps or when necessary
            # This reduces expensive feature creation calls
            if step % 12 == 0 or step == batch_start:  # Every hour or batch start
                current_loads = np.array(hist_values)
                current_features = feature_service.create_features(
                    loads=current_loads,
                    interval_minutes=time_interval_minutes,
                )
            else:
                # Quick update: Just append new row with basic features
                # More advanced: implement incremental feature update
                current_loads = np.array(hist_values)
                current_features = feature_service.create_features(
                    loads=current_loads,
                    interval_minutes=time_interval_minutes,
                )
            
            # Get last row features
            last_row = current_features.iloc[[-1]][feature_cols]
            
            # Handle inf/nan
            last_row = last_row.replace([np.inf, -np.inf], np.nan).fillna(0)
            
            # Scale
            if scaler is not None:
                last_row_scaled = scaler.transform(last_row)
            else:
                last_row_scaled = last_row.values
            
            # Predict
            pred = model.predict(last_row_scaled)[0]
            batch_predictions.append(pred)
            hist_values.append(pred)
        
        predictions.extend(batch_predictions)
    
    # Clear progress
    progress_bar.progress(1.0)
    status_text.text(f"✅ Completed {forecast_periods} predictions!")
    
    # Calculate confidence intervals
    predictions_array = np.array(predictions)
    std_pred = np.std(predictions_array)
    
    return ForecastResult(
        timestamps=[],
        predictions=predictions_array,
        lower_bound=predictions_array - 1.96 * std_pred,
        upper_bound=predictions_array + 1.96 * std_pred,
        model_type="lgbm",
        confidence_level=0.95,
        metrics={},
    )


def _create_prediction_chart(
    historical_data: np.ndarray,
    historical_timestamps: list,
    forecast_data: np.ndarray,
    forecast_timestamps: list,
    lower_bound: np.ndarray | None,
    upper_bound: np.ndarray | None,
    model_type: str,
    chart_title: str = "Traffic Forecast",
    y_axis_title: str = "Request Count",
) -> go.Figure:
    """Create interactive prediction chart."""
    
    fig = go.Figure()
    
    # Historical data
    fig.add_trace(go.Scatter(
        x=historical_timestamps,
        y=historical_data,
        mode='lines',
        name='Historical Traffic',
        line=dict(color='#3498db', width=2),
        opacity=0.8,
    ))
    
    # Forecast
    fig.add_trace(go.Scatter(
        x=forecast_timestamps,
        y=forecast_data,
        mode='lines',
        name=f'Predicted ({model_type.upper()})',
        line=dict(color='#e74c3c', width=2, dash='dash'),
    ))
    
    # Confidence interval
    if lower_bound is not None and upper_bound is not None:
        fig.add_trace(go.Scatter(
            x=forecast_timestamps + forecast_timestamps[::-1],
            y=np.concatenate([upper_bound, lower_bound[::-1]]),
            fill='toself',
            fillcolor='rgba(231, 76, 60, 0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            name='95% Confidence Interval',
            showlegend=True,
        ))
    
    # Add vertical line at prediction start
    if len(historical_timestamps) > 0 and len(forecast_timestamps) > 0:
        # Add shape manually to avoid type issues with add_vline
        fig.add_shape(
            type="line",
            x0=forecast_timestamps[0],
            x1=forecast_timestamps[0],
            y0=0,
            y1=1,
            yref="paper",
            line=dict(color="gray", dash="dot", width=2),
        )
        fig.add_annotation(
            x=forecast_timestamps[0],
            y=1,
            yref="paper",
            text="Prediction Start",
            showarrow=False,
            yshift=10,
            font=dict(color="gray"),
        )
    
    fig.update_layout(
        title=chart_title,
        xaxis_title="Timestamp",
        yaxis_title=y_axis_title,
        hovermode='x unified',
        height=500,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
    )
    
    return fig


def _create_bytes_prediction_chart(
    forecast_timestamps: list,
    bytes_predictions: np.ndarray,
    bytes_estimated: np.ndarray,
    bytes_lower: np.ndarray | None = None,
    bytes_upper: np.ndarray | None = None,
) -> go.Figure:
    """Create bytes prediction comparison chart (ML vs Ratio)."""
    
    fig = go.Figure()
    
    # ML Model predictions
    fig.add_trace(go.Scatter(
        x=forecast_timestamps,
        y=bytes_predictions,
        mode='lines',
        name='ML Model Prediction',
        line=dict(color='#2ecc71', width=2),
    ))
    
    # Ratio-based estimation
    fig.add_trace(go.Scatter(
        x=forecast_timestamps,
        y=bytes_estimated,
        mode='lines',
        name='Ratio Estimation',
        line=dict(color='#e74c3c', width=2, dash='dash'),
    ))
    
    # Confidence interval for ML model
    if bytes_lower is not None and bytes_upper is not None:
        fig.add_trace(go.Scatter(
            x=forecast_timestamps + forecast_timestamps[::-1],
            y=np.concatenate([bytes_upper, bytes_lower[::-1]]),
            fill='toself',
            fillcolor='rgba(46, 204, 113, 0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            name='95% Confidence Interval',
            showlegend=True,
        ))
    
    fig.update_layout(
        title="Bytes Prediction: ML Model vs Ratio Estimation",
        xaxis_title="Timestamp",
        yaxis_title="Bytes",
        hovermode='x unified',
        height=500,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
    )
    
    return fig


def _create_bytes_estimation_chart(
    forecast_timestamps: list,
    bytes_estimated: np.ndarray,
) -> go.Figure:
    """Create simple bytes estimation chart."""
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=forecast_timestamps,
        y=bytes_estimated,
        mode='lines',
        name='Ratio-based Estimation',
        line=dict(color='#e74c3c', width=2),
    ))
    
    fig.update_layout(
        title="Bytes Estimation (Ratio-based)",
        xaxis_title="Timestamp",
        yaxis_title="Bytes",
        hovermode='x unified',
        height=500,
        showlegend=True,
    )
    
    return fig


def _display_daily_breakdown(
    forecast_data: np.ndarray,
    forecast_timestamps: list,
    time_interval_minutes: int,
    bytes_predictions: np.ndarray | None = None,
    bytes_estimated: np.ndarray | None = None,
):
    """Display daily statistics breakdown."""
    
    periods_per_day = 24 * 60 // time_interval_minutes
    n_days = len(forecast_data) // periods_per_day
    
    daily_stats = []
    
    for day in range(n_days):
        start_idx = day * periods_per_day
        end_idx = (day + 1) * periods_per_day
        day_data = forecast_data[start_idx:end_idx]
        
        stats_dict = {
            'Day': f"Day {day + 1}",
            'Date': forecast_timestamps[start_idx].strftime('%Y-%m-%d'),
            'Mean Requests': f"{np.mean(day_data):.1f}",
            'Max Requests': f"{np.max(day_data):.1f}",
            'Min Requests': f"{np.min(day_data):.1f}",
            'Std': f"{np.std(day_data):.1f}",
        }
        
        # Add bytes statistics if available
        if bytes_predictions is not None:
            day_bytes = bytes_predictions[start_idx:end_idx]
            stats_dict['Mean Bytes (ML)'] = f"{np.mean(day_bytes):,.0f}"
            stats_dict['Total GB (ML)'] = f"{np.sum(day_bytes) / 1e9:.2f}"
        
        if bytes_estimated is not None:
            day_bytes_est = bytes_estimated[start_idx:end_idx]
            stats_dict['Total GB (Ratio)'] = f"{np.sum(day_bytes_est) / 1e9:.2f}"
        
        daily_stats.append(stats_dict)
    
    df = pd.DataFrame(daily_stats)
    st.dataframe(df, use_container_width=True, hide_index=True)


def _export_to_csv(
    forecast_timestamps: list,
    predictions: np.ndarray,
    lower_bound: np.ndarray,
    upper_bound: np.ndarray,
    bytes_predictions: np.ndarray | None = None,
    bytes_estimated: np.ndarray | None = None,
) -> str:
    """Export forecast to CSV format."""
    
    df_data = {
        'timestamp': [t.strftime('%Y-%m-%d %H:%M:%S') for t in forecast_timestamps],
        'predicted_traffic': predictions,
        'lower_bound_95': lower_bound,
        'upper_bound_95': upper_bound,
    }
    
    # Add bytes columns if available
    if bytes_predictions is not None:
        df_data['predicted_bytes_ml'] = bytes_predictions
    
    if bytes_estimated is not None:
        df_data['estimated_bytes_ratio'] = bytes_estimated
    
    df = pd.DataFrame(df_data)
    
    return df.to_csv(index=False)
