"""FastAPI application for autoscaling predictions."""

import logging
from datetime import datetime
from enum import Enum
from threading import Lock
from typing import Annotated

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field, field_validator

from src.scaling.config import ScalingConfig, BALANCED_CONFIG, CONSERVATIVE_CONFIG, AGGRESSIVE_CONFIG
from src.scaling.policy import (
    ScalingPolicy,
    PredictivePolicy,
    ReactivePolicy,
    ScalingAction,
)
from src.scaling.simulator import CostSimulator, SimulationMetrics

logger = logging.getLogger(__name__)

# Constants
FORECAST_WINDOW_SIZE = 12
FORECAST_PERIOD_CYCLE = 12
FORECAST_TREND_FACTOR = 0.01
FORECAST_PERIODICITY_FACTOR = 0.1
CONFIDENCE_Z_SCORE = 1.96
MIN_HISTORICAL_PERIODS = 3
MAX_LOAD_VALUE = 1e9
MAX_LOADS_LENGTH = 100_000


class PolicyType(str, Enum):
    """Available scaling policy types."""

    REACTIVE = "reactive"
    PREDICTIVE = "predictive"
    BALANCED = "balanced"


class ConfigPreset(str, Enum):
    """Available configuration presets."""

    CONSERVATIVE = "conservative"
    BALANCED = "balanced"
    AGGRESSIVE = "aggressive"


app = FastAPI(
    title="Autoscaling API",
    description="Traffic prediction and autoscaling recommendation service for NASA web server logs",
    version="1.0.0",
)


class ForecastRequest(BaseModel):
    """Request body for forecast endpoint."""

    historical_loads: list[float] = Field(
        ...,
        description="Historical load values for prediction",
        min_length=1,
        max_length=MAX_LOADS_LENGTH,
    )
    features: dict[str, float] | None = Field(
        default=None,
        description="Optional feature dictionary for ML model",
    )

    @field_validator("historical_loads")
    @classmethod
    def validate_loads(cls, v: list[float]) -> list[float]:
        """Validate that loads are non-negative and within bounds."""
        for val in v:
            if val < 0:
                raise ValueError("Historical loads must be non-negative")
            if val > MAX_LOAD_VALUE:
                raise ValueError(f"Historical load values must not exceed {MAX_LOAD_VALUE}")
        return v


class ForecastResponse(BaseModel):
    """Response body for forecast endpoint."""

    horizon: int
    predictions: list[float]
    lower_bound: list[float] | None = None
    upper_bound: list[float] | None = None
    model_used: str
    timestamp: datetime


class ScalingRequest(BaseModel):
    """Request body for scaling recommendation endpoint."""

    predicted_loads: list[float] = Field(
        ...,
        description="Predicted future load values",
        min_length=1,
        max_length=MAX_LOADS_LENGTH,
    )
    current_servers: int = Field(
        default=1,
        description="Current number of active servers",
        ge=1,
        le=1000,
    )
    policy_type: PolicyType = Field(
        default=PolicyType.PREDICTIVE,
        description="Scaling policy type",
    )
    config_preset: ConfigPreset = Field(
        default=ConfigPreset.BALANCED,
        description="Configuration preset",
    )

    @field_validator("predicted_loads")
    @classmethod
    def validate_loads(cls, v: list[float]) -> list[float]:
        """Validate that loads are non-negative and within bounds."""
        for val in v:
            if val < 0:
                raise ValueError("Predicted loads must be non-negative")
            if val > MAX_LOAD_VALUE:
                raise ValueError(f"Predicted load values must not exceed {MAX_LOAD_VALUE}")
        return v


class ScalingResponse(BaseModel):
    """Response body for scaling recommendation endpoint."""

    action: str
    current_servers: int
    target_servers: int
    utilization: float
    load: float
    reason: str
    timestamp: datetime


class MetricsResponse(BaseModel):
    """Response body for metrics endpoint."""

    model_name: str
    rmse: float | None
    mape: float | None
    mae: float | None
    r2: float | None
    last_updated: datetime | None
    training_samples: int | None


class CostReportRequest(BaseModel):
    """Request body for cost report endpoint."""

    loads: list[float] = Field(
        ...,
        description="Time series of request loads",
        min_length=1,
        max_length=MAX_LOADS_LENGTH,
    )
    config_preset: ConfigPreset = Field(
        default=ConfigPreset.BALANCED,
        description="Configuration preset",
    )
    compare_fixed: bool = Field(
        default=True,
        description="Whether to compare with fixed server strategies",
    )

    @field_validator("loads")
    @classmethod
    def validate_loads(cls, v: list[float]) -> list[float]:
        """Validate that loads are non-negative and within bounds."""
        for val in v:
            if val < 0:
                raise ValueError("Loads must be non-negative")
            if val > MAX_LOAD_VALUE:
                raise ValueError(f"Load values must not exceed {MAX_LOAD_VALUE}")
        return v


class CostReportResponse(BaseModel):
    """Response body for cost report endpoint."""

    total_cost: float
    avg_cost_per_hour: float
    avg_servers: float
    max_servers: int
    min_servers: int
    avg_utilization: float
    max_utilization: float
    sla_violations: int
    sla_violation_rate: float
    scaling_events: int
    scale_out_events: int
    scale_in_events: int
    comparison: dict | None = None


# Thread-safe application state
_state_lock = Lock()

_model_metrics: dict = {
    "model_name": "LightGBM",
    "rmse": None,
    "mape": None,
    "mae": None,
    "r2": None,
    "last_updated": None,
    "training_samples": None,
}

_scaling_state: dict = {
    "current_servers": 1,
    "last_action_time": None,
}

_CONFIG_MAP: dict[str, ScalingConfig] = {
    "conservative": CONSERVATIVE_CONFIG,
    "balanced": BALANCED_CONFIG,
    "aggressive": AGGRESSIVE_CONFIG,
}

_POLICY_MAP: dict[str, type] = {
    "reactive": ReactivePolicy,
    "predictive": PredictivePolicy,
    "balanced": ScalingPolicy,
}


def _get_config(preset: str) -> ScalingConfig:
    """Get scaling configuration by preset name."""
    return _CONFIG_MAP.get(preset, BALANCED_CONFIG)


def _get_policy(policy_type: str, config: ScalingConfig) -> ScalingPolicy:
    """Get scaling policy by type."""
    policy_class = _POLICY_MAP.get(policy_type, PredictivePolicy)
    return policy_class(config)


@app.get("/")
async def root():
    """Root endpoint with API info."""
    return {
        "name": "Autoscaling API",
        "version": "1.0.0",
        "endpoints": {
            "forecast": "POST /forecast?horizon=30",
            "recommend_scaling": "POST /recommend-scaling",
            "metrics": "GET /metrics",
            "cost_report": "POST /cost-report",
        },
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


@app.post("/forecast", response_model=ForecastResponse)
async def forecast(
    request: ForecastRequest,
    horizon: Annotated[int, Query(ge=1, le=288, description="Forecast horizon in periods")] = 30,
):
    """Predict traffic for next N periods.

    Uses historical load patterns to forecast future traffic.
    For production, this would use a trained LightGBM model.
    """
    historical = request.historical_loads

    if len(historical) < MIN_HISTORICAL_PERIODS:
        raise HTTPException(
            status_code=400,
            detail=f"At least {MIN_HISTORICAL_PERIODS} historical load values required for prediction",
        )

    window = min(FORECAST_WINDOW_SIZE, len(historical))
    recent_avg = sum(historical[-window:]) / window
    recent_std = (
        sum((x - recent_avg) ** 2 for x in historical[-window:]) / window
    ) ** 0.5

    predictions = []
    for i in range(horizon):
        trend = FORECAST_TREND_FACTOR * i
        periodicity = FORECAST_PERIODICITY_FACTOR * ((i % FORECAST_PERIOD_CYCLE) - 6) / 6
        pred = recent_avg * (1 + trend + periodicity)
        predictions.append(max(0, pred))

    lower_bound = [max(0, p - CONFIDENCE_Z_SCORE * recent_std) for p in predictions]
    upper_bound = [p + CONFIDENCE_Z_SCORE * recent_std for p in predictions]

    logger.info("Generated forecast with horizon=%d, model=moving_average_baseline", horizon)

    return ForecastResponse(
        horizon=horizon,
        predictions=predictions,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        model_used="moving_average_baseline",
        timestamp=datetime.now(),
    )


@app.post("/recommend-scaling", response_model=ScalingResponse)
async def recommend_scaling(request: ScalingRequest):
    """Get scaling recommendation based on predicted loads.

    Analyzes predicted load patterns and current server state
    to recommend scale out, scale in, or hold actions.
    """
    config = _get_config(request.config_preset.value)
    policy = _get_policy(request.policy_type.value, config)

    policy.set_servers(request.current_servers)

    timestamp = datetime.now()

    if isinstance(policy, PredictivePolicy):
        current_load = request.predicted_loads[0]
        future_loads = request.predicted_loads[1:] if len(request.predicted_loads) > 1 else []
        decision = policy.recommend_with_forecast(current_load, future_loads, timestamp)
    else:
        decision = policy.recommend_proactive(request.predicted_loads, timestamp)

    with _state_lock:
        _scaling_state["current_servers"] = decision.target_servers
        _scaling_state["last_action_time"] = timestamp

    logger.info(
        "Scaling recommendation: action=%s, servers=%d->%d",
        decision.action.value, decision.current_servers, decision.target_servers,
    )

    return ScalingResponse(
        action=decision.action.value,
        current_servers=decision.current_servers,
        target_servers=decision.target_servers,
        utilization=decision.utilization,
        load=decision.load,
        reason=decision.reason,
        timestamp=timestamp,
    )


@app.get("/metrics", response_model=MetricsResponse)
async def get_metrics():
    """Get current model performance metrics.

    Returns the latest evaluation metrics from the trained model.
    Metrics are updated after each model retraining.
    """
    with _state_lock:
        return MetricsResponse(
            model_name=_model_metrics["model_name"],
            rmse=_model_metrics["rmse"],
            mape=_model_metrics["mape"],
            mae=_model_metrics["mae"],
            r2=_model_metrics["r2"],
            last_updated=_model_metrics["last_updated"],
            training_samples=_model_metrics["training_samples"],
        )


@app.put("/metrics")
async def update_metrics(
    rmse: float | None = None,
    mape: float | None = None,
    mae: float | None = None,
    r2: float | None = None,
    training_samples: int | None = None,
):
    """Update model metrics (internal use for model training pipeline)."""
    with _state_lock:
        if rmse is not None:
            _model_metrics["rmse"] = rmse
        if mape is not None:
            _model_metrics["mape"] = mape
        if mae is not None:
            _model_metrics["mae"] = mae
        if r2 is not None:
            _model_metrics["r2"] = r2
        if training_samples is not None:
            _model_metrics["training_samples"] = training_samples
        _model_metrics["last_updated"] = datetime.now()

        metrics_snapshot = dict(_model_metrics)

    logger.info("Model metrics updated")
    return {"status": "updated", "metrics": metrics_snapshot}


@app.post("/cost-report", response_model=CostReportResponse)
async def cost_report(request: CostReportRequest):
    """Generate cost analysis report.

    Simulates autoscaling behavior on the provided load time series
    and calculates costs, utilization, and SLA metrics.
    """
    config = _get_config(request.config_preset.value)
    simulator = CostSimulator(config)
    policy = ScalingPolicy(config)

    metrics = simulator.simulate(request.loads, policy)

    comparison = None
    if request.compare_fixed:
        fixed_min = simulator.simulate_fixed(request.loads, config.min_servers)
        fixed_max = simulator.simulate_fixed(request.loads, config.max_servers)

        avg_load = sum(request.loads) / len(request.loads)
        optimal_fixed = config.get_required_servers(avg_load, target_utilization=0.7)
        fixed_optimal = simulator.simulate_fixed(request.loads, optimal_fixed)

        comparison = {
            "fixed_min_servers": {
                "servers": config.min_servers,
                "total_cost": fixed_min.total_cost,
                "sla_violations": fixed_min.sla_violations,
            },
            "fixed_max_servers": {
                "servers": config.max_servers,
                "total_cost": fixed_max.total_cost,
                "sla_violations": fixed_max.sla_violations,
            },
            "fixed_optimal": {
                "servers": optimal_fixed,
                "total_cost": fixed_optimal.total_cost,
                "sla_violations": fixed_optimal.sla_violations,
            },
            "autoscale_savings_vs_max": {
                "cost_saved": fixed_max.total_cost - metrics.total_cost,
                "percentage": (
                    (fixed_max.total_cost - metrics.total_cost) / fixed_max.total_cost * 100
                    if fixed_max.total_cost > 0 else 0
                ),
            },
        }

    return CostReportResponse(
        total_cost=metrics.total_cost,
        avg_cost_per_hour=metrics.avg_cost_per_hour,
        avg_servers=metrics.avg_servers,
        max_servers=metrics.max_servers,
        min_servers=metrics.min_servers,
        avg_utilization=metrics.avg_utilization,
        max_utilization=metrics.max_utilization,
        sla_violations=metrics.sla_violations,
        sla_violation_rate=metrics.sla_violation_rate,
        scaling_events=metrics.scaling_events,
        scale_out_events=metrics.scale_out_events,
        scale_in_events=metrics.scale_in_events,
        comparison=comparison,
    )


@app.get("/config/{preset}")
async def get_config_endpoint(preset: ConfigPreset):
    """Get scaling configuration for a preset."""
    config = _get_config(preset.value)
    return {
        "preset": preset.value,
        "config": config.to_dict(),
    }


@app.get("/status")
async def get_status():
    """Get current scaling status."""
    with _state_lock:
        return {
            "current_servers": _scaling_state["current_servers"],
            "last_action_time": _scaling_state["last_action_time"],
            "model_metrics": dict(_model_metrics),
        }


def run_server():
    """Run the FastAPI server with uvicorn."""
    import uvicorn
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
        timeout_graceful_shutdown=5,  # Fast shutdown
    )


if __name__ == "__main__":
    run_server()
