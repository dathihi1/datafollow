"""Unit tests for the FastAPI application."""

from datetime import datetime

import pytest
from fastapi.testclient import TestClient

from src.api.main import app, _model_metrics, _state_lock


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


@pytest.fixture(autouse=True)
def reset_state():
    """Reset global state before each test."""
    with _state_lock:
        _model_metrics["rmse"] = None
        _model_metrics["mape"] = None
        _model_metrics["mae"] = None
        _model_metrics["r2"] = None
        _model_metrics["last_updated"] = None
        _model_metrics["training_samples"] = None
    yield


class TestRootEndpoint:
    """Tests for root endpoint."""

    def test_root(self, client):
        """Test root endpoint returns API info."""
        response = client.get("/")

        assert response.status_code == 200
        data = response.json()
        assert "name" in data
        assert data["name"] == "Autoscaling API"
        assert "version" in data
        assert "endpoints" in data


class TestHealthEndpoint:
    """Tests for health check endpoint."""

    def test_health_check(self, client):
        """Test health check returns healthy status."""
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data


class TestForecastEndpoint:
    """Tests for forecast endpoint."""

    def test_forecast_basic(self, client):
        """Test basic forecast request."""
        response = client.post(
            "/forecast?horizon=10",
            json={"historical_loads": [100, 120, 150, 180, 200]},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["horizon"] == 10
        assert len(data["predictions"]) == 10
        assert "model_used" in data
        assert "timestamp" in data

    def test_forecast_with_default_horizon(self, client):
        """Test forecast with default horizon."""
        response = client.post(
            "/forecast",
            json={"historical_loads": [100, 120, 150, 180, 200]},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["horizon"] == 30

    def test_forecast_with_confidence_intervals(self, client):
        """Test forecast returns confidence intervals."""
        response = client.post(
            "/forecast?horizon=5",
            json={"historical_loads": [100, 120, 150, 180, 200]},
        )

        assert response.status_code == 200
        data = response.json()
        assert "lower_bound" in data
        assert "upper_bound" in data
        assert len(data["lower_bound"]) == 5
        assert len(data["upper_bound"]) == 5

    def test_forecast_too_few_historical_values(self, client):
        """Test forecast fails with too few historical values."""
        response = client.post(
            "/forecast?horizon=10",
            json={"historical_loads": [100, 120]},
        )

        assert response.status_code == 400
        assert "historical load values required" in response.json()["detail"].lower()

    def test_forecast_invalid_horizon(self, client):
        """Test forecast fails with invalid horizon."""
        response = client.post(
            "/forecast?horizon=0",
            json={"historical_loads": [100, 120, 150, 180, 200]},
        )

        assert response.status_code == 422

    def test_forecast_horizon_too_large(self, client):
        """Test forecast fails with horizon too large."""
        response = client.post(
            "/forecast?horizon=500",
            json={"historical_loads": [100, 120, 150, 180, 200]},
        )

        assert response.status_code == 422

    def test_forecast_negative_loads_rejected(self, client):
        """Test that negative loads are rejected."""
        response = client.post(
            "/forecast?horizon=5",
            json={"historical_loads": [100, -50, 200]},
        )

        assert response.status_code == 422

    def test_forecast_predictions_non_negative(self, client):
        """Test that predictions are non-negative."""
        response = client.post(
            "/forecast?horizon=10",
            json={"historical_loads": [100, 120, 150, 180, 200]},
        )

        assert response.status_code == 200
        data = response.json()
        assert all(p >= 0 for p in data["predictions"])


class TestScalingEndpoint:
    """Tests for scaling recommendation endpoint."""

    def test_recommend_scaling_basic(self, client):
        """Test basic scaling recommendation."""
        response = client.post(
            "/recommend-scaling",
            json={
                "predicted_loads": [100, 150, 200],
                "current_servers": 2,
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert "action" in data
        assert "current_servers" in data
        assert "target_servers" in data
        assert "utilization" in data
        assert "reason" in data

    def test_recommend_scaling_with_policy_type(self, client):
        """Test scaling with different policy types."""
        for policy_type in ["reactive", "predictive", "balanced"]:
            response = client.post(
                "/recommend-scaling",
                json={
                    "predicted_loads": [100, 150, 200],
                    "current_servers": 2,
                    "policy_type": policy_type,
                },
            )

            assert response.status_code == 200

    def test_recommend_scaling_with_config_preset(self, client):
        """Test scaling with different config presets."""
        for preset in ["conservative", "balanced", "aggressive"]:
            response = client.post(
                "/recommend-scaling",
                json={
                    "predicted_loads": [100, 150, 200],
                    "current_servers": 2,
                    "config_preset": preset,
                },
            )

            assert response.status_code == 200

    def test_recommend_scaling_valid_actions(self, client):
        """Test that returned action is valid."""
        response = client.post(
            "/recommend-scaling",
            json={
                "predicted_loads": [100, 150, 200],
                "current_servers": 2,
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["action"] in ["scale_out", "scale_in", "hold"]

    def test_recommend_scaling_empty_loads(self, client):
        """Test scaling fails with empty loads."""
        response = client.post(
            "/recommend-scaling",
            json={
                "predicted_loads": [],
                "current_servers": 2,
            },
        )

        assert response.status_code == 422

    def test_recommend_scaling_invalid_policy_type(self, client):
        """Test scaling fails with invalid policy type."""
        response = client.post(
            "/recommend-scaling",
            json={
                "predicted_loads": [100, 150, 200],
                "current_servers": 2,
                "policy_type": "invalid_policy",
            },
        )

        assert response.status_code == 422

    def test_recommend_scaling_invalid_config_preset(self, client):
        """Test scaling fails with invalid config preset."""
        response = client.post(
            "/recommend-scaling",
            json={
                "predicted_loads": [100, 150, 200],
                "current_servers": 2,
                "config_preset": "invalid_preset",
            },
        )

        assert response.status_code == 422

    def test_recommend_scaling_negative_loads_rejected(self, client):
        """Test that negative loads are rejected."""
        response = client.post(
            "/recommend-scaling",
            json={
                "predicted_loads": [100, -50, 200],
                "current_servers": 2,
            },
        )

        assert response.status_code == 422


class TestMetricsEndpoint:
    """Tests for metrics endpoint."""

    def test_get_metrics(self, client):
        """Test getting model metrics."""
        response = client.get("/metrics")

        assert response.status_code == 200
        data = response.json()
        assert "model_name" in data
        assert "rmse" in data
        assert "mape" in data
        assert "mae" in data
        assert "r2" in data

    def test_update_metrics(self, client):
        """Test updating model metrics."""
        response = client.put(
            "/metrics",
            params={
                "rmse": 35.5,
                "mape": 12.0,
                "mae": 25.0,
                "r2": 0.85,
                "training_samples": 10000,
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "updated"

        get_response = client.get("/metrics")
        get_data = get_response.json()
        assert get_data["rmse"] == 35.5
        assert get_data["mape"] == 12.0


class TestCostReportEndpoint:
    """Tests for cost report endpoint."""

    def test_cost_report_basic(self, client):
        """Test basic cost report."""
        response = client.post(
            "/cost-report",
            json={
                "loads": [100, 150, 200, 250, 300, 250, 200, 150, 100],
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert "total_cost" in data
        assert "avg_cost_per_hour" in data
        assert "avg_servers" in data
        assert "sla_violations" in data
        assert "scaling_events" in data

    def test_cost_report_with_comparison(self, client):
        """Test cost report with fixed strategy comparison."""
        response = client.post(
            "/cost-report",
            json={
                "loads": [100, 150, 200, 250, 300, 250, 200, 150, 100],
                "compare_fixed": True,
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert "comparison" in data
        assert data["comparison"] is not None
        assert "fixed_min_servers" in data["comparison"]
        assert "fixed_max_servers" in data["comparison"]
        assert "autoscale_savings_vs_max" in data["comparison"]

    def test_cost_report_without_comparison(self, client):
        """Test cost report without comparison."""
        response = client.post(
            "/cost-report",
            json={
                "loads": [100, 150, 200, 250, 300],
                "compare_fixed": False,
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["comparison"] is None

    def test_cost_report_different_presets(self, client):
        """Test cost report with different config presets."""
        for preset in ["conservative", "balanced", "aggressive"]:
            response = client.post(
                "/cost-report",
                json={
                    "loads": [100, 150, 200, 250, 300],
                    "config_preset": preset,
                },
            )

            assert response.status_code == 200

    def test_cost_report_empty_loads(self, client):
        """Test cost report fails with empty loads."""
        response = client.post(
            "/cost-report",
            json={
                "loads": [],
            },
        )

        assert response.status_code == 422

    def test_cost_report_negative_loads_rejected(self, client):
        """Test that negative loads are rejected."""
        response = client.post(
            "/cost-report",
            json={
                "loads": [100, -50, 200],
            },
        )

        assert response.status_code == 422


class TestConfigEndpoint:
    """Tests for config endpoint."""

    def test_get_config_balanced(self, client):
        """Test getting balanced config."""
        response = client.get("/config/balanced")

        assert response.status_code == 200
        data = response.json()
        assert data["preset"] == "balanced"
        assert "config" in data
        assert "scale_out_threshold" in data["config"]

    def test_get_config_conservative(self, client):
        """Test getting conservative config."""
        response = client.get("/config/conservative")

        assert response.status_code == 200
        data = response.json()
        assert data["preset"] == "conservative"

    def test_get_config_aggressive(self, client):
        """Test getting aggressive config."""
        response = client.get("/config/aggressive")

        assert response.status_code == 200
        data = response.json()
        assert data["preset"] == "aggressive"

    def test_get_config_invalid_preset(self, client):
        """Test getting invalid preset fails."""
        response = client.get("/config/invalid")

        assert response.status_code == 422


class TestStatusEndpoint:
    """Tests for status endpoint."""

    def test_get_status(self, client):
        """Test getting current status."""
        response = client.get("/status")

        assert response.status_code == 200
        data = response.json()
        assert "current_servers" in data
        assert "model_metrics" in data
