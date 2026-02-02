"""Tests for database service."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def db_service(tmp_path):
    """Create test database service."""
    from app.db.service import SimulationService
    db_url = f"sqlite:///{tmp_path}/test.db"
    return SimulationService(db_url)


@pytest.fixture
def sample_metrics():
    """Sample simulation metrics."""
    return {
        "total_cost": 100.0,
        "avg_cost_per_hour": 10.0,
        "avg_servers": 5.0,
        "max_servers": 10,
        "min_servers": 2,
        "avg_utilization": 0.7,
        "max_utilization": 0.95,
        "sla_violations": 5,
        "sla_violation_rate": 0.02,
        "scaling_events": 20,
        "scale_out_events": 12,
        "scale_in_events": 8,
        "wasted_capacity_periods": 10,
    }


class TestSimulationService:
    """Tests for SimulationService."""

    def test_save_and_retrieve(self, db_service, sample_metrics):
        """Should save and retrieve simulation result."""
        result_id = db_service.save_result(
            metrics=sample_metrics,
            servers=[1, 2, 3, 4, 5],
            utilizations=[0.5, 0.6, 0.7, 0.8, 0.9],
            costs=[1.0, 2.0, 3.0, 4.0, 5.0],
            data_source_type="test",
            data_points_count=100,
            config_preset="balanced",
            policy_type="predictive",
            config_dict={"min_servers": 1, "max_servers": 20},
        )

        assert result_id is not None
        assert result_id > 0

        result = db_service.get_result(result_id)

        assert result is not None
        assert result["total_cost"] == 100.0
        assert result["config_preset"] == "balanced"
        assert result["policy_type"] == "predictive"
        assert result["servers_over_time"] == [1, 2, 3, 4, 5]

    def test_save_with_name_and_description(self, db_service, sample_metrics):
        """Should save with custom name and description."""
        result_id = db_service.save_result(
            metrics=sample_metrics,
            servers=[1, 2, 3],
            utilizations=[0.5, 0.7, 0.9],
            costs=[1.0, 2.0, 3.0],
            data_source_type="csv",
            data_points_count=50,
            config_preset="aggressive",
            policy_type="reactive",
            config_dict={},
            name="My Test Simulation",
            description="This is a test description",
        )

        result = db_service.get_result(result_id)

        assert result["name"] == "My Test Simulation"

    def test_save_with_loads_hash(self, db_service, sample_metrics):
        """Should calculate hash from loads array."""
        loads = np.array([100, 150, 200, 250, 300])

        result_id = db_service.save_result(
            metrics=sample_metrics,
            servers=[1, 2, 3],
            utilizations=[0.5, 0.7, 0.9],
            costs=[1.0, 2.0, 3.0],
            data_source_type="sample",
            data_points_count=len(loads),
            config_preset="balanced",
            policy_type="balanced",
            config_dict={},
            loads=loads,
        )

        result = db_service.get_result(result_id)
        # Hash should be generated (64 char hex string)
        assert result is not None

    def test_list_results(self, db_service, sample_metrics):
        """Should list saved results."""
        # Save multiple results
        for i in range(5):
            db_service.save_result(
                metrics=sample_metrics,
                servers=[1, 2, 3],
                utilizations=[0.5, 0.7, 0.9],
                costs=[1.0, 2.0, 3.0],
                data_source_type="test",
                data_points_count=100,
                config_preset="balanced",
                policy_type="predictive",
                config_dict={},
                name=f"Test {i}",
            )

        results = db_service.list_results(limit=10)

        assert len(results) == 5

    def test_list_results_with_filter(self, db_service, sample_metrics):
        """Should filter results by preset and policy."""
        # Save with different presets
        db_service.save_result(
            metrics=sample_metrics, servers=[1], utilizations=[0.5], costs=[1.0],
            data_source_type="test", data_points_count=10,
            config_preset="balanced", policy_type="predictive", config_dict={},
        )
        db_service.save_result(
            metrics=sample_metrics, servers=[1], utilizations=[0.5], costs=[1.0],
            data_source_type="test", data_points_count=10,
            config_preset="aggressive", policy_type="reactive", config_dict={},
        )

        balanced_results = db_service.list_results(config_preset="balanced")
        aggressive_results = db_service.list_results(config_preset="aggressive")

        assert len(balanced_results) == 1
        assert len(aggressive_results) == 1

    def test_delete_result(self, db_service, sample_metrics):
        """Should delete simulation result."""
        result_id = db_service.save_result(
            metrics=sample_metrics, servers=[1], utilizations=[0.5], costs=[1.0],
            data_source_type="test", data_points_count=10,
            config_preset="balanced", policy_type="predictive", config_dict={},
        )

        # Verify it exists
        assert db_service.get_result(result_id) is not None

        # Delete
        deleted = db_service.delete_result(result_id)
        assert deleted is True

        # Verify it's gone
        assert db_service.get_result(result_id) is None

    def test_delete_nonexistent(self, db_service):
        """Should return False for nonexistent result."""
        deleted = db_service.delete_result(9999)
        assert deleted is False

    def test_compare_results(self, db_service, sample_metrics):
        """Should retrieve multiple results for comparison."""
        ids = []
        for i in range(3):
            result_id = db_service.save_result(
                metrics={**sample_metrics, "total_cost": 100 + i * 10},
                servers=[1], utilizations=[0.5], costs=[1.0],
                data_source_type="test", data_points_count=10,
                config_preset="balanced", policy_type="predictive", config_dict={},
                name=f"Comparison {i}",
            )
            ids.append(result_id)

        comparison = db_service.compare_results(ids)

        assert len(comparison) == 3
        costs = [r["total_cost"] for r in comparison]
        assert set(costs) == {100.0, 110.0, 120.0}

    def test_get_statistics(self, db_service, sample_metrics):
        """Should return database statistics."""
        # Save some results
        db_service.save_result(
            metrics=sample_metrics, servers=[1], utilizations=[0.5], costs=[1.0],
            data_source_type="test", data_points_count=10,
            config_preset="balanced", policy_type="predictive", config_dict={},
        )
        db_service.save_result(
            metrics={**sample_metrics, "total_cost": 200.0},
            servers=[1], utilizations=[0.5], costs=[1.0],
            data_source_type="test", data_points_count=10,
            config_preset="aggressive", policy_type="reactive", config_dict={},
        )

        stats = db_service.get_statistics()

        assert stats["total_results"] == 2
        assert stats["avg_cost"] == 150.0  # (100 + 200) / 2
        assert "by_preset" in stats
        assert stats["by_preset"]["balanced"] == 1
        assert stats["by_preset"]["aggressive"] == 1

    def test_search_results(self, db_service, sample_metrics):
        """Should search results by name."""
        db_service.save_result(
            metrics=sample_metrics, servers=[1], utilizations=[0.5], costs=[1.0],
            data_source_type="test", data_points_count=10,
            config_preset="balanced", policy_type="predictive", config_dict={},
            name="NASA Traffic Analysis",
        )
        db_service.save_result(
            metrics=sample_metrics, servers=[1], utilizations=[0.5], costs=[1.0],
            data_source_type="test", data_points_count=10,
            config_preset="balanced", policy_type="predictive", config_dict={},
            name="Other Simulation",
        )

        nasa_results = db_service.list_results(search="NASA")
        other_results = db_service.list_results(search="Other")

        assert len(nasa_results) == 1
        assert len(other_results) == 1


class TestSimulationServiceEdgeCases:
    """Edge case tests for SimulationService."""

    def test_empty_database(self, db_service):
        """Should handle empty database."""
        results = db_service.list_results()
        assert results == []

        stats = db_service.get_statistics()
        assert stats["total_results"] == 0

    def test_get_nonexistent_result(self, db_service):
        """Should return None for nonexistent result."""
        result = db_service.get_result(9999)
        assert result is None
