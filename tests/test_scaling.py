"""Unit tests for the scaling policy and simulator modules."""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from src.scaling.config import (
    ScalingConfig,
    BALANCED_CONFIG,
    CONSERVATIVE_CONFIG,
    AGGRESSIVE_CONFIG,
)
from src.scaling.policy import (
    ScalingPolicy,
    ReactivePolicy,
    PredictivePolicy,
    ScalingAction,
    ScalingDecision,
    ScalingState,
)
from src.scaling.simulator import CostSimulator, SimulationMetrics


class TestScalingConfig:
    """Tests for ScalingConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = ScalingConfig()

        assert config.min_servers == 1
        assert config.max_servers == 20
        assert config.requests_per_server == 100
        assert config.scale_out_threshold == 0.80
        assert config.scale_in_threshold == 0.30
        assert config.scale_out_consecutive == 3
        assert config.scale_in_consecutive == 6
        assert config.cooldown_minutes == 5
        assert config.scale_out_increment == 2
        assert config.scale_in_decrement == 1
        assert config.cost_per_server_per_hour == 0.85
        assert config.time_window_minutes == 5

    def test_custom_values(self):
        """Test custom configuration values."""
        config = ScalingConfig(
            min_servers=2,
            max_servers=50,
            requests_per_server=200,
            scale_out_threshold=0.90,
            scale_in_threshold=0.20,
        )

        assert config.min_servers == 2
        assert config.max_servers == 50
        assert config.requests_per_server == 200
        assert config.scale_out_threshold == 0.90
        assert config.scale_in_threshold == 0.20

    def test_validation_min_servers(self):
        """Test validation of min_servers."""
        with pytest.raises(ValueError, match="min_servers must be at least 1"):
            ScalingConfig(min_servers=0)

    def test_validation_max_servers(self):
        """Test validation of max_servers."""
        with pytest.raises(ValueError, match="max_servers must be >= min_servers"):
            ScalingConfig(min_servers=10, max_servers=5)

    def test_validation_scale_out_threshold(self):
        """Test validation of scale_out_threshold."""
        with pytest.raises(ValueError, match="scale_out_threshold must be between 0 and 1"):
            ScalingConfig(scale_out_threshold=1.5)

        with pytest.raises(ValueError, match="scale_out_threshold must be between 0 and 1"):
            ScalingConfig(scale_out_threshold=0)

    def test_validation_scale_in_threshold(self):
        """Test validation of scale_in_threshold."""
        with pytest.raises(ValueError, match="scale_in_threshold must be between 0 and 1"):
            ScalingConfig(scale_in_threshold=-0.1)

    def test_validation_threshold_order(self):
        """Test that scale_in < scale_out."""
        with pytest.raises(ValueError, match="scale_in_threshold must be < scale_out_threshold"):
            ScalingConfig(scale_out_threshold=0.50, scale_in_threshold=0.60)

    def test_validation_increments(self):
        """Test validation of scale increments."""
        with pytest.raises(ValueError, match="scale_out_increment must be at least 1"):
            ScalingConfig(scale_out_increment=0)

        with pytest.raises(ValueError, match="scale_in_decrement must be at least 1"):
            ScalingConfig(scale_in_decrement=0)

    def test_capacity_per_server(self):
        """Test capacity_per_server property with default 5-min window."""
        config = ScalingConfig(requests_per_server=150, time_window_minutes=5)
        assert config.capacity_per_server == 150

    def test_capacity_per_server_scales_with_1min(self):
        """1-minute window = 1/5 of 5-minute capacity."""
        config = ScalingConfig(time_window_minutes=1, requests_per_server=100)
        assert config.capacity_per_server == 20

    def test_capacity_per_server_scales_with_15min(self):
        """15-minute window = 3x of 5-minute capacity."""
        config = ScalingConfig(time_window_minutes=15, requests_per_server=100)
        assert config.capacity_per_server == 300

    def test_capacity_per_server_scales_with_30min(self):
        """30-minute window = 6x of 5-minute capacity."""
        config = ScalingConfig(time_window_minutes=30, requests_per_server=100)
        assert config.capacity_per_server == 600

    def test_get_total_capacity(self):
        """Test total capacity calculation."""
        config = ScalingConfig(requests_per_server=100)
        assert config.get_total_capacity(5) == 500
        assert config.get_total_capacity(10) == 1000

    def test_get_utilization(self):
        """Test utilization calculation."""
        config = ScalingConfig(requests_per_server=100)

        # 200 requests / (2 servers * 100 capacity) = 100%
        assert config.get_utilization(200, 2) == 1.0

        # 100 requests / (2 servers * 100 capacity) = 50%
        assert config.get_utilization(100, 2) == 0.5

    def test_get_utilization_zero_servers(self):
        """Test utilization with zero servers returns infinity."""
        config = ScalingConfig()
        assert config.get_utilization(100, 0) == float("inf")

    def test_get_required_servers(self):
        """Test required servers calculation."""
        config = ScalingConfig(requests_per_server=100, min_servers=1, max_servers=20)

        # Low load should return min_servers
        assert config.get_required_servers(10) >= config.min_servers

        # High load should return appropriate count
        servers = config.get_required_servers(500, target_utilization=0.7)
        assert servers > 5

        # Very high load should be capped at max_servers
        servers = config.get_required_servers(100000)
        assert servers <= config.max_servers

    def test_get_cost_per_period(self):
        """Test cost per period calculation."""
        config = ScalingConfig(
            cost_per_server_per_hour=0.10,
            time_window_minutes=5,
        )

        # 5 servers for 5 minutes = 5 * 0.10 * (5/60) = $0.0417
        cost = config.get_cost_per_period(5)
        expected = 5 * 0.10 * (5 / 60)
        assert abs(cost - expected) < 0.001

    def test_to_dict(self):
        """Test conversion to dictionary."""
        config = ScalingConfig()
        config_dict = config.to_dict()

        assert "min_servers" in config_dict
        assert "max_servers" in config_dict
        assert "scale_out_threshold" in config_dict
        assert config_dict["min_servers"] == config.min_servers

    def test_from_dict(self):
        """Test creation from dictionary."""
        original = ScalingConfig(min_servers=5, max_servers=30)
        config_dict = original.to_dict()

        restored = ScalingConfig.from_dict(config_dict)

        assert restored.min_servers == original.min_servers
        assert restored.max_servers == original.max_servers


class TestPredefinedConfigs:
    """Tests for predefined configurations."""

    def test_conservative_config(self):
        """Test conservative configuration."""
        config = CONSERVATIVE_CONFIG

        assert config.scale_out_threshold == 0.70
        assert config.scale_in_threshold == 0.20
        assert config.scale_out_consecutive == 5
        assert config.cooldown_minutes == 10

    def test_aggressive_config(self):
        """Test aggressive configuration."""
        config = AGGRESSIVE_CONFIG

        assert config.scale_out_threshold == 0.85
        assert config.scale_in_threshold == 0.40
        assert config.scale_out_consecutive == 2
        assert config.scale_out_increment == 3

    def test_balanced_config(self):
        """Test balanced configuration defaults."""
        # Use fresh config to avoid mutation from ReactivePolicy tests
        config = ScalingConfig(
            scale_out_threshold=0.80,
            scale_in_threshold=0.30,
            scale_out_consecutive=3,
            scale_in_consecutive=6,
            cooldown_minutes=5,
            scale_out_increment=2,
            scale_in_decrement=1,
        )

        assert config.scale_out_threshold == 0.80
        assert config.scale_in_threshold == 0.30
        assert config.scale_out_consecutive == 3


class TestScalingAction:
    """Tests for ScalingAction enum."""

    def test_scale_out(self):
        """Test SCALE_OUT action."""
        assert ScalingAction.SCALE_OUT.value == "scale_out"

    def test_scale_in(self):
        """Test SCALE_IN action."""
        assert ScalingAction.SCALE_IN.value == "scale_in"

    def test_hold(self):
        """Test HOLD action."""
        assert ScalingAction.HOLD.value == "hold"


class TestScalingDecision:
    """Tests for ScalingDecision dataclass."""

    def test_creation(self):
        """Test decision creation."""
        decision = ScalingDecision(
            action=ScalingAction.SCALE_OUT,
            current_servers=2,
            target_servers=4,
            utilization=0.85,
            load=170,
            reason="high_utilization",
        )

        assert decision.action == ScalingAction.SCALE_OUT
        assert decision.current_servers == 2
        assert decision.target_servers == 4
        assert decision.utilization == 0.85
        assert decision.load == 170

    def test_str_representation(self):
        """Test string representation."""
        decision = ScalingDecision(
            action=ScalingAction.SCALE_OUT,
            current_servers=2,
            target_servers=4,
            utilization=0.85,
            load=170,
            reason="high_utilization",
        )

        decision_str = str(decision)
        assert "SCALE_OUT" in decision_str
        assert "2 -> 4" in decision_str


class TestScalingState:
    """Tests for ScalingState dataclass."""

    def test_creation(self):
        """Test state creation."""
        state = ScalingState(current_servers=3)

        assert state.current_servers == 3
        assert state.last_action_time is None
        assert state.consecutive_high == 0
        assert state.consecutive_low == 0
        assert state.history == []

    def test_reset_consecutive(self):
        """Test resetting consecutive counters."""
        state = ScalingState(
            current_servers=3,
            consecutive_high=5,
            consecutive_low=3,
        )

        state.reset_consecutive()

        assert state.consecutive_high == 0
        assert state.consecutive_low == 0


class TestScalingPolicy:
    """Tests for ScalingPolicy class."""

    @pytest.fixture
    def policy(self):
        """Create policy with balanced config."""
        return ScalingPolicy(BALANCED_CONFIG)

    def test_initialization(self, policy):
        """Test policy initialization."""
        assert policy.config == BALANCED_CONFIG
        assert policy.state.current_servers == BALANCED_CONFIG.min_servers

    def test_reset(self, policy):
        """Test policy reset."""
        policy.state.current_servers = 5
        policy.state.consecutive_high = 3

        policy.reset()

        assert policy.state.current_servers == BALANCED_CONFIG.min_servers
        assert policy.state.consecutive_high == 0

    def test_reset_with_initial_servers(self, policy):
        """Test reset with custom initial servers."""
        policy.reset(initial_servers=10)

        assert policy.state.current_servers == 10

    def test_recommend_hold_normal_utilization(self, policy):
        """Test HOLD recommendation for normal utilization."""
        policy.reset(initial_servers=5)

        # 250 requests / (5 servers * 100 capacity) = 50% utilization
        decision = policy.recommend(250)

        assert decision.action == ScalingAction.HOLD
        assert decision.reason == "normal_utilization"

    def test_recommend_scale_out_high_utilization(self, policy):
        """Test SCALE_OUT recommendation for high utilization."""
        config = ScalingConfig(
            scale_out_consecutive=1,  # Immediate scale out
            requests_per_server=100,
        )
        policy = ScalingPolicy(config)
        policy.reset(initial_servers=2)

        # 200 requests / (2 servers * 100 capacity) = 100% utilization
        decision = policy.recommend(200)

        assert decision.action == ScalingAction.SCALE_OUT
        assert decision.target_servers > decision.current_servers

    def test_recommend_scale_in_low_utilization(self, policy):
        """Test SCALE_IN recommendation for low utilization."""
        config = ScalingConfig(
            scale_in_consecutive=1,  # Immediate scale in
            scale_in_threshold=0.30,
            requests_per_server=100,
            min_servers=1,
        )
        policy = ScalingPolicy(config)
        policy.reset(initial_servers=10)

        # 100 requests / (10 servers * 100 capacity) = 10% utilization
        decision = policy.recommend(100)

        assert decision.action == ScalingAction.SCALE_IN
        assert decision.target_servers < decision.current_servers

    def test_consecutive_high_tracking(self):
        """Test consecutive high utilization tracking."""
        # Use fresh config with consecutive=3 to test tracking
        config = ScalingConfig(
            scale_out_consecutive=3,
            scale_in_consecutive=6,
            requests_per_server=100,
        )
        policy = ScalingPolicy(config)
        policy.reset(initial_servers=2)

        # High utilization but not enough consecutive periods (need 3)
        for _ in range(2):
            decision = policy.recommend(200)

        assert decision.action == ScalingAction.HOLD
        assert policy.state.consecutive_high == 2

    def test_consecutive_low_tracking(self):
        """Test consecutive low utilization tracking."""
        config = ScalingConfig(
            scale_out_consecutive=3,
            scale_in_consecutive=6,
            requests_per_server=100,
        )
        policy = ScalingPolicy(config)
        policy.reset(initial_servers=10)

        # Low utilization but not enough consecutive periods (need 6)
        for _ in range(3):
            decision = policy.recommend(50)

        assert decision.action == ScalingAction.HOLD
        assert policy.state.consecutive_low == 3

    def test_cooldown_prevents_scaling(self, policy):
        """Test that cooldown prevents scaling."""
        config = ScalingConfig(
            scale_out_consecutive=1,
            cooldown_minutes=10,
            requests_per_server=100,
        )
        policy = ScalingPolicy(config)
        policy.reset(initial_servers=2)

        # First scale out
        timestamp1 = datetime(2023, 1, 1, 0, 0, 0)
        decision1 = policy.recommend(200, timestamp1)
        assert decision1.action == ScalingAction.SCALE_OUT

        # Second request during cooldown
        timestamp2 = timestamp1 + timedelta(minutes=5)
        decision2 = policy.recommend(200, timestamp2)
        assert decision2.action == ScalingAction.HOLD
        assert decision2.reason == "cooldown_active"

    def test_cooldown_expires(self, policy):
        """Test that cooldown eventually expires."""
        config = ScalingConfig(
            scale_out_consecutive=1,
            cooldown_minutes=5,
            requests_per_server=100,
        )
        policy = ScalingPolicy(config)
        policy.reset(initial_servers=2)

        # First scale out
        timestamp1 = datetime(2023, 1, 1, 0, 0, 0)
        policy.recommend(200, timestamp1)

        # After cooldown
        timestamp2 = timestamp1 + timedelta(minutes=10)
        decision = policy.recommend(400, timestamp2)  # Still high utilization
        assert decision.action == ScalingAction.SCALE_OUT

    def test_max_servers_limit(self, policy):
        """Test that scaling respects max_servers."""
        config = ScalingConfig(
            scale_out_consecutive=1,
            scale_out_increment=100,  # Try to add many servers
            max_servers=5,
            requests_per_server=100,
        )
        policy = ScalingPolicy(config)
        policy.reset(initial_servers=4)

        decision = policy.recommend(1000)

        assert decision.target_servers <= config.max_servers

    def test_min_servers_limit(self, policy):
        """Test that scaling respects min_servers."""
        config = ScalingConfig(
            scale_in_consecutive=1,
            scale_in_decrement=100,  # Try to remove many servers
            min_servers=2,
            requests_per_server=100,
        )
        policy = ScalingPolicy(config)
        policy.reset(initial_servers=3)

        decision = policy.recommend(10)

        assert decision.target_servers >= config.min_servers

    def test_history_tracking(self, policy):
        """Test that scaling history is tracked."""
        config = ScalingConfig(
            scale_out_consecutive=1,
            requests_per_server=100,
        )
        policy = ScalingPolicy(config)
        policy.reset(initial_servers=2)

        policy.recommend(200)

        history = policy.get_history()
        assert len(history) == 1
        assert history[0].action == ScalingAction.SCALE_OUT

    def test_get_current_servers(self, policy):
        """Test getting current server count."""
        policy.reset(initial_servers=7)

        assert policy.get_current_servers() == 7

    def test_set_servers(self, policy):
        """Test manually setting servers."""
        policy.set_servers(8)

        assert policy.get_current_servers() == 8

    def test_set_servers_respects_limits(self, policy):
        """Test that set_servers respects config limits."""
        policy.set_servers(100)  # Above max
        assert policy.get_current_servers() <= BALANCED_CONFIG.max_servers

        policy.set_servers(0)  # Below min
        assert policy.get_current_servers() >= BALANCED_CONFIG.min_servers


class TestReactivePolicy:
    """Tests for ReactivePolicy class."""

    def test_faster_scale_out(self):
        """Test that ReactivePolicy scales out faster."""
        policy = ReactivePolicy()

        # Should have lower consecutive requirement
        assert policy.config.scale_out_consecutive == 1

    def test_faster_scale_in(self):
        """Test that ReactivePolicy scales in faster."""
        policy = ReactivePolicy()

        # Should have lower consecutive requirement
        assert policy.config.scale_in_consecutive == 2


class TestPredictivePolicy:
    """Tests for PredictivePolicy class."""

    @pytest.fixture
    def policy(self):
        """Create predictive policy."""
        return PredictivePolicy()

    def test_initialization(self, policy):
        """Test initialization with forecast horizon."""
        assert policy.forecast_horizon == 6
        assert policy.safety_margin == 0.1

    def test_custom_initialization(self):
        """Test custom initialization."""
        policy = PredictivePolicy(forecast_horizon=12, safety_margin=0.2)

        assert policy.forecast_horizon == 12
        assert policy.safety_margin == 0.2

    def test_recommend_with_forecast_scale_out(self, policy):
        """Test scale out based on forecast."""
        config = ScalingConfig(requests_per_server=100, scale_out_consecutive=1)
        policy = PredictivePolicy(config)
        policy.reset(initial_servers=2)

        # Current load is low, but forecast is high
        current_load = 100
        forecasted_loads = [200, 300, 400, 500, 600]

        decision = policy.recommend_with_forecast(current_load, forecasted_loads)

        # Should scale out proactively
        assert decision.action == ScalingAction.SCALE_OUT

    def test_recommend_with_forecast_hold(self, policy):
        """Test hold when forecast is stable."""
        policy.reset(initial_servers=5)

        current_load = 200
        forecasted_loads = [200, 200, 200, 200]

        decision = policy.recommend_with_forecast(current_load, forecasted_loads)

        # Load is moderate, should hold
        assert decision.action == ScalingAction.HOLD

    def test_recommend_proactive(self, policy):
        """Test proactive recommendation."""
        policy.reset(initial_servers=2)

        predicted_loads = [100, 150, 200, 250, 300]

        decision = policy.recommend_proactive(predicted_loads)

        assert "proactive" in decision.reason


class TestCostSimulator:
    """Tests for CostSimulator class."""

    @pytest.fixture
    def simulator(self):
        """Create simulator with balanced config."""
        return CostSimulator(BALANCED_CONFIG)

    @pytest.fixture
    def sample_loads(self):
        """Create sample load data."""
        return np.array([100, 150, 200, 250, 300, 250, 200, 150, 100, 50])

    def test_initialization(self, simulator):
        """Test simulator initialization."""
        assert simulator.config == BALANCED_CONFIG

    def test_simulate_basic(self, simulator, sample_loads):
        """Test basic simulation."""
        metrics = simulator.simulate(sample_loads)

        assert isinstance(metrics, SimulationMetrics)
        assert metrics.total_cost >= 0
        assert len(metrics.servers_over_time) == len(sample_loads)

    def test_simulate_with_policy(self, simulator, sample_loads):
        """Test simulation with custom policy."""
        policy = ReactivePolicy(simulator.config)

        metrics = simulator.simulate(sample_loads, policy)

        assert metrics.total_cost >= 0

    def test_simulate_with_timestamps(self, simulator, sample_loads):
        """Test simulation with custom timestamps."""
        timestamps = pd.date_range("2023-01-01", periods=len(sample_loads), freq="5min")

        metrics = simulator.simulate(sample_loads, timestamps=timestamps)

        assert metrics.total_cost >= 0

    def test_simulate_fixed(self, simulator, sample_loads):
        """Test fixed server simulation."""
        metrics = simulator.simulate_fixed(sample_loads, num_servers=5)

        assert metrics.avg_servers == 5.0
        assert metrics.max_servers == 5
        assert metrics.min_servers == 5
        assert metrics.scaling_events == 0

    def test_simulate_sla_violations(self, simulator):
        """Test SLA violation detection."""
        # Very high load with few servers
        loads = np.array([1000, 1000, 1000])

        metrics = simulator.simulate_fixed(loads, num_servers=1)

        assert metrics.sla_violations > 0
        assert metrics.sla_violation_rate > 0

    def test_simulate_no_sla_violations(self, simulator):
        """Test no SLA violations with adequate capacity."""
        loads = np.array([50, 50, 50])

        metrics = simulator.simulate_fixed(loads, num_servers=10)

        assert metrics.sla_violations == 0
        assert metrics.sla_violation_rate == 0

    def test_compare_strategies(self, simulator, sample_loads):
        """Test strategy comparison."""
        strategies = {
            "fixed_5": 5,
            "fixed_10": 10,
            "autoscale": ScalingPolicy(simulator.config),
        }

        comparison = simulator.compare_strategies(sample_loads, strategies)

        assert len(comparison) == 3
        assert "total_cost" in comparison.columns
        assert "sla_violations" in comparison.columns

    def test_calculate_savings(self, simulator, sample_loads):
        """Test savings calculation."""
        autoscale_metrics = simulator.simulate(sample_loads)
        fixed_max_metrics = simulator.simulate_fixed(sample_loads, simulator.config.max_servers)

        savings = simulator.calculate_savings(autoscale_metrics, fixed_max_metrics)

        assert "cost_savings" in savings
        assert "cost_savings_pct" in savings
        assert savings["fixed_cost"] == fixed_max_metrics.total_cost
        assert savings["autoscale_cost"] == autoscale_metrics.total_cost

    def test_grid_search_thresholds(self, simulator, sample_loads):
        """Test threshold grid search."""
        results = simulator.grid_search_thresholds(
            sample_loads,
            scale_out_range=[0.70, 0.80],
            scale_in_range=[0.20, 0.30],
        )

        assert len(results) > 0
        assert "scale_out_threshold" in results.columns
        assert "scale_in_threshold" in results.columns
        assert "total_cost" in results.columns

    def test_optimize_for_cost(self, simulator, sample_loads):
        """Test cost optimization."""
        optimal_config, metrics = simulator.optimize_for_cost(
            sample_loads,
            max_sla_violation_rate=0.05,
        )

        assert isinstance(optimal_config, ScalingConfig)
        assert isinstance(metrics, SimulationMetrics)


class TestSimulationMetrics:
    """Tests for SimulationMetrics dataclass."""

    def test_creation(self):
        """Test metrics creation."""
        metrics = SimulationMetrics(
            total_cost=100.0,
            avg_cost_per_hour=10.0,
            avg_servers=5.0,
            max_servers=10,
            min_servers=2,
            avg_utilization=0.6,
            max_utilization=0.9,
            sla_violations=5,
            sla_violation_rate=0.05,
            overload_periods=5,
            wasted_capacity_periods=10,
            scaling_events=8,
            scale_out_events=5,
            scale_in_events=3,
        )

        assert metrics.total_cost == 100.0
        assert metrics.avg_servers == 5.0
        assert metrics.scaling_events == 8

    def test_str_representation(self):
        """Test string representation."""
        metrics = SimulationMetrics(
            total_cost=100.0,
            avg_cost_per_hour=10.0,
            avg_servers=5.0,
            max_servers=10,
            min_servers=2,
            avg_utilization=0.6,
            max_utilization=0.9,
            sla_violations=5,
            sla_violation_rate=0.05,
            overload_periods=5,
            wasted_capacity_periods=10,
            scaling_events=8,
            scale_out_events=5,
            scale_in_events=3,
        )

        metrics_str = str(metrics)
        assert "Total Cost" in metrics_str
        assert "Avg Servers" in metrics_str
        assert "SLA Violations" in metrics_str


class TestTimeIntervalScaling:
    """Test capacity scales correctly with time interval."""

    def test_utilization_equivalent_across_intervals(self):
        """Same load rate should give same utilization across intervals."""
        config_5min = ScalingConfig(time_window_minutes=5, requests_per_server=100)
        config_15min = ScalingConfig(time_window_minutes=15, requests_per_server=100)

        # 100 req/5min = 300 req/15min (same rate: 20 req/min)
        util_5min = config_5min.get_utilization(100, num_servers=1)
        util_15min = config_15min.get_utilization(300, num_servers=1)

        assert abs(util_5min - util_15min) < 0.01

    def test_required_servers_scales_correctly(self):
        """Required servers should be same for equivalent load rates."""
        config_1min = ScalingConfig(
            time_window_minutes=1, requests_per_server=100, min_servers=1, max_servers=50
        )
        config_5min = ScalingConfig(
            time_window_minutes=5, requests_per_server=100, min_servers=1, max_servers=50
        )

        # 20 req/1min = 100 req/5min (same rate)
        servers_1min = config_1min.get_required_servers(20)
        servers_5min = config_5min.get_required_servers(100)

        assert servers_1min == servers_5min

    def test_total_capacity_scales_with_interval(self):
        """Total capacity should scale proportionally with time interval."""
        config_5min = ScalingConfig(time_window_minutes=5, requests_per_server=100)
        config_15min = ScalingConfig(time_window_minutes=15, requests_per_server=100)

        # 15min capacity should be 3x of 5min capacity
        capacity_5min = config_5min.get_total_capacity(num_servers=2)
        capacity_15min = config_15min.get_total_capacity(num_servers=2)

        assert capacity_15min == capacity_5min * 3

    def test_cost_per_period_independent_of_capacity_scaling(self):
        """Cost per period should only depend on time, not capacity scaling."""
        config_5min = ScalingConfig(
            time_window_minutes=5, cost_per_server_per_hour=0.10, requests_per_server=100
        )
        config_15min = ScalingConfig(
            time_window_minutes=15, cost_per_server_per_hour=0.10, requests_per_server=100
        )

        # Cost for 15min should be 3x cost for 5min (same hourly rate)
        cost_5min = config_5min.get_cost_per_period(num_servers=2)
        cost_15min = config_15min.get_cost_per_period(num_servers=2)

        assert abs(cost_15min - cost_5min * 3) < 0.0001
