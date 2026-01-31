"""Cost simulation for autoscaling strategies."""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Callable

import numpy as np
import pandas as pd

from src.scaling.config import ScalingConfig
from src.scaling.policy import ScalingPolicy, ScalingAction, ScalingDecision


@dataclass
class SimulationMetrics:
    """Metrics from a simulation run."""

    # Cost metrics
    total_cost: float
    avg_cost_per_hour: float

    # Capacity metrics
    avg_servers: float
    max_servers: int
    min_servers: int

    # Utilization metrics
    avg_utilization: float
    max_utilization: float

    # SLA metrics
    sla_violations: int
    sla_violation_rate: float
    overload_periods: int

    # Efficiency metrics
    wasted_capacity_periods: int  # Periods with < 30% utilization
    scaling_events: int
    scale_out_events: int
    scale_in_events: int

    # Time series
    servers_over_time: list = field(default_factory=list)
    utilization_over_time: list = field(default_factory=list)
    cost_over_time: list = field(default_factory=list)

    def __str__(self) -> str:
        return (
            f"Simulation Results:\n"
            f"  Total Cost: ${self.total_cost:.2f}\n"
            f"  Avg Cost/Hour: ${self.avg_cost_per_hour:.4f}\n"
            f"  Avg Servers: {self.avg_servers:.2f}\n"
            f"  Avg Utilization: {self.avg_utilization:.1%}\n"
            f"  SLA Violations: {self.sla_violations} ({self.sla_violation_rate:.1%})\n"
            f"  Scaling Events: {self.scaling_events} (out: {self.scale_out_events}, in: {self.scale_in_events})"
        )


class CostSimulator:
    """Simulate autoscaling behavior and calculate costs.

    Runs historical or predicted load through a scaling policy
    and calculates resulting costs, utilization, and SLA metrics.
    """

    def __init__(self, config: ScalingConfig | None = None):
        """Initialize simulator.

        Args:
            config: Scaling configuration
        """
        self.config = config or ScalingConfig()

    def simulate(
        self,
        loads: pd.Series | np.ndarray | list,
        policy: ScalingPolicy | None = None,
        timestamps: pd.DatetimeIndex | list | None = None,
        initial_servers: int | None = None,
    ) -> SimulationMetrics:
        """Run simulation with given loads and policy.

        Args:
            loads: Time series of request loads
            policy: Scaling policy to use (creates default if None)
            timestamps: Timestamps for each load (generates if None)
            initial_servers: Starting number of servers

        Returns:
            SimulationMetrics with results
        """
        loads = np.array(loads)
        n_periods = len(loads)

        # Create policy if not provided
        if policy is None:
            policy = ScalingPolicy(self.config)
        policy.reset(initial_servers or self.config.min_servers)

        # Generate timestamps if not provided
        if timestamps is None:
            start = datetime(2023, 1, 1, 0, 0, 0)
            timestamps = [
                start + timedelta(minutes=i * self.config.time_window_minutes)
                for i in range(n_periods)
            ]

        # Tracking arrays
        servers = []
        utilizations = []
        costs = []
        decisions = []

        # Run simulation
        for i, (load, ts) in enumerate(zip(loads, timestamps)):
            # Get scaling decision
            decision = policy.recommend(load, ts)
            decisions.append(decision)

            # Record metrics
            current_servers = policy.get_current_servers()
            servers.append(current_servers)

            utilization = self.config.get_utilization(load, current_servers)
            utilizations.append(utilization)

            cost = self.config.get_cost_per_period(current_servers)
            costs.append(cost)

        # Calculate metrics
        servers = np.array(servers)
        utilizations = np.array(utilizations)
        costs = np.array(costs)

        # Count scaling events
        scale_out_events = sum(1 for d in decisions if d.action == ScalingAction.SCALE_OUT)
        scale_in_events = sum(1 for d in decisions if d.action == ScalingAction.SCALE_IN)

        # SLA violations (utilization > 100%)
        sla_violations = np.sum(utilizations > 1.0)

        # Wasted capacity (utilization < 30%)
        wasted_capacity = np.sum(utilizations < 0.3)

        # Calculate time
        total_hours = (n_periods * self.config.time_window_minutes) / 60

        return SimulationMetrics(
            total_cost=float(np.sum(costs)),
            avg_cost_per_hour=float(np.sum(costs) / total_hours) if total_hours > 0 else 0,
            avg_servers=float(np.mean(servers)),
            max_servers=int(np.max(servers)),
            min_servers=int(np.min(servers)),
            avg_utilization=float(np.mean(utilizations)),
            max_utilization=float(np.max(utilizations)),
            sla_violations=int(sla_violations),
            sla_violation_rate=float(sla_violations / n_periods) if n_periods > 0 else 0,
            overload_periods=int(sla_violations),
            wasted_capacity_periods=int(wasted_capacity),
            scaling_events=scale_out_events + scale_in_events,
            scale_out_events=scale_out_events,
            scale_in_events=scale_in_events,
            servers_over_time=servers.tolist(),
            utilization_over_time=utilizations.tolist(),
            cost_over_time=costs.tolist(),
        )

    def simulate_fixed(
        self,
        loads: pd.Series | np.ndarray | list,
        num_servers: int,
    ) -> SimulationMetrics:
        """Simulate with fixed number of servers (no scaling).

        Args:
            loads: Time series of request loads
            num_servers: Fixed number of servers

        Returns:
            SimulationMetrics with results
        """
        loads = np.array(loads)
        n_periods = len(loads)

        # Calculate metrics with fixed servers
        utilizations = loads / (num_servers * self.config.requests_per_server)
        cost_per_period = self.config.get_cost_per_period(num_servers)
        costs = np.full(n_periods, cost_per_period)

        # SLA violations
        sla_violations = np.sum(utilizations > 1.0)
        wasted_capacity = np.sum(utilizations < 0.3)

        total_hours = (n_periods * self.config.time_window_minutes) / 60

        return SimulationMetrics(
            total_cost=float(np.sum(costs)),
            avg_cost_per_hour=float(np.sum(costs) / total_hours) if total_hours > 0 else 0,
            avg_servers=float(num_servers),
            max_servers=num_servers,
            min_servers=num_servers,
            avg_utilization=float(np.mean(utilizations)),
            max_utilization=float(np.max(utilizations)),
            sla_violations=int(sla_violations),
            sla_violation_rate=float(sla_violations / n_periods) if n_periods > 0 else 0,
            overload_periods=int(sla_violations),
            wasted_capacity_periods=int(wasted_capacity),
            scaling_events=0,
            scale_out_events=0,
            scale_in_events=0,
            servers_over_time=[num_servers] * n_periods,
            utilization_over_time=utilizations.tolist(),
            cost_over_time=costs.tolist(),
        )

    def compare_strategies(
        self,
        loads: pd.Series | np.ndarray | list,
        strategies: dict[str, ScalingPolicy | int],
        timestamps: pd.DatetimeIndex | list | None = None,
    ) -> pd.DataFrame:
        """Compare multiple scaling strategies.

        Args:
            loads: Time series of request loads
            strategies: Dict of strategy name -> policy or fixed server count
            timestamps: Timestamps for each load

        Returns:
            DataFrame with comparison metrics
        """
        results = []

        for name, strategy in strategies.items():
            if isinstance(strategy, int):
                # Fixed server count
                metrics = self.simulate_fixed(loads, strategy)
            else:
                # Scaling policy
                metrics = self.simulate(loads, strategy, timestamps)

            results.append({
                "strategy": name,
                "total_cost": metrics.total_cost,
                "avg_cost_per_hour": metrics.avg_cost_per_hour,
                "avg_servers": metrics.avg_servers,
                "avg_utilization": metrics.avg_utilization,
                "sla_violations": metrics.sla_violations,
                "sla_violation_rate": metrics.sla_violation_rate,
                "scaling_events": metrics.scaling_events,
                "wasted_capacity_periods": metrics.wasted_capacity_periods,
            })

        return pd.DataFrame(results).set_index("strategy")

    def calculate_savings(
        self,
        autoscale_metrics: SimulationMetrics,
        fixed_max_metrics: SimulationMetrics,
    ) -> dict:
        """Calculate savings from autoscaling vs fixed max servers.

        Args:
            autoscale_metrics: Metrics from autoscaling simulation
            fixed_max_metrics: Metrics from fixed max servers simulation

        Returns:
            Dictionary with savings metrics
        """
        cost_savings = fixed_max_metrics.total_cost - autoscale_metrics.total_cost
        cost_savings_pct = (cost_savings / fixed_max_metrics.total_cost * 100
                           if fixed_max_metrics.total_cost > 0 else 0)

        return {
            "cost_savings": cost_savings,
            "cost_savings_pct": cost_savings_pct,
            "fixed_cost": fixed_max_metrics.total_cost,
            "autoscale_cost": autoscale_metrics.total_cost,
            "avg_servers_saved": fixed_max_metrics.avg_servers - autoscale_metrics.avg_servers,
            "sla_impact": autoscale_metrics.sla_violations - fixed_max_metrics.sla_violations,
        }

    def grid_search_thresholds(
        self,
        loads: pd.Series | np.ndarray | list,
        scale_out_range: list[float] = [0.70, 0.75, 0.80, 0.85],
        scale_in_range: list[float] = [0.20, 0.25, 0.30, 0.35],
        timestamps: pd.DatetimeIndex | list | None = None,
    ) -> pd.DataFrame:
        """Grid search for optimal threshold parameters.

        Args:
            loads: Time series of request loads
            scale_out_range: Scale out thresholds to test
            scale_in_range: Scale in thresholds to test
            timestamps: Timestamps for each load

        Returns:
            DataFrame with results for each parameter combination
        """
        results = []

        for scale_out in scale_out_range:
            for scale_in in scale_in_range:
                if scale_in >= scale_out:
                    continue  # Invalid combination

                # Create config with these thresholds
                config = ScalingConfig(
                    scale_out_threshold=scale_out,
                    scale_in_threshold=scale_in,
                    min_servers=self.config.min_servers,
                    max_servers=self.config.max_servers,
                    requests_per_server=self.config.requests_per_server,
                    cost_per_server_per_hour=self.config.cost_per_server_per_hour,
                )

                policy = ScalingPolicy(config)
                metrics = self.simulate(loads, policy, timestamps)

                results.append({
                    "scale_out_threshold": scale_out,
                    "scale_in_threshold": scale_in,
                    "total_cost": metrics.total_cost,
                    "avg_utilization": metrics.avg_utilization,
                    "sla_violations": metrics.sla_violations,
                    "sla_violation_rate": metrics.sla_violation_rate,
                    "scaling_events": metrics.scaling_events,
                    "avg_servers": metrics.avg_servers,
                })

        return pd.DataFrame(results)

    def optimize_for_cost(
        self,
        loads: pd.Series | np.ndarray | list,
        max_sla_violation_rate: float = 0.01,
        timestamps: pd.DatetimeIndex | list | None = None,
    ) -> tuple[ScalingConfig, SimulationMetrics]:
        """Find optimal config that minimizes cost while meeting SLA.

        Args:
            loads: Time series of request loads
            max_sla_violation_rate: Maximum acceptable SLA violation rate
            timestamps: Timestamps for each load

        Returns:
            Tuple of (optimal config, simulation metrics)
        """
        grid_results = self.grid_search_thresholds(loads, timestamps=timestamps)

        # Filter to configs that meet SLA
        valid = grid_results[grid_results["sla_violation_rate"] <= max_sla_violation_rate]

        if len(valid) == 0:
            # No config meets SLA, return most conservative
            best_row = grid_results.loc[grid_results["sla_violation_rate"].idxmin()]
        else:
            # Find lowest cost among valid configs
            best_row = valid.loc[valid["total_cost"].idxmin()]

        # Create optimal config
        optimal_config = ScalingConfig(
            scale_out_threshold=best_row["scale_out_threshold"],
            scale_in_threshold=best_row["scale_in_threshold"],
            min_servers=self.config.min_servers,
            max_servers=self.config.max_servers,
            requests_per_server=self.config.requests_per_server,
            cost_per_server_per_hour=self.config.cost_per_server_per_hour,
        )

        # Run final simulation with optimal config
        policy = ScalingPolicy(optimal_config)
        metrics = self.simulate(loads, policy, timestamps)

        return optimal_config, metrics
