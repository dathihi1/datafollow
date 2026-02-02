"""Simulation service for running scaling simulations."""

from dataclasses import dataclass, field
from typing import Literal
import numpy as np
import streamlit as st

from src.scaling.config import ScalingConfig, BALANCED_CONFIG, CONSERVATIVE_CONFIG, AGGRESSIVE_CONFIG
from src.scaling.policy import ScalingPolicy, PredictivePolicy, ReactivePolicy
from src.scaling.simulator import CostSimulator


@dataclass
class SimulationResult:
    """Container for simulation results."""
    
    config_name: str
    policy_name: str
    metrics: dict
    servers_over_time: list[int]
    utilization_over_time: list[float]
    cost_over_time: list[float]
    config: ScalingConfig = None
    
    @property
    def total_cost(self) -> float:
        return self.metrics.get('total_cost', 0.0)
    
    @property
    def avg_servers(self) -> float:
        return self.metrics.get('avg_servers', 0.0)
    
    @property
    def sla_violations(self) -> int:
        return self.metrics.get('sla_violations', 0)
    
    @property
    def sla_violation_rate(self) -> float:
        return self.metrics.get('sla_violation_rate', 0.0)


@dataclass
class ComparisonMatrix:
    """Matrix of simulation results for all config/policy combinations."""
    
    results: dict[tuple[str, str], SimulationResult] = field(default_factory=dict)
    
    def get(self, config_name: str, policy_name: str) -> SimulationResult | None:
        return self.results.get((config_name, policy_name))
    
    def add(self, result: SimulationResult):
        self.results[(result.config_name, result.policy_name)] = result
    
    @property
    def best_cost(self) -> SimulationResult | None:
        if not self.results:
            return None
        return min(self.results.values(), key=lambda r: r.total_cost)
    
    @property
    def best_sla(self) -> SimulationResult | None:
        if not self.results:
            return None
        return min(self.results.values(), key=lambda r: r.sla_violations)
    
    @property
    def best_balanced(self) -> SimulationResult | None:
        """Find result with best balance of cost and SLA."""
        if not self.results:
            return None
        
        # Normalize scores
        costs = [r.total_cost for r in self.results.values()]
        slas = [r.sla_violations for r in self.results.values()]
        
        min_cost, max_cost = min(costs), max(costs)
        min_sla, max_sla = min(slas), max(slas)
        
        def score(r: SimulationResult) -> float:
            cost_score = (r.total_cost - min_cost) / (max_cost - min_cost + 1e-10)
            sla_score = (r.sla_violations - min_sla) / (max_sla - min_sla + 1e-10)
            return cost_score + sla_score  # Lower is better
        
        return min(self.results.values(), key=score)
    
    def to_dataframe(self):
        """Convert to pandas DataFrame for display."""
        import pandas as pd
        
        data = []
        for (config_name, policy_name), result in self.results.items():
            data.append({
                "Config": config_name,
                "Policy": policy_name,
                "Total Cost": f"${result.total_cost:.2f}",
                "Avg Servers": f"{result.avg_servers:.1f}",
                "SLA Violations": result.sla_violations,
                "Violation Rate": f"{result.sla_violation_rate:.1%}",
            })
        
        return pd.DataFrame(data)


class SimulatorService:
    """Service for running scaling simulations."""
    
    CONFIGS = {
        "conservative": CONSERVATIVE_CONFIG,
        "balanced": BALANCED_CONFIG,
        "aggressive": AGGRESSIVE_CONFIG,
    }
    
    POLICIES = {
        "balanced": ScalingPolicy,
        "reactive": ReactivePolicy,
        "predictive": PredictivePolicy,
    }
    
    def __init__(self):
        self._cache: dict[str, SimulationResult] = {}
    
    def run_simulation(
        self,
        loads: np.ndarray,
        config: ScalingConfig,
        policy_type: str = "balanced",
        config_name: str = "custom",
    ) -> SimulationResult | None:
        """Run a single simulation."""
        cache_key = f"{hash(loads.tobytes())}_{config_name}_{policy_type}"
        
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        try:
            simulator = CostSimulator(config)
            
            if policy_type == "reactive":
                policy = ReactivePolicy(config)
            elif policy_type == "predictive":
                policy = PredictivePolicy(config)
            else:
                policy = ScalingPolicy(config)
            
            metrics = simulator.simulate(loads, policy)
            
            result = SimulationResult(
                config_name=config_name,
                policy_name=policy_type,
                metrics=metrics.__dict__,
                servers_over_time=metrics.servers_over_time,
                utilization_over_time=metrics.utilization_over_time,
                cost_over_time=metrics.cost_over_time,
                config=config,
            )
            
            self._cache[cache_key] = result
            return result
            
        except Exception as e:
            st.error(f"Simulation failed: {e}")
            return None
    
    def run_all_combinations(
        self,
        loads: np.ndarray,
        configs: list[str] | None = None,
        policies: list[str] | None = None,
    ) -> ComparisonMatrix:
        """Run simulations for all config/policy combinations."""
        configs = configs or list(self.CONFIGS.keys())
        policies = policies or list(self.POLICIES.keys())
        
        matrix = ComparisonMatrix()
        total = len(configs) * len(policies)
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        idx = 0
        for config_name in configs:
            config = self.CONFIGS.get(config_name, BALANCED_CONFIG)
            
            for policy_name in policies:
                idx += 1
                status_text.text(f"Running {config_name} + {policy_name}... ({idx}/{total})")
                progress_bar.progress(idx / total)
                
                result = self.run_simulation(
                    loads=loads,
                    config=config,
                    policy_type=policy_name,
                    config_name=config_name,
                )
                
                if result:
                    matrix.add(result)
        
        progress_bar.empty()
        status_text.empty()
        
        return matrix
    
    def compare_with_fixed(
        self,
        loads: np.ndarray,
        config: ScalingConfig,
    ) -> dict[str, SimulationResult]:
        """Compare autoscaling with fixed capacity strategies."""
        simulator = CostSimulator(config)
        results = {}
        
        # Fixed min
        fixed_min = simulator.simulate_fixed(loads, config.min_servers)
        results["fixed_min"] = SimulationResult(
            config_name=f"Fixed ({config.min_servers})",
            policy_name="fixed",
            metrics=fixed_min.__dict__,
            servers_over_time=[config.min_servers] * len(loads),
            utilization_over_time=fixed_min.utilization_over_time,
            cost_over_time=fixed_min.cost_over_time,
        )
        
        # Fixed max
        fixed_max = simulator.simulate_fixed(loads, config.max_servers)
        results["fixed_max"] = SimulationResult(
            config_name=f"Fixed ({config.max_servers})",
            policy_name="fixed",
            metrics=fixed_max.__dict__,
            servers_over_time=[config.max_servers] * len(loads),
            utilization_over_time=fixed_max.utilization_over_time,
            cost_over_time=fixed_max.cost_over_time,
        )
        
        # Fixed optimal (based on average load)
        avg_load = np.mean(loads)
        optimal_servers = config.get_required_servers(avg_load, target_utilization=0.7)
        fixed_optimal = simulator.simulate_fixed(loads, optimal_servers)
        results["fixed_optimal"] = SimulationResult(
            config_name=f"Fixed Optimal ({optimal_servers})",
            policy_name="fixed",
            metrics=fixed_optimal.__dict__,
            servers_over_time=[optimal_servers] * len(loads),
            utilization_over_time=fixed_optimal.utilization_over_time,
            cost_over_time=fixed_optimal.cost_over_time,
        )
        
        return results
    
    def clear_cache(self):
        """Clear simulation cache."""
        self._cache.clear()
