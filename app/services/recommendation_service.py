"""Recommendation service for optimal configuration suggestions."""

from dataclasses import dataclass, field
from typing import Literal
import numpy as np

from src.scaling.config import ScalingConfig, BALANCED_CONFIG, CONSERVATIVE_CONFIG, AGGRESSIVE_CONFIG
from app.services.simulator_service import SimulatorService, SimulationResult, ComparisonMatrix


@dataclass
class Recommendation:
    """Container for configuration recommendation."""
    
    config_name: str
    policy_name: str
    config: ScalingConfig
    expected_cost: float
    expected_sla_violations: int
    sla_violation_rate: float
    avg_servers: float
    
    # Comparisons
    savings_vs_fixed_max: float = 0.0
    savings_vs_fixed_max_pct: float = 0.0
    
    # Risk analysis
    cost_variance: float = 0.0
    sla_risk_level: Literal["low", "medium", "high"] = "medium"
    peak_hour_risk: Literal["low", "medium", "high"] = "medium"
    
    # Reasoning
    reasoning: str = ""
    
    @property
    def min_servers(self) -> int:
        return self.config.min_servers
    
    @property
    def max_servers(self) -> int:
        return self.config.max_servers
    
    @property
    def scale_out_threshold(self) -> float:
        return self.config.scale_out_threshold
    
    @property
    def scale_in_threshold(self) -> float:
        return self.config.scale_in_threshold


@dataclass
class RiskAnalysis:
    """Risk analysis for a configuration."""
    
    sla_violation_probability: float
    peak_hour_risk: Literal["low", "medium", "high"]
    cost_variance: float
    worst_case_cost: float
    best_case_cost: float
    risk_factors: list[str] = field(default_factory=list)


class RecommendationService:
    """Service for generating optimal configuration recommendations."""
    
    CONFIGS = {
        "conservative": CONSERVATIVE_CONFIG,
        "balanced": BALANCED_CONFIG,
        "aggressive": AGGRESSIVE_CONFIG,
    }
    
    def __init__(self, simulator_service: SimulatorService):
        self.simulator = simulator_service
    
    def find_optimal(
        self,
        loads: np.ndarray,
        priority: Literal["cost", "sla", "balanced"] = "balanced",
        max_sla_violation_rate: float = 0.05,
        max_cost: float | None = None,
    ) -> Recommendation | None:
        """Find optimal configuration based on priorities."""
        
        # Run all combinations
        matrix = self.simulator.run_all_combinations(loads)
        
        if not matrix.results:
            return None
        
        # Filter by constraints
        valid_results = []
        for result in matrix.results.values():
            if result.sla_violation_rate > max_sla_violation_rate:
                continue
            if max_cost and result.total_cost > max_cost:
                continue
            valid_results.append(result)
        
        # If no valid results, relax constraints
        if not valid_results:
            valid_results = list(matrix.results.values())
        
        # Select best based on priority
        if priority == "cost":
            best = min(valid_results, key=lambda r: r.total_cost)
            reasoning = "Selected for lowest cost while meeting SLA constraints."
        elif priority == "sla":
            best = min(valid_results, key=lambda r: r.sla_violations)
            reasoning = "Selected for best SLA performance with minimal violations."
        else:  # balanced
            best = self._find_balanced(valid_results)
            reasoning = "Selected for optimal balance between cost and SLA performance."
        
        # Calculate savings vs fixed max
        fixed_results = self.simulator.compare_with_fixed(loads, best.config)
        fixed_max_cost = fixed_results["fixed_max"].total_cost
        savings = fixed_max_cost - best.total_cost
        savings_pct = (savings / fixed_max_cost * 100) if fixed_max_cost > 0 else 0
        
        # Risk analysis
        risk = self._analyze_risk(loads, best)
        
        return Recommendation(
            config_name=best.config_name,
            policy_name=best.policy_name,
            config=best.config,
            expected_cost=best.total_cost,
            expected_sla_violations=best.sla_violations,
            sla_violation_rate=best.sla_violation_rate,
            avg_servers=best.avg_servers,
            savings_vs_fixed_max=savings,
            savings_vs_fixed_max_pct=savings_pct,
            cost_variance=risk.cost_variance,
            sla_risk_level=self._sla_risk_level(best.sla_violation_rate),
            peak_hour_risk=risk.peak_hour_risk,
            reasoning=reasoning,
        )
    
    def _find_balanced(self, results: list[SimulationResult]) -> SimulationResult:
        """Find result with best cost/SLA balance."""
        if not results:
            raise ValueError("No results to evaluate")
        
        costs = [r.total_cost for r in results]
        slas = [r.sla_violations for r in results]
        
        min_cost, max_cost = min(costs), max(costs)
        min_sla, max_sla = min(slas), max(slas)
        
        def score(r: SimulationResult) -> float:
            # Normalize both metrics to 0-1 scale
            cost_score = (r.total_cost - min_cost) / (max_cost - min_cost + 1e-10)
            sla_score = (r.sla_violations - min_sla) / (max_sla - min_sla + 1e-10)
            # Equal weight
            return cost_score + sla_score
        
        return min(results, key=score)
    
    def _analyze_risk(
        self,
        loads: np.ndarray,
        result: SimulationResult,
    ) -> RiskAnalysis:
        """Analyze risks for a configuration."""
        
        # Cost variance from simulation
        costs = result.cost_over_time
        cost_variance = np.std(costs) * len(costs)  # Total variance
        
        # Peak hour risk
        peak_load = np.max(loads)
        avg_load = np.mean(loads)
        peak_ratio = peak_load / avg_load if avg_load > 0 else 1
        
        if peak_ratio > 3:
            peak_risk = "high"
        elif peak_ratio > 2:
            peak_risk = "medium"
        else:
            peak_risk = "low"
        
        # Estimate worst/best case costs
        worst_case = result.total_cost * 1.2  # 20% buffer
        best_case = result.total_cost * 0.9
        
        risk_factors = []
        if peak_ratio > 2:
            risk_factors.append(f"High peak load ratio ({peak_ratio:.1f}x)")
        if result.sla_violation_rate > 0.02:
            risk_factors.append(f"Elevated SLA violation rate ({result.sla_violation_rate:.1%})")
        
        return RiskAnalysis(
            sla_violation_probability=result.sla_violation_rate,
            peak_hour_risk=peak_risk,
            cost_variance=cost_variance,
            worst_case_cost=worst_case,
            best_case_cost=best_case,
            risk_factors=risk_factors,
        )
    
    def _sla_risk_level(self, violation_rate: float) -> Literal["low", "medium", "high"]:
        """Determine SLA risk level."""
        if violation_rate < 0.01:
            return "low"
        elif violation_rate < 0.05:
            return "medium"
        else:
            return "high"
    
    def generate_report(
        self,
        recommendation: Recommendation,
        comparison_matrix: ComparisonMatrix,
    ) -> str:
        """Generate a text report of the recommendation."""
        lines = [
            "=" * 60,
            "AUTOSCALING CONFIGURATION RECOMMENDATION REPORT",
            "=" * 60,
            "",
            "RECOMMENDED CONFIGURATION",
            "-" * 30,
            f"  Configuration: {recommendation.config_name.upper()}",
            f"  Policy: {recommendation.policy_name.upper()}",
            "",
            "EXPECTED PERFORMANCE",
            "-" * 30,
            f"  Total Cost: ${recommendation.expected_cost:.2f}",
            f"  Average Servers: {recommendation.avg_servers:.1f}",
            f"  SLA Violations: {recommendation.expected_sla_violations}",
            f"  Violation Rate: {recommendation.sla_violation_rate:.2%}",
            "",
            "COST SAVINGS",
            "-" * 30,
            f"  vs Fixed Max: ${recommendation.savings_vs_fixed_max:.2f} ({recommendation.savings_vs_fixed_max_pct:.1f}%)",
            "",
            "RECOMMENDED SETTINGS",
            "-" * 30,
            f"  Min Servers: {recommendation.min_servers}",
            f"  Max Servers: {recommendation.max_servers}",
            f"  Scale Out Threshold: {recommendation.scale_out_threshold:.0%}",
            f"  Scale In Threshold: {recommendation.scale_in_threshold:.0%}",
            "",
            "RISK ANALYSIS",
            "-" * 30,
            f"  SLA Risk Level: {recommendation.sla_risk_level.upper()}",
            f"  Peak Hour Risk: {recommendation.peak_hour_risk.upper()}",
            f"  Cost Variance: ±${recommendation.cost_variance:.2f}",
            "",
            "REASONING",
            "-" * 30,
            f"  {recommendation.reasoning}",
            "",
            "=" * 60,
        ]
        
        return "\n".join(lines)
