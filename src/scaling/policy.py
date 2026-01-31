"""Autoscaling policy implementation."""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Literal

import numpy as np
import pandas as pd

from src.scaling.config import ScalingConfig


class ScalingAction(Enum):
    """Possible scaling actions."""

    SCALE_OUT = "scale_out"
    SCALE_IN = "scale_in"
    HOLD = "hold"


@dataclass
class ScalingDecision:
    """Result of a scaling decision."""

    action: ScalingAction
    current_servers: int
    target_servers: int
    utilization: float
    load: float
    reason: str
    timestamp: datetime | None = None

    def __str__(self) -> str:
        return (
            f"{self.action.value.upper()}: {self.current_servers} -> {self.target_servers} servers "
            f"(util={self.utilization:.1%}, load={self.load:.0f}, reason={self.reason})"
        )


@dataclass
class ScalingState:
    """Internal state of the scaling policy."""

    current_servers: int
    last_action_time: datetime | None = None
    consecutive_high: int = 0
    consecutive_low: int = 0
    history: list = field(default_factory=list)

    def reset_consecutive(self):
        """Reset consecutive counters."""
        self.consecutive_high = 0
        self.consecutive_low = 0


class ScalingPolicy:
    """Autoscaling policy for server management.

    Implements scale out/in decisions based on utilization thresholds
    with cooldown periods and hysteresis to prevent flapping.
    """

    def __init__(self, config: ScalingConfig | None = None):
        """Initialize scaling policy.

        Args:
            config: Scaling configuration
        """
        self.config = config or ScalingConfig()
        self.state = ScalingState(current_servers=self.config.min_servers)

    def reset(self, initial_servers: int | None = None):
        """Reset policy state.

        Args:
            initial_servers: Initial number of servers
        """
        servers = initial_servers or self.config.min_servers
        self.state = ScalingState(current_servers=servers)

    def recommend(
        self,
        load: float,
        timestamp: datetime | None = None,
    ) -> ScalingDecision:
        """Get scaling recommendation for current load.

        Args:
            load: Current request load
            timestamp: Current timestamp (for cooldown tracking)

        Returns:
            ScalingDecision with action and details
        """
        timestamp = timestamp or datetime.now()
        current = self.state.current_servers
        utilization = self.config.get_utilization(load, current)

        # Check cooldown
        if not self._is_cooldown_expired(timestamp):
            return ScalingDecision(
                action=ScalingAction.HOLD,
                current_servers=current,
                target_servers=current,
                utilization=utilization,
                load=load,
                reason="cooldown_active",
                timestamp=timestamp,
            )

        # Check for scale out
        if utilization > self.config.scale_out_threshold:
            self.state.consecutive_high += 1
            self.state.consecutive_low = 0

            if self.state.consecutive_high >= self.config.scale_out_consecutive:
                target = min(
                    current + self.config.scale_out_increment,
                    self.config.max_servers,
                )
                if target > current:
                    decision = ScalingDecision(
                        action=ScalingAction.SCALE_OUT,
                        current_servers=current,
                        target_servers=target,
                        utilization=utilization,
                        load=load,
                        reason=f"high_utilization_{self.state.consecutive_high}_periods",
                        timestamp=timestamp,
                    )
                    self._apply_decision(decision, timestamp)
                    return decision

            return ScalingDecision(
                action=ScalingAction.HOLD,
                current_servers=current,
                target_servers=current,
                utilization=utilization,
                load=load,
                reason=f"waiting_scale_out_{self.state.consecutive_high}/{self.config.scale_out_consecutive}",
                timestamp=timestamp,
            )

        # Check for scale in
        elif utilization < self.config.scale_in_threshold:
            self.state.consecutive_low += 1
            self.state.consecutive_high = 0

            if self.state.consecutive_low >= self.config.scale_in_consecutive:
                target = max(
                    current - self.config.scale_in_decrement,
                    self.config.min_servers,
                )
                if target < current:
                    decision = ScalingDecision(
                        action=ScalingAction.SCALE_IN,
                        current_servers=current,
                        target_servers=target,
                        utilization=utilization,
                        load=load,
                        reason=f"low_utilization_{self.state.consecutive_low}_periods",
                        timestamp=timestamp,
                    )
                    self._apply_decision(decision, timestamp)
                    return decision

            return ScalingDecision(
                action=ScalingAction.HOLD,
                current_servers=current,
                target_servers=current,
                utilization=utilization,
                load=load,
                reason=f"waiting_scale_in_{self.state.consecutive_low}/{self.config.scale_in_consecutive}",
                timestamp=timestamp,
            )

        # Normal utilization - reset counters
        else:
            self.state.reset_consecutive()
            return ScalingDecision(
                action=ScalingAction.HOLD,
                current_servers=current,
                target_servers=current,
                utilization=utilization,
                load=load,
                reason="normal_utilization",
                timestamp=timestamp,
            )

    def recommend_proactive(
        self,
        predicted_loads: list[float],
        timestamp: datetime | None = None,
        lookahead_weight: float = 0.7,
    ) -> ScalingDecision:
        """Get proactive scaling recommendation using predicted future loads.

        Args:
            predicted_loads: List of predicted future loads
            timestamp: Current timestamp
            lookahead_weight: Weight for future predictions vs current

        Returns:
            ScalingDecision with action and details
        """
        if not predicted_loads:
            return self.recommend(0, timestamp)

        # Use weighted average of predictions (more weight on near future)
        weights = np.array([1.0 / (i + 1) for i in range(len(predicted_loads))])
        weights = weights / weights.sum()
        weighted_load = np.sum(np.array(predicted_loads) * weights)

        # Also consider peak load
        peak_load = max(predicted_loads)

        # Blend current-looking and peak-looking load
        effective_load = lookahead_weight * weighted_load + (1 - lookahead_weight) * peak_load

        decision = self.recommend(effective_load, timestamp)
        decision.reason = f"proactive_{decision.reason}"

        return decision

    def _is_cooldown_expired(self, timestamp: datetime) -> bool:
        """Check if cooldown period has expired.

        Args:
            timestamp: Current timestamp

        Returns:
            True if cooldown expired or no previous action
        """
        if self.state.last_action_time is None:
            return True

        cooldown = timedelta(minutes=self.config.cooldown_minutes)
        return timestamp >= self.state.last_action_time + cooldown

    def _apply_decision(self, decision: ScalingDecision, timestamp: datetime):
        """Apply a scaling decision to the state.

        Args:
            decision: Scaling decision to apply
            timestamp: Current timestamp
        """
        if decision.action != ScalingAction.HOLD:
            self.state.current_servers = decision.target_servers
            self.state.last_action_time = timestamp
            self.state.reset_consecutive()
            self.state.history.append(decision)

    def get_history(self) -> list[ScalingDecision]:
        """Get history of scaling decisions.

        Returns:
            List of past scaling decisions
        """
        return self.state.history.copy()

    def get_current_servers(self) -> int:
        """Get current number of servers.

        Returns:
            Current server count
        """
        return self.state.current_servers

    def set_servers(self, num_servers: int):
        """Manually set number of servers.

        Args:
            num_servers: New server count
        """
        self.state.current_servers = max(
            self.config.min_servers,
            min(num_servers, self.config.max_servers),
        )


class ReactivePolicy(ScalingPolicy):
    """Reactive scaling policy that responds immediately to load changes."""

    def __init__(self, config: ScalingConfig | None = None):
        super().__init__(config)
        # Override consecutive requirements for faster response
        self.config.scale_out_consecutive = 1
        self.config.scale_in_consecutive = 2


class PredictivePolicy(ScalingPolicy):
    """Predictive scaling policy that uses forecasts to scale proactively."""

    def __init__(
        self,
        config: ScalingConfig | None = None,
        forecast_horizon: int = 6,
        safety_margin: float = 0.1,
    ):
        """Initialize predictive policy.

        Args:
            config: Scaling configuration
            forecast_horizon: Number of periods to look ahead
            safety_margin: Additional capacity margin (0-1)
        """
        super().__init__(config)
        self.forecast_horizon = forecast_horizon
        self.safety_margin = safety_margin

    def recommend_with_forecast(
        self,
        current_load: float,
        forecasted_loads: list[float],
        timestamp: datetime | None = None,
    ) -> ScalingDecision:
        """Get recommendation considering forecasted loads.

        Args:
            current_load: Current request load
            forecasted_loads: Predicted future loads
            timestamp: Current timestamp

        Returns:
            ScalingDecision with action and details
        """
        # Consider both current and future loads
        all_loads = [current_load] + forecasted_loads[:self.forecast_horizon]

        # Use peak expected load with safety margin
        peak_load = max(all_loads) * (1 + self.safety_margin)

        # Calculate required servers for peak
        required = self.config.get_required_servers(peak_load, target_utilization=0.7)

        timestamp = timestamp or datetime.now()
        current = self.state.current_servers
        utilization = self.config.get_utilization(current_load, current)

        # If we need more servers than current, scale out
        if required > current and self._is_cooldown_expired(timestamp):
            target = min(required, self.config.max_servers)
            decision = ScalingDecision(
                action=ScalingAction.SCALE_OUT,
                current_servers=current,
                target_servers=target,
                utilization=utilization,
                load=current_load,
                reason=f"predictive_scale_out_for_peak_{peak_load:.0f}",
                timestamp=timestamp,
            )
            self._apply_decision(decision, timestamp)
            return decision

        # Otherwise use normal reactive logic for scale-in
        return self.recommend(current_load, timestamp)
