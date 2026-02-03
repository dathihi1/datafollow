"""Configuration for autoscaling policies."""

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class ScalingConfig:
    """Configuration for autoscaling policy.

    Attributes:
        min_servers: Minimum number of servers (cannot scale below)
        max_servers: Maximum number of servers (cannot scale above)
        requests_per_server: Requests each server can handle per time unit

        scale_out_threshold: Utilization threshold to trigger scale out (0-1)
        scale_in_threshold: Utilization threshold to trigger scale in (0-1)

        scale_out_consecutive: Consecutive periods above threshold before scaling out
        scale_in_consecutive: Consecutive periods below threshold before scaling in

        cooldown_minutes: Minutes to wait between scaling actions

        scale_out_increment: Number of servers to add when scaling out
        scale_in_decrement: Number of servers to remove when scaling in

        cost_per_server_per_hour: Cost of running one server per hour

        time_window_minutes: Time window for aggregation (5 for 5-minute data)
    """

    # Capacity limits
    min_servers: int = 1
    max_servers: int = 20
    requests_per_server: int = 100  # Requests per server per 5-min window

    # Thresholds (0-1 utilization)
    scale_out_threshold: float = 0.80  # 80% utilization triggers scale out
    scale_in_threshold: float = 0.30   # 30% utilization triggers scale in

    # Timing (consecutive periods before action)
    scale_out_consecutive: int = 3   # 3 periods (15 min for 5-min data)
    scale_in_consecutive: int = 6    # 6 periods (30 min for 5-min data)
    cooldown_minutes: int = 5        # Wait 5 min between scaling actions

    # Scaling increments
    scale_out_increment: int = 2     # Add 2 servers at once
    scale_in_decrement: int = 1      # Remove 1 server at a time

    # Cost (realistic AWS EC2 pricing)
    cost_per_server_per_hour: float = 0.85  # $0.85 per server per hour (t3.medium)

    # Time window
    time_window_minutes: int = 5  # 5-minute aggregation

    def __post_init__(self):
        """Validate configuration after initialization."""
        self._validate()

    def _validate(self):
        """Validate configuration parameters."""
        if self.min_servers < 1:
            raise ValueError("min_servers must be at least 1")
        if self.max_servers < self.min_servers:
            raise ValueError("max_servers must be >= min_servers")
        if not 0 < self.scale_out_threshold <= 1:
            raise ValueError("scale_out_threshold must be between 0 and 1")
        if not 0 <= self.scale_in_threshold < 1:
            raise ValueError("scale_in_threshold must be between 0 and 1")
        if self.scale_in_threshold >= self.scale_out_threshold:
            raise ValueError("scale_in_threshold must be < scale_out_threshold")
        if self.scale_out_increment < 1:
            raise ValueError("scale_out_increment must be at least 1")
        if self.scale_in_decrement < 1:
            raise ValueError("scale_in_decrement must be at least 1")

    @property
    def capacity_per_server(self) -> int:
        """Get capacity per server for the configured time window.

        Base capacity is requests_per_server (100) per 5 minutes.
        Scale proportionally for other time windows.
        Example: 1min = 20, 5min = 100, 15min = 300, 30min = 600
        """
        scale_factor = self.time_window_minutes / 5
        return int(self.requests_per_server * scale_factor)

    def get_total_capacity(self, num_servers: int) -> int:
        """Calculate total capacity for given number of servers.

        Args:
            num_servers: Number of active servers

        Returns:
            Total request capacity
        """
        return num_servers * self.capacity_per_server

    def get_utilization(self, load: float, num_servers: int) -> float:
        """Calculate utilization for given load and servers.

        Args:
            load: Current request load
            num_servers: Number of active servers

        Returns:
            Utilization as fraction (0-1+)
        """
        capacity = self.get_total_capacity(num_servers)
        if capacity == 0:
            return float("inf")
        return load / capacity

    def get_required_servers(self, load: float, target_utilization: float = 0.7) -> int:
        """Calculate required servers for given load.

        Args:
            load: Expected request load
            target_utilization: Target utilization level

        Returns:
            Number of servers needed
        """
        if load <= 0:
            return self.min_servers

        required = int(load / (self.capacity_per_server * target_utilization)) + 1
        return max(self.min_servers, min(required, self.max_servers))

    def get_cost_per_period(self, num_servers: int) -> float:
        """Calculate cost for one time period.

        Args:
            num_servers: Number of active servers

        Returns:
            Cost for one period
        """
        hours_per_period = self.time_window_minutes / 60
        return num_servers * self.cost_per_server_per_hour * hours_per_period

    def to_dict(self) -> dict:
        """Convert config to dictionary.

        Returns:
            Configuration as dictionary
        """
        return {
            "min_servers": self.min_servers,
            "max_servers": self.max_servers,
            "requests_per_server": self.requests_per_server,
            "scale_out_threshold": self.scale_out_threshold,
            "scale_in_threshold": self.scale_in_threshold,
            "scale_out_consecutive": self.scale_out_consecutive,
            "scale_in_consecutive": self.scale_in_consecutive,
            "cooldown_minutes": self.cooldown_minutes,
            "scale_out_increment": self.scale_out_increment,
            "scale_in_decrement": self.scale_in_decrement,
            "cost_per_server_per_hour": self.cost_per_server_per_hour,
            "time_window_minutes": self.time_window_minutes,
        }

    @classmethod
    def from_dict(cls, config_dict: dict) -> "ScalingConfig":
        """Create config from dictionary.

        Args:
            config_dict: Configuration dictionary

        Returns:
            ScalingConfig instance
        """
        return cls(**config_dict)


# Predefined configurations
CONSERVATIVE_CONFIG = ScalingConfig(
    scale_out_threshold=0.70,
    scale_in_threshold=0.20,
    scale_out_consecutive=5,
    scale_in_consecutive=10,
    cooldown_minutes=10,
    scale_out_increment=1,
    scale_in_decrement=1,
)

AGGRESSIVE_CONFIG = ScalingConfig(
    scale_out_threshold=0.85,
    scale_in_threshold=0.40,
    scale_out_consecutive=2,
    scale_in_consecutive=4,
    cooldown_minutes=3,
    scale_out_increment=3,
    scale_in_decrement=2,
)

BALANCED_CONFIG = ScalingConfig(
    scale_out_threshold=0.80,
    scale_in_threshold=0.30,
    scale_out_consecutive=3,
    scale_in_consecutive=6,
    cooldown_minutes=5,
    scale_out_increment=2,
    scale_in_decrement=1,
)
