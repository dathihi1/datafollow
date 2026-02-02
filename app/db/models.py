"""Database models for simulation result persistence.

Supports SQLite (default) and PostgreSQL (production).
"""

from datetime import datetime
from pathlib import Path
from typing import Optional

from sqlalchemy import (
    JSON,
    Column,
    DateTime,
    Float,
    Integer,
    String,
    Text,
    create_engine,
)
from sqlalchemy.orm import DeclarativeBase, sessionmaker

# Use standard JSON type for SQLite/PostgreSQL compatibility
JSON_TYPE = JSON


# Ensure DATA directory exists
DATA_DIR = Path(__file__).parent.parent.parent / "DATA"
DATA_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_DB_URL = f"sqlite:///{DATA_DIR}/simulations.db"


class Base(DeclarativeBase):
    """Base class for all models."""
    pass


class SimulationResult(Base):
    """Persisted simulation result.

    Stores all metrics and time series data from a simulation run,
    allowing users to save, compare, and reload results.
    """

    __tablename__ = "simulation_results"

    # Primary key
    id = Column(Integer, primary_key=True, autoincrement=True)

    # Identification
    name = Column(String(255), nullable=True, index=True)
    description = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)

    # Data source info
    data_source_type = Column(String(50), nullable=False)  # sample, csv, txt, manual
    data_points_count = Column(Integer, nullable=False)
    data_hash = Column(String(64), nullable=True)  # SHA256 of input data

    # Configuration
    config_preset = Column(String(50), nullable=False, index=True)
    policy_type = Column(String(50), nullable=False, index=True)
    config_json = Column(JSON_TYPE, nullable=False)

    # Results - Summary metrics
    total_cost = Column(Float, nullable=False)
    avg_cost_per_hour = Column(Float, nullable=False)
    avg_servers = Column(Float, nullable=False)
    max_servers = Column(Integer, nullable=False)
    min_servers = Column(Integer, nullable=False)
    avg_utilization = Column(Float, nullable=False)
    max_utilization = Column(Float, nullable=False)
    sla_violations = Column(Integer, nullable=False)
    sla_violation_rate = Column(Float, nullable=False)
    scaling_events = Column(Integer, nullable=False)
    scale_out_events = Column(Integer, nullable=False)
    scale_in_events = Column(Integer, nullable=False)
    wasted_capacity_periods = Column(Integer, nullable=False)

    # Results - Time series (stored as JSON for flexibility)
    servers_over_time = Column(JSON_TYPE, nullable=True)
    utilization_over_time = Column(JSON_TYPE, nullable=True)
    cost_over_time = Column(JSON_TYPE, nullable=True)

    # Comparison data
    comparison_json = Column(JSON_TYPE, nullable=True)

    def to_dict(self) -> dict:
        """Convert to dictionary for API/display.

        Returns:
            Dictionary representation of the result
        """
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "data_source_type": self.data_source_type,
            "data_points_count": self.data_points_count,
            "config_preset": self.config_preset,
            "policy_type": self.policy_type,
            "config_json": self.config_json,
            "total_cost": self.total_cost,
            "avg_cost_per_hour": self.avg_cost_per_hour,
            "avg_servers": self.avg_servers,
            "max_servers": self.max_servers,
            "min_servers": self.min_servers,
            "avg_utilization": self.avg_utilization,
            "max_utilization": self.max_utilization,
            "sla_violations": self.sla_violations,
            "sla_violation_rate": self.sla_violation_rate,
            "scaling_events": self.scaling_events,
            "scale_out_events": self.scale_out_events,
            "scale_in_events": self.scale_in_events,
            "wasted_capacity_periods": self.wasted_capacity_periods,
        }

    def to_full_dict(self) -> dict:
        """Convert to dictionary including time series data.

        Returns:
            Full dictionary representation including time series
        """
        result = self.to_dict()
        result.update({
            "servers_over_time": self.servers_over_time,
            "utilization_over_time": self.utilization_over_time,
            "cost_over_time": self.cost_over_time,
            "comparison_json": self.comparison_json,
        })
        return result


def create_db_engine(db_url: str = DEFAULT_DB_URL):
    """Create database engine and tables.

    Args:
        db_url: Database connection URL (SQLite or PostgreSQL)

    Returns:
        SQLAlchemy engine
    """
    # Use check_same_thread=False for SQLite to allow multi-threading
    connect_args = {}
    if db_url.startswith("sqlite"):
        connect_args["check_same_thread"] = False

    engine = create_engine(db_url, echo=False, connect_args=connect_args)
    Base.metadata.create_all(engine)
    return engine


def get_session(engine):
    """Get database session.

    Args:
        engine: SQLAlchemy engine

    Returns:
        New session instance
    """
    Session = sessionmaker(bind=engine)
    return Session()
