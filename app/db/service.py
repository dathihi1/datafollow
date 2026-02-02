"""Database service for simulation persistence.

Provides high-level operations for saving, loading, and comparing
simulation results.
"""

import hashlib
from datetime import datetime
from typing import Optional

import numpy as np
from sqlalchemy import desc

from app.db.models import (
    DEFAULT_DB_URL,
    SimulationResult,
    create_db_engine,
    get_session,
)


class SimulationService:
    """Service for managing simulation results.

    Provides CRUD operations and comparison functionality for
    persisted simulation results.

    Example:
        service = SimulationService()
        result_id = service.save_result(
            metrics=metrics,
            servers=[1, 2, 3],
            ...
        )
        result = service.get_result(result_id)
    """

    def __init__(self, db_url: str = DEFAULT_DB_URL):
        """Initialize service with database connection.

        Args:
            db_url: Database URL (defaults to SQLite in DATA folder)
        """
        self.engine = create_db_engine(db_url)

    def save_result(
        self,
        metrics: dict,
        servers: list,
        utilizations: list,
        costs: list,
        data_source_type: str,
        data_points_count: int,
        config_preset: str,
        policy_type: str,
        config_dict: dict,
        name: Optional[str] = None,
        description: Optional[str] = None,
        loads: Optional[np.ndarray] = None,
        comparison: Optional[dict] = None,
    ) -> int:
        """Save simulation result to database.

        Args:
            metrics: Dictionary of simulation metrics
            servers: List of server counts over time
            utilizations: List of utilization values over time
            costs: List of costs over time
            data_source_type: Type of data source (sample, csv, txt, manual)
            data_points_count: Number of data points in input
            config_preset: Configuration preset name
            policy_type: Scaling policy type
            config_dict: Configuration as dictionary
            name: Optional name for the result
            description: Optional description
            loads: Optional input data for hashing
            comparison: Optional comparison data

        Returns:
            ID of saved result
        """
        session = get_session(self.engine)

        try:
            # Calculate data hash if loads provided
            data_hash = None
            if loads is not None:
                data_hash = hashlib.sha256(loads.tobytes()).hexdigest()

            # Generate default name if not provided
            if not name:
                name = f"Simulation {datetime.now().strftime('%Y-%m-%d %H:%M')}"

            result = SimulationResult(
                name=name,
                description=description,
                data_source_type=data_source_type,
                data_points_count=data_points_count,
                data_hash=data_hash,
                config_preset=config_preset,
                policy_type=policy_type,
                config_json=config_dict,
                total_cost=metrics["total_cost"],
                avg_cost_per_hour=metrics["avg_cost_per_hour"],
                avg_servers=metrics["avg_servers"],
                max_servers=metrics["max_servers"],
                min_servers=metrics["min_servers"],
                avg_utilization=metrics["avg_utilization"],
                max_utilization=metrics["max_utilization"],
                sla_violations=metrics["sla_violations"],
                sla_violation_rate=metrics["sla_violation_rate"],
                scaling_events=metrics["scaling_events"],
                scale_out_events=metrics["scale_out_events"],
                scale_in_events=metrics["scale_in_events"],
                wasted_capacity_periods=metrics["wasted_capacity_periods"],
                servers_over_time=servers,
                utilization_over_time=utilizations,
                cost_over_time=costs,
                comparison_json=comparison,
            )

            session.add(result)
            session.commit()

            return result.id
        finally:
            session.close()

    def get_result(self, result_id: int) -> Optional[dict]:
        """Get simulation result by ID.

        Args:
            result_id: Result ID

        Returns:
            Full result dictionary or None if not found
        """
        session = get_session(self.engine)
        try:
            result = session.query(SimulationResult).filter_by(id=result_id).first()
            return result.to_full_dict() if result else None
        finally:
            session.close()

    def list_results(
        self,
        limit: int = 50,
        offset: int = 0,
        config_preset: Optional[str] = None,
        policy_type: Optional[str] = None,
        search: Optional[str] = None,
    ) -> list[dict]:
        """List simulation results with optional filtering.

        Args:
            limit: Maximum results to return
            offset: Offset for pagination
            config_preset: Filter by config preset
            policy_type: Filter by policy type
            search: Search in name/description

        Returns:
            List of result dictionaries (summary only)
        """
        session = get_session(self.engine)
        try:
            query = session.query(SimulationResult)

            if config_preset:
                query = query.filter_by(config_preset=config_preset)
            if policy_type:
                query = query.filter_by(policy_type=policy_type)
            if search:
                search_pattern = f"%{search}%"
                query = query.filter(
                    SimulationResult.name.ilike(search_pattern) |
                    SimulationResult.description.ilike(search_pattern)
                )

            results = (
                query
                .order_by(desc(SimulationResult.created_at))
                .offset(offset)
                .limit(limit)
                .all()
            )

            return [r.to_dict() for r in results]
        finally:
            session.close()

    def delete_result(self, result_id: int) -> bool:
        """Delete simulation result.

        Args:
            result_id: Result ID to delete

        Returns:
            True if deleted, False if not found
        """
        session = get_session(self.engine)
        try:
            result = session.query(SimulationResult).filter_by(id=result_id).first()
            if result:
                session.delete(result)
                session.commit()
                return True
            return False
        finally:
            session.close()

    def compare_results(self, result_ids: list[int]) -> list[dict]:
        """Get multiple results for comparison.

        Args:
            result_ids: List of result IDs to compare

        Returns:
            List of result dictionaries
        """
        session = get_session(self.engine)
        try:
            results = (
                session.query(SimulationResult)
                .filter(SimulationResult.id.in_(result_ids))
                .all()
            )
            return [r.to_dict() for r in results]
        finally:
            session.close()

    def get_statistics(self) -> dict:
        """Get database statistics.

        Returns:
            Dictionary with statistics
        """
        session = get_session(self.engine)
        try:
            from sqlalchemy import func

            total = session.query(func.count(SimulationResult.id)).scalar()

            avg_cost = session.query(func.avg(SimulationResult.total_cost)).scalar()
            avg_sla = session.query(func.avg(SimulationResult.sla_violation_rate)).scalar()

            by_preset = dict(
                session.query(
                    SimulationResult.config_preset,
                    func.count(SimulationResult.id)
                )
                .group_by(SimulationResult.config_preset)
                .all()
            )

            by_policy = dict(
                session.query(
                    SimulationResult.policy_type,
                    func.count(SimulationResult.id)
                )
                .group_by(SimulationResult.policy_type)
                .all()
            )

            return {
                "total_results": total or 0,
                "avg_cost": avg_cost or 0,
                "avg_sla_violation_rate": avg_sla or 0,
                "by_preset": by_preset,
                "by_policy": by_policy,
            }
        finally:
            session.close()
