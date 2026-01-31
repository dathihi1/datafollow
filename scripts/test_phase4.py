"""Test script for Phase 4 autoscaling modules."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd


def test_scaling_modules():
    """Test all Phase 4 scaling modules."""
    print("=" * 60)
    print("PHASE 4 AUTOSCALING MODULE TESTS")
    print("=" * 60)

    # Test 1: Imports
    print("\n1. Testing imports...")
    try:
        from src.scaling.config import ScalingConfig, BALANCED_CONFIG, CONSERVATIVE_CONFIG, AGGRESSIVE_CONFIG
        from src.scaling.policy import ScalingPolicy, ScalingAction, ReactivePolicy, PredictivePolicy
        from src.scaling.simulator import CostSimulator, SimulationMetrics
        print("   All imports successful!")
    except ImportError as e:
        print(f"   Import error: {e}")
        return False

    # Test 2: Configuration
    print("\n2. Testing ScalingConfig...")
    config = ScalingConfig()
    print(f"   Default config: min={config.min_servers}, max={config.max_servers}")
    print(f"   Scale thresholds: out={config.scale_out_threshold}, in={config.scale_in_threshold}")

    # Test validation
    try:
        bad_config = ScalingConfig(scale_in_threshold=0.9, scale_out_threshold=0.5)
        print("   ERROR: Should have raised validation error!")
        return False
    except ValueError:
        print("   Validation works correctly!")

    # Test 3: Policy
    print("\n3. Testing ScalingPolicy...")
    policy = ScalingPolicy(config)
    print(f"   Initial servers: {policy.get_current_servers()}")

    from datetime import datetime
    decision = policy.recommend(load=150, timestamp=datetime.now())
    print(f"   Decision for load=150: {decision.action.value}, util={decision.utilization:.1%}")

    # Test 4: Simulator
    print("\n4. Testing CostSimulator...")
    simulator = CostSimulator(config)

    sample_loads = np.array([100, 150, 200, 180, 120, 80, 50, 100, 200, 300])
    policy.reset()
    metrics = simulator.simulate(sample_loads, policy)

    print(f"   Simulation results:")
    print(f"     Total cost: ${metrics.total_cost:.2f}")
    print(f"     Avg utilization: {metrics.avg_utilization:.1%}")
    print(f"     Scaling events: {metrics.scaling_events}")
    print(f"     SLA violations: {metrics.sla_violations}")

    # Test 5: Fixed simulation
    print("\n5. Testing fixed capacity simulation...")
    fixed_metrics = simulator.simulate_fixed(sample_loads, num_servers=5)
    print(f"   Fixed (5 servers): cost=${fixed_metrics.total_cost:.2f}, util={fixed_metrics.avg_utilization:.1%}")

    # Test 6: Grid search
    print("\n6. Testing grid search...")
    grid_results = simulator.grid_search_thresholds(
        sample_loads,
        scale_out_range=[0.7, 0.8],
        scale_in_range=[0.2, 0.3]
    )
    print(f"   Grid search tested {len(grid_results)} combinations")

    # Test 7: Strategy comparison
    print("\n7. Testing strategy comparison...")
    strategies = {
        "Fixed (5)": 5,
        "Balanced": ScalingPolicy(BALANCED_CONFIG),
    }
    comparison = simulator.compare_strategies(sample_loads, strategies)
    print(f"   Comparison results:\n{comparison.to_string()}")

    # Test 8: Load real data if available
    print("\n8. Testing with real data...")
    data_path = project_root / "DATA" / "processed" / "test_features_5m.parquet"
    if data_path.exists():
        df = pd.read_parquet(data_path)
        real_loads = df['request_count'].values[:100]  # First 100 periods

        policy.reset()
        real_metrics = simulator.simulate(real_loads, policy)
        print(f"   Real data simulation (first 100 periods):")
        print(f"     Cost: ${real_metrics.total_cost:.2f}")
        print(f"     Avg servers: {real_metrics.avg_servers:.2f}")
        print(f"     SLA violations: {real_metrics.sla_violations}")
    else:
        print(f"   Skipped (data file not found at {data_path})")

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED!")
    print("=" * 60)
    return True


if __name__ == "__main__":
    success = test_scaling_modules()
    sys.exit(0 if success else 1)
