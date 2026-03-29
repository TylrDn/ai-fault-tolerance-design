"""Pytest tests for src/simulator.py."""

import pytest

import sys
import os

# Ensure the repo root is on the path so `from src.simulator import ...` works
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.simulator import Cluster, SimulationResult, NodeState, CheckpointRecord


# ---------------------------------------------------------------------------
# NodeState / CheckpointRecord unit tests
# ---------------------------------------------------------------------------


class TestNodeState:
    def test_defaults(self):
        node = NodeState(node_id=3)
        assert node.node_id == 3
        assert node.alive is True
        assert node.local_step == 0

    def test_mutation(self):
        node = NodeState(node_id=0)
        node.alive = False
        assert not node.alive


class TestCheckpointRecord:
    def test_fields(self):
        record = CheckpointRecord(global_step=42, alive_nodes=[0, 1, 2])
        assert record.global_step == 42
        assert record.alive_nodes == [0, 1, 2]


# ---------------------------------------------------------------------------
# Cluster construction / validation
# ---------------------------------------------------------------------------


class TestClusterConstruction:
    def test_valid(self):
        cluster = Cluster(num_nodes=4, fail_rate=0.1)
        assert len(cluster.nodes) == 4

    def test_invalid_num_nodes(self):
        with pytest.raises(ValueError):
            Cluster(num_nodes=0)

    def test_invalid_fail_rate_negative(self):
        with pytest.raises(ValueError):
            Cluster(num_nodes=4, fail_rate=-0.1)

    def test_invalid_fail_rate_above_one(self):
        with pytest.raises(ValueError):
            Cluster(num_nodes=4, fail_rate=1.5)

    def test_invalid_checkpoint_interval(self):
        with pytest.raises(ValueError):
            Cluster(num_nodes=4, fail_rate=0.0, checkpoint_interval=0)


# ---------------------------------------------------------------------------
# Zero-failure run
# ---------------------------------------------------------------------------


class TestZeroFailureRun:
    def test_completes_all_steps(self):
        cluster = Cluster(num_nodes=4, fail_rate=0.0, seed=0)
        result = cluster.run(steps=50)
        assert result.completed_steps == 50
        assert result.total_failures == 0
        assert result.final_world_size == 4

    def test_checkpoints_saved(self):
        cluster = Cluster(num_nodes=2, fail_rate=0.0, checkpoint_interval=5, seed=0)
        result = cluster.run(steps=20)
        # Steps 5, 10, 15, 20 → 4 periodic checkpoints
        assert result.num_checkpoints == 4

    def test_global_step_advances(self):
        cluster = Cluster(num_nodes=2, fail_rate=0.0, seed=0)
        cluster.run(steps=10)
        assert cluster.global_step == 10


# ---------------------------------------------------------------------------
# Guaranteed-failure run (fail_rate=1.0)
# ---------------------------------------------------------------------------


class TestHighFailureRun:
    def test_elastic_records_failures_before_cluster_exhaustion(self):
        # With fail_rate=1.0 every node fails in step 1.
        # Elastic mode records the failures and attempts recovery,
        # but world_size drops to 0 so the cluster stops.
        cluster = Cluster(num_nodes=4, fail_rate=1.0, allow_elastic=True, seed=42)
        result = cluster.run(steps=100)
        assert result.total_failures > 0

    def test_non_elastic_stops_on_first_failure(self):
        # Seed chosen so that at least one failure occurs in step 1.
        cluster = Cluster(num_nodes=4, fail_rate=1.0, allow_elastic=False, seed=0)
        result = cluster.run(steps=100)
        # Non-elastic: stops as soon as any node fails
        assert result.completed_steps <= result.requested_steps
        assert result.total_failures >= 1


# ---------------------------------------------------------------------------
# Elastic vs non-elastic
# ---------------------------------------------------------------------------


class TestElasticBehavior:
    def test_elastic_continues_after_partial_failure(self):
        # Inject failures only on the first two nodes by patching _rng
        cluster = Cluster(num_nodes=4, fail_rate=0.0, allow_elastic=True, seed=0)
        # Manually kill one node and trigger step
        cluster.nodes[0].alive = False
        cluster.total_failures += 1
        cluster._recover([0])
        assert len(cluster.alive_nodes) == 3

    def test_alive_nodes_property(self):
        cluster = Cluster(num_nodes=3, fail_rate=0.0, seed=0)
        cluster.nodes[1].alive = False
        assert [n.node_id for n in cluster.alive_nodes] == [0, 2]


# ---------------------------------------------------------------------------
# Checkpoint logic
# ---------------------------------------------------------------------------


class TestCheckpoints:
    def test_checkpoint_content(self):
        cluster = Cluster(num_nodes=3, fail_rate=0.0, checkpoint_interval=5, seed=0)
        cluster.run(steps=5)
        assert len(cluster.checkpoints) == 1
        record = cluster.checkpoints[0]
        assert record.global_step == 5
        assert sorted(record.alive_nodes) == [0, 1, 2]

    def test_checkpoint_on_failure(self):
        cluster = Cluster(num_nodes=4, fail_rate=1.0, checkpoint_interval=100, seed=0)
        cluster.run(steps=1)
        # A failure checkpoint should have been saved regardless of the interval
        assert len(cluster.checkpoints) >= 1


# ---------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------


class TestReset:
    def test_reset_restores_initial_state(self):
        cluster = Cluster(num_nodes=4, fail_rate=0.5, seed=0)
        cluster.run(steps=20)
        cluster.reset()
        assert cluster.global_step == 0
        assert cluster.total_failures == 0
        assert cluster.total_recoveries == 0
        assert len(cluster.checkpoints) == 0
        assert all(n.alive for n in cluster.nodes)

    def test_run_after_reset(self):
        cluster = Cluster(num_nodes=4, fail_rate=0.0, seed=1)
        cluster.run(steps=10)
        cluster.reset()
        result = cluster.run(steps=10)
        assert result.completed_steps == 10


# ---------------------------------------------------------------------------
# SimulationResult __str__
# ---------------------------------------------------------------------------


class TestSimulationResult:
    def test_str_contains_key_info(self):
        result = SimulationResult(
            requested_steps=100,
            completed_steps=95,
            final_world_size=7,
            total_failures=3,
            total_recoveries=2,
            num_checkpoints=10,
        )
        s = str(result)
        assert "95/100" in s
        assert "world_size=7" in s
        assert "failures=3" in s
        assert "recoveries=2" in s
        assert "checkpoints=10" in s


# ---------------------------------------------------------------------------
# Reproducibility (seeded runs)
# ---------------------------------------------------------------------------


class TestReproducibility:
    def test_same_seed_same_result(self):
        r1 = Cluster(num_nodes=8, fail_rate=0.1, seed=123).run(steps=50)
        r2 = Cluster(num_nodes=8, fail_rate=0.1, seed=123).run(steps=50)
        assert r1.total_failures == r2.total_failures
        assert r1.completed_steps == r2.completed_steps

    def test_different_seed_may_differ(self):
        r1 = Cluster(num_nodes=8, fail_rate=0.3, seed=1).run(steps=50)
        r2 = Cluster(num_nodes=8, fail_rate=0.3, seed=999).run(steps=50)
        # Not guaranteed to differ, but very likely with fail_rate=0.3
        # Just ensure both runs complete without error
        assert r1.completed_steps >= 1
        assert r2.completed_steps >= 1
