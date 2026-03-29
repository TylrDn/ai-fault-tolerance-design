"""Distributed training fault-tolerance simulator.

Simulates a cluster of training nodes where any node may fail with a
configurable probability at each training step.  When a failure is
detected the simulation checkpoints, optionally rescales, and resumes –
mirroring the recovery flow described in docs/design.md.

Usage::

    python src/simulator.py --nodes 8 --fail-rate 0.1 --steps 200

    # Or from Python:
    from src.simulator import Cluster
    cluster = Cluster(num_nodes=8, fail_rate=0.1)
    result = cluster.run(steps=100)
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import random
from typing import List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class NodeState:
    """Runtime state of a single training node."""

    node_id: int
    alive: bool = True
    local_step: int = 0


@dataclasses.dataclass
class CheckpointRecord:
    """Immutable record of a saved checkpoint."""

    global_step: int
    alive_nodes: List[int]


# ---------------------------------------------------------------------------
# Cluster simulator
# ---------------------------------------------------------------------------


class Cluster:
    """Simulates a distributed training cluster with fault injection.

    Parameters
    ----------
    num_nodes:
        Initial world size (number of training ranks).
    fail_rate:
        Per-node, per-step probability of a crash (0.0 – 1.0).
    checkpoint_interval:
        Save a full checkpoint every this many steps.
    allow_elastic:
        When *True* the cluster continues with a reduced world size after
        failures.  When *False* the simulation stops if any node fails.
    seed:
        Optional RNG seed for reproducibility.
    """

    def __init__(
        self,
        num_nodes: int = 4,
        fail_rate: float = 0.05,
        checkpoint_interval: int = 10,
        allow_elastic: bool = True,
        seed: Optional[int] = None,
    ) -> None:
        if num_nodes < 1:
            raise ValueError("num_nodes must be >= 1")
        if not 0.0 <= fail_rate <= 1.0:
            raise ValueError("fail_rate must be in [0, 1]")
        if checkpoint_interval < 1:
            raise ValueError("checkpoint_interval must be >= 1")

        self.num_nodes = num_nodes
        self.fail_rate = fail_rate
        self.checkpoint_interval = checkpoint_interval
        self.allow_elastic = allow_elastic
        self._rng = random.Random(seed)

        self.nodes: List[NodeState] = [NodeState(node_id=i) for i in range(num_nodes)]
        self.checkpoints: List[CheckpointRecord] = []
        self.global_step: int = 0
        self.total_failures: int = 0
        self.total_recoveries: int = 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @property
    def alive_nodes(self) -> List[NodeState]:
        return [n for n in self.nodes if n.alive]

    def _inject_failures(self) -> List[int]:
        """Randomly kill nodes according to *fail_rate*.

        Returns a list of node IDs that failed this step.
        """
        failed: List[int] = []
        for node in self.alive_nodes:
            if self._rng.random() < self.fail_rate:
                node.alive = False
                failed.append(node.node_id)
                logger.debug("  Node %d failed at step %d", node.node_id, self.global_step)
        return failed

    def _save_checkpoint(self) -> CheckpointRecord:
        """Persist a checkpoint of the current cluster state."""
        record = CheckpointRecord(
            global_step=self.global_step,
            alive_nodes=[n.node_id for n in self.alive_nodes],
        )
        self.checkpoints.append(record)
        logger.debug("  Checkpoint saved at step %d (%d nodes)", self.global_step, len(record.alive_nodes))
        return record

    def _recover(self, failed_node_ids: List[int]) -> None:
        """Handle recovery after one or more node failures.

        Strategy:
        1. Save a checkpoint.
        2. If elastic mode is enabled, mark failed nodes as permanently
           removed and continue with the reduced world size.
        3. Increment the recovery counter.
        """
        self._save_checkpoint()
        if self.allow_elastic:
            logger.info(
                "Step %d – elastic recovery: removing nodes %s, continuing with %d/%d nodes",
                self.global_step,
                failed_node_ids,
                len(self.alive_nodes),
                self.num_nodes,
            )
            self.total_recoveries += 1
        else:
            logger.info(
                "Step %d – non-elastic mode, stopping after node failures: %s",
                self.global_step,
                failed_node_ids,
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def step(self) -> bool:
        """Execute one training step.

        Returns
        -------
        bool
            *True* if the cluster is still running, *False* if it has stopped.
        """
        self.global_step += 1

        # Periodic checkpoint
        if self.global_step % self.checkpoint_interval == 0:
            self._save_checkpoint()

        # Fault injection
        failed = self._inject_failures()
        if failed:
            self.total_failures += len(failed)
            self._recover(failed)
            if not self.allow_elastic:
                return False

        # Advance local step on surviving nodes
        for node in self.alive_nodes:
            node.local_step = self.global_step

        return len(self.alive_nodes) > 0

    def run(self, steps: int = 100) -> "SimulationResult":
        """Run the simulation for *steps* training steps.

        Parameters
        ----------
        steps:
            Maximum number of training steps to simulate.

        Returns
        -------
        SimulationResult
            Summary of the run.
        """
        logger.info(
            "Starting simulation: nodes=%d, fail_rate=%.3f, steps=%d, elastic=%s",
            self.num_nodes,
            self.fail_rate,
            steps,
            self.allow_elastic,
        )

        completed_steps = 0
        for _ in range(steps):
            still_running = self.step()
            completed_steps += 1
            if not still_running:
                logger.info("Cluster stopped at step %d", self.global_step)
                break

        result = SimulationResult(
            requested_steps=steps,
            completed_steps=completed_steps,
            final_world_size=len(self.alive_nodes),
            total_failures=self.total_failures,
            total_recoveries=self.total_recoveries,
            num_checkpoints=len(self.checkpoints),
        )
        logger.info("Simulation finished: %s", result)
        return result

    def reset(self) -> None:
        """Reset the cluster to its initial state."""
        self.nodes = [NodeState(node_id=i) for i in range(self.num_nodes)]
        self.checkpoints = []
        self.global_step = 0
        self.total_failures = 0
        self.total_recoveries = 0


# ---------------------------------------------------------------------------
# Result summary
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class SimulationResult:
    """Summary statistics returned by :meth:`Cluster.run`."""

    requested_steps: int
    completed_steps: int
    final_world_size: int
    total_failures: int
    total_recoveries: int
    num_checkpoints: int

    def __str__(self) -> str:
        return (
            f"steps={self.completed_steps}/{self.requested_steps} "
            f"world_size={self.final_world_size} "
            f"failures={self.total_failures} "
            f"recoveries={self.total_recoveries} "
            f"checkpoints={self.num_checkpoints}"
        )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Distributed training fault-tolerance simulator")
    parser.add_argument("--nodes", type=int, default=8, help="Number of training nodes (default: 8)")
    parser.add_argument(
        "--fail-rate",
        type=float,
        default=0.05,
        help="Per-node per-step failure probability (default: 0.05)",
    )
    parser.add_argument("--steps", type=int, default=100, help="Number of training steps (default: 100)")
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=10,
        help="Steps between periodic checkpoints (default: 10)",
    )
    parser.add_argument("--no-elastic", action="store_true", help="Disable elastic training (stop on any failure)")
    parser.add_argument("--seed", type=int, default=None, help="RNG seed for reproducibility")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    return parser.parse_args(argv)


def main(argv=None) -> SimulationResult:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )
    cluster = Cluster(
        num_nodes=args.nodes,
        fail_rate=args.fail_rate,
        checkpoint_interval=args.checkpoint_interval,
        allow_elastic=not args.no_elastic,
        seed=args.seed,
    )
    result = cluster.run(steps=args.steps)
    print(result)
    return result


if __name__ == "__main__":
    main()
