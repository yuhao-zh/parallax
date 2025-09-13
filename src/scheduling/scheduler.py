"""
Scheduler for Layer Allocation and Request Routing.
"""

from __future__ import annotations

import time
from collections import deque
from typing import Deque, Dict, List, Literal, Optional, Tuple

from parallax_utils.logging_config import get_logger
from scheduling.layer_allocation import (
    DynamicProgrammingLayerAllocator,
    GreedyLayerAllocator,
)
from scheduling.model_info import ModelInfo
from scheduling.node import Node, RequestSignal
from scheduling.request_routing import DynamicProgrammingRouting

logger = get_logger(__name__)


class Scheduler:
    """Coordinates allocation, node materialization, and request routing."""

    def __init__(
        self,
        model_info: ModelInfo,
        nodes: List[Node],
        min_nodes_before_allocation: int = 1,
        strategy: Literal["greedy", "dp"] = "dp",
        *,
        request_arrival_horizon_sec: float = 600.0,
        layer_hosting_power_memory_score: float = 0.5,
        rebalance_threshold: float = float("inf"),
        water_filling_max_iterations: int = 40,
        request_warm_up_for_reshard: int = 0,
        heartbeat_timeout: float = 60.0,
    ) -> None:
        self.model_info = model_info
        self.num_layers = model_info.num_layers
        self.nodes = nodes

        allocator_class = (
            GreedyLayerAllocator if strategy == "greedy" else DynamicProgrammingLayerAllocator
        )
        self.layer_allocator = allocator_class(
            model_info,
            nodes,
            min_nodes=min_nodes_before_allocation,
            layer_hosting_power_memory_score=layer_hosting_power_memory_score,
            rebalance_threshold=rebalance_threshold,
            water_filling_max_iterations=water_filling_max_iterations,
        )
        self.node_id_to_node: Dict[str, Node] = self.layer_allocator.node_id_to_node

        self.request_router = DynamicProgrammingRouting()
        self.request_warm_up_for_reshard = request_warm_up_for_reshard

        self.wait_pool: Deque[RequestSignal] = deque()
        self.request_arrival_horizon_sec = request_arrival_horizon_sec
        self.heartbeat_timeout = heartbeat_timeout
        self._arrival_ts: Deque[float] = deque()

        # Event queues for main loop orchestration
        self._pending_joins: Deque[Node] = deque()
        self._pending_leaves: Deque[str] = deque()
        self._pending_node_updates: Deque[
            Tuple[Node, Optional[int], Optional[float], Optional[Dict[str, float]]]
        ] = deque()
        self._running: bool = False

        self.initialize()

    # Orchestration helpers
    def initialize(self) -> None:
        """Run allocation, materialize nodes, attach RTT getters, and warm-up."""
        logger.info("Initializing layer allocator")
        if not self.layer_allocator.initialize():
            raise ValueError("Failed to initialize layer allocator")
        assignments = self.list_node_allocations()
        logger.info(f"Layer allocator assignments: {assignments}")
        # Optional warm-up to find turning points and truncate node ranges
        if self.request_warm_up_for_reshard > 0:
            self._run_warmup_and_truncate()
            assignments = self.list_node_allocations()
            logger.info(f"Layer allocator assignments after turn-point warm-up: {assignments}")
        # TODO: send results to the nodes
        # for node_id, start_layer, end_layer in assignments:

    def list_node_allocations(self) -> List[Tuple[str, int, int]]:
        """List the allocations of all nodes."""
        return self.layer_allocator.list_node_allocations()

    # Warm-up and re-shard
    def _run_warmup_and_truncate(self) -> None:
        """Run a brief warm-up to detect truncation points and shrink shards.

        Uses layer-level DP turning points (node_id, layer_idx, kind):
        - kind == "tail": drop [layer_idx, end) on that node
        - kind == "head": drop [start, layer_idx) on that node
        """
        nodes_list = list(self.nodes)
        if not nodes_list:
            return
        num_layers = self.model_info.num_layers
        # The number of warm-up requests can be used to repeat detection, but a
        # single pass is sufficient with our DP model; we repeat to smooth noise.
        agg_turns: Dict[Tuple[str, int, str], int] = {}
        for _ in range(self.request_warm_up_for_reshard):
            turns = self.request_router.find_turning_points(nodes_list, num_layers)
            for t in turns:
                agg_turns[t] = agg_turns.get(t, 0) + 1
        # Apply truncation for consistently observed turning points
        for node_id, layer_idx, kind in agg_turns:
            node = next((n for n in self.nodes if n.node_id == node_id), None)
            if node is None or node.start_layer is None or node.end_layer is None:
                continue
            start, end = node.start_layer, node.end_layer
            if kind == "tail":
                if layer_idx < end:
                    node.set_layer_allocation(start, layer_idx)
            elif kind == "head":
                if layer_idx > start:
                    node.set_layer_allocation(layer_idx, end)

    def update_node_info(
        self,
        node: Node,
        *,
        current_requests: Optional[int] = None,
        layer_latency_ms: Optional[float] = None,
        new_rtt_to_nodes: Optional[Dict[str, float]] = None,
    ) -> None:
        """Update the info of a node."""
        if current_requests is not None:
            node.current_requests = current_requests
        if layer_latency_ms is not None:
            node.set_layer_latency_ms(layer_latency_ms)
        if new_rtt_to_nodes is not None:
            node.rtt_to_nodes.update(new_rtt_to_nodes)
        node.last_heartbeat = time.time()

    # Async-style event enqueuers for main loop
    def enqueue_join(self, node: Node) -> None:
        """Enqueue a join event."""
        self._pending_joins.append(node)

    def enqueue_leave(self, node_id: str) -> None:
        """Enqueue a leave event."""
        self._pending_leaves.append(node_id)

    def enqueue_node_update(
        self,
        node: Node,
        *,
        current_requests: Optional[int] = None,
        layer_latency_ms: Optional[float] = None,
        new_rtt_to_nodes: Optional[Dict[str, float]] = None,
    ) -> None:
        """Enqueue a node update event."""
        self._pending_node_updates.append(
            (node, current_requests, layer_latency_ms, new_rtt_to_nodes)
        )

    def checking_node_heartbeat(self) -> None:
        """Check the heartbeat of all nodes."""
        for node in self.nodes:
            if time.time() - node.last_heartbeat > self.heartbeat_timeout:
                logger.info(f"Node {node.node_id} heartbeat timeout")
                self.leave(node.node_id)

    # Dynamic node management
    def join(self, node: Node) -> None:
        """Add a node to allocation and refresh plan and materialized nodes."""
        self.nodes.append(node)
        self.layer_allocator.join(node)

    def leave(self, node_id: str) -> None:
        """Remove a node from allocation and refresh plan and materialized nodes."""
        if node_id not in self.layer_allocator.node_id_to_node:
            raise ValueError(f"Node {node_id} not found in nodes")
        node = self.node_id_to_node[node_id]
        self.layer_allocator.leave(node_id)
        self.nodes.remove(node)
        if self.layer_allocator.should_global_rebalance():
            logger.info("Global rebalance triggered due to node leave")
            # TODO: send a signal to the nodes to stop running requests
            for n in self.nodes:
                if n.start_layer is not None and n.end_layer is not None:
                    self.layer_allocator.deallocate(n)
            self.layer_allocator.initialize()

    def receive_request(self, request: RequestSignal) -> None:
        """Add a request to the wait pool."""
        self.wait_pool.append(request)
        now = time.time()
        self._arrival_ts.append(now)
        # Trim old timestamps to keep arrival-rate window bounded
        horizon = self.request_arrival_horizon_sec
        while self._arrival_ts and now - self._arrival_ts[0] > horizon:
            self._arrival_ts.popleft()

    def dispatch_next_request(self) -> Optional[Tuple[str, List[str], float]]:
        """Route the next request in the wait pool; returns (request_id, path, latency)."""
        req = self.wait_pool.popleft() if self.wait_pool else None
        if req is None:
            return None
        path, latency = self.request_router.find_optimal_path(self.nodes, self.num_layers)
        req.routing_table = path
        # Update simple load counters
        for node_id in path:
            n = self.node_id_to_node[node_id]
            if n is not None:
                n.add_request()
        return req.request_id, path, latency

    def run(self, *, poll_interval: float = 0.05) -> None:
        """Main loop to process joins/leaves/updates, heartbeats, and dispatch requests.

        This runs forever until `stop()` is called. In a real server, external code
        should enqueue events via `enqueue_join`, `enqueue_leave`, `enqueue_node_update`,
        and `receive_request`.
        """
        self._running = True
        last_hb_check = 0.0
        try:
            while self._running:
                # Apply pending node updates
                while self._pending_node_updates:
                    node, cur, lat, rtts = self._pending_node_updates.popleft()
                    self.update_node_info(
                        node, current_requests=cur, layer_latency_ms=lat, new_rtt_to_nodes=rtts
                    )

                # Handle joins
                joins_happened = False
                while self._pending_joins:
                    # TODO: should be NodeHardwareInfo
                    node = self._pending_joins.popleft()
                    self.join(node)
                    joins_happened = True

                # Run initialize after enough nodes have joined and no full pipeline
                if joins_happened and not self.layer_allocator.has_full_pipeline():
                    if len(self.nodes) >= self.layer_allocator.min_nodes:
                        logger.info("Running initialization after joins")
                        self.layer_allocator.initialize()

                # Handle leaves
                while self._pending_leaves:
                    node_id = self._pending_leaves.popleft()
                    try:
                        self.leave(node_id)
                    except Exception as exc:  # pylint: disable=broad-except
                        logger.warning(f"Leave failed for {node_id}: {exc}")

                # Periodic heartbeat checks
                now = time.time()
                if now - last_hb_check >= max(0.5, poll_interval):
                    self.checking_node_heartbeat()
                    last_hb_check = now

                # Dispatch requests (one per tick to avoid starvation)
                _ = self.dispatch_next_request()

                # Sleep/poll interval
                time.sleep(poll_interval)
        finally:
            self._running = False

    def stop(self) -> None:
        """Stop the main loop if running."""
        self._running = False
