"""
Scheduler for Phase 1 + Phase 2 orchestration.

- Runs layer allocation (greedy or DP) to produce a `LayerAllocationPlan`.
- Materializes runtime `Node` objects and maintains a request wait pool.
- Performs a prefix DP routing phase ("warm-up" of routing) and then finalizes
  shard-level routing using the DP path with a shard coalescing step.
- Handles node join/leave using the selected allocation strategy's dynamic
  handlers.
- Tracks request arrival rate and overall system load; supports dispatching
  requests using the request router and updating node loads accordingly.
"""

from __future__ import annotations

import time
from collections import deque
from typing import Callable, Deque, Dict, List, Literal, Optional, Tuple

from parallax.scheduling.layer_allocation import (
    DynamicProgrammingLayerAllocator,
    GreedyLayerAllocator,
    LayerAllocationPlan,
)
from parallax.scheduling.model_info import ModelInfo
from parallax.scheduling.node import Node, NodeInfo
from parallax.scheduling.request_routing import DynamicProgrammingRouting
from parallax.server.request import Request


class Scheduler:
    """Coordinates allocation, node materialization, and request routing."""

    def __init__(
        self,
        model_info: ModelInfo,
        nodes: Dict[str, NodeInfo],
        strategy: Literal["greedy", "dp"] = "dp",
        prefix_layers: int = 0,
        rtt_provider: Optional[Callable[[Node, Node], float]] = None,
        request_warm_up_for_reshard: int = 0,
        request_arrival_horizon_sec: float = 600.0,
    ) -> None:
        self.model_info = model_info
        self.node_infos = nodes
        self.strategy = strategy
        self.prefix_layers = max(0, int(prefix_layers))
        self.rtt_provider = rtt_provider
        self.request_warm_up_for_reshard = max(0, int(request_warm_up_for_reshard))
        # cache for measured RTTs
        self._rtts: Dict[Tuple[str, str], float] = {}
        self.allocator = (
            GreedyLayerAllocator(self.model_info, list(self.node_infos.values()))
            if strategy == "greedy"
            else DynamicProgrammingLayerAllocator(self.model_info, list(self.node_infos.values()))
        )

        self.plan: Optional[LayerAllocationPlan] = None
        self.nodes: Dict[str, Node] = {}
        self.wait_pool: Deque[Request] = deque()
        self.request_shards: Dict[str, List[Tuple[str, int, int]]] = {}
        # Arrival timestamps for rate calculation
        self._arrival_ts: Deque[float] = deque()
        self.request_arrival_horizon_sec = request_arrival_horizon_sec
        # DP request router
        self._router = DynamicProgrammingRouting()

    def run_allocation(self) -> LayerAllocationPlan:
        """Run the allocator and record the allocation plan."""
        return self.allocator.allocate()

    def materialize_node(self, node_id: str, start_layer: int, end_layer: int) -> Node:
        """Materialize a single node from an assignment."""
        info = self.plan.node_id_to_node_info[node_id]
        node = Node(node_id=node_id, node_info=info)
        # Node uses [start, end) range
        node.set_layer_allocation(start_layer, end_layer)
        # Attach RTT getter if provided
        node.rtt_getter = self._rtt_getter
        return node

    def materialize_nodes(self, plan: LayerAllocationPlan) -> Dict[str, Node]:
        """Create dynamic `Node` objects and set their current layer ranges.

        Converts [start, end) assignments to runtime `Node` objects.
        """
        if self.plan is None:
            raise ValueError("Allocation plan not available. Call run_allocation() first.")

        nodes = {}
        for node_id, assignment in plan.node_assignments.items():
            node = self.materialize_node(node_id, assignment.start_layer, assignment.end_layer)
            nodes[node_id] = node
        return nodes

    # Orchestration helpers
    def initialize(self) -> None:
        """Run allocation, materialize nodes, attach RTT getters, and warm-up."""
        self.plan = self.run_allocation()
        self.nodes = self.materialize_nodes(self.plan)
        # Optional warm-up to find turning points and truncate node ranges
        if self.request_warm_up_for_reshard > 0:
            self._run_warmup_and_truncate()

    def add_request(self, request: Request) -> None:
        """Add a request to the wait pool."""
        self.wait_pool.append(request)
        now = time.time()
        self._arrival_ts.append(now)
        # Trim old timestamps to keep arrival-rate window bounded
        horizon = self.request_arrival_horizon_sec
        while self._arrival_ts and now - self._arrival_ts[0] > horizon:
            self._arrival_ts.popleft()

    def pop_next_request(self) -> Optional[Request]:
        """Pop the next request from the wait pool."""
        return self.wait_pool.popleft() if self.wait_pool else None

    def _update_pair_rtt(self, src: Node, dst: Node) -> Optional[float]:
        """Measure or fetch RTT for a pair and update dynamic Nodes."""
        if self.rtt_provider is None:
            return None
        rtt = float(self.rtt_provider(src, dst))
        src.update_rtt(dst.node_id, rtt)
        return rtt

    def _rtt_getter(self, src: Node, dst: Node) -> float:
        """Getter passed into DP router; measures and updates as needed."""
        # Return cached value if exists
        if dst.node_id in src.rtt_to_nodes:
            return src.rtt_to_nodes[dst.node_id]
        measured = self._update_pair_rtt(src, dst)
        return float("inf") if measured is None else measured

    # Warm-up and re-shard
    def _run_warmup_and_truncate(self) -> None:
        """Run a brief warm-up to detect truncation points and shrink shards.

        Uses layer-level DP turning points (node_id, layer_idx, kind):
        - kind == "tail": drop [layer_idx, end) on that node
        - kind == "head": drop [start, layer_idx) on that node
        """
        nodes_list = list(self.nodes.values())
        if not nodes_list:
            return
        num_layers = self.model_info.num_layers
        # The number of warm-up requests can be used to repeat detection, but a
        # single pass is sufficient with our DP model; we repeat to smooth noise.
        agg_turns: Dict[Tuple[str, int, str], int] = {}
        for _ in range(self.request_warm_up_for_reshard):
            turns = self._router.find_turning_points(nodes_list, num_layers)
            for t in turns:
                agg_turns[t] = agg_turns.get(t, 0) + 1
        # Apply truncation for consistently observed turning points
        for node_id, layer_idx, kind in agg_turns:
            node = self.nodes.get(node_id)
            if node is None or node.current_layers is None:
                continue
            start, end = node.current_layers
            if kind == "tail":
                if layer_idx < end:
                    node.set_layer_allocation(start, layer_idx)
            elif kind == "head":
                if layer_idx > start:
                    node.set_layer_allocation(layer_idx, end)

    # Request queue management and routing
    def dispatch_next_request(self) -> Optional[Tuple[str, List[str], float]]:
        """Route the next request in the wait pool; returns (request_id, path, latency)."""
        req = self.pop_next_request()
        if req is None:
            return None
        nodes_list = list(self.nodes.values())
        path, latency = self._router.find_optimal_path(nodes_list, self.model_info.num_layers)
        req.routing_table = path
        # Update simple load counters
        for node_id in path:
            n = self.nodes.get(node_id)
            if n is not None:
                n.add_request()
        return req.request_id, path, latency

    def dispatch_all_pending(self) -> List[Tuple[str, List[str], float]]:
        """Route all pending requests and return their assignments."""
        assignments: List[Tuple[str, List[str], float]] = []
        while True:
            res = self.dispatch_next_request()
            if res is None:
                break
            assignments.append(res)
        return assignments

    # Dynamic node management
    def add_node(self, node_info: NodeInfo) -> None:
        """Add a node to allocation and refresh plan and materialized nodes."""
        self.node_infos[node_info.node_id] = node_info
        start_layer, end_layer = self.allocator.add_node(node_info)
        self.materialize_node(node_info.node_id, start_layer, end_layer)

    def remove_node(self, node_id: str) -> None:
        """Remove a node from allocation and refresh plan and materialized nodes."""
        self.allocator.remove_node(node_id)
        del self.nodes[node_id]

    def rebalance(self) -> None:
        """Re-run allocation and apply."""
        self.plan = self.allocator.rebalance()
        self.materialize_nodes(self.plan)
