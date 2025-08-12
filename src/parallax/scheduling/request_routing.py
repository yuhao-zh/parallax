# pylint: disable=too-many-locals, too-many-branches
"""
Phase 2: Request routing.

Provides:
- A base strategy interface.
- A dynamic-programming router that minimizes end-to-end latency across nodes.

Routing is at node granularity: once a request enters a node, it runs all layers
hosted by that node. We can optionally compute layer-level turning points for a
warm-up phase (to inform rebalancing), then perform shard-level DP to produce the
final node path and total latency.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

from parallax.scheduling.node import Node


class RequestRoutingStrategy(ABC):
    """Base abstract class for request routing strategies."""

    @abstractmethod
    def find_turning_points(self, nodes: List[Node], num_layers: int) -> List[Tuple[str, int]]:
        """Find turning points for nodes with layer-level DP path.

        Turning points are points where the request decides to switch to a different node.
        even if current node also hosts that next layer.

        Args:
            nodes: List of nodes.
            num_layers: Number of layers.

        Returns:
            List of tuples (node_id, layer_index) for each turning point.
        """

    @abstractmethod
    def find_optimal_path(self, nodes: List[Node], num_layers: int) -> Tuple[List[str], float]:
        """Shard-level DP path across nodes. Returns (node_ids, latency)."""


class DynamicProgrammingRouting(RequestRoutingStrategy):
    """
    Dynamic-programming router.

    - Warm-up: run a layer-level DP to identify turning points (where the optimal
      path switches nodes even if the current node still hosts the next layer).
    - Routing: run a shard-level DP over node assignments (contiguous layer ranges),
      using per-node execution latency and RTT via `Node.get_rtt_to`, to obtain a
      minimum-latency node sequence and total latency.
    """

    def find_turning_points(self, nodes: List[Node], num_layers: int) -> List[Tuple[str, int]]:
        """Find turning points for nodes with layer-level DP path.

        DP state is (layer l, node i that hosts l). Node cost: per-layer exec latency.
        Edge cost: RTT between nodes.
        """
        if num_layers <= 0 or not nodes:
            return []

        # Build host lists per layer
        layer_hosts: List[List[int]] = []
        for l in range(num_layers):
            hosts = [i for i, n in enumerate(nodes) if n.hosts_layer(l)]
            layer_hosts.append(hosts)

        # If any layer lacks a host, return empty
        if any(len(h) == 0 for h in layer_hosts):
            return []

        # layer_id: node_id -> cost
        dp: List[Dict[int, float]] = [{i: float("inf") for i in layer_hosts[0]}]
        back: List[Dict[int, Optional[int]]] = [{i: None for i in layer_hosts[0]}]

        # Init layer 0
        for i in layer_hosts[0]:
            dp[0][i] = nodes[i].layer_latency_ms

        # Recurrrence: dp[l+1][g] = min_g' (dp[l][g] + rtt(g,g') + latency(g'))
        for l in range(1, num_layers):
            curr: Dict[int, float] = {i: float("inf") for i in layer_hosts[l]}
            prev_back: Dict[int, Optional[int]] = {i: None for i in layer_hosts[l]}
            for i in layer_hosts[l]:
                node_i = nodes[i]
                best_cost = float("inf")
                best_j: Optional[int] = None
                for j, prev_cost in dp[l - 1].items():
                    if prev_cost == float("inf"):
                        continue
                    node_j = nodes[j]
                    trans = 0.0 if i == j else node_j.get_rtt_to(node_i)
                    total = prev_cost + trans + node_i.layer_latency_ms
                    if total < best_cost:
                        best_cost = total
                        best_j = j
                curr[i] = best_cost
                prev_back[i] = best_j
            dp.append(curr)
            back.append(prev_back)

        # Backtrack optimal node index per layer
        last = dp[-1]
        end_i = min(last, key=lambda k: last[k])
        path_idx: List[int] = [end_i]
        for l in range(num_layers - 1, 0, -1):
            prev_i = back[l][path_idx[-1]]
            if prev_i is None:
                break
            path_idx.append(prev_i)
        path_idx.reverse()

        # Identify turning points: when switching away from a node that still hosts layer l
        turning: List[Tuple[str, int]] = []
        for l in range(1, len(path_idx)):
            prev_i = path_idx[l - 1]
            cur_i = path_idx[l]
            if prev_i == cur_i:
                continue
            prev_node = nodes[prev_i]
            if prev_node.hosts_layer(l):
                turning.append((nodes[prev_i].node_id, l))
        return turning

    def find_optimal_path(self, nodes: List[Node], num_layers: int) -> Tuple[List[str], float]:
        """Shard-level DP path across node ranges using `Node` APIs."""
        if num_layers <= 0 or not nodes:
            return [], 0.0

        # Collect vertices from nodes with valid layer ranges
        starts: Dict[int, List[int]] = {}
        ends: Dict[int, List[int]] = {}
        for idx, n in enumerate(nodes):
            if n.current_layers is None:
                continue
            start, end = n.current_layers
            starts.setdefault(start, []).append(idx)
            ends.setdefault(end, []).append(idx)

        # DP over vertices sorted by (start, end)
        order = [
            i
            for i, n in sorted(
                [(i, n) for i, n in enumerate(nodes) if n.current_layers is not None],
                key=lambda p: (p[1].current_layers[0], p[1].current_layers[1]),
            )
        ]

        dp: Dict[int, float] = {i: float("inf") for i in order}
        parent: Dict[int, Optional[int]] = {i: None for i in order}

        # Initialize with nodes starting at layer 0
        for i in starts.get(0, []):
            dp[i] = float(nodes[i].layer_latency_ms)
            parent[i] = None

        # Transitions: j -> i if end(j) == start(i)
        for i in order:
            if dp[i] == float("inf"):
                # Not reachable yet; still try to relax successors using INF + ... won't help
                pass
            n_i = nodes[i]
            if n_i.current_layers is None:
                continue
            start_i, _ = n_i.current_layers
            for j in ends.get(start_i, []):
                if dp[j] == float("inf"):
                    continue
                n_j = nodes[j]
                trans = 0.0 if n_j.node_id == n_i.node_id else float(n_j.get_rtt_to(n_i))
                cand = dp[j] + trans + float(n_i.layer_latency_ms)
                if cand < dp[i]:
                    dp[i] = cand
                    parent[i] = j

        # Pick best terminal node that ends at num_layers
        terminals = ends.get(num_layers, [])
        if not terminals:
            return [], float("inf")
        end_idx = min(terminals, key=lambda k: dp.get(k, float("inf")))
        if dp.get(end_idx, float("inf")) == float("inf"):
            return [], float("inf")

        # Reconstruct path
        path_indices: List[int] = []
        cur: Optional[int] = end_idx
        while cur is not None:
            path_indices.append(cur)
            cur = parent[cur]
        path_indices.reverse()
        return [nodes[i].node_id for i in path_indices], dp[end_idx]
