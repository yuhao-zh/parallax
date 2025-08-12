"""
Phase 2 of scheduling: Request Routing

RequestRoutingStrategy: generic class;
DynamicProgramingRouting:
    - Setup:
        - Layer-index DAG
        - Node `(l, g)` denote replication of layer `l` on node `g`
            - Node weight: layer latency `tau_{gl}`
            - Note: we can relax this to simply `tau[g]` as each layer should be done with the same latency
        - Edge: `(l, g) -> (l + 1, g')`, i.e. built from consecutive layers
            - Edge weight: RTTs `pho_{gg'}`
            - Note: when building the edge, we may need to run RTTs measurement in parallel?
        - Dynamic Graph: nodes broadcast real time layer latency through DHT
            - If maximum number of requests reached, set the latency to infinity, essentially ‘closing’ the node
    - Dynamic Programming:
        - Initialization: maintain cost table stands for cumulative latency to reach layer `l` on node `g`
            - `dp[0][g] = tau_{g}`
            - others entries will be infinity
        - Recurrence:
            - `dp[l+1][g'] = min(dp[l+1][g'], dp[l][g] + pho[g][g'] + tau[g']`
        - Backtracking for path extraction.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple

from parallax.scheduling.layer_allocation import LayerAllocationPlan


class RequestRoutingStrategy(ABC):
    """Base abstract class for request routing strategies."""

    @abstractmethod
    def find_optimal_path(self) -> Tuple[List[str], float]:
        """Find the optimal path for a request, returning the path and total latency."""


class DynamicProgrammingRouting(RequestRoutingStrategy):
    """
    Finds the optimal request path using dynamic programming.

    This strategy models the layer-and-node combinations as a Directed Acyclic
    Graph (DAG) and uses DP to find the shortest path, where "shortest" means
    minimum cumulative latency.
    """

    def __init__(
        self,
        allocation_plan: LayerAllocationPlan,
        rtts: Dict[Tuple[str, str], float],
    ):
        self.plan = allocation_plan
        self.rtts = rtts
        self.num_layers = self.plan.num_total_layers
        self.nodes = self.plan.node_id_to_node_info

        # Pre-build a map of layer_id -> list of node_ids hosting it
        self.layer_to_nodes = [[] for _ in range(self.num_layers)]
        for node_id, assignment in self.plan.node_assignments.items():
            for i in range(assignment.start_layer, assignment.end_layer):
                self.layer_to_nodes[i].append(node_id)

    def _get_execution_latency(self, node_id: str) -> float:
        """
        Gets the execution latency for a single layer on a given node.
        For now, this assumes all layers on a node have the same latency.
        """
        node_info = self.nodes.get(node_id)
        if node_info is None or node_info.per_layer_latency_ms is None:
            return float("inf")
        return node_info.per_layer_latency_ms

    def find_optimal_path(self) -> Tuple[List[str], float]:
        """
        Calculates the minimum latency path through the layer-node DAG.

        Returns:
            A tuple containing:
            - A list of node IDs representing the optimal path.
            - The total minimum latency for that path in milliseconds.
        """
        # dp[l][g] = min latency to reach layer l on node g
        dp = [{node_id: float("inf") for node_id in self.nodes} for _ in range(self.num_layers)]
        # path[l][g] = predecessor node ID at layer l-1
        path = [{node_id: "" for node_id in self.nodes} for _ in range(self.num_layers)]

        # --- Initialization (Layer 0) ---
        for node_id in self.layer_to_nodes[0]:
            dp[0][node_id] = self._get_execution_latency(node_id)

        # --- Recurrence ---
        for l in range(self.num_layers - 1):
            for g_curr in self.layer_to_nodes[l]:
                if dp[l][g_curr] == float("inf"):
                    continue

                # Consider transitions to all nodes hosting the next layer
                for g_next in self.layer_to_nodes[l + 1]:
                    rtt = self.rtts.get((g_curr, g_next), float("inf"))
                    exec_latency = self._get_execution_latency(g_next)

                    new_cost = dp[l][g_curr] + rtt + exec_latency

                    if new_cost < dp[l + 1][g_next]:
                        dp[l + 1][g_next] = new_cost
                        path[l + 1][g_next] = g_curr

        # --- Backtracking ---
        # Find the best ending node at the last layer
        last_layer_idx = self.num_layers - 1
        min_latency = float("inf")
        end_node = ""

        for node_id in self.layer_to_nodes[last_layer_idx]:
            if dp[last_layer_idx][node_id] < min_latency:
                min_latency = dp[last_layer_idx][node_id]
                end_node = node_id

        if end_node == "":
            return [], float("inf")  # No valid path found

        # Reconstruct the path
        optimal_path = [""] * self.num_layers
        optimal_path[last_layer_idx] = end_node

        for l in range(last_layer_idx, 0, -1):
            pred_node = path[l][optimal_path[l]]
            optimal_path[l - 1] = pred_node

        return optimal_path, min_latency
