"""
Layer allocation and dynamic rebalancing primitives (Phase 1 of scheduling).

Key components:
- LayerLoad: hosting power per layer (memory + FLOPs), kept in a min-heap
- NodeAssignment: contiguous [start, end) layer range hosted by a node
- LayerAllocationPlan: mutable plan with node id to assignment mapping and heap-backed layer loads
- PipelineRebalancer: rebalances layers within a pipeline to be proportional to TFLOPS of the nodes.
    - Uses water-filling algorithm to find the optimal lambda.
    - Uses minimal-movement integerization to round the solution.
    - Applied in all LayerAllocator strategies.
- Generic Abstract classes:
    - DynamicNodeHandler: Abstract strategy interface for node join/leave events, takes in LayerAllocationPlan
    - LayerAllocator: Abstract strategy interface for layer allocation, takes in ModelInfo and List[NodeInfo]
- Concrete classes:
    - GapPatchDynamicNodeHandler: a DynamicNodeHandler that
        - fills gaps by assigning lightest layers first,
        - rebalances if load imbalance is too high
    - GreedyLayerAllocator: a LayerAllocator that
        - assigns layers to nodes in a greedy manner.
    - DPAllocator: a LayerAllocator that
        - uses dynamic programming to find an optimal balance between
          maximizing pipelines and minimizing stages.
        - uses a simple scalar taking number of pipelines and average stages as objective
"""

import heapq
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from functools import lru_cache
from math import floor
from typing import Dict, List, Optional, Set, Tuple

from parallax.scheduling.model_info import ModelInfo
from parallax.scheduling.node import NodeInfo


@dataclass
class LayerLoad:
    """Tracks the load and hosting power for a specific layer."""

    layer_id: int
    current_memory_size: int
    current_flops: float
    hosting_nodes: Set[str] = field(default_factory=set)

    def add_node(self, node: NodeInfo):
        """Add a node to the layer load."""
        self.hosting_nodes.add(node.node_id)
        assert (
            node.per_layer_flops is not None and node.per_layer_memory is not None
        ), "Node must have per_layer_flops and per_layer_memory"
        self.current_memory_size += node.per_layer_memory
        self.current_flops += node.per_layer_flops

    def remove_node(self, node: NodeInfo):
        """Remove a node from the layer load."""
        if node.node_id in self.hosting_nodes:
            self.hosting_nodes.remove(node.node_id)
            assert (
                node.per_layer_flops is not None and node.per_layer_memory is not None
            ), "Node must have per_layer_flops and per_layer_memory"
            self.current_memory_size -= node.per_layer_memory
            self.current_flops -= node.per_layer_flops

    def __lt__(self, other):
        """For heap ordering: prioritize layers with lower hosting power.

        Primary: less memory (less hosting power)
        Secondary: lower FLOPs (less compute allocated)
        Tertiary: lower layer ID (less layers hosted)
        """
        return (self.current_memory_size, self.current_flops, self.layer_id) < (
            other.current_memory_size,
            other.current_flops,
            other.layer_id,
        )


@dataclass
class NodeAssignment:
    """Tracks the hosting range for a node."""

    node_id: str
    start_layer: int  # inclusive
    end_layer: int  # exclusive

    @property
    def num_layers(self) -> int:
        """Number of layers hosted by this node."""
        return self.end_layer - self.start_layer

    @property
    def layer_range(self) -> Tuple[int, int]:
        """Get the layer range as a tuple."""
        return (self.start_layer, self.end_layer)

    def contains_layer(self, layer_id: int) -> bool:
        """Check if this node hosts the given layer."""
        return self.start_layer <= layer_id < self.end_layer


class LayerAllocationPlan:
    """
    Manages the allocation plan for layers across nodes.
    """

    def __init__(self, num_total_layers: int):
        self.num_total_layers = num_total_layers
        self.node_assignments: Dict[str, NodeAssignment] = {}
        self.layer_to_load: Dict[int, LayerLoad] = {}
        self.node_id_to_node_info: Dict[str, NodeInfo] = {}

        # Pipeline endpoints for routing
        self.embedding_node_ids: List[str] = []
        self.lm_head_node_ids: List[str] = []

        # Dynamic state for rebalancing strategies
        # TODO: not necessarily a heap; maybe make it a general type
        self.layer_loads_heap: List[LayerLoad] = []
        for layer_id in range(self.num_total_layers):
            layer_load = LayerLoad(layer_id=layer_id, current_memory_size=0, current_flops=0.0)
            self.layer_to_load[layer_id] = layer_load
        self._update_layer_loads_heap()

    def _update_layer_loads_heap(self):
        """Rebuild the layer loads heap."""
        self.layer_loads_heap = list(self.layer_to_load.values())
        heapq.heapify(self.layer_loads_heap)

    def add_to_allocation(self, start_layer: int, end_layer: int, node: NodeInfo):
        """
        Add a node to host a range of layers.

        Args:
            start_layer: The start layer (inclusive)
            end_layer: The end layer (exclusive)
            node: The node to host these layers
        """
        self.node_id_to_node_info[node.node_id] = node
        assignment = NodeAssignment(
            node_id=node.node_id, start_layer=start_layer, end_layer=end_layer
        )
        self.node_assignments[node.node_id] = assignment

        # Update pipeline endpoints
        if start_layer == 0:
            self.embedding_node_ids.append(node.node_id)
        if end_layer == self.num_total_layers:
            self.lm_head_node_ids.append(node.node_id)

        # Update layer loads
        for layer_id in range(start_layer, end_layer):
            assert layer_id in self.layer_to_load, "Layer not found in layer_to_load"
            node.set_holding_layers(end_layer - start_layer)
            self.layer_to_load[layer_id].add_node(node)
        self._update_layer_loads_heap()

    def remove_from_allocation(self, node_id: str):
        """Remove a node from the allocation."""
        node_info = self.node_id_to_node_info.get(node_id)
        assignment = self.node_assignments.get(node_id)
        if node_info is None or assignment is None:
            raise ValueError(f"Node {node_id} not found in allocation")

        # Remove from pipeline endpoints
        if node_id in self.embedding_node_ids:
            self.embedding_node_ids.remove(node_id)
        if node_id in self.lm_head_node_ids:
            self.lm_head_node_ids.remove(node_id)

        # Update layer loads
        for layer_id in range(assignment.start_layer, assignment.end_layer):
            if layer_id in self.layer_to_load:
                self.layer_to_load[layer_id].remove_node(node_info)

        # Remove node assignment
        del self.node_assignments[node_id]
        del self.node_id_to_node_info[node_id]

        self._update_layer_loads_heap()

    def get_lightest_layer(self) -> Optional[LayerLoad]:
        """Get the lightest layer without removing it from heap."""
        if self.layer_loads_heap:
            return self.layer_loads_heap[0]
        return None


class WaterFillingPipelineRebalancer:
    """
    Proportional per-pipeline layer allocator (logical stage split).
    - Sorts nodes by decoder-layer capacity (desc)
    - Water-filling by TFLOPS
    - Integerizes with largest remainders under per-node caps
    - First node reserves input embedding; last reserves LM head

    - Goal:
        - t_i prop flops_i
        - 0 <= t_i <= capacity_i
        - Sum_i t_i = total_layers
    - Solving for lambda:
        - t_i(lambda) = min(c_i, lambda F_i)
        - sum t_i(lambda) = total_layers
    """

    def __init__(self, num_total_layers: int, max_iterations: int = 40):
        self.num_total_layers = num_total_layers
        self.max_iterations = max_iterations

    # pylint: disable=too-many-locals,too-many-branches,too-many-statements
    def rebalance(
        self, pipeline_nodes: List[NodeInfo], assume_sorted: bool = False
    ) -> LayerAllocationPlan:
        """
        Rebalance the layers within a pipeline to be proportional to TFLOPS of the nodes.

        Args:
            pipeline_nodes: The nodes in the pipeline.
            assume_sorted: Whether the nodes are already sorted by capacity.

        Returns:
            The rebalanced allocation plan.
        """
        total_layers = self.num_total_layers
        plan = LayerAllocationPlan(total_layers)
        if not pipeline_nodes or total_layers <= 0:
            raise ValueError("No nodes or total layers is non-positive")

        nodes = (
            pipeline_nodes
            if assume_sorted
            else sorted(pipeline_nodes, key=lambda n: n.capacity_layers(), reverse=True)
        )
        n = len(nodes)

        caps: List[int] = []
        flops: List[float] = []
        for i, node in enumerate(nodes):
            if i == 0:
                cap = node.capacity_layers(include_input_embed=True)
            elif i == n - 1:
                cap = node.capacity_layers(include_lm_head=True)
            else:
                cap = node.capacity_layers()
            if cap <= 0:
                raise ValueError(f"Node {node.node_id} has non-positive capacity: {cap}")
            caps.append(cap)
            flops.append(node.tflops_fp16)

        if sum(caps) < total_layers:
            raise ValueError(f"Total capacity {sum(caps)} is less than total layers {total_layers}")

        # Water-filling: find lambda s.t. sum min(c_i, λ F_i) == L
        def total_at(lmbd: float) -> float:
            return sum(min(caps[i], lmbd * flops[i]) for i in range(n))

        lo, hi = 0.0, max((caps[i] / flops[i]) for i in range(n))
        for _ in range(self.max_iterations):
            mid = 0.5 * (lo + hi)
            if total_at(mid) >= total_layers:
                hi = mid
            else:
                lo = mid
        lam = hi

        target = [min(caps[i], lam * flops[i]) for i in range(n)]

        # Integerization: floor + largest remainders (respect caps)
        stage_layer_counts = [min(caps[i], int(floor(target[i]))) for i in range(n)]
        assigned = sum(stage_layer_counts)
        remaining = total_layers - assigned
        if remaining > 0:
            frac = [(target[i] - stage_layer_counts[i], -i) for i in range(n)]
            for _, negi in sorted(frac, reverse=True):
                i = -negi
                if remaining == 0:
                    break
                room = caps[i] - stage_layer_counts[i]
                if room > 0:
                    stage_layer_counts[i] += 1
                    remaining -= 1
        elif remaining < 0:
            raise ValueError(f"Remaining {remaining} is negative")

        # Final clamp (safety) and residual distribute, if any
        extra = 0
        for i in range(n):
            if stage_layer_counts[i] > caps[i]:
                extra += stage_layer_counts[i] - caps[i]
                stage_layer_counts[i] = caps[i]
        if extra > 0:
            for i in range(n):
                if extra == 0:
                    break
                room = caps[i] - stage_layer_counts[i]
                take = min(room, extra)
                stage_layer_counts[i] += take
                extra -= take

        # Build per-pipeline allocation plan with contiguous stages in stage order
        start_layer = 0
        for idx, node in enumerate(nodes):
            layers = stage_layer_counts[idx]
            if layers <= 0:
                continue
            end_layer = start_layer + layers
            plan.add_to_allocation(start_layer, end_layer, node)
            start_layer = end_layer

        return plan


class DynamicNodeHandler(ABC):
    """Abstract strategy for handling dynamic node changes."""

    @abstractmethod
    def handle_node_join(self, allocation_plan: LayerAllocationPlan, new_node: NodeInfo) -> None:
        """Handle a new node joining the cluster (modifies plan in-place)."""

    @abstractmethod
    def handle_node_leave(
        self,
        allocation_plan: LayerAllocationPlan,
        departed_node_id: str,
    ) -> None:
        """Handle a node leaving the cluster (modifies plan in-place)."""

    @abstractmethod
    def should_global_rebalance(self, allocation_plan: LayerAllocationPlan) -> bool:
        """Determine if a full global rebalance is needed."""


class GapPatchDynamicNodeHandler(DynamicNodeHandler):
    """Gap-patch dynamic node handler: fills gaps with lightest layers."""

    def __init__(self, model_info: ModelInfo, rebalance_threshold: float = 0.25):
        self.model_info = model_info
        self.rebalance_threshold = rebalance_threshold

    def handle_node_join(
        self, allocation_plan: LayerAllocationPlan, new_node: NodeInfo
    ) -> Tuple[int, int]:
        """Assign lightest layers to the new node greedily up to its capacity."""

        node_capacity = new_node.capacity_layers()

        lightest_layer = allocation_plan.get_lightest_layer()
        if lightest_layer is None:
            raise ValueError("No layers to assign")

        # Assign consecutive layers starting from the lightest layer
        start_layer = lightest_layer.layer_id
        end_layer = min(start_layer + node_capacity, allocation_plan.num_total_layers)

        allocation_plan.add_to_allocation(start_layer, end_layer, new_node)
        return start_layer, end_layer

    def handle_node_leave(
        self,
        allocation_plan: LayerAllocationPlan,
        departed_node_id: str,
    ) -> None:
        """Remove the departed node from allocation."""

        allocation_plan.remove_from_allocation(departed_node_id)

    def _calculate_combined_load(
        self, layer: LayerLoad, total_cluster_memory: int, total_cluster_flops: float
    ) -> float:
        """Calculates a normalized, combined load metric for a layer."""
        if total_cluster_memory == 0 or total_cluster_flops == 0:
            return 0.0

        normalized_memory = layer.current_memory_size / total_cluster_memory
        normalized_flops = layer.current_flops / total_cluster_flops

        # Using a 50/50 weighted average for memory and FLOPs
        return 0.5 * normalized_memory + 0.5 * normalized_flops

    def should_global_rebalance(self, allocation_plan: LayerAllocationPlan) -> bool:
        """
        Trigger global rebalance if load imbalance is too high.

        The method calculates a combined, normalized load for each layer based
        on its memory and FLOPs usage relative to the total cluster capacity.
        It then computes the coefficient of variation (std_dev / mean) of these
        loads. If this value exceeds a configurable threshold, it indicates
        significant imbalance and returns True.
        """

        layer_heap = allocation_plan.layer_loads_heap
        if len(layer_heap) < 2:
            return False

        nodes = allocation_plan.node_id_to_node_info.values()
        total_cluster_memory = sum(node.memory_gb for node in nodes)
        total_cluster_flops = sum(node.tflops_fp16 for node in nodes)

        if total_cluster_memory == 0 or total_cluster_flops == 0:
            raise ValueError("Total cluster memory or flops is zero")

        loads = [
            self._calculate_combined_load(layer, total_cluster_memory, total_cluster_flops)
            for layer in layer_heap
        ]

        if not loads:
            return False

        avg_load = sum(loads) / len(loads)
        if avg_load == 0:
            return False

        variance = sum((load - avg_load) ** 2 for load in loads) / len(loads)
        std_dev = variance**0.5

        coefficient_of_variation = std_dev / avg_load

        return coefficient_of_variation > self.rebalance_threshold


class LayerAllocator(ABC):
    """Base abstract class for layer allocation strategies."""

    def __init__(self, model_info: ModelInfo, nodes: List[NodeInfo]):
        self.model_info = model_info
        self.nodes = nodes

    def _sort_nodes_by_capacity(self) -> List[NodeInfo]:
        """Sort nodes by capacity."""
        return sorted(self.nodes, key=lambda node: node.capacity_layers(), reverse=True)

    @abstractmethod
    def allocate(self) -> LayerAllocationPlan:
        """Allocate layers to nodes."""

    @abstractmethod
    def add_node(self, node: NodeInfo) -> Tuple[int, int]:
        """Add a node to the allocator."""

    @abstractmethod
    def remove_node(self, node_id: str) -> None:
        """Remove a node from the allocator."""

    @abstractmethod
    def rebalance(self) -> LayerAllocationPlan:
        """Rebalance the layers across the nodes."""


class GreedyLayerAllocator(LayerAllocator):
    """
    Greedy layer allocator that assigns layers to nodes trying to
    1. Minimize number of stages in each pipeline;
    2. Maximize number of pipelines.

    The algorithm proceeds as follows:
    1.  Initialization: Nodes are sorted by their layer capacity in descending
        order, so nodes with more memory are considered first.

    2.  Iterative Pipeline Construction: The allocator attempts to build one
        pipeline at a time. It iterates through the sorted nodes, adding them
        to the current pipeline until the total number of layers required by
        the model is met. Greedy in the sense that it always assigns the max
        layers allowed by capacity of the current node to the pipeline.
        So to minimize number of stages.

    3.  Look-Ahead Optimization: Before selecting a node, the algorithm "looks ahead"
        to see if the remaining nodes have enough capacity to form at least one more
        full pipeline.
        - If so, it chooses the *smallest* node that can complete the current
          pipeline. This preserves the larger nodes for future pipelines.
        - If not, it falls back to the default greedy behavior and picks the
          *largest* available node to complete the pipeline as efficiently as
          possible.
        So to maximize number of pipelines.

    4.  In-Place Rebalancing: After each pipeline is formed, the layers within
        it are immediately rebalanced to be proportional to the computing power
        (TFLOPS) of the nodes. Capped by capacity. This ensures that the stages
        are balanced and system throughput is maximized.
        We use water-filling algorithm.


    This process repeats until no more complete pipelines can be formed from the
    remaining nodes.
    """

    def __init__(self, model_info: ModelInfo, nodes: List[NodeInfo]):
        super().__init__(model_info, nodes)
        self.nodes = self._sort_nodes_by_capacity()
        self._rebalancer = WaterFillingPipelineRebalancer(model_info.num_layers)
        self._dynamic_node_handler = GapPatchDynamicNodeHandler(model_info)
        self.allocation_plan: Optional[LayerAllocationPlan] = None

    # pylint: disable=too-many-locals, too-many-nested-blocks, too-many-branches
    def allocate(self) -> LayerAllocationPlan:
        """
        Allocate layers to nodes greedily to maximize the number of pipelines.
        """
        num_total_layers = self.model_info.num_layers
        allocation_plan = LayerAllocationPlan(num_total_layers)

        available_nodes = self.nodes.copy()

        while available_nodes:
            total_remaining_capacity = sum(node.capacity_layers() for node in available_nodes)
            if total_remaining_capacity < num_total_layers:
                break

            pipeline_nodes: List[NodeInfo] = []
            remaining_layers = num_total_layers

            current_pipeline_total_capacity = total_remaining_capacity

            while remaining_layers > 0 and available_nodes:
                is_start = len(pipeline_nodes) == 0
                # Look-ahead optimization, addition one account for LM head
                look_ahead_possible = (
                    current_pipeline_total_capacity - remaining_layers >= num_total_layers + 1
                )
                # turns off look-ahead if the pipeline is not yet started
                if not is_start:
                    look_ahead_possible = False
                best_fit_idx = -1
                if look_ahead_possible:
                    # Find smallest node that can complete the pipeline
                    for i, node in enumerate(available_nodes):
                        node_i_capacity = node.capacity_layers(include_lm_head=True)
                        if node_i_capacity >= remaining_layers:
                            # Check if remaining nodes can form a pipeline
                            # The sum is optimized by using the running total.
                            remaining_nodes_capacity = (
                                current_pipeline_total_capacity - node_i_capacity
                            )
                            if remaining_nodes_capacity >= num_total_layers:
                                # We take the last one which will be the smallest capacity
                                # because nodes are sorted descending
                                best_fit_idx = i

                if best_fit_idx != -1:
                    node_to_add = available_nodes.pop(best_fit_idx)
                else:
                    node_to_add = available_nodes.pop(0)

                pipeline_nodes.append(node_to_add)
                # Update running totals
                node_capacity = node_to_add.capacity_layers(include_input_embed=is_start)
                remaining_layers -= node_capacity
                if remaining_layers <= 0:
                    # TODO: remove this
                    if is_start:
                        raise ValueError("Can't map full model on a single node")

                    remaining_layers += node_capacity
                    node_capacity = node_to_add.capacity_layers(
                        include_input_embed=is_start, include_lm_head=True
                    )
                    remaining_layers -= node_capacity

                current_pipeline_total_capacity -= node_capacity

            if remaining_layers <= 0:
                # Rebalance layers within the pipeline and merge into the global plan.
                pipeline_plan = self._rebalancer.rebalance(pipeline_nodes)
                for node_id, assignment in pipeline_plan.node_assignments.items():
                    node_info = pipeline_plan.node_id_to_node_info[node_id]
                    allocation_plan.add_to_allocation(
                        assignment.start_layer, assignment.end_layer, node_info
                    )
            else:
                # Cannot form a complete pipeline, put nodes back
                available_nodes.extend(pipeline_nodes)
                break

        if not allocation_plan.node_assignments:
            raise ValueError("No valid allocation found")
        self.allocation_plan = allocation_plan
        return allocation_plan

    def add_node(self, node: NodeInfo) -> Tuple[int, int]:
        """Add a node to the allocator and re-calculates the allocation plan."""
        self.nodes.append(node)
        self.nodes = self._sort_nodes_by_capacity()
        start_layer, end_layer = self._dynamic_node_handler.handle_node_join(
            self.allocation_plan, node
        )
        return start_layer, end_layer

    def remove_node(self, node_id: str) -> None:
        """Remove a node from the allocator and re-calculates the allocation plan."""
        if node_id in self.nodes:
            self.nodes.remove(node_id)
        if node_id in self.allocation_plan.node_assignments:
            self._dynamic_node_handler.handle_node_leave(self.allocation_plan, node_id)

    def rebalance(self) -> LayerAllocationPlan:
        """Re-runs the allocation and rebalancing process."""
        if self._dynamic_node_handler.should_global_rebalance(self.allocation_plan):
            self.allocation_plan = self.allocate()
            return self.allocation_plan
        return self.allocation_plan


class DynamicProgrammingLayerAllocator(LayerAllocator):
    """
    Dynamic programming based allocator that balances two objectives:
     - Concurrency (number of pipelines)
     - Latency (number of stages per pipeline).

    Why DP (vs greedy): interleaving constructions of multiple pipelines
    so cases like capacities (40, 40, 20, 20, 10, 10) with total 70 layers
    yields (40, 20, 10) + (40, 20, 10) instead of a single 2-stage pipe line (40 + 30).

    We want to maximize a simple scalar objective function:
        Z(k) = k / (s*(k) / k) = k^2 / s*(k)
    where:
        k: number of pipelines
        s*(k): minimum total number of stages realizing k pipelines

    DP State:
        dp(i, open_residuals, finshed_pipes) := min total stages needed using GPUs with index >= i;
        where:
            i: GPU index, in [0, N]
            open_residuals: sorted tuple of remaining layers for all open pipelines (values in 1..L-1)
            finished_pipes: number of already fully closed pipelines
        Transitions (for node i with capacity c_i):
            1. Skip node: dp(i + 1, open_residuals, finished_pipes)
            2. Assign to an existing open pipeline j:
               r' = r_j - c_i. If r' <= 0, try closing with LM head; if still <= 0 -> close (remove j, finished+1),
               else keep open with updated residual r'.
            3. Start a new pipeline (if finished + len(open_residuals) < k_target):
               r = L - c_i (with input embedding). If r <= 0, it closes immediately (finished+1), else append r.

    Finally:
        Compute objective Z(k) = (k**alpha) / (T_comp + (total_stages/k)*r_RTT)
        Choose the best k and backtrack to recover assignments
    """

    def __init__(
        self,
        model_info: ModelInfo,
        nodes: List[NodeInfo],
        alpha: float = 2.0,
    ) -> None:
        super().__init__(model_info, nodes)
        # Sort GPUs by layer capacity descending
        self.nodes = self._sort_nodes_by_capacity()
        self.alpha = alpha
        self._path: Dict[Tuple[int, Tuple[int, ...], int], Tuple] = {}
        self._rebalancer = WaterFillingPipelineRebalancer(model_info.num_layers)
        self._dynamic_node_handler = GapPatchDynamicNodeHandler(model_info)
        self.allocation_plan: Optional[LayerAllocationPlan] = None

    # pylint: disable=too-many-locals, too-many-statements
    def allocate(self) -> LayerAllocationPlan:
        num_nodes = len(self.nodes)
        num_layers = int(self.model_info.num_layers)
        total_cap = sum(node.capacity_layers() for node in self.nodes)

        if num_layers <= 0 or num_nodes == 0 or total_cap < num_layers:
            raise ValueError("No valid allocation found")
        # used for pruning
        suffix_sum = [0] * (num_nodes + 1)
        for i in range(num_nodes - 1, -1, -1):
            suffix_sum[i] = suffix_sum[i + 1] + self.nodes[i].capacity_layers()

        max_num_pipes = min(num_nodes, total_cap // num_layers)
        best_num_pipes = 0
        best_score: float = float("-inf")

        best_path: Dict[Tuple[int, Tuple[int, ...], int], Tuple] = {}
        for k_target in range(1, max_num_pipes + 1):
            path: Dict[Tuple[int, Tuple[int, ...], int], Tuple] = {}

            # pylint: disable=too-many-branches, too-many-locals, cell-var-from-loop, too-many-statements
            @lru_cache(maxsize=None)
            def dp(i: int, open_residuals: Tuple[int, ...], finished_pipes: int) -> int:
                # Completed target with no open pipelines
                if finished_pipes == k_target and len(open_residuals) == 0:
                    path[(i, open_residuals, finished_pipes)] = ("done",)
                    return 0
                if i == num_nodes:
                    return float("inf")

                new_needed = k_target - finished_pipes - len(open_residuals)
                need_open = sum(open_residuals)
                remaining_cap = suffix_sum[i]

                # Pruning
                # 1. already have more (finished + open) than target;
                # 2. remaining capacity is not enough to close ongoing pipelines
                #    and unfulfilled new pipelines
                # 3. remaining nodes are not enough to fulfill new pipelines
                if (
                    new_needed < 0
                    or remaining_cap < need_open + max(0, new_needed) * num_layers
                    or finished_pipes + len(open_residuals) + (num_nodes - i) < k_target
                ):
                    return float("inf")

                # Option 1: Skip this node
                best_cost = dp(i + 1, open_residuals, finished_pipes)
                best_action: Tuple = ("skip",)

                # Option 2: Assign to existing open pipeline
                for j, rj in enumerate(open_residuals):
                    c_norm = self.nodes[i].capacity_layers()
                    r_after = rj - c_norm
                    if r_after <= 0:
                        # try closing with LM head allowance
                        c_close = self.nodes[i].capacity_layers(include_lm_head=True)
                        r_after_close = rj - c_close
                        if r_after_close <= 0:
                            new_open = list(open_residuals)
                            new_open.pop(j)
                            cost = 1 + dp(i + 1, tuple(new_open), finished_pipes + 1)
                            if cost < best_cost:
                                best_cost = cost
                                best_action = ("assign", j, True)
                        else:
                            new_open = list(open_residuals)
                            new_open[j] = r_after_close
                            new_open.sort()
                            cost = 1 + dp(i + 1, tuple(new_open), finished_pipes)
                            if cost < best_cost:
                                best_cost = cost
                                best_action = ("assign", j, False)
                    else:
                        new_open = list(open_residuals)
                        new_open[j] = r_after
                        new_open.sort()
                        cost = 1 + dp(i + 1, tuple(new_open), finished_pipes)
                        if cost < best_cost:
                            best_cost = cost
                            best_action = ("assign", j, False)

                # Option 3: start a new pipeline (if we still need more)
                if new_needed > 0:
                    c_start = self.nodes[i].capacity_layers(include_input_embed=True)
                    r_new = num_layers - c_start
                    if r_new <= 0:
                        cost = 1 + dp(i + 1, open_residuals, finished_pipes + 1)
                        if cost < best_cost:
                            best_cost = cost
                            best_action = ("start", 0, True)
                    else:
                        new_open = list(open_residuals) + [r_new]
                        new_open.sort()
                        cost = 1 + dp(i + 1, tuple(new_open), finished_pipes)
                        if cost < best_cost:
                            best_cost = cost
                            best_action = ("start", r_new, False)

                path[(i, open_residuals, finished_pipes)] = best_action
                return best_cost

            s_star = dp(0, tuple(), 0)
            if s_star < float("inf"):
                score = (k_target * k_target) / s_star  # Z(k) = k^2 / s*(k)
                if score > best_score:
                    best_score, best_num_pipes = score, k_target
                    best_path = dict(path)

        if best_num_pipes is None or best_num_pipes == 0:
            raise ValueError("No valid allocation found")
        self._path = best_path
        pipelines = self._backtrack(best_num_pipes, num_nodes)

        plan = LayerAllocationPlan(num_layers)
        # Merge pipelines into the final plan with rebalancing
        for pl_nodes in pipelines:
            if not pl_nodes:
                continue
            pipeline_plan = self._rebalancer.rebalance(pl_nodes, assume_sorted=False)
            for node_id, assignment in pipeline_plan.node_assignments.items():
                node_info = pipeline_plan.node_id_to_node_info[node_id]
                plan.add_to_allocation(assignment.start_layer, assignment.end_layer, node_info)

        self.allocation_plan = plan
        return plan

    def _backtrack(self, best_num_pipes: int, num_nodes: int) -> List[List[NodeInfo]]:
        # Reconstruct pipelines
        pipelines: List[List[NodeInfo]] = [[] for _ in range(best_num_pipes)]
        # (residual, nodes list)
        open_list: List[Tuple[int, List[NodeInfo]]] = []
        i = 0
        finished = 0
        while i < num_nodes and finished < best_num_pipes:
            open_tuple = tuple(sorted(r for r, _ in open_list))
            action = self._path.get((i, open_tuple, finished))
            if action is None:
                break
            kind = action[0]
            if kind in ("skip", "done"):
                i += 1
                continue
            node = self.nodes[i]
            if kind == "assign":
                j, closed = action[1], action[2]
                # ensure open_list sorted like open_tuple
                open_list.sort(key=lambda x: x[0])
                rj, nodes_seq = open_list[j]
                c_norm = node.capacity_layers()
                r_after = rj - c_norm
                if r_after <= 0:
                    c_close = node.capacity_layers(include_lm_head=True)
                    r_after = rj - c_close
                nodes_seq.append(node)
                if r_after <= 0 or closed:
                    # pipeline closes here
                    pipelines[finished].extend(nodes_seq)
                    open_list.pop(j)
                    finished += 1
                else:
                    open_list[j] = (r_after, nodes_seq)
                i += 1
            elif kind == "start":
                r_new, closed = action[1], action[2]
                if closed:
                    pipelines[finished].append(node)
                    finished += 1
                else:
                    open_list.append((r_new, [node]))
                i += 1
            else:
                # Unknown action; advance to avoid infinite loop
                i += 1

        return pipelines

    def add_node(self, node: NodeInfo) -> Tuple[int, int]:
        """Add a node to the allocator and re-calculates the allocation plan."""
        self.nodes.append(node)
        self.nodes = self._sort_nodes_by_capacity()
        return self._dynamic_node_handler.handle_node_join(self.allocation_plan, node)

    def remove_node(self, node_id: str) -> None:
        """Remove a node from the allocator and re-calculates the allocation plan."""
        if node_id in self.nodes:
            self.nodes.remove(node_id)
        if node_id in self.allocation_plan.node_assignments:
            self._dynamic_node_handler.handle_node_leave(self.allocation_plan, node_id)

    def rebalance(self) -> LayerAllocationPlan:
        """Re-runs the allocation and rebalancing process."""
        if self._dynamic_node_handler.should_global_rebalance(self.allocation_plan):
            self._path.clear()
            self.allocation_plan = self.allocate()
            return self.allocation_plan
        return self.allocation_plan
