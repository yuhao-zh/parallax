"""
Node information classes for distributed LLM inference scheduling.

This module provides core abstractions for representing compute nodes in a distributed
inference swarm, enabling intelligent layer allocation and request routing decisions.

NodeInfo: Static hardware capabilities and model configuration for phase 1 scheduling.
    - Hardware specs: TFLOPS, memory, bandwidth, quantization speedup
    - Model information: embedded ModelInfo for layer-specific calculations
    - Roofline model computation: compute vs memory-bound latency estimation
    - Capacity calculation: maximum layers a node can host given memory constraints

Node: Dynamic runtime state for phase 2 request routing and load management
    - Layer allocation: tracks (start_layer, end_layer) range assignment
    - Request management: current load, capacity limits, overload detection
    - Performance tracking: empirical latency, RTT to other nodes
    - Load balancing: effective latency with dynamic load penalties
"""

import time
from dataclasses import dataclass
from math import floor
from typing import Callable, Dict, Optional, Tuple

from parallax.scheduling.model_info import ModelInfo


@dataclass
class NodeInfo:
    """
    Static information about a node's capabilities.
    Used in phase 1 layer allocation scheduling.
    """

    node_id: str
    tflops_fp16: float
    memory_gb: float
    model_info: ModelInfo
    param_hosting_ratio: float = 0.5
    kv_cache_ratio: float = 0.3
    memory_bandwidth_gbps: float = 100.0
    # Roofline speedup factor for quantized models (e.g., 2.0 for INT8, 4.0 for INT4)
    quantization_speedup: float = 1
    # Get per-layer info once num layers holding is set
    num_layers_holding: Optional[int] = None
    per_layer_flops: Optional[int] = None
    per_layer_memory: Optional[int] = None

    def get_compute_roofline_latency_ms(self, flops: int) -> float:
        """Compute roofline latency based on FLOPs and node capabilities."""
        return flops / (self.quantization_speedup * self.tflops_fp16 * 1e9)

    def get_io_roofline_latency_ms(self, io_bytes: int) -> float:
        """Compute io latency based on data size and node bandwidth."""
        return io_bytes / (self.memory_bandwidth_gbps * 1e6)

    def roofline_layer_latency_ms(
        self,
        include_input_embed: bool = False,
        include_lm_head: bool = False,
        num_decoder_layers: int = 1,
    ) -> float:
        """
        Compute roofline latency for all decoder layers with specified embedding types.

        Args:
            include_input_embed: Whether to include embedding layer computation
            include_lm_head: Whether to include LM head layer computation
            num_decoder_layers: Number of decoder layers

        Returns:
            Latency in milliseconds for the specified layer types
        """
        decoder_layer_compute_latency = self.get_compute_roofline_latency_ms(
            self.model_info.decoder_layer_flops
        )
        decoder_layer_io_latency = self.get_io_roofline_latency_ms(
            self.model_info.decoder_layer_io_bytes()
        )

        flops, io_bytes = 0, 0
        if include_input_embed:
            # compute is negligible for look-up
            io_bytes += self.model_info.embedding_io_bytes

        if include_lm_head:
            flops += self.model_info.lm_head_flops
            io_bytes += self.model_info.embedding_io_bytes

        compute_time_ms = self.get_compute_roofline_latency_ms(flops)
        io_time_ms = self.get_io_roofline_latency_ms(io_bytes)
        return num_decoder_layers * max(
            decoder_layer_compute_latency, decoder_layer_io_latency
        ) + max(compute_time_ms, io_time_ms)

    def capacity_layers(
        self, include_input_embed: bool = False, include_lm_head: bool = False
    ) -> int:
        """
        Check number of layers a node can hold.

        Args:
            include_input_embed: Whether to include embedding layer
            include_lm_head: Whether to include LM head layer

        Returns:
            Number of decoder layers this node can accommodate
        """
        # TODO: remove this assumption
        if include_input_embed and include_lm_head:
            raise ValueError("A node cannot host both input and output embedding")

        available_memory_bytes = floor(
            self.memory_gb * 1024 * 1024 * 1024 * self.param_hosting_ratio
        )
        if include_input_embed or include_lm_head:
            available_memory_bytes -= self.model_info.embedding_io_bytes

        return floor(available_memory_bytes / self.model_info.decoder_layer_io_bytes(active=False))

    def set_holding_layers(
        self,
        num_layers: int,
    ):
        """Set the number of layers this node can hold and per-layer flops/memory."""
        self.num_layers_holding = num_layers
        self.per_layer_memory = floor(
            self.memory_gb * self.kv_cache_ratio / self.num_layers_holding
        )
        self.per_layer_flops = floor(self.tflops_fp16 / self.num_layers_holding)


@dataclass
class Node:
    """
    Dynamic state of a node.
    Used in phase 2 request routing and runtime management.
    """

    node_id: str
    node_info: NodeInfo
    # start: inclusive, end: exclusive
    current_layers: Optional[Tuple[int, int]] = None
    current_requests: int = 0
    max_requests: int = 8
    avg_layer_latency_ms: float = 0.0
    rtt_to_nodes: Dict[str, float] = None
    is_active: bool = True
    last_heartbeat: float = 0.0
    # Optional RTT provider for measuring latency to other nodes
    rtt_getter: Optional[Callable[["Node", "Node"], float]] = None

    def __post_init__(self):
        if self.rtt_to_nodes is None:
            self.rtt_to_nodes = {}
        if self.last_heartbeat == 0.0:
            self.last_heartbeat = time.time()

    @property
    def num_current_layers(self) -> int:
        """Number of currently allocated layers."""
        if self.current_layers is None:
            return 0
        start_layer, end_layer = self.current_layers
        return end_layer - start_layer

    @property
    def has_embedding(self) -> bool:
        """Check if this node hosts the embedding layer (layer 0)."""
        if self.current_layers is None:
            return False
        start_layer, _ = self.current_layers
        return start_layer == 0

    @property
    def has_lm_head(self) -> bool:
        """Check if this node hosts the LM head layer (last layer)."""
        if self.current_layers is None:
            return False
        _, end_layer = self.current_layers
        return end_layer == self.node_info.model_info.num_layers

    @property
    def is_overloaded(self) -> bool:
        """Check if node is at capacity for requests."""
        return self.current_requests >= self.max_requests

    @property
    def load_factor(self) -> float:
        """Current load as a fraction of maximum capacity (0.0 to 1.0)."""
        if self.max_requests == 0:
            return 1.0
        return min(1.0, self.current_requests / self.max_requests)

    def set_layer_allocation(self, start_layer: int, end_layer: int):
        """Set the layer range allocated to this node."""
        self.current_layers = (start_layer, end_layer)

    def clear_layer_allocation(self):
        """Clear the layer allocation for this node."""
        self.current_layers = None

    @property
    def layer_latency_ms(self) -> float:
        """Get effective layer latency considering both roofline and load."""
        if self.is_overloaded:
            return float("inf")

        base_latency = self.node_info.roofline_layer_latency_ms(
            self.has_embedding, self.has_lm_head, self.num_current_layers
        )
        load_penalty = 1.0 + (self.load_factor * 0.5)  # Up to 50% penalty at full load
        return base_latency * load_penalty

    def update_rtt(self, target_node_id: str, rtt_ms: float):
        """Update RTT measurement to another node."""
        self.rtt_to_nodes[target_node_id] = rtt_ms

    def get_rtt_to(self, other: "Node") -> float:
        """Get RTT to another node, measuring via `rtt_getter` if needed.

        Falls back to 0.0 if no getter is provided and no cached RTT exists.
        """
        if self == other:
            return 0.0
        if other.node_id in self.rtt_to_nodes:
            return self.rtt_to_nodes[other.node_id]
        if self.rtt_getter is None:
            return 0.0
        rtt_ms = float(self.rtt_getter(self, other))
        self.update_rtt(other.node_id, rtt_ms)
        return rtt_ms

    def hosts_layer(self, layer_id: int) -> bool:
        """Return True if this node hosts the given layer id.

        Interprets `current_layers` as a half-open interval [start, end).
        """
        if self.current_layers is None:
            return False
        start_layer, end_layer = self.current_layers
        return start_layer <= layer_id < end_layer

    def add_request(self):
        """Add a request to this node."""
        self.current_requests += 1

    def remove_request(self):
        """Remove a request from this node."""
        self.current_requests -= 1
