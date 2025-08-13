"""
Reusable test helpers for model/node builders and RTT utilities.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Dict, Iterable, List, Tuple

from parallax.scheduling.model_info import ModelInfo
from parallax.scheduling.node import Node, NodeInfo


def build_model_info(num_layers: int) -> ModelInfo:
    """Build a model config used across tests (matches allocation tests)."""
    return ModelInfo(
        model_name=f"GPUOss-{num_layers}L",
        head_size=64,
        hidden_dim=2880,
        intermediate_dim=2880,
        num_attention_heads=64,
        num_kv_heads=8,
        vocab_size=201088,
        num_layers=num_layers,
        ffn_num_projections=3,
        num_local_experts=128,
        num_experts_per_tok=4,
        batch_size=1,
        target_seq_len=1,
        source_seq_len=4096,
        param_bytes_per_element=1,
        cache_bytes_per_element=2,
        embedding_bytes_per_element=2,
    )

def build_node_info(
    gpu_type: str,
    model_info: ModelInfo,
    id_suffix: str = "",
) -> NodeInfo:
    """Build NodeInfo by GPU type (used by allocation tests)."""
    if gpu_type == "a100-80g":
        return NodeInfo(node_id=f"a100-80g{id_suffix}", tflops_fp16=312.0, memory_gb=80.0, model_info=model_info)
    if gpu_type == "a100-40g":
        return NodeInfo(node_id=f"a100-40g{id_suffix}", tflops_fp16=312.0, memory_gb=40.0, model_info=model_info)
    if gpu_type == "rtx5090":
        return NodeInfo(node_id=f"rtx5090{id_suffix}", tflops_fp16=165.0, memory_gb=32.0, model_info=model_info)
    # rtx4090 default
    return NodeInfo(node_id=f"rtx4090{id_suffix}", tflops_fp16=82.6, memory_gb=24.0, model_info=model_info)

def build_node_infos_by_counts(
    model_info: ModelInfo,
    counts: Dict[str, int],
) -> Dict[str, NodeInfo]:
    """Build NodeInfos by GPU type counts dict."""
    infos = {}
    for gpu_type, count in counts.items():
        for i in range(count):
            node = build_node_info(gpu_type, model_info, id_suffix=f"-{i}")
            infos[node.node_id] = node
    return infos


@dataclass
class GeoNodeInfo(NodeInfo):
    """Test-only NodeInfo with coordinates for RTT synthesis."""

    x: float = 0.0
    y: float = 0.0


def build_node(
    node_id: str,
    model: ModelInfo,
    tflops: float = 200.0,
    mem_gb: float = 80.0,
    x: float = 0.0,
    y: float = 0.0,
    mem_bandwidth_gbps: float = 100.0,
) -> Node:
    """Create a `Node` with `GeoNodeInfo` and optional coordinates/bandwidth."""
    info = GeoNodeInfo(
        node_id=node_id,
        tflops_fp16=tflops,
        memory_gb=mem_gb,
        model_info=model,
        x=x,
        y=y,
        memory_bandwidth_gbps=mem_bandwidth_gbps,
    )
    return Node(node_id=node_id, node_info=info)


def compute_rtts_from_coords(nodes: Iterable[Node]) -> Dict[Tuple[str, str], float]:
    """Map Euclidean distances between nodes' (x, y) to RTTs in [10, 200] ms."""
    node_list = list(nodes)
    if not node_list:
        return {}
    coords: Dict[str, Tuple[float, float]] = {
        n.node_id: (
            float(getattr(n.node_info, "x", 0.0)),
            float(getattr(n.node_info, "y", 0.0)),
        )
        for n in node_list
    }
    ids = [n.node_id for n in node_list]

    max_dist = 0.0
    for i, aid in enumerate(ids):
        ax, ay = coords[aid]
        for bid in ids[i + 1 :]:
            bx, by = coords[bid]
            d = sqrt((ax - bx) ** 2 + (ay - by) ** 2)
            max_dist = max(max_dist, d)

    def to_latency(d: float) -> float:
        return 10.0 if max_dist <= 0 else 10.0 + 190.0 * (d / max_dist)

    rtts: Dict[Tuple[str, str], float] = {(nid, nid): 10.0 for nid in ids}
    for i, aid in enumerate(ids):
        ax, ay = coords[aid]
        for bid in ids[i + 1 :]:
            bx, by = coords[bid]
            d = sqrt((ax - bx) ** 2 + (ay - by) ** 2)
            lat = to_latency(d)
            rtts[(aid, bid)] = lat
            rtts[(bid, aid)] = lat
    return rtts


def set_rtt_from_coords(nodes: List[Node]) -> None:
    """Attach an RTT getter to each node based on their coordinates."""
    rtts = compute_rtts_from_coords(nodes)

    def getter(src: Node, dst: Node) -> float:
        if src.node_id == dst.node_id:
            return 0.0
        return rtts.get((src.node_id, dst.node_id), 200.0)

    for n in nodes:
        n.rtt_getter = getter


def build_node_infos(model: ModelInfo, counts: Tuple[int, int, int]) -> List[NodeInfo]:
    """Create NodeInfos by counts tuple: (n_a100_80g, n_rtx5090, n_rtx4090)."""
    n_a100, n_5090, n_4090 = counts
    infos: List[NodeInfo] = []
    for i in range(n_a100):
        infos.append(build_node_info("a100-80g", model, id_suffix=f"-{i}"))
    for i in range(n_5090):
        infos.append(build_node_info("rtx5090", model, id_suffix=f"-{i}"))
    for i in range(n_4090):
        infos.append(build_node_info("rtx4090", model, id_suffix=f"-{i}"))
    return infos


def geo_rtt_provider(positions: Dict[str, Tuple[float, float]]):
    """Create an RTT provider mapping Euclidean distance to [10, 200] ms.

    Scales by the maximum pairwise distance among provided positions.
    """
    ids = list(positions.keys())
    # Compute max pairwise distance for scaling
    max_dist = 0.0
    for i, aid in enumerate(ids):
        ax, ay = positions[aid]
        for bid in ids[i + 1 :]:
            bx, by = positions[bid]
            d = ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5
            if d > max_dist:
                max_dist = d

    def to_latency(d: float) -> float:
        return 10.0 if max_dist <= 0 else 10.0 + 190.0 * (d / max_dist)

    def provider(src: Node, dst: Node) -> float:
        if src.node_id == dst.node_id:
            return 0.0
        sx, sy = positions.get(src.node_id, (0.0, 0.0))
        dx, dy = positions.get(dst.node_id, (0.0, 0.0))
        dist = ((sx - dx) ** 2 + (sy - dy) ** 2) ** 0.5
        return to_latency(dist)

    return provider


