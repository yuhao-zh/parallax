# pylint: disable=too-many-locals
"""
Unit tests for DynamicProgrammingRouting.

Covers:
- Shard-level DP path using `Node` APIs (latency + RTT)
- Turning point detection via layer-level DP
- Parametrized scenarios with different splits/overlaps
"""

from dataclasses import dataclass
from math import sqrt
from typing import Iterable, List, Tuple

import pytest

from parallax.scheduling.model_info import ModelInfo
from parallax.scheduling.node import Node, NodeInfo
from parallax.scheduling.request_routing import DynamicProgrammingRouting


def build_model(num_layers: int) -> ModelInfo:
    """Construct a small `ModelInfo` for tests with given number of layers."""
    return ModelInfo(
        model_name=f"TestModel-{num_layers}L",
        head_size=64,
        hidden_dim=2048,
        intermediate_dim=2048,
        num_attention_heads=32,
        num_kv_heads=8,
        vocab_size=32000,
        num_layers=num_layers,
        ffn_num_projections=3,
        batch_size=1,
        target_seq_len=1,
        source_seq_len=256,
        param_bytes_per_element=1,
        cache_bytes_per_element=1,
        embedding_bytes_per_element=1,
    )


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
    """Create a `Node` with `GeoNodeInfo` and optional coordinates."""
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


def compute_rtts_from_coords(nodes: Iterable[Node]) -> dict[tuple[str, str], float]:
    """Map Euclidean distances between nodes' (x, y) to RTTs in [10, 200] ms.

    Returns a symmetric mapping of (src_id, dst_id) -> RTT.
    """
    node_list = list(nodes)
    if not node_list:
        return {}
    # Extract coords and ids
    coords: dict[str, Tuple[float, float]] = {
        n.node_id: (
            float(getattr(n.node_info, "x", 0.0)),
            float(getattr(n.node_info, "y", 0.0)),
        )
        for n in node_list
    }
    ids = [n.node_id for n in node_list]

    # Max pairwise distance
    max_dist = 0.0
    for i, aid in enumerate(ids):
        ax, ay = coords[aid]
        for bid in ids[i + 1 :]:
            bx, by = coords[bid]
            d = sqrt((ax - bx) ** 2 + (ay - by) ** 2)
            max_dist = max(max_dist, d)

    def to_latency(d: float) -> float:
        return 10.0 if max_dist <= 0 else 10.0 + 190.0 * (d / max_dist)

    # Build RTT map
    rtts: dict[tuple[str, str], float] = {(nid, nid): 10.0 for nid in ids}
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


def test_optimal_path_simple_chain():
    """Two-node chain [0, k) -> [k, L): verify path and latency sum."""
    num_layers = 12
    split = 7
    model = build_model(num_layers)
    n1 = build_node("n1", model, tflops=200.0, x=0.0, y=0.0)
    n2 = build_node("n2", model, tflops=200.0, x=1.0, y=0.0)
    n1.set_layer_allocation(0, split)
    n2.set_layer_allocation(split, num_layers)
    set_rtt_from_coords([n1, n2])

    router = DynamicProgrammingRouting()
    node_ids, latency = router.find_optimal_path([n1, n2], num_layers)

    assert node_ids == ["n1", "n2"]
    expected = float(n1.layer_latency_ms) + float(n1.get_rtt_to(n2)) + float(n2.layer_latency_ms)
    assert latency == pytest.approx(expected, rel=1e-6)


def test_optimal_path_single_node():
    """Single-node [0, L) path should have no hops and sum only node latency."""
    num_layers = 10
    model = build_model(num_layers)
    n = build_node("solo", model, tflops=250.0, x=0.0, y=0.0)
    n.set_layer_allocation(0, num_layers)

    router = DynamicProgrammingRouting()
    node_ids, latency = router.find_optimal_path([n], num_layers)

    assert node_ids == ["solo"]
    assert latency == pytest.approx(float(n.layer_latency_ms), rel=1e-6)


@pytest.mark.parametrize(
    "num_layers,segments,expected_path",
    [
        (
            15,
            [("a", 0, 5, 200.0, 0.0), ("b", 5, 10, 200.0, 1.0), ("c", 10, 15, 200.0, 2.0)],
            ["a", "b", "c"],
        ),
        (
            12,
            [("full", 0, 12, 150.0, 0.0), ("tail", 6, 12, 300.0, 1.0)],
            ["full"],
        ),
    ],
)
def test_optimal_path_parametrized(
    num_layers: int,
    segments: list[tuple[str, int, int, float, float]],
    expected_path: list[str],
):
    """Parameterized shard-level routing across contiguous and overlap cases."""
    model = build_model(num_layers)
    nodes: list[Node] = []
    for node_id, start, end, tflops, x in segments:
        n = build_node(node_id, model, tflops=tflops, x=x, y=0.0)
        n.set_layer_allocation(start, end)
        nodes.append(n)

    router = DynamicProgrammingRouting()
    set_rtt_from_coords(nodes)
    node_ids, latency = router.find_optimal_path(nodes, num_layers)
    assert node_ids == expected_path
    # Sanity: latency equals sum of node latencies along path plus RTTs
    total = 0.0
    for i, nid in enumerate(node_ids):
        n = next(n for n in nodes if n.node_id == nid)
        total += float(n.layer_latency_ms)
        if i > 0:
            prev = next(n for n in nodes if n.node_id == node_ids[i - 1])
            total += float(prev.get_rtt_to(n))
    assert latency == pytest.approx(total, rel=1e-6)


@pytest.mark.parametrize(
    "num_layers,segments,expected_turns",
    [
        # tail has dramatically larger I/O and close-enough y so small RTT
        # 'head' should 'turn' to the tail ASAP
        (10, [("head", 0, 6, 1.0, 0.0), ("tail", 4, 10, 3000.0, 0.01)], [("head", 4)]),
        (8, [("solo", 0, 8, 250.0, 0.0)], []),
    ],
)
def test_turning_points(
    num_layers: int,
    segments: list[tuple[str, int, int, float, float]],
    expected_turns: list[tuple[str, int]],
):
    """Parameterized turning-point detection across contiguous and overlap cases."""
    model = build_model(num_layers)
    nodes: list[Node] = []
    for node_id, start, end, io, x in segments:
        n = build_node(node_id, model, mem_bandwidth_gbps=io, x=x, y=0.0)
        n.set_layer_allocation(start, end)
        nodes.append(n)

    router = DynamicProgrammingRouting()
    set_rtt_from_coords(nodes)
    turns = router.find_turning_points(nodes, num_layers)
    # Order of turning points is by layer progression; compare directly
    assert turns == expected_turns
