"""
Tests for Phase 1: layer allocation and rebalancing.

Covers:
- Capacity sanity for `Node`
- Water-filling rebalancing within a pipeline (stage splits across heterogeneous nodes)
- Gap-patch dynamic rebalancing (join/leave behavior)
- Greedy and DP allocators producing contiguous [start, end) layer ranges
"""

from collections import Counter
from typing import Literal

import pytest

from scheduling.layer_allocation import (
    BaseLayerAllocator,
    DynamicProgrammingLayerAllocator,
    GreedyLayerAllocator,
)
from scheduling.model_info import ModelInfo
from scheduling.node import Node, NodeHardwareInfo

from .test_utils import build_model_info


def _build_node(gpu_type: str, model: ModelInfo, id_suffix: str = "") -> Node:
    hw_map = {
        "a100-80g": NodeHardwareInfo("a100-80g" + id_suffix, 312.0, 80.0, 2039.0),
        "a100-40g": NodeHardwareInfo("a100-40g" + id_suffix, 312.0, 40.0, 1935.0),
        "rtx5090": NodeHardwareInfo("rtx5090" + id_suffix, 165, 32.0, 1792.0),
        "rtx4090": NodeHardwareInfo("rtx4090" + id_suffix, 82.6, 24.0, 1008.0),
    }
    hw = hw_map[gpu_type]
    return Node(node_id=hw.node_id, hardware=hw, model_info=model)


def test_capacity_sanity_check():
    """Sanity check for capacity calculations."""
    # 36 layers for actual GPT-OSS 120B
    model = build_model_info(36)
    print(f"decoder layer flops: {model.decoder_layer_flops}")
    print(f"lm head flops: {model.lm_head_flops}")
    print(f"decoder layer io in GB: {model.decoder_layer_io_bytes(roofline=False) / (1024 ** 3)}")
    print(f"embedding table in GB: {model.embedding_io_bytes / (1024 ** 3)}")

    for gpu_type in ["a100-80g", "a100-40g", "rtx5090", "rtx4090"]:
        # (capacity, with embed) -> (13, 13), (6, 6), (5, 5), (4, 3)
        node = _build_node(gpu_type, model)
        capacity = node.get_decoder_layer_capacity()
        capacity_with_embed = node.get_decoder_layer_capacity(include_input_embed=True)
        assert capacity_with_embed <= capacity


@pytest.mark.parametrize(
    "num_layers,gpu_types,expected_layers",
    [
        (21, ["a100-80g", "rtx5090", "rtx4090"], [13, 5, 3]),
        (15, ["a100-80g", "rtx5090"], [10, 5]),
        # (20 * 312 : 20 * 165 : 20 * 82.6) / 559.6 = 11.1 : 5.8 : 2.9 -> 12 : 5 : 3
        (20, ["a100-80g", "rtx5090", "rtx4090"], [12, 5, 3]),
        (25, ["a100-80g", "rtx5090", "rtx4090", "rtx4090"], [13, 5, 4, 3]),
        (29, ["rtx4090", "a100-80g", "rtx5090", "rtx5090", "rtx4090"], [3, 13, 5, 5, 3]),
        (9, ["rtx5090", "rtx5090"], [5, 4]),
        (7, ["a100-40g", "rtx5090"], [5, 2]),
    ],
)
def test_water_filling_rebalance(num_layers: int, gpu_types: list[str], expected_layers: list[int]):
    """Test water-filling rebalancer with various GPU configurations."""
    model = build_model_info(num_layers)

    nodes = [_build_node(g, model, id_suffix=f"-{i}") for i, g in enumerate(gpu_types)]

    allocator = GreedyLayerAllocator(model, nodes)
    allocator.adjust_pipeline_layers(nodes, assume_sorted=False)

    actual_layers = []
    for node in nodes:
        assert node.start_layer is not None and node.end_layer is not None
        actual_layers.append(node.end_layer - node.start_layer)

    assert sum(actual_layers) == num_layers
    assert actual_layers == expected_layers

    for i, node in enumerate(nodes):
        cap = node.get_decoder_layer_capacity(
            include_input_embed=(i == 0), include_lm_head=(i == len(nodes) - 1)
        )
        assert actual_layers[i] <= cap


def _test_gap_patch_rebalance(allocator: BaseLayerAllocator):
    """Sanity checks for gap-patch dynamic rebalancing using allocator state."""
    assert allocator.layer_loads_heap, "Layer loads heap should not be empty"
    assert allocator.node_id_to_node, "Allocator should have node mappings"
    model_info = allocator.model_info

    # Heap top should include a host with minimal per-layer KV memory
    per_node_mem = {
        nid: (node.per_decoder_layer_kv_cache_memory or 0)
        for nid, node in allocator.node_id_to_node.items()
    }
    min_mem = min(per_node_mem.values()) if per_node_mem else 0

    top_load = allocator.layer_loads_heap[0]
    assert top_load.hosting_nodes
    top_hosts_min = min(per_node_mem.get(h, float("inf")) for h in top_load.hosting_nodes)
    assert top_hosts_min == min_mem

    # Join a new small GPU and verify hosting set updates
    before_layer_id = top_load.layer_id
    before_mem = allocator.layer_to_load[before_layer_id].current_kv_size

    new_node = _build_node("rtx4090", model_info, id_suffix="-gap")
    allocator.join(new_node)

    after_mem = allocator.layer_to_load[before_layer_id].current_kv_size
    assert after_mem >= before_mem
    assert new_node.node_id in allocator.layer_to_load[before_layer_id].hosting_nodes

    allocator.leave(new_node.node_id)
    restored_mem = allocator.layer_to_load[before_layer_id].current_kv_size
    assert new_node.node_id not in allocator.layer_to_load[before_layer_id].hosting_nodes
    assert restored_mem == before_mem


@pytest.mark.parametrize(
    "num_layers,counts,expected_ranges,strategy",
    [
        # Six A100-80g: expect two pipelines, 12 each per stage in creation order
        (36, (6, 0, 0, 0), [(0, 12), (12, 24), (24, 36), (0, 12), (12, 24), (24, 36)], "greedy"),
        (36, (6, 0, 0, 0), [(0, 12), (12, 24), (24, 36), (0, 12), (12, 24), (24, 36)], "dp"),
        # 22 Layers, capacity (13, 13, 6, 6, 3, 3) -> greedy assigns (11, 11)
        (
            22,
            (2, 2, 0, 2),
            [
                (0, 11),
                (11, 22),
            ],
            "greedy",
        ),
        # For DP, we expect two pipelines, 13 each per stage in creation order
        (
            22,
            (2, 2, 0, 2),
            [
                (0, 13),
                (13, 19),
                (19, 22),
                (0, 13),
                (13, 19),
                (19, 22),
            ],
            "dp",
        ),
        # 14 Layers, capacity (13, 5, 5, 3, 3) -> greedy assigns (9, 5)
        (
            14,
            (1, 0, 2, 2),
            [
                (0, 9),
                (9, 14),
            ],
            "greedy",
        ),
        # 7 Layers, capacity (6, 5, 5, 3, 3) -> greedy assigns (5, 2, 5, 2)
        (
            7,
            (0, 1, 2, 2),
            [
                (0, 5),
                (5, 7),
                (0, 5),
                (5, 7),
            ],
            "greedy",
        ),
    ],
)
def test_allocator(
    num_layers: int,
    counts: tuple[int, int, int, int],
    expected_ranges: list[tuple[int, int]],
    strategy: Literal["greedy", "dp"],
):
    """Test allocator with greedy and dp strategies."""
    # pylint: disable=too-many-locals
    model = build_model_info(num_layers)

    n_a100_80g, n_a100_40g, n_5090, n_4090 = counts

    nodes: list[Node] = []
    for i in range(n_a100_80g):
        nodes.append(_build_node("a100-80g", model, id_suffix=f"-{i}"))
    for i in range(n_a100_40g):
        nodes.append(_build_node("a100-40g", model, id_suffix=f"-{i}"))
    for i in range(n_5090):
        nodes.append(_build_node("rtx5090", model, id_suffix=f"-{i}"))
    for i in range(n_4090):
        nodes.append(_build_node("rtx4090", model, id_suffix=f"-{i}"))

    allocator = (
        GreedyLayerAllocator(model, nodes)
        if strategy == "greedy"
        else DynamicProgrammingLayerAllocator(model, nodes)
    )
    allocator.initialize()
    _test_gap_patch_rebalance(allocator)

    # Collect (start,end) per node in creation order
    actual_ranges: list[tuple[int, int]] = []
    for node in nodes:
        if node.start_layer is None or node.end_layer is None:
            actual_ranges.append((0, 0))
        else:
            actual_ranges.append((node.start_layer, node.end_layer))

    # Trim trailing (0,0) if no assignment expected
    actual_trimmed = [r for r in actual_ranges if r != (0, 0)]
    expected_total = sum(e - s for (s, e) in expected_ranges)
    assert sum(e - s for (s, e) in actual_trimmed) == expected_total
    # Order-insensitive comparison: ranges represent stages; allow pipeline reordering
    assert Counter(actual_trimmed) == Counter(
        expected_ranges
    ), f"Stage ranges mismatch (order-insensitive):\nactual={actual_trimmed}\nexpected={expected_ranges}"
