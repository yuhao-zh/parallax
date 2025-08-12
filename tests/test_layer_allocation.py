"""
Parametrized tests for water-filling rebalancer.

Covers:
- Real GPU setup (A100, RTX 5090, RTX 4090) with realistic FLOPS and RAM
- Mock "tight" setups where sum of capacities barely exceeds total layers
"""

from collections import Counter
from typing import Literal

import pytest

from parallax.scheduling.layer_allocation import (
    GreedyLayerAllocator,
    WaterFillingPipelineRebalancer,
)
from parallax.scheduling.model_info import ModelInfo
from parallax.scheduling.node import NodeInfo


def build_model_info(num_layers: int) -> ModelInfo:
    """Build GPT-OSS model with arbitrary number of layers, real model has 36."""
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
    gpu_type: Literal["a100-80g", "a100-40g", "rtx5090", "rtx4090"],
    model_info: ModelInfo,
    id_suffix: str = "",
) -> NodeInfo:
    """Build node info object for a given GPU type., assuming 50% of RAM is used to host params."""
    if gpu_type == "a100-80g":
        # Decoder Layer Capcity: 13; with embedding: 13
        return NodeInfo(
            node_id=f"a100-80g{id_suffix}", tflops_fp16=312.0, memory_gb=80.0, model_info=model_info
        )
    if gpu_type == "a100-40g":
        # Decoder Layer Capcity: 6; with embedding: 6
        return NodeInfo(
            node_id=f"a100-40g{id_suffix}", tflops_fp16=312.0, memory_gb=40.0, model_info=model_info
        )
    if gpu_type == "rtx5090":
        # Decoder Layer Capcity: 5; with embedding: 5
        return NodeInfo(
            node_id=f"rtx5090{id_suffix}", tflops_fp16=165.0, memory_gb=32.0, model_info=model_info
        )
    # 4090
    # Decoder Layer Capcity: 4; with embedding: 3
    return NodeInfo(
        node_id=f"rtx4090{id_suffix}", tflops_fp16=82.6, memory_gb=24.0, model_info=model_info
    )


def test_capacity_sanity_check():
    """Sanity check for capacity calculations."""
    # 36 layers for actual GPT-OSS 120B
    model = build_model_info(36)
    print(f"decoder layer flops: {model.decoder_layer_flops}")
    print(f"lm head flops: {model.lm_head_flops}")
    print(f"decoder layer io in GB: {model.decoder_layer_io_bytes(active=False) / (1024 ** 3)}")
    print(f"embedding table in GB: {model.embedding_io_bytes / (1024 ** 3)}")

    for gpu_type in ["a100-80g", "a100-40g", "rtx5090", "rtx4090"]:
        # (capacity, with embed) -> (13, 13), (6, 6), (5, 5), (4, 3)
        node = build_node_info(gpu_type, model)
        capacity = node.capacity_layers()
        print(f"{gpu_type} capacity layers: {capacity}")
        capacity_with_embed = node.capacity_layers(include_input_embed=True)
        print(f"{gpu_type} capacity layers with embedding: {capacity_with_embed}")
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

    # Build nodes based on gpu_types
    nodes = []
    for i, gpu_type in enumerate(gpu_types):
        node = build_node_info(gpu_type, model, id_suffix=f"-{i}")
        nodes.append(node)

    # Run rebalancer
    rebalancer = WaterFillingPipelineRebalancer(model.num_layers)
    plan = rebalancer.rebalance(nodes)

    # Extract assigned layers per node (in order of nodes)
    actual_layers = []
    for node in nodes:
        assignment = plan.node_assignments.get(node.node_id)
        assert assignment is not None, f"Node {node.node_id} not assigned"
        actual_layers.append(assignment.end_layer - assignment.start_layer)

    # Sum match
    assert (
        sum(actual_layers) == num_layers
    ), f"Total layers mismatch: {sum(actual_layers)} != {num_layers}"
    # Per-node layers match
    assert (
        actual_layers == expected_layers
    ), f"Per-node layers mismatch: {actual_layers} != {expected_layers}"
    # No node exceeds effective capacity (embedding on first, lm_head on last)
    for i, node in enumerate(nodes):
        cap = node.capacity_layers(
            include_input_embed=(i == 0),
            include_lm_head=(i == len(nodes) - 1),
        )
        assert actual_layers[i] <= cap, (
            f"Node {node.node_id} assigned {actual_layers[i]} > effective cap {cap} "
            f"(i={i}, first_has_embed={i==0}, last_has_lm={i==len(nodes)-1})"
        )


@pytest.mark.parametrize(
    "num_layers,counts,expected_ranges, strategy",
    [
        # Six A100-80g: expect two pipelines, 12 each per stage in creation order
        (36, (6, 0, 0, 0), [(0, 12), (12, 24), (24, 36), (0, 12), (12, 24), (24, 36)], "greedy"),
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
def test_allocator_expected_plan(
    num_layers: int,
    counts: tuple[int, int, int, int],
    expected_ranges: list[tuple[int, int]],
    strategy: Literal["greedy", "dynamic"],
):
    """Test allocator with greedy and dp strategies."""
    # pylint: disable=too-many-locals
    model = build_model_info(num_layers)

    n_a100_80g, n_a100_40g, n_5090, n_4090 = counts

    # Build nodes in capacity-desc order as specified
    nodes: list[NodeInfo] = []
    for i in range(n_a100_80g):
        nodes.append(build_node_info("a100-80g", model, id_suffix=f"-{i}"))
    for i in range(n_a100_40g):
        nodes.append(build_node_info("a100-40g", model, id_suffix=f"-{i}"))
    for i in range(n_5090):
        nodes.append(build_node_info("rtx5090", model, id_suffix=f"-{i}"))
    for i in range(n_4090):
        nodes.append(build_node_info("rtx4090", model, id_suffix=f"-{i}"))

    if strategy == "greedy":
        allocator = GreedyLayerAllocator(model, nodes)
    plan = allocator.allocate()

    # Collect (start,end) per node in creation order
    actual_ranges: list[tuple[int, int]] = []
    for node in nodes:
        asn = plan.node_assignments.get(node.node_id)
        if asn is None:
            actual_ranges.append((0, 0))
        else:
            actual_ranges.append((asn.start_layer, asn.end_layer))

    # Trim trailing (0,0) if no assignment expected
    actual_trimmed = [r for r in actual_ranges if r != (0, 0)]
    expected_total = sum(e - s for (s, e) in expected_ranges)
    assert sum(e - s for (s, e) in actual_trimmed) == expected_total
    # Order-insensitive comparison: ranges represent stages; allow pipeline reordering
    assert Counter(actual_trimmed) == Counter(
        expected_ranges
    ), f"Stage ranges mismatch (order-insensitive):\nactual={actual_trimmed}\nexpected={expected_ranges}"
