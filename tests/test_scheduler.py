# pylint: disable=use-dict-literal, protected-access

"""
Lightweight tests for the Scheduler orchestrator.

Covers:
- Allocation and node materialization
- Warm-up turning points and truncation (internal helper)
- Request queue management and shard-level routing
- Dynamic node add/remove with rebalance
"""

from __future__ import annotations

from parallax.scheduling.node import Node, NodeInfo
from parallax.scheduling.scheduler import Scheduler
from parallax.server.request import Request

from .test_utils import build_model_info, build_node_infos_by_counts, geo_rtt_provider


def test_scheduler_initialize_and_dispatch():
    """End-to-end: allocate, materialize, enqueue one request and dispatch it."""
    model = build_model_info(12)
    # node_infos = build_node_infos(model, counts=(2, 1, 0))
    node_infos = build_node_infos_by_counts(model, counts=dict(a100_80g=2, rtx5090=1, rtx4090=0))
    # positions = {ni.node_id: (idx * 1.0, 0.0) for idx, ni in enumerate(node_infos)}
    positions = {ni: (idx * 1.0, 0.0) for idx, ni in enumerate(node_infos)}
    sched = Scheduler(model, node_infos, strategy="dp", rtt_provider=geo_rtt_provider(positions))

    sched.initialize()
    assert sched.plan is not None
    assert sched.nodes, "Nodes should be materialized"

    # Push a request and dispatch
    req = Request()
    sched.add_request(req)
    assignment = sched.dispatch_next_request()
    assert assignment is not None
    req_id, path, latency = assignment
    assert req_id == req.request_id
    assert path, "Path should be non-empty"
    assert all(nid in sched.nodes for nid in path)
    assert latency >= 0.0


def test_scheduler_warmup_truncate_internal():
    """Warm-up should detect a turning point and truncate a slow head shard."""
    model = build_model_info(10)
    # Build two Node objects manually with overlapping ranges to force a turning
    slow_head = Node(
        node_id="head",
        node_info=NodeInfo(
            node_id="head",
            tflops_fp16=50.0,
            memory_gb=24.0,
            memory_bandwidth_gbps=10.0,
            model_info=model,
        ),
    )
    fast_tail = Node(
        node_id="tail",
        node_info=NodeInfo(
            node_id="tail",
            tflops_fp16=300.0,
            memory_gb=24.0,
            memory_bandwidth_gbps=100.0,
            model_info=model,
        ),
    )
    slow_head.set_layer_allocation(0, 6)
    fast_tail.set_layer_allocation(4, 10)

    positions = {"head": (0.0, 0.0), "tail": (0.2, 0.0)}
    # Attach RTT getters
    getter = geo_rtt_provider(positions)
    slow_head.rtt_getter = getter
    fast_tail.rtt_getter = getter

    sched = Scheduler(model, {}, strategy="dp", request_warm_up_for_reshard=1)
    # Inject our custom nodes and run warm-up truncation
    sched.nodes = {"head": slow_head, "tail": fast_tail}
    sched._run_warmup_and_truncate()  # internal helper; safe for tests

    assert slow_head.current_layers is not None
    start, end = slow_head.current_layers
    assert (start, end) == (0, 4), f"Head should truncate at turning layer: got {(start, end)}"


def test_scheduler_dynamic_add_remove_and_rebalance():
    """Adding/removing nodes should allow rebalancing and fresh materialization."""
    model = build_model_info(16)
    base_nodes = build_node_infos_by_counts(model, counts=dict(a100_80g=1, rtx5090=1, rtx4090=0))
    sched = Scheduler(model, base_nodes, strategy="greedy")
    sched.initialize()
    before_num_assignments = len(sched.plan.node_assignments) if sched.plan else 0

    # Add a new node and rebalance
    new_info = NodeInfo(node_id="rtx4090-extra", tflops_fp16=82.6, memory_gb=12.0, model_info=model)
    sched.add_node(new_info)
    sched.rebalance()
    after_add_assignments = len(sched.plan.node_assignments) if sched.plan else 0
    assert after_add_assignments >= before_num_assignments

    # Remove the node and rebalance
    sched.remove_node("rtx4090-extra")
    sched.rebalance()
    after_remove_assignments = len(sched.plan.node_assignments) if sched.plan else 0
    assert after_remove_assignments <= after_add_assignments
