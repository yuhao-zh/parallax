"""
Minimal tests for the Scheduler orchestrator.
"""

from __future__ import annotations

from scheduling.model_info import ModelInfo
from scheduling.node import Node, NodeHardwareInfo, RequestSignal
from scheduling.scheduler import Scheduler

from .test_utils import build_model_info


def _build_node(node_id: str, model: ModelInfo, *, tflops: float, mem_gb: float) -> Node:
    hw = NodeHardwareInfo(
        node_id=node_id,
        tflops_fp16=tflops,
        memory_gb=mem_gb,
        memory_bandwidth_gbps=1000.0,
    )
    n = Node(node_id=node_id, hardware=hw, model_info=model)
    # Ensure latency estimation uses a defined speedup
    setattr(n, "quantization_speedup", 1.0)
    return n


def test_scheduler_initialize_and_dispatch():
    """Allocate, then enqueue one request and dispatch it."""
    model = build_model_info(12)
    n1 = _build_node("a100-0", model, tflops=312.0, mem_gb=80.0)
    n2 = _build_node("a100-1", model, tflops=312.0, mem_gb=80.0)

    sched = Scheduler(model, [n1, n2], strategy="greedy", min_nodes_before_allocation=1)
    allocs = sched.list_node_allocations()
    assert allocs, "Allocator should assign at least one pipeline"
    # Check coverage equals total layers
    total = sum(e - s for _, s, e in allocs)
    assert total >= model.num_layers

    # Push a request and dispatch
    req = RequestSignal(request_id="req-1")
    sched.receive_request(req)
    assignment = sched.dispatch_next_request()
    assert assignment is not None
    req_id, path, latency = assignment
    assert req_id == req.request_id
    assert path, "Path should be non-empty"
    assert latency >= 0.0


def test_scheduler_join_and_leave():
    """New node can join and be assigned; leave removes it and may rebalance."""
    model = build_model_info(12)
    n1 = _build_node("a100-0", model, tflops=312.0, mem_gb=80.0)
    n2 = _build_node("a100-1", model, tflops=312.0, mem_gb=80.0)
    sched = Scheduler(model, [n1, n2], strategy="greedy", min_nodes_before_allocation=1)

    # Join a new node
    n3 = _build_node("rtx4090-x", model, tflops=82.6, mem_gb=24.0)
    sched.join(n3)
    assert n3.start_layer is not None and n3.end_layer is not None

    # Leave
    sched.leave(n3.node_id)
    assert n3 not in sched.nodes
