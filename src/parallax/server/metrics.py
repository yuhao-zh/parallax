"""
Thread-safe, in-process metrics registry for executor-node telemetry.

Exposes functions to update and retrieve per-node metrics that are consumed by
the P2P server announcements (e.g., current_requests, layer_latency_ms).
"""

from __future__ import annotations

import threading
import time
from typing import Any, Dict, Optional

_lock = threading.Lock()
_metrics: Dict[str, Any] = {
    "current_requests": 0,
    "layer_latency_ms": None,  # Exponentially smoothed per-layer latency
    "_last_update_ts": 0.0,
}


def update_metrics(
    *,
    current_requests: Optional[int] = None,
    layer_latency_ms_sample: Optional[float] = None,
    ewma_alpha: float = 0.2,
) -> None:
    """Update metrics with optional fields and EWMA smoothing for latency.

    Args:
        current_requests: Number of in-flight requests on this node.
        layer_latency_ms_sample: A new sample of per-layer latency in ms.
        ewma_alpha: Smoothing factor in [0, 1] for latency EWMA.
    """
    global _metrics
    with _lock:
        if current_requests is not None:
            _metrics["current_requests"] = int(current_requests)
        if layer_latency_ms_sample is not None:
            prev = _metrics.get("layer_latency_ms")
            if prev is None:
                _metrics["layer_latency_ms"] = float(layer_latency_ms_sample)
            else:
                _metrics["layer_latency_ms"] = float(
                    (1.0 - ewma_alpha) * float(prev) + ewma_alpha * float(layer_latency_ms_sample)
                )
        _metrics["_last_update_ts"] = time.time()


def get_metrics() -> Dict[str, Any]:
    """Return a shallow copy of current metrics suitable for JSON serialization."""
    with _lock:
        return dict(_metrics)
