"""
High-level executor for managing model shards, scheduler, and cache pool on each Peer.

Executor handles
1. Loading model shards from the repository;
2. Instantiate scheduler, kv cache manager;
3. Handles tokenization / detokenization if needed;
4. Keep listening to RPC to get requests, feed these to scheduler's request pool;
5. Get batched requests from the scheduler,
    - prepare the MLX tensor input
    - rebuild KV cache
    - feed to model runner;
6. Run model forward, our model will returned updated caches, kv cache manager will handle updating caches per layer;
7. Get the hidden-states from the model execution.
"""

# pylint: disable=too-many-locals,too-many-statements,too-many-branches
import time
from typing import Dict, List, Optional, Tuple, Type

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import zmq

from parallax.server.kv_cache import PagedKVCache
from parallax.server.model import ShardedModel
from parallax.server.request import (
    InitialRequest,
    IntermediateRequest,
    Request,
    RequestStatus,
)
from parallax.server.scheduler import Scheduler
from parallax.server.shard_loader import MLXModelLoader
from parallax.utils.logging_config import get_logger
from parallax.utils.utils import get_zmq_socket, pad_inputs

logger = get_logger(__name__)


class Executor:
    """High-level executor for managing model shards, scheduler, and cache pool on each Peer."""

    def __init__(
        self,
        model_repo: str,
        start_layer: int,
        end_layer: int,
        total_model_layers: int,
        block_class: Type[nn.Module],
        dtype: mx.Dtype = mx.float16,
        # Scheduler Configs
        max_num_tokens_in_batch: int = 1024,
        prefill_priority: int = 0,
        micro_batch_ratio: int = 2,
        scheduler_wait_ms: int = 500,
        # KV Cache Configs
        max_batch_size: int = 16,
        kv_block_size: int = 16,
        kv_cache_memory_fraction: float = 0.8,
        kv_max_tokens_in_cache: Optional[int] = None,
        # ZMQ Configs
        rpc_listen_addr: str = "ipc:///tmp/parallax_executor_listen",
        next_peer_addr: Optional[str] = None,
        first_peer_addr: Optional[str] = None,
    ):
        self.is_first_peer = start_layer == 0
        self.is_last_peer = end_layer == total_model_layers
        self.start_layer = start_layer
        self.end_layer = end_layer
        self.num_shard_layers = end_layer - start_layer

        # Sharded Model
        self.shard_loader = MLXModelLoader(model_repo, start_layer=start_layer, end_layer=end_layer)
        self.config, self.tokenizer = self.shard_loader.load_config_and_tokenizer()
        self.model_shard = ShardedModel(
            config=self.config,
            model_id=model_repo,
            start_layer=start_layer,
        )
        self.dtype = dtype

        if self.tokenizer.pad_token_id is None:
            self.pad_token_id = self.tokenizer.eos_token_id
        else:
            self.pad_token_id = self.tokenizer.pad_token_id

        self.eos_token_id = self.tokenizer.eos_token_id
        if isinstance(self.eos_token_id, list):
            self.eos_token_id = self.eos_token_id[0]

        # Scheduler
        self.scheduler = Scheduler(
            max_batch_size=max_batch_size,
            max_num_tokens=max_num_tokens_in_batch,
            prefill_priority=prefill_priority,
            scheduler_wait_ms=scheduler_wait_ms,
            micro_batch_ratio=micro_batch_ratio,
        )

        # KV Cache Manager
        self.kv_cache_manager = PagedKVCache(
            block_size=kv_block_size,
            num_kv_heads=self.config.num_key_value_heads,
            head_dim=self.config.head_dim,
            num_layers=self.num_shard_layers,
            dtype=self.dtype,
            kv_cache_memory_fraction=kv_cache_memory_fraction,
            max_tokens=kv_max_tokens_in_cache,
        )

        # Communication Related
        self.next_peer_addr = next_peer_addr
        self.first_peer_addr = first_peer_addr
        self.zmq_context = zmq.Context(2)
        self.recv_from_rpc = get_zmq_socket(
            self.zmq_context, zmq.DEALER, rpc_listen_addr, False, bind=True
        )

        self.send_to_next_peer_socket: Optional[zmq.Socket] = None
        if self.next_peer_addr and not self.is_last_peer:
            self.send_to_next_peer_socket = get_zmq_socket(
                self.zmq_context, zmq.DEALER, self.next_peer_addr, True, bind=False
            )

        self.send_to_first_peer_socket: Optional[zmq.Socket] = None
        if self.first_peer_addr and self.is_last_peer:
            self.send_to_first_peer_socket = get_zmq_socket(
                self.zmq_context, zmq.DEALER, self.first_peer_addr, True, bind=False
            )

        logger.info(
            f"Executor initialized. First Peer: {self.is_first_peer}, Last Peer: {self.is_last_peer}"
        )

    def recv_requests_from_rpc(self) -> List[Request]:
        """
        Receives requests from the RPC server.
        """
        recv_reqs = []
        while True:
            try:
                recv_req = self.recv_from_rpc.recv_pyobj(zmq.NOBLOCK)
                recv_reqs.append(recv_req)
            except zmq.ZMQError:
                break
            except Exception as e:
                logger.error(f"Error receiving or deserializing request: {e}")
                break
        return recv_reqs

    def _prepare_prefill_batch(self, batched_requests: List[Request]) -> List[Request]:
        """Prepares inputs for ShardedModel from a batch of requests."""
        batch_size = len(batched_requests)
        if batch_size == 0:
            return None

        h = []
        lengths = []
        for req in batched_requests:
            assert req.is_prefill, f"Request {req.request_id} is not a prefill request."
            h.append(req.input_ids)
            lengths.append(req.input_length)

        padded_inputs, mask = pad_inputs(self.pad_token_id, h)

        return {
            "h_or_tokens": padded_inputs,
            "cache": None,
            "lengths": mx.array(lengths),
            "mask": mask,
        }

    def _prepare_decode_batch(self, batched_requests: List[Request]) -> List[Request]:
        """Prepares inputs for ShardedModel from a batch of requests."""
        batch_size = len(batched_requests)
        if batch_size == 0:
            return None

        h = []
        lengths = []
        k_cache = []
        v_cache = []
        for req in batched_requests:
            assert req.is_decoding, f"Request {req.request_id} is not a decode request."
            h.append(req.hidden_states)
            lengths.append(req.total_length)
            kv_cache = self.kv_cache_manager.gather_kv_cache(req.request_id)
            k_cache.append(kv_cache[0])
            v_cache.append(kv_cache[1])

        padded_inputs, mask = pad_inputs(self.pad_token_id, h)
        # pad kv cache

        raise NotImplementedError("Not implemented.")

    def _prepare_batch_inputs(self, batched_requests: List[Request]) -> Optional[Dict[str, Any]]:
        """Prepares inputs for ShardedModel from a batch of requests."""
        raise NotImplementedError("Not implemented.")

    def _process_batch(self, prepared_inputs: Dict[str, Any]):
        raise NotImplementedError("Not implemented.")

    def run_loop(self):
        logger.info(
            f"Executor for layers {self.start_layer}-{self.end_layer-1} starting run loop..."
        )
        while True:
            incoming_requests = self.recv_requests_from_rpc()
            for req in incoming_requests:
                self.scheduler.enque_request(req)

            if not self.scheduler.should_dispatch():
                continue

            batch_to_process = self.scheduler.form_batch()
            if not batch_to_process:
                continue

            try:
                prepared_inputs = self._prepare_batch_inputs(batch_to_process)
                if prepared_inputs:
                    self._process_batch_v2(prepared_inputs)
            except Exception as e:
                logger.exception(f"Error processing batch: {e}")
                for req in batch_to_process:  # Mark requests in the failed batch as errored
                    if req.request_id in self.scheduler._running_requests:
                        failed_req_obj = self.scheduler._running_requests[req.request_id]
                        if isinstance(failed_req_obj, InitialRequest):
                            failed_req_obj.update_status(RequestStatus.ERROR)
                    if req.request_id in self.kv_cache_manager._sequences:
                        self.kv_cache_manager.release_request(req.request_id)
                    # Clean from scheduler's running state as well if appropriate
                    self.scheduler._running_requests.pop(req.request_id, None)

    def shutdown(self):
        logger.info("Executor shutting down...")
        self.recv_from_rpc.close()
        if self.send_to_next_peer_socket:
            self.send_to_next_peer_socket.close()
        if self.send_to_first_peer_socket:
            self.send_to_first_peer_socket.close()
        self.zmq_context.term()
        logger.info("Executor shutdown complete.")


if __name__ == "__main__":
    logger.info("Executor conceptual example finished (not actually run).")
