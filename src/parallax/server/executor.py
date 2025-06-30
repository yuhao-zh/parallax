# pylint: disable=too-many-locals,too-few-public-methods,too-many-statements,too-many-branches, broad-exception-caught, pointless-string-statement,c-extension-no-member
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
    For now we process prefill and decode requests separately.
    Later when we have Ragged Paged Flash Attention kernel, we can process both in one batch.
6. Run model forward, our model will returned updated caches,
    kv cache manager will handle updating caches per layer;
7. Get the hidden-states from the model execution.
"""

import time
from typing import Any, Dict, List, Optional, Type

import mlx.core as mx
import zmq
from mlx import nn

from parallax.server.kv_cache import PagedKVCache
from parallax.server.request import InitialRequest, IntermediateRequest, Request
from parallax.server.scheduler import Scheduler
from parallax.server.shard_loader import MLXModelLoader
from parallax.utils.logging_config import get_logger
from parallax.utils.utils import (
    combine_padding_and_causal_masks,
    create_causal_mask,
    get_zmq_socket,
    pad_inputs,
)

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
        rpc_listen_addr: str = "ipc:///tmp/parallax_executor.ipc",
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
        self.model_shard, self.config, self.tokenizer = self.shard_loader.load(
            block_class=block_class
        )
        self.dtype = dtype
        self.num_key_value_heads = self.config.get("num_key_value_heads")
        self.head_dim = self.config.get("head_dim")

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
            is_first_peer=self.is_first_peer,
            tokenizer=self.tokenizer,
        )

        # KV Cache Manager
        self.kv_cache_manager = PagedKVCache(
            block_size=kv_block_size,
            num_kv_heads=self.num_key_value_heads,
            head_dim=self.head_dim,
            num_layers=self.num_shard_layers,
            dtype=self.dtype,
            kv_cache_memory_fraction=kv_cache_memory_fraction,
            max_tokens=kv_max_tokens_in_cache,
        )

        # Communication Related
        self.next_peer_addr = next_peer_addr
        self.first_peer_addr = first_peer_addr
        self.zmq_context = zmq.Context()
        self.recv_from_rpc = get_zmq_socket(self.zmq_context, zmq.PULL, rpc_listen_addr, bind=True)

        self.send_to_next_peer_socket: Optional[zmq.Socket] = None
        if self.next_peer_addr and not self.is_last_peer:
            self.send_to_next_peer_socket = get_zmq_socket(
                self.zmq_context, zmq.PUSH, self.next_peer_addr, bind=False
            )

        self.send_to_first_peer_socket: Optional[zmq.Socket] = None
        if self.first_peer_addr and self.is_last_peer:
            self.send_to_first_peer_socket = get_zmq_socket(
                self.zmq_context, zmq.PUSH, self.first_peer_addr, bind=False
            )

    def recv_requests_from_rpc(self) -> List[Request]:
        """Receives requests from the RPC server."""
        recv_reqs = []
        while True:
            try:
                recv_req = self.recv_from_rpc.recv_pyobj(zmq.NOBLOCK)
                # First peer is responsible for tokenization
                if self.is_first_peer and isinstance(recv_req, InitialRequest):
                    recv_req.input_ids = self.tokenizer.encode(recv_req.prompt)
                    recv_req.prompt_len = len(recv_req.input_ids)
                    recv_req.max_total_length = min(
                        recv_req.max_total_length, recv_req.prompt_len + recv_req.max_new_tokens
                    )

                recv_reqs.append(recv_req)
            except zmq.ZMQError:
                break
            except Exception as e:
                logger.error(f"Error receiving or deserializing request: {e}")
                break
        return recv_reqs

    def _prepare_prefill_batch(self, batched_requests: List[Request]) -> Dict[str, Any]:
        """Prepares inputs for ShardedModel from a batch of prefill requests."""
        batch_size = len(batched_requests)
        if batch_size == 0:
            return None

        h = []
        lengths = []
        for req in batched_requests:
            assert req.is_prefill, f"Request {req.request_id} is not a prefill request."
            if hasattr(req, "input_ids"):
                assert (
                    isinstance(req, InitialRequest) and self.is_first_peer
                ), f"Request {req.request_id} should be in FirstPeer."
                h.append(req.input_ids)
            else:
                assert isinstance(
                    req, IntermediateRequest
                ), f"Request {req.request_id} should not be in FirstPeer."
                h.append(req.hidden_states)
            lengths.append(req.total_length)

        if self.is_first_peer:
            padded_inputs, padding_mask = pad_inputs(self.pad_token_id, h, self.dtype)
        else:
            padded_inputs, padding_mask = pad_inputs(0, h, self.dtype)

        causal_mask = create_causal_mask(padded_inputs.shape[1])
        mask = combine_padding_and_causal_masks(padding_mask, causal_mask)

        return {
            "h_or_tokens": padded_inputs,
            "cache": None,
            "lengths": mx.array(lengths),
            "mask": mask,
            "requests": batched_requests,
        }

    def _prepare_decode_batch(self, batched_requests: List[Request]) -> Optional[Dict[str, Any]]:
        """Prepares inputs for ShardedModel from a batch of decode requests."""
        batch_size = len(batched_requests)
        if batch_size == 0:
            return None

        h_list = []

        for req in batched_requests:
            assert req.is_decoding, f"Request {req.request_id} is not a decode request."

            if self.is_first_peer:
                assert isinstance(req, InitialRequest)
                # First peer input is the last generated token
                h_list.append([req.output_ids[-1]])
            else:
                assert isinstance(req, IntermediateRequest)
                assert req.hidden_states is not None and req.hidden_states.shape[0] == 1
                h_list.append(req.hidden_states)

        padded_inputs, _ = pad_inputs(0, h_list)

        return {
            "h_or_tokens": padded_inputs,
            "requests": batched_requests,
            "cache": None,
            "lengths": None,
            "mask": None,
        }

    def _prepare_batch_inputs(self, batched_requests: List[Request]) -> Optional[Dict[str, Any]]:
        """Prepares inputs for ShardedModel from a batch of requests.
        Args:
            batched_requests: A list of requests to prepare inputs for.

        Returns:
            A dictionary containing the prepared inputs for the ShardedModel.
            The dictionary contains "prefill_batch" and "decode_batch",
            with the prepared inputs for the corresponding request type.

            For now we process prefill and decode requests separately.
            Later when we have Ragged Paged Flash Attention kernel,
            we can process both in one batch.
        """
        if len(batched_requests) == 0:
            return None

        prefill_reqs = []
        decode_reqs = []
        for req in batched_requests:
            if req.is_prefill:
                prefill_reqs.append(req)
            elif req.is_decoding:
                decode_reqs.append(req)
        prefill_batch = self._prepare_prefill_batch(prefill_reqs)
        decode_batch = self._prepare_decode_batch(decode_reqs)
        if prefill_batch is None and decode_batch is None:
            return None
        return {
            "prefill_batch": prefill_batch,
            "decode_batch": decode_batch,
        }

    def _handle_input_requests(self, requests: List[Request]):
        """Update requests states and status in scheduler and cache manager."""
        if not requests:
            return

        if self.is_first_peer:
            # First peer can receive InitialRequests from the client RPC,
            # or IntermediateRequests from the last peer.
            for req in requests:
                if isinstance(req, InitialRequest):
                    if not self.kv_cache_manager.has_request(req.request_id):
                        self.kv_cache_manager.add_request(req, req.total_length)
                    self.scheduler.enque_request(req)
                elif isinstance(req, IntermediateRequest):
                    original_req = self.scheduler.get_running_request(req.request_id)
                    if original_req is None:
                        raise ValueError(
                            f"Recieved Request {req.request_id} should be in request pool"
                        )
                    if not self.kv_cache_manager.has_request(req.request_id):
                        raise ValueError(
                            f"Recieved Request {req.request_id} should be in cache manager"
                        )

                    assert req.hidden_states is not None
                    token_id = int(req.hidden_states[0])
                    original_req.commit_new_token(token_id)

                    # Check for termination.
                    if self.scheduler.check_and_update_request_status(original_req):
                        logger.info(f"Releasing resources for finished request {req.request_id}")
                        self.kv_cache_manager.release_request(original_req.request_id)
                    else:
                        self.scheduler.enque_request(original_req)
                else:
                    raise TypeError(f"First peer received unexpected request type: {type(req)}")

        else:
            # Intermediate and Last peers receive IntermediateRequests from the previous peer.
            for req in requests:
                assert isinstance(
                    req, IntermediateRequest
                ), "Non-first peers must receive IntermediateRequests."
                if req.is_finished or req.hidden_states is None:
                    logger.info(f"Releasing resources for finished request {req.request_id}")
                    self.kv_cache_manager.release_request(req.request_id)
                    self.scheduler.evict_request(req.request_id, req.status)
                else:
                    # This is an active request, add it to the scheduler queue to be processed.
                    self.scheduler.enque_request(req)
                    if not self.kv_cache_manager.has_request(req.request_id):
                        self.kv_cache_manager.add_request(req, req.total_length)

    def _prepare_next_single_request(self, request: Request, hidden_states: mx.array) -> Request:
        """Handle request state changes both inter and intra peers.

        This function prepares the request object to be sent to the *next* peer in the
        pipeline, or back to the first peer if this is the last peer.

        Args:
            request: The request that was just processed by this peer.
            hidden_states: The output hidden_states/output_ids from the model for this request.

        Returns:
            A new Request object ready to be sent to the next destination.
        """
        if self.is_last_peer:
            # Last peer decodes a token and sends it back to the first peer.
            # The token is wrapped in an IntermediateRequest.
            assert isinstance(
                request, IntermediateRequest
            ), "Last peer must receive an IntermediateRequest."
            assert hidden_states.dtype == mx.uint32, "Last peer must receive an output_id."
            return IntermediateRequest(
                request_id=request.request_id,
                status=request.status,  # Status remains DECODING or PREFILLING
                current_position=request.total_length + 1,
                hidden_states=hidden_states,
            )

        # This peer is the first or an intermediate peer.
        if self.is_first_peer:
            # First peer converts its InitialRequest to an IntermediateRequest.
            assert isinstance(request, InitialRequest), "First peer must process an InitialRequest."
            if request.is_finished:
                hidden_states = None
            return IntermediateRequest.from_initial_request(request, hidden_states=hidden_states)
        # Intermediate peer passes along an updated IntermediateRequest.
        assert isinstance(
            request, IntermediateRequest
        ), "Intermediate peer must process an IntermediateRequest."
        return IntermediateRequest.from_intermediate_request(request, hidden_states)

    def _prepare_next_batch_requests(
        self, requests: List[Request], hidden_states: mx.array, lengths: mx.array
    ) -> List[Request]:
        """Prepares a batch of requests for the next stage of the pipeline."""
        batched_requests = []
        for i, src_request in enumerate(requests):
            if self.is_last_peer:
                # Last peer gets a 1D array of token IDs
                hidden_state_for_req = hidden_states[i : i + 1]
            else:
                # Other peers get a 3D array of hidden states
                true_length = int(lengths[i])
                hidden_state_for_req = hidden_states[i, :true_length, :]

            next_req = self._prepare_next_single_request(src_request, hidden_state_for_req)
            batched_requests.append(next_req)

        return batched_requests

    def process_batch(
        self, prepared_inputs: Dict[str, Any], return_decoded_tokens: bool = True
    ) -> mx.array:
        """
        Process a batch of requests.

        Args:
            prepared_inputs: A dictionary containing the prepared inputs for the ShardedModel.
            return_decoded_tokens: Whether to return decoded tokens.

        Returns:
            A tensor of shape (B, L, D) containing the hidden states for the next peer.
            or (B,) containing the decoded tokens.
        """

        # Run model and get updated cache
        hidden_states, (k_caches, v_caches) = self.model_shard(
            h_or_tokens=prepared_inputs["h_or_tokens"],
            cache=prepared_inputs["cache"],
            lengths=prepared_inputs["lengths"],
            mask=prepared_inputs["mask"],
            requests=prepared_inputs["requests"],
            kv_cache_manager=self.kv_cache_manager,
        )
        return_decoded_tokens = return_decoded_tokens and self.is_last_peer
        # k_caches shape: (num_layers, B, num_kv_heads, L_padded, head_dim)
        logger.debug(
            f"Processed batch with {len(prepared_inputs['requests'])} requests, "
            f"request status: {prepared_inputs['requests'][0].status}, "
            f"hidden_states shape: {hidden_states.shape}, "
            f"k_caches shape: {k_caches.shape}, "
            f"v_caches shape: {v_caches.shape}"
        )

        lengths = []
        for i, req in enumerate(prepared_inputs["requests"]):
            # Slice the KV updates for the current request
            k_update_for_req = k_caches[:, i]
            v_update_for_req = v_caches[:, i]

            num_tokens_to_update = req.total_length
            if req.is_prefill:
                token_indices = list(range(num_tokens_to_update))
                k_update_sliced = k_update_for_req[:, :, :num_tokens_to_update, :]
                v_update_sliced = v_update_for_req[:, :, :num_tokens_to_update, :]
                lengths.append(prepared_inputs["lengths"][i])
                # Transpose from: (num_layers, num_kv_heads, L_padded, head_dim)
                # to (n_layers, L_true, n_kv_h, h_dim)
                k_to_write = k_update_sliced.transpose(0, 2, 1, 3)
                v_to_write = v_update_sliced.transpose(0, 2, 1, 3)

                self.kv_cache_manager.update_kv_cache(
                    req.request_id, token_indices, k_to_write, v_to_write
                )

            elif req.is_decoding:
                lengths.append(1)
            else:
                continue

                # Process last peer: need additional sampling + detokenization
        if return_decoded_tokens:
            return mx.array(self.model_shard.logits_to_tokens(hidden_states, mx.array(lengths)))

        return hidden_states

    def run_loop(self):
        # pylint: disable=too-many-nested-blocks
        """The main loop of the executor."""
        logger.info(
            f"Executor for layers {self.start_layer}-{self.end_layer-1} starting run loop..."
        )
        while True:
            # 1. Ingest new requests from the RPC server
            incoming_requests = self.recv_requests_from_rpc()
            self._handle_input_requests(incoming_requests)

            # 2. Check if we should form a batch
            if not self.scheduler.should_dispatch():
                time.sleep(0.01)  # prevent busy waiting
                continue

            # 3. Form a batch from the scheduler's queue
            batch_to_process = self.scheduler.form_batch()
            if not batch_to_process:
                continue

            # 4. Process the batch
            try:
                prepared_inputs_dict = self._prepare_batch_inputs(batch_to_process)

                # We will process prefill and decode batches separately for now
                for batch_type in ["prefill_batch", "decode_batch"]:
                    if prepared_inputs_dict and prepared_inputs_dict.get(batch_type):
                        prepared_inputs = prepared_inputs_dict[batch_type]

                        output = self.process_batch(
                            prepared_inputs, return_decoded_tokens=self.is_last_peer
                        )

                        # 5. Prepare requests for the next stage in the pipeline
                        next_batch = self._prepare_next_batch_requests(
                            requests=prepared_inputs["requests"],
                            hidden_states=output,
                            lengths=prepared_inputs["lengths"],
                        )

                        # 6. Dispatch to the appropriate destination
                        if self.is_last_peer:
                            # Last peer sends feedback to the first peer
                            if self.send_to_first_peer_socket:
                                for req in next_batch:
                                    self.send_to_first_peer_socket.send_pyobj(req)
                            # If this is a single-node setup, handle it locally
                            if self.is_first_peer:
                                self._handle_input_requests(next_batch)
                        else:
                            # First or intermediate peer sends to the next peer
                            if self.send_to_next_peer_socket:
                                for req in next_batch:
                                    self.send_to_next_peer_socket.send_pyobj(req)

            except Exception as e:
                logger.exception(f"Error processing batch: {e}")
                # Naive error handling: release and evict all requests in the batch
                for req in batch_to_process:
                    self.kv_cache_manager.release_request(req.request_id)
                    self.scheduler.evict_request(req.request_id, req.status)

    def shutdown(self):
        """Shuts down the executor."""
        logger.info("Executor shutting down...")
        self.recv_from_rpc.close()
        if self.send_to_next_peer_socket:
            self.send_to_next_peer_socket.close()
        if self.send_to_first_peer_socket:
            self.send_to_first_peer_socket.close()
        self.zmq_context.term()
        logger.info("Executor shutdown complete.")


if __name__ == "__main__":
    # check test_executor.py for exmample
    logger.info("Executor conceptual example finished (not actually run).")
