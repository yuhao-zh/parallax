# pylint: disable=too-many-locals,too-few-public-methods,too-many-statements,too-many-branches, broad-exception-caught, pointless-string-statement
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

import argparse
import time
import zmq
import uuid
from typing import Any, Dict, List, Optional

import mlx.core as mx
from mlx_lm.server import process_message_content, convert_chat

from parallax.p2p.message_util import proto_to_request, request_to_proto
from parallax.p2p.proto import forward_pb2
from parallax.server.kv_cache import KVCacheManager
from parallax.server.request import (
    InitialRequest,
    IntermediateRequest,
    Request,
    RequestStatus,
)
from parallax.server.scheduler import Scheduler
from parallax.server.shard_loader import MLXModelLoader
from parallax.server.sampling.sampling_params import SamplingParams
from parallax.server.sampling.sampler import SamplingBatchInfo
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
        # Model Configs
        model_repo: str,
        start_layer: int,
        end_layer: int,
        dtype: mx.Dtype = mx.float16,
        # Scheduler Configs
        max_batch_size: int = 16,
        max_num_tokens_in_batch: int = 1024,
        prefill_priority: int = 0,
        micro_batch_ratio: int = 2,
        scheduler_wait_ms: int = 500,
        # KV Cache Configs
        kv_block_size: int = 64,
        kv_cache_memory_fraction: float = 0.8,
        kv_max_tokens_in_cache: Optional[int] = None,
        # P2P Communication Configs
        send_to_peer_addr: Optional[str] = None,
        recv_from_peer_addr: Optional[str] = None,
        # IPC Communication Configs
        executor_input_ipc_addr: Optional[str] = None,
        executor_output_ipc_addr: Optional[str] = None,
    ):
        # Sharded Model
        self.shard_loader = MLXModelLoader(model_repo, start_layer=start_layer, end_layer=end_layer)
        self.model_shard, self.config, self.tokenizer = self.shard_loader.load()

        self.start_layer = start_layer
        self.end_layer = end_layer
        self.is_first_peer = start_layer == 0
        self.is_last_peer = end_layer == self.config.get("num_hidden_layers")
        self.num_shard_layers = end_layer - start_layer

        self.dtype = dtype
        self.num_key_value_heads = self.config.get("num_key_value_heads")
        self.head_dim = self.config.get("head_dim") or self.config.get(
            "hidden_size"
        ) // self.config.get("num_attention_heads")

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
        self.kv_cache_manager = KVCacheManager(
            block_size=kv_block_size,
            num_kv_heads=self.num_key_value_heads,
            head_dim=self.head_dim,
            num_layers=self.num_shard_layers,
            dtype=self.dtype,
            cache_memory_fraction=kv_cache_memory_fraction,
            max_num_tokens=kv_max_tokens_in_cache,
        )

        # Communication Related
        self.zmq_context = zmq.Context()
        if recv_from_peer_addr:
            self.recv_from_peer_socket = get_zmq_socket(
                self.zmq_context, zmq.PULL, recv_from_peer_addr, bind=False
            )
        if send_to_peer_addr:
            self.send_to_peer_socket = get_zmq_socket(
                self.zmq_context, zmq.PUSH, send_to_peer_addr, bind=False
            )
        if executor_input_ipc_addr:
            self.recv_from_ipc_socket = get_zmq_socket(
                self.zmq_context, zmq.PULL, executor_input_ipc_addr, bind=False
            )
        if executor_output_ipc_addr:
            self.send_to_ipc_socket = get_zmq_socket(
                self.zmq_context, zmq.PUSH, executor_output_ipc_addr, bind=False
            )

    @classmethod
    def create_from_args(cls, args: argparse.Namespace):
        """Create executor from command line arguments."""
        return cls(**create_executor_config(args))
    
    def recv_requests_from_http(self) -> List[Request]:
        recv_reqs = []
        while True:
            try:
                raw_request = self.recv_from_ipc_socket.recv_pyobj(zmq.NOBLOCK)
                # Do tokenization and form InitialRequest
                req = self._handle_raw_request(raw_request)
                recv_reqs.append(req)
            except zmq.ZMQError:
                break
            except Exception as e:
                logger.exception(f"Error receiving http request: {e}")
        return recv_reqs
            

    def recv_requests_from_peer(self) -> List[Request]:
        """Receives requests from the RPC server."""
        recv_reqs = []
        while True:
            try:
                recv_req = self.recv_from_peer_socket.recv_multipart(zmq.NOBLOCK)
                if recv_req[0] == b"forward":
                    # Create a new ForwardRequest instance and parse from bytes
                    forward_request = forward_pb2.ForwardRequest()
                    forward_request.ParseFromString(recv_req[1])
                    recv_req = proto_to_request(forward_request)
                    recv_reqs.extend(recv_req)
                elif recv_req[0] == b"abort":
                    # TODO: handle abort request
                    pass
                else:
                    raise ValueError(f"Unknown request type: {recv_req[0]}")
                # First peer is responsible for tokenization
                # if self.is_first_peer and isinstance(recv_req, InitialRequest):
                #     recv_req.input_ids = self.tokenizer.encode(recv_req.prompt)
                #     recv_req.prompt_len = len(recv_req.input_ids)
                #     recv_req.max_total_length = min(
                #         recv_req.max_total_length, recv_req.prompt_len + recv_req.max_new_tokens
                #     )

            except zmq.ZMQError:
                break
            except Exception as e:
                logger.exception(f"Error receiving or deserializing request: {e}")
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
            if self.is_first_peer:
                assert (
                    hasattr(req, "input_ids")
                ), f"Request {req.request_id} should has attribute input_ids in FirstPeer."
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
        cache_lengths = []
        kv_cache_list = []

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

            num_tokens_in_cache = self.kv_cache_manager.request_length(req.request_id)
            cache_lengths.append(num_tokens_in_cache)
            kv_cache = self.kv_cache_manager.gather_kv_cache(req.request_id)
            kv_cache_list.append(kv_cache)

        padded_inputs, _ = pad_inputs(0, h_list)

        if not kv_cache_list:
            raise ValueError("No KV cache found for request.")

        k_caches = [kv[0] for kv in kv_cache_list]
        v_caches = [kv[1] for kv in kv_cache_list]

        k_batched, k_padding_mask = pad_inputs(0, k_caches, self.dtype)
        v_batched, _ = pad_inputs(0, v_caches, self.dtype)

        # The mask from padding K is for the PAST tokens. It has shape (B, 1, 1, source_len_padded).
        # We need to add a '1' for the CURRENT token so the final mask can be broadcast
        # to the attention weights of shape (B, n_heads, 1, source_len_padded + 1).
        ones_for_current_token = mx.ones((k_padding_mask.shape[0], 1, 1, 1), dtype=self.dtype)
        final_padding_mask = mx.concatenate([k_padding_mask, ones_for_current_token], axis=3)
        attention_mask = (1.0 - final_padding_mask) * -1e9

        model_lengths = mx.array([kv[0].shape[2] for kv in kv_cache_list])

        return {
            "h_or_tokens": padded_inputs,
            "cache": (k_batched, v_batched),
            "lengths": model_lengths,
            "mask": attention_mask,
            "requests": batched_requests,
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
    
    def _handle_raw_request(self, raw_request: Dict):
        assert "messages" in raw_request, "Request did not contain messages"

        rid = raw_request["rid"]
        if self.tokenizer.chat_template:
            messages = raw_request["messages"]
            process_message_content(messages)
            prompt = self.tokenizer.apply_chat_template(
                messages,
                raw_request.get("tools") or None,
                add_generation_prompt=True,
                # **self.model_provider.cli_args.chat_template_args,  # TODO: add chat template
            )
        else:
            prompt = convert_chat(raw_request["messages"], raw_request.get("role_mapping"))
            prompt = self.tokenizer.encode(prompt)

        max_new_tokens = raw_request.get("max_tokens")
        if max_new_tokens is None:
            max_new_tokens = 2048
        max_total_length = len(prompt) + max_new_tokens

        raw_sampling_params = raw_request.get("sampling_params")
        if raw_sampling_params is None:
            sampling_params = SamplingParams()
        else:
            # TODO
            sampling_params = SamplingParams()
        
        req = InitialRequest(
            request_id=rid,
            output_ids=None,
            input_ids=prompt,
            sampling_params=sampling_params,
            max_new_tokens=max_new_tokens,
            max_total_length=max_total_length,
        )
        return req

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

                    assert req.next_token_id is not None
                    original_req.commit_new_token(req.next_token_id)

                    # detokenize and send to http server
                    cur_text = self.tokenizer.decode(req.next_token_id)
                    req_dict = {
                        "output": cur_text,
                        "rid": req.request_id,
                    }
                    if req.next_token_id == self.tokenizer.eos_token_id:
                        req_dict["eos"] = True
                    self.send_to_ipc_socket.send_pyobj(req_dict)

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
                    self.kv_cache_manager.release_request(req.request_id)
                    logger.info(
                        f"Released resources for finished request {req.request_id}, "
                        f"kv cache manager has {self.kv_cache_manager.tokens_in_cache} tokens, "
                        f"memory usage: {mx.get_active_memory() / 1024**3 :.3f} GB"
                    )
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
            next_token_id = int(hidden_states[0])
            return IntermediateRequest(
                request_id=request.request_id,
                status=RequestStatus.DECODING,  # Last peer always changes status to DECODING
                current_position=request.total_length + 1,
                input_ids=request.input_ids,
                hidden_states=hidden_states,
                next_token_id=next_token_id,
                routing_table=request.routing_table,
            )

        # This peer is the first or an intermediate peer.
        if self.is_first_peer:
            assert isinstance(request, InitialRequest), "First peer must process an InitialRequest."
            if request.is_finished:
                hidden_states = None
            return IntermediateRequest.from_initial_request(request, hidden_states=hidden_states)
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

        lengths = mx.zeros((len(prepared_inputs["requests"]),), dtype=mx.int32)
        requests = prepared_inputs["requests"]
        for i, req in enumerate(requests):
            if req.is_prefill:
                lengths[i] = prepared_inputs["lengths"][i]
            elif req.is_decoding:
                lengths[i] = 1
            else:
                continue
        self.kv_cache_manager.update_requests(requests, k_caches, v_caches, lengths)

        # Process last peer: need additional sampling + detokenization
        if return_decoded_tokens:
            sampling_info = SamplingBatchInfo.from_reqs(requests)
            return mx.array(self.model_shard.logits_to_tokens(hidden_states, lengths, sampling_info))

        return hidden_states

    def run_loop(self):
        # pylint: disable=too-many-nested-blocks
        """The main loop of the executor."""
        logger.info(
            f"Executor for layers [{self.start_layer}, {self.end_layer}) starting run loop..."
        )
        mx.set_wired_limit(mx.metal.device_info()["max_recommended_working_set_size"])
        while True:
            # 1. Ingest new requests from the http frontend
            if self.is_first_peer:
                time.sleep(0.1)
                http_requests = self.recv_requests_from_http()
                self._handle_input_requests(http_requests)

            # 2. Ingest new requests from the RPC server
            incoming_requests = self.recv_requests_from_peer()
            self._handle_input_requests(incoming_requests)

            # 3. Check if we should form a batch
            if not self.scheduler.should_dispatch():
                time.sleep(0.01)  # prevent busy waiting
                continue

            # 4. Form a batch from the scheduler's queue
            batch_to_process = self.scheduler.form_batch()
            if not batch_to_process:
                continue

            # 5. Process the batch
            try:
                prepared_inputs_dict = self._prepare_batch_inputs(batch_to_process)

                # We will process prefill and decode batches separately for now
                for batch_type in ["prefill_batch", "decode_batch"]:
                    if prepared_inputs_dict and prepared_inputs_dict.get(batch_type):
                        prepared_inputs = prepared_inputs_dict[batch_type]

                        start_time = time.time()
                        output = self.process_batch(
                            prepared_inputs, return_decoded_tokens=self.is_last_peer
                        )
                        # 6. Prepare requests for the next stage in the pipeline
                        next_batch = self._prepare_next_batch_requests(
                            requests=prepared_inputs["requests"],
                            hidden_states=output,
                            lengths=prepared_inputs["lengths"],
                        )

                        # 7. Dispatch to the appropriate destination
                        if self.is_last_peer and self.is_first_peer:
                            # Single node: handle locally
                            self._handle_input_requests(next_batch)
                        else:
                            # Send output to next peer
                            self.send_to_peer_socket.send_multipart(
                                [b"forward", request_to_proto(next_batch).SerializeToString()]
                            )
                            logger.info(
                                f"Processed batch of type {batch_type} with {len(next_batch)} requests "
                                f"in {(time.time() - start_time) * 1000:.3f} ms"
                            )

            except Exception as e:
                logger.exception(f"Error processing batch: {e}")
                # Naive error handling: release and evict all requests in the batch
                for req in batch_to_process:
                    self.kv_cache_manager.release_request(req.request_id)
                    self.scheduler.evict_request(req.request_id, req.status)

    def run_loop_in_background(self):
        """Run the executor loop in the background."""

    def shutdown(self):
        """Shuts down the executor."""
        logger.info("Executor shutting down...")
        self.recv_from_peer_socket.close()
        self.send_to_peer_socket.close()
        self.zmq_context.term()
        logger.info("Executor shutdown complete.")


def create_executor_config(args):
    """Create executor configuration from command line arguments."""

    config = {
        "model_repo": args.model_path,
        "start_layer": args.start_layer,
        "end_layer": args.end_layer,
        "dtype": args.dtype,
        "max_batch_size": args.max_batch_size,
        "kv_block_size": args.kv_block_size,
        "kv_cache_memory_fraction": args.kv_cache_memory_fraction,
        "kv_max_tokens_in_cache": args.kv_max_tokens_in_cache,
        "max_num_tokens_in_batch": args.max_num_tokens_in_batch,
        "prefill_priority": args.prefill_priority,
        "micro_batch_ratio": args.micro_batch_ratio,
        "scheduler_wait_ms": args.scheduler_wait_ms,
        "send_to_peer_addr": getattr(args, "send_to_peer_addr", None),
        "recv_from_peer_addr": getattr(args, "recv_from_peer_addr", None),
        "executor_input_ipc_addr": args.executor_input_ipc,
        "executor_output_ipc_addr": args.executor_output_ipc,
    }
    return config
