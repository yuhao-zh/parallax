"""
Utility functions for message serialization and deserialization.

This module contains utility functions for serializing and deserializing messages
between the P2P server and the executor.
"""

import io
from typing import List

import mlx.core as mx

from parallax.p2p.proto import forward_pb2
from parallax.server.request import IntermediateRequest, RequestStatus
from parallax.server.sampling.sampling_params import SamplingParams


def request_to_proto(requests: List[IntermediateRequest]) -> forward_pb2.ForwardRequest:
    """
    Convert a list of IntermediateRequest objects to a ForwardRequest protobuf message.
    IntermediateRequest contains request_id, current_position, status, and hidden_states.
    """
    forward_request = forward_pb2.ForwardRequest()
    assert len(requests) > 0, "No requests to convert"
    assert all(
        request.status == requests[0].status for request in requests
    ), "All requests must have the same status"
    if requests[0].status == RequestStatus.PREFILLING:
        forward_request.forward_mode = forward_pb2.ForwardMode.EXTEND
    elif requests[0].status == RequestStatus.DECODING:
        forward_request.forward_mode = forward_pb2.ForwardMode.DECODE
    else:
        raise ValueError(f"Invalid status: {requests[0].status}")

    # Collect all hidden_states and next_token_ids
    all_hidden_states = []
    all_residual = []

    for request in requests:
        proto_req = forward_pb2.Req()
        proto_req.rid = request.request_id
        proto_req.output_length = request.current_position - len(request.input_ids)
        proto_req.input_ids.extend(request.input_ids)
        proto_req.routing_table.extend(request.routing_table)
        forward_request.reqs.append(proto_req)

        # Assert that hidden_states is mlx array
        assert isinstance(
            request.hidden_states, mx.array
        ), f"hidden_states must be mlx.array, got {type(request.hidden_states)}"

        # Check if hidden_states contains a single token_id (from Last Peer to First Peer)
        if hasattr(request.hidden_states, "shape"):
            if request.hidden_states.shape == (1,) or (
                len(request.hidden_states.shape) == 1 and request.hidden_states.shape[0] == 1
            ):
                # This is a token_id from Last Peer, add to next_token_ids
                token_id = int(request.hidden_states[0])
                forward_request.next_token_ids.append(token_id)
            else:
                # This is actual hidden states, collect for concatenation
                all_hidden_states.append(request.hidden_states)
                # Add a zero residual for gpu executors
                residual = mx.zeros(request.hidden_states.shape, dtype=request.hidden_states.dtype)
                all_residual.append(residual)
                # Pass the previous token_id to the next peer.
                if request.next_token_id is not None:
                    forward_request.next_token_ids.append(request.next_token_id)

    # Concatenate all hidden_states into a single tensor if any exist
    if all_hidden_states:
        # Concatenate along the first dimension (batch dimension)
        concatenated_hidden_states = mx.concatenate(all_hidden_states, axis=0)
        concatenated_residual = mx.concatenate(all_residual, axis=0)

        # Create a single named tensor for all hidden states
        named_tensor = forward_pb2.NamedTensor()
        named_tensor.name = "hidden_states"
        named_tensor.tensor.CopyFrom(tensor_to_proto(concatenated_hidden_states))
        forward_request.pp_proxy_tensors.tensors.append(named_tensor)
        # Create a single named tensor for all residual
        residual_tensor = forward_pb2.NamedTensor()
        residual_tensor.name = "residual"
        residual_tensor.tensor.CopyFrom(tensor_to_proto(concatenated_residual))
        forward_request.pp_proxy_tensors.tensors.append(residual_tensor)

    return forward_request


def proto_to_hidden_states(proto: forward_pb2.PPProxyTensorsMessage) -> mx.array:
    """
    Convert a PPProxyTensorsMessage protobuf message to a list of mlx arrays.
    """
    if proto is None:
        return None

    hidden_states = None
    for named_tensor in proto.tensors:
        if named_tensor.name in ("hidden_states", "residual"):
            if hidden_states is None:
                hidden_states = proto_to_tensor(named_tensor.tensor)
            else:
                hidden_states = hidden_states + proto_to_tensor(named_tensor.tensor)
    return hidden_states


def proto_to_request(proto_request: forward_pb2.ForwardRequest) -> List[IntermediateRequest]:
    """
    Convert a ForwardRequest protobuf message to a IntermediateRequest object.
    """

    requests = []

    hidden_states = proto_to_hidden_states(proto_request.pp_proxy_tensors)
    if hidden_states is None:
        status = RequestStatus.FINISHED_EOS
    elif proto_request.forward_mode == forward_pb2.ForwardMode.EXTEND:
        status = RequestStatus.PREFILLING
    elif proto_request.forward_mode == forward_pb2.ForwardMode.DECODE:
        status = RequestStatus.DECODING
    else:
        raise ValueError(f"Invalid forward mode: {proto_request.forward_mode}")

    token_index = 0
    for index, proto_req in enumerate(proto_request.reqs):
        current_position = len(proto_req.input_ids) + proto_req.output_length

        current_hidden_states = None
        if status == RequestStatus.PREFILLING:
            seq_len = len(proto_req.input_ids)
            if seq_len == 0:
                seq_len = len(proto_req.input_ids) + proto_req.output_length
            current_hidden_states = hidden_states[token_index : token_index + seq_len]
            token_index += len(proto_req.input_ids)
        elif status == RequestStatus.DECODING:
            current_hidden_states = hidden_states[token_index : token_index + 1]
            token_index += 1

        next_token_id = None
        if proto_request.next_token_ids:
            next_token_id = proto_request.next_token_ids[index]

        sampling_params = proto_to_sampling_params(proto_req.sampling_params)

        request = IntermediateRequest(
            request_id=proto_req.rid,
            current_position=current_position,
            status=status,
            input_ids=list(proto_req.input_ids),
            hidden_states=current_hidden_states,
            routing_table=proto_req.routing_table,
            next_token_id=next_token_id,
            sampling_params=sampling_params,
        )

        requests.append(request)

    return requests


def proto_to_sampling_params(proto: forward_pb2.SamplingParams) -> SamplingParams:
    """Convert protobuf message to SamplingParams."""
    if proto is None:
        return SamplingParams()
    sampling_params = SamplingParams(
        max_new_tokens=proto.max_new_tokens,
        min_new_tokens=proto.min_new_tokens,
        temperature=proto.temperature,
        top_p=proto.top_p,
        min_p=proto.min_p,
        top_k=proto.top_k,
        stop_strs=list(proto.stop_strs),
        stop_token_ids=list(proto.stop_token_ids),
        ignore_eos=proto.ignore_eos,
        repetition_penalty=proto.repetition_penalty,
        presence_penalty=proto.presence_penalty,
        frequency_penalty=proto.frequency_penalty,
        json_schema=proto.json_schema,
    )
    return sampling_params


def sampling_params_to_proto(params: SamplingParams) -> forward_pb2.SamplingParams:
    """Convert SamplingParams to protobuf message."""
    proto = forward_pb2.SamplingParams()

    proto.max_new_tokens = params.max_new_tokens
    proto.min_new_tokens = params.min_new_tokens
    proto.temperature = params.temperature
    proto.top_p = params.top_p
    proto.min_p = params.min_p
    proto.top_k = params.top_k
    if params.stop_strs is not None:
        proto.stop_strs.extend(params.stop_strs)
    if params.stop_token_ids is not None:
        proto.stop_token_ids.extend(params.stop_token_ids)
    proto.ignore_eos = params.ignore_eos
    proto.repetition_penalty = params.repetition_penalty
    proto.presence_penalty = params.presence_penalty
    proto.frequency_penalty = params.frequency_penalty
    proto.json_schema = params.json_schema
    return proto


def tensor_to_proto(mlx_tensor: mx.array) -> forward_pb2.Tensor:
    """Convert MLX array to protobuf Tensor using safetensor serialization."""
    assert mlx_tensor.size > 0, "Tensor must have size > 0"
    buffer = io.BytesIO()
    mx.save_safetensors(buffer, {"tensor": mlx_tensor})
    tensor = forward_pb2.Tensor()
    tensor.buffer = buffer.getvalue()
    return tensor


def proto_to_tensor(tensor: forward_pb2.Tensor) -> mx.array:
    """Convert protobuf Tensor (safetensor format) to MLX array."""
    buffer = io.BytesIO(tensor.buffer)
    tensors_dict = mx.load(buffer, format="safetensors")
    return tensors_dict["tensor"]
