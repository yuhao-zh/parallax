"""Utility functions."""

from typing import List

import mlx.core as mx
import psutil
import zmq


def get_zmq_socket(context: zmq.Context, socket_type: zmq.SocketType, endpoint: str, bind: bool):
    """Create and configure a ZeroMQ socket.

    Ported from SGLang.
    """
    mem = psutil.virtual_memory()
    total_mem = mem.total / 1024**3
    available_mem = mem.available / 1024**3
    if total_mem > 32 and available_mem > 16:
        buf_size = int(0.5 * 1024**3)
    else:
        buf_size = -1

    socket = context.socket(socket_type)
    if endpoint.find("[") != -1:
        socket.setsockopt(zmq.IPV6, 1)

    def set_send_opt():
        socket.setsockopt(zmq.SNDHWM, 0)
        socket.setsockopt(zmq.SNDBUF, buf_size)

    def set_recv_opt():
        socket.setsockopt(zmq.RCVHWM, 0)
        socket.setsockopt(zmq.RCVBUF, buf_size)

    if socket_type == zmq.PUSH:
        set_send_opt()
    elif socket_type == zmq.PULL:
        set_recv_opt()
    elif socket_type == zmq.DEALER:
        set_send_opt()
        set_recv_opt()
    else:
        raise ValueError(f"Unsupported socket type: {socket_type}")

    if bind:
        socket.bind(endpoint)
    else:
        socket.connect(endpoint)

    return socket


def batch_tokenize(tokenizer, list_of_strings: List[str]):
    """
    Tokenize a list of strings and pad them to the same length.
    """
    tokenized_sequences = [tokenizer.encode(text) for text in list_of_strings]
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id  # Common practice for many autoregressive LMs
    if pad_token_id is None:
        raise ValueError("Tokenizer does not have a pad_token_id or eos_token_id for padding.")

    max_len = 0
    actual_lengths = []
    for tokens in tokenized_sequences:
        current_length = len(tokens)
        actual_lengths.append(current_length)
        max_len = max(max_len, current_length)

    padded_input_ids = []
    attention_masks = []

    for tokens in tokenized_sequences:
        num_padding_tokens = max_len - len(tokens)

        # Padded sequence
        padded_tokens = tokens + [pad_token_id] * num_padding_tokens
        padded_input_ids.append(padded_tokens)

        # Attention mask
        mask = [1] * len(tokens) + [0] * num_padding_tokens
        attention_masks.append(mask)

    batch_input_ids = mx.array(padded_input_ids)
    # 4d mask: (batch, n_q_heads, target_len, source_len)
    batch_attention_mask = mx.array(attention_masks)[:, None, None, :]
    return batch_input_ids, batch_attention_mask
