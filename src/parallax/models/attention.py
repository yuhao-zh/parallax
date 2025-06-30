# pylint: disable=c-extension-no-member
"""
Wrapper for Parallalx Attention Layer with Paged Attention Support.
"""

from dataclasses import dataclass
from typing import List

import mlx.core as mx

from parallax.server.kv_cache import PagedKVCache
from parallax.server.request import Request


@dataclass
class PagedAttentionInput:
    """
    q: (num_seqs, num_heads, head_dim) - The query tensor for the current batch of requests.
    k_cache: (num_blocks, num_kv_heads, head_dim // x, block_size, x)
    v_cache: (num_blocks, num_kv_heads, head_dim, block_size)
    block_tables: (num_seqs, max_num_blocks_per_seq) with mx.uint32. Maps sequence to its block IDs.
    context_lens: (num_seqs,) with mx.uint32.
        Hodling the token length of each sequence in the batch.
    max_context_len: int - The maximum context length across all sequences in the batch.
    softmax_scale: float - The softmax scale for the attention.
    """

    q: mx.array
    k_cache: mx.array
    v_cache: mx.array
    block_tables: mx.array
    context_lens: mx.array
    max_context_len: int
    softmax_scale: float

    @classmethod
    def construct_from_requests(
        cls,
        query: mx.array,
        requests: List[Request],
        kv_cache: PagedKVCache,
        layer_idx: int,
        new_key: mx.array,
        new_val: mx.array,
    ) -> "PagedAttentionInput":
        """
        Construct a PagedAttentionInput from a list of requests.

        Args:
            query: (num_seqs, num_heads, head_dim).
            requests: List[Request] - The list of requests to construct the input for.
            kv_cache: PagedKVCache - The KV cache to use.
            layer_idx: int - The index of the layer to construct the input for.
            new_key: (batch, n_kv_heads, target_len, head_dim) - The new key tensor.
            new_val: (batch, n_kv_heads, target_len, head_dim) - The new value tensor.

        Returns:
            PagedAttentionInput - The constructed input.
        """
        block_tables, context_lens = kv_cache.gather_block_tables(requests)
        max_context_len = max(context_lens)
        softmax_scale = 1.0 / (kv_cache.head_dim**0.5)

        # update the kv cache
        # TODO: handle OOM for decoding
        kv_cache.update_kv_cache(requests, new_key, new_val)

        return PagedAttentionInput(
            q=query,
            k_cache=kv_cache.get_k_cache_pool(layer_idx),
            v_cache=kv_cache.get_v_cache_pool(layer_idx),
            block_tables=mx.array(block_tables, dtype=mx.uint32),
            context_lens=mx.array(context_lens, dtype=mx.uint32),
            max_context_len=max_context_len,
            softmax_scale=softmax_scale,
        )
