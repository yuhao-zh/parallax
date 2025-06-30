# pylint: disable=too-many-locals,c-extension-no-member
"""
Defines Qwen3 model on Parallax.

TODO: Our current implementation is temporary. We will remove cache manager dependency.
"""

from typing import List, Optional, Tuple

import mlx.core as mx
from mlx_lm.models.base import scaled_dot_product_attention
from mlx_lm.models.qwen3 import Attention as MLXQwen3Attention
from mlx_lm.models.qwen3 import ModelArgs
from mlx_lm.models.qwen3 import TransformerBlock as MLXQwen3Block

# Ugly dependency for paged kv kernel
# TODO: refactor.
from parallax.models.attention import PagedAttentionInput
from parallax.server.kv_cache import PagedKVCache
from parallax.server.request import Request


class ParallaxQwen3Attention(MLXQwen3Attention):
    """A custom attention module for Parallax, extending the Qwen3 Attention class.

    We apply explicit KV cache handling and passing in `offset` directly from Request.
    This version returns the new K and V states for external caching.
    """

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Tuple[mx.array, mx.array]] = None,
        offset: int = 0,
        *,
        requests: Optional[List[Request]] = None,
        cache_manager: Optional[PagedKVCache] = None,
        layer_idx: Optional[int] = None,
    ) -> Tuple[mx.array, Tuple[mx.array, mx.array]]:
        """
        Attention forward pass with explicit KV cache handling.

        Args:
            x: (batch, target_len, hidden_dim) - Input hidden states for the current query segment.
            mask: (batch, n_q_heads, target_len, source_len)
            cache: Optional tuple (past_k, past_v).
                   shape: (batch, n_kv_heads, S_past_padded, head_dim)
            offset: S_past_padded (scalar, used for RoPE calculation).
            requests: Optional list of Request objects for the current batch.
            cache_manager: Optional PagedKVCache instance for managing KV cache.
            layer_idx: Optional int, if provided, use the layer_idx-th KV cache.

        Returns:
            output_h: (batch, target_len, hidden_dim) - Output hidden states.
            new_k: (batch, n_kv_heads, target_len, head_dim) - New keys for this segment.
            new_v: (batch, n_kv_heads, target_len, head_dim) - New values for this segment.
        """
        batch, target_len, _ = x.shape

        queries_new = self.q_proj(x)
        keys_new = self.k_proj(x)
        values_new = self.v_proj(x)

        # TODO: avoid reshape, fuse RoPE with qk Norm.
        queries_new = self.q_norm(
            queries_new.reshape(batch, target_len, self.n_heads, -1)
        ).transpose(0, 2, 1, 3)
        keys_new = self.k_norm(keys_new.reshape(batch, target_len, self.n_kv_heads, -1)).transpose(
            0, 2, 1, 3
        )
        values_new = values_new.reshape(batch, target_len, self.n_kv_heads, -1).transpose(
            0, 2, 1, 3
        )
        queries_rotated = self.rope(queries_new, offset=offset)
        keys_rotated = self.rope(keys_new, offset=offset)
        final_keys_for_attn = keys_rotated
        final_values_for_attn = values_new

        paged_input = None
        if cache_manager is not None and requests is not None:
            assert layer_idx is not None
            paged_input = PagedAttentionInput.construct_from_requests(
                query=queries_rotated,
                requests=requests,
                kv_cache=cache_manager,
                layer_idx=layer_idx,
                new_key=keys_rotated,
                new_val=values_new,
            )
        else:
            if cache is not None:
                past_k, past_v = cache
                if past_k is not None and past_v is not None:
                    past_k = past_k.transpose(0, 2, 1, 3)
                    past_v = past_v.transpose(0, 2, 1, 3)
                    if past_k.shape[2] != offset:  # Check against S_past_padded
                        raise ValueError(
                            f"ParallaxAttention: Expected past_k sequence length {past_k.shape[2]} "
                            f"to match RoPE offset {offset} (S_past_padded)."
                        )
                    final_keys_for_attn = mx.concatenate([past_k, keys_rotated], axis=2)
                    final_values_for_attn = mx.concatenate([past_v, values_new], axis=2)
                else:
                    raise ValueError("cache was provided but one of k/v was None.")

        if paged_input is not None:
            output = scaled_dot_product_attention(
                queries_rotated,
                final_keys_for_attn,
                final_values_for_attn,
                scale=self.scale,
                mask=mask,
                cache=None,
            )
        else:
            output = mx.core.paged_attention(
                q=paged_input.q,
                k_cache=paged_input.k_cache,
                v_cache=paged_input.v_cache,
                block_tables=paged_input.block_tables,
                context_lens=paged_input.context_lens,
                max_context_len=paged_input.max_context_len,
                softmax_scale=paged_input.softmax_scale,
            )

        output = output.transpose(0, 2, 1, 3).reshape(batch, target_len, -1)
        return self.o_proj(output), (keys_rotated, values_new)


class ParallaxQwen3Block(MLXQwen3Block):
    """A custom transformer block for Parallax, extending the Qwen3 Block class.
    This version handles the KV cache explicitly and returns new K and V states.
    """

    def __init__(self, args: ModelArgs):
        super().__init__(args)
        self.self_attn = ParallaxQwen3Attention(args)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Tuple[mx.array, mx.array]] = None,
        offset: int = 0,
        *,
        requests: Optional[List[Request]] = None,
        cache_manager: Optional[PagedKVCache] = None,
    ):
        r, (k_cache, v_cache) = self.self_attn(
            self.input_layernorm(x),
            mask,
            cache,
            offset=offset,
            requests=requests,
            cache_manager=cache_manager,
        )
        h = x + r
        r = self.mlp(self.post_attention_layernorm(h))
        out = h + r
        return out, (k_cache, v_cache)
