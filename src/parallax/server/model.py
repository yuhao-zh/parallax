# pylint: disable=too-many-locals
"""
Defines the ShardedModel class for distributing MLX models across multiple devices.
"""

from typing import Optional, Tuple, Type

import mlx.core as mx
from mlx import nn
from mlx_lm.models.base import BaseModelArgs, create_causal_mask


class ShardedModel(nn.Module):
    """A general class for MLX sharded model, adapted for Parallax KV cache.

    Assumes self.layers are composed of modules (e.g., TransformerBlocks)
    that internally use ParallaxAttention and their __call__ method returns
    (hidden_state, new_k_for_layer, new_v_for_layer).
    """

    def __init__(
        self,
        config: BaseModelArgs,
        model_id: str,
        start_layer: int,
        end_layer: int,
        block_class: Type[nn.Module],
        *,
        has_norm_in: bool = False,
        dtype: Optional[mx.Dtype] = None,
    ):
        super().__init__()
        self.config = config
        self.model_id = model_id
        self.start_layer = start_layer
        self.end_layer = end_layer
        self.block_class = block_class
        self.has_norm_in = has_norm_in
        self.dtype = dtype if dtype is not None else mx.float16

        self.hidden_size = config.hidden_size
        self.vocab_size = config.vocab_size

        self.is_first_shard = start_layer == 0
        self.is_last_shard = end_layer == config.num_hidden_layers
        self.n_layers_in_shard = end_layer - start_layer

        if self.is_first_shard:
            self.embed_tokens = nn.Embedding(self.vocab_size, self.hidden_size)
            if has_norm_in:
                self.norm_in = nn.RMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        else:
            self.embed_tokens = None
            self.norm_in = None

        self.layers = [block_class(config) for _ in range(self.n_layers_in_shard)]

        if self.is_last_shard:
            self.norm = nn.RMSNorm(self.hidden_size, eps=config.rms_norm_eps)
            self.lm_head = nn.Linear(self.hidden_size, self.vocab_size, bias=False)
        else:
            self.norm = None
            self.lm_head = None

    def __call__(
        self,
        h_or_tokens: mx.array,
        cache: Optional[Tuple[mx.array, mx.array]] = None,
        lengths: Optional[mx.array] = None,
        mask: Optional[mx.array] = None,
        window_size: Optional[int] = None,
    ) -> Tuple[mx.array, Tuple[mx.array, mx.array]]:
        # pylint: disable=too-many-branches
        """
        Args:
            h_or_tokens:
                (batch, target_len_padded, D) or (batch, target_len_padded) for prefill,
                (batch, 1, D) or (batch, 1) for decode.
            cache: Optional tuple of (k_past_shard, v_past_shard) for this shard.
                   each of k/v cache has shape:
                       (batch, n_layers_in_shard, n_kv_heads, source_len_padded, head_dim)
            lengths: (batch,) true lengths of each sequence in batch.
            mask: Optional causal mask for the current segment.
            window_size: Optional int, if provided, will use a sliding window attention mask.

        Returns:
            h: (batch, L_padded, D) or (batch, L_padded, vocab_size) if last_shard
            (stacked_k_updates, stacked_v_updates):
                new KV for current segment.
                (n_layers_in_shard, batch, n_kv_heads, L_padded, head_dim)
        """
        h = h_or_tokens
        batch = h.shape[0]
        target_len = h.shape[1]  # Padded length of current query segment

        if self.is_first_shard:
            if self.embed_tokens is None:
                raise ValueError("embed_tokens is None for the first shard.")
            h = self.embed_tokens(h)
            if self.has_norm_in and self.norm_in:
                h = self.norm_in(h)

        source_len = 0
        k_past_all_layers, v_past_all_layers = None, None

        if cache is not None:
            k_past_all_layers, v_past_all_layers = cache
            if k_past_all_layers is not None:
                assert (
                    k_past_all_layers.ndim == 5
                ), f"Unexpected k_past_all_layers ndim: {k_past_all_layers.ndim}"
                source_len = k_past_all_layers.shape[3]

        if lengths is None:
            lengths = mx.full((batch,), target_len + source_len, dtype=mx.int32)
        else:
            # Validate cumulative_true_lengths shape
            assert lengths.shape == (
                batch,
            ), f"lengths shape mismatch: expected ({batch},), got {lengths.shape}"

        # Create causal mask if not provided
        offset = source_len - 1 if target_len == 1 else 0
        if mask is None:
            if target_len > 1:
                mask = create_causal_mask(
                    N=target_len,
                    offset=offset,
                    lengths=lengths,
                    window_size=window_size,
                )

        collected_k_updates = []
        collected_v_updates = []

        for i, layer_module in enumerate(self.layers):
            current_layer_past_kv = None
            if k_past_all_layers is not None and v_past_all_layers is not None:
                layer_k_past_slice = k_past_all_layers[:, i, ...]
                layer_v_past_slice = v_past_all_layers[:, i, ...]
                current_layer_past_kv = (layer_k_past_slice, layer_v_past_slice)

            h, (new_k, new_v) = layer_module(
                h,
                mask=mask,
                cache=current_layer_past_kv,
                offset=offset,
            )
            collected_k_updates.append(new_k)
            collected_v_updates.append(new_v)

        if self.is_last_shard:
            if self.norm is None or self.lm_head is None:
                raise ValueError("ShardedModel: norm or lm_head is None for the last shard.")
            h = self.norm(h)
            h = self.lm_head(h)

        # Stack the collected KV updates for this shard.
        # (batch, n_layers_in_shard, n_kv_heads, source_len, head_dim)
        stacked_k_updates = mx.stack(collected_k_updates, axis=0)
        stacked_v_updates = mx.stack(collected_v_updates, axis=0)

        return h, (stacked_k_updates, stacked_v_updates)

    def forward(
        self,
        h_or_tokens: mx.array,
        cache: Optional[Tuple[mx.array, mx.array]] = None,
        cumulative_true_lengths: Optional[mx.array] = None,
    ) -> Tuple[mx.array, Tuple[mx.array, mx.array]]:
        """Alias for __call__."""
        return self(h_or_tokens, cache, cumulative_true_lengths)
