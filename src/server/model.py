"""
Defines the ShardedModel class for distributing MLX models across multiple devices.
"""

from typing import Any, List, Optional, Type

import mlx.core as mx
from mlx import nn
from mlx_lm.models.base import BaseModelArgs, create_attention_mask
from mlx_lm.tokenizer_utils import TokenizerWrapper


class ShardedModel(nn.Module):
    # pylint: disable=too-many-instance-attributes, too-many-arguments, too-many-positional-arguments
    """A general class for MLX sharded model."""

    def __init__(
        self,
        config: BaseModelArgs,
        model_id: str,
        start_layer: int,
        end_layer: int,
        block_class: Type[nn.Module],
        *,
        has_norm_in: bool = False,
        tokenizer: Optional[TokenizerWrapper] = None,
        dtype: Optional[mx.Dtype] = None,
    ):
        super().__init__()
        self.config = config
        self.model_id = model_id
        self.start_layer = start_layer
        self.end_layer = end_layer
        self.block_class = block_class
        self.has_norm_in = has_norm_in
        self.dtype = dtype

        # Get essential model parameters from the config
        self.hidden_size = config.hidden_size
        self.vocab_size = config.vocab_size
        self.num_hidden_layers = config.num_hidden_layers

        # Determine the roles of this shard
        self.is_first_shard = start_layer == 0
        self.is_last_shard = end_layer == self.num_hidden_layers

        # Instantiate modules based on the shard's role
        if self.is_first_shard:
            assert tokenizer is not None
            self.tokenizer = tokenizer
            self.embed_tokens = nn.Embedding(self.vocab_size, self.hidden_size)
            if has_norm_in:
                self.norm_in = nn.RMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        else:
            self.embed_tokens = None
            self.norm_in = None

        # Instantiate the slice of transformer blocks
        self.layers = [block_class(config) for _ in range(start_layer, end_layer)]

        if self.is_last_shard:
            self.norm = nn.RMSNorm(self.hidden_size, eps=config.rms_norm_eps)
            self.lm_head = nn.Linear(self.hidden_size, self.vocab_size, bias=False)
        else:
            self.norm = None
            self.lm_head = None

    def __call__(
        self,
        x: mx.array | List[str],
        cache: Optional[Any] = None,
        mask: Optional[mx.array] = None,
    ) -> mx.array:
        if self.is_first_shard:
            if isinstance(x, list):
                h = self.tokenizer.encode(
                    x,
                    padding=True,
                    return_tensors="mlx",
                )
            if self.embed_tokens:
                h = self.embed_tokens(h)
            if self.has_norm_in and self.norm_in:
                h = self.norm_in(h)
        else:
            assert isinstance(x, mx.array), "Input must be an mx.array for subsequent shards"
            h = x

        if mask is None:
            # Assuming x (or h) is the input to the first layer of this shard
            mask = create_attention_mask(h, cache)

        if cache is None:
            # Create a list of Nones with the same length as self.layers
            cache = [None] * len(self.layers)
        elif not isinstance(cache, list) or len(cache) != len(self.layers):
            # Handle incorrect cache format if necessary, or raise error
            # For now, let's assume if cache is provided, it's in the correct list format
            # This part might need more robust handling depending on expected cache structure
            pass

        for i, layer in enumerate(self.layers):
            layer_cache = cache[i] if cache and i < len(cache) else None
            h = layer(h, mask, layer_cache)

        if self.is_last_shard and self.norm and self.lm_head:
            h = self.norm(h)
            return self.lm_head(h)

        return h

    def forward(
        self, x: mx.array, cache: Optional[Any] = None, mask: Optional[mx.array] = None
    ) -> mx.array:
        """Alias for __call__."""
        return self(x, cache, mask)
