"""
Smoke‑tests for the ShardedModel loader utilities.
These tests download *only* the shards required for the given layer slices
from `mlx-community/Qwen3-0.6B-bf16` and verify that a forward pass runs
without errors.  They are intentionally lightweight so they can run on a
laptop with limited RAM.
"""

from typing import List, Tuple

import mlx.core as mx
import pytest
from mlx_lm.models.qwen3 import TransformerBlock as Qwen3Block
from mlx_lm.tokenizer_utils import load_tokenizer
from mlx_lm.utils import get_model_path, load_model

from server.shard_loader import MLXModelLoader

REPO_ID = "mlx-community/Qwen3-0.6B-bf16"
TOTAL_LAYERS = 28


model_path = get_model_path(REPO_ID)
ref_model, ref_config = load_model(model_path)
ref_tokenizer = load_tokenizer(model_path, eos_token_ids=ref_config.get("eos_token_id", None))


@pytest.mark.parametrize(
    "layers_config",
    [
        [(0, 12), (12, TOTAL_LAYERS)],
        [(0, 8), (8, 16), (16, TOTAL_LAYERS)],
    ],
)
def test_shard_forward(layers_config: List[Tuple[int, int]]) -> None:
    """Load sharded model based on layers_config and

    compare its forward pass with a full reference model.
    """
    print(f"\nTesting with sharding configuration: {layers_config}")
    model_shards = []
    for i, (layer_from, layer_to) in enumerate(layers_config):
        print(f"  Loading shard {i+1}: layers {layer_from} to {layer_to-1}")
        loader = MLXModelLoader(
            model_path_or_hf_repo=REPO_ID,
            start_layer=layer_from,
            end_layer=layer_to,
        )
        model_shard_instance, _ = loader.load(block_class=Qwen3Block)
        model_shards.append(model_shard_instance)

    assert len(model_shards) == len(layers_config), "Number of loaded shards should match config"

    x = ["This is a test.", "This is yet another test."]
    if ref_tokenizer.pad_token is None:
        ref_tokenizer.pad_token = ref_tokenizer.eos_token
    tokenized_batch = ref_tokenizer.encode(
        x,
        padding=True,
        return_tensors="mlx",
    )

    # Forward pass through the reference model
    print("  Running forward pass on reference model...")
    ref_out = ref_model(tokenized_batch)

    for i, shard in enumerate(model_shards):
        x = shard(x, cache=None, mask=None)

    assert mx.allclose(x, ref_out, atol=1e-3, rtol=1e-3)
