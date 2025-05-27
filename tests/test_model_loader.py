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

from server.server_info import ShardedModelInfo
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
    model_shards = []
    for layer_from, layer_to in layers_config:
        loader = MLXModelLoader(
            model_path_or_hf_repo=REPO_ID,
            start_layer=layer_from,
            end_layer=layer_to,
        )
        model_shard_instance, _ = loader.load(block_class=Qwen3Block)
        model_info = ShardedModelInfo.from_sharded_model(model_shard_instance)
        print(model_info)

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
    ref_out = ref_model(tokenized_batch)

    for shard in model_shards:
        x = shard(x, cache=None, mask=None)

    assert mx.allclose(x, ref_out, atol=1e-3, rtol=1e-3)
