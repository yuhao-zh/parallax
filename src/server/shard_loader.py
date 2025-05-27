"""
Loads sharded MLX models from Hugging Face Hub or local paths.
"""

import glob
import importlib
import logging
from typing import Any, Dict, Optional, Tuple, Type

import mlx.core as mx
from mlx import nn  # R0402
from mlx_lm.utils import get_model_path, load_config

from .model import ShardedModel

logger = logging.getLogger(__name__)


class MLXModelLoader:
    # pylint: disable=too-few-public-methods
    """
    Handles downloading model assets from Hugging Face (if needed) and loading
    a specified shard of an MLX model.
    """

    def __init__(
        self,
        model_path_or_hf_repo: str,
        *,
        start_layer: Optional[int] = None,
        end_layer: Optional[int] = None,
    ):
        """
        Initializes the model loader.

        Args:
            model_path_or_hf_repo (str): The Hugging Face Hub model ID or a local path
                                         to the model directory.
            start_layer (Optional[int]): The starting layer index for the shard (inclusive).
                                         Defaults to the beginning of the model.
            end_layer (Optional[int]): The ending layer index for the shard (exclusive).
                                       Defaults to the end of the model.
        """
        self.model_path_str = model_path_or_hf_repo
        self.start_layer = start_layer
        self.end_layer = end_layer

    def load(
        self, lazy: bool = False, strict: bool = True, *, block_class: Type[nn.Module]
    ) -> Tuple[nn.Module, Dict[str, Any]]:
        # pylint: disable=too-many-arguments,too-many-locals,too-many-branches,too-many-statements
        """
        Loads the specified model shard.

        Args:
            lazy (bool): If False, evaluates model parameters to ensure they are loaded
                         into memory before returning. Otherwise, they are loaded on demand.
                         Defaults to False.
            strict (bool): If True, raises an exception if weights in checkpoint files
                           do not match the model structure. Defaults to True.
            block_class (Type[nn.Module]): The class to use for instantiating transformer blocks
                                          (e.g., a specific AttentionBlock or DecoderLayer).

        Returns:
            Tuple[nn.Module, dict]: A tuple containing the loaded sharded MLX model
                                    and its configuration dictionary.

        Raises:
            FileNotFoundError: If no .safetensors weight files are found in the model path
                               and strict mode is enabled.
            ValueError: If the model type specified in the config is not found in mlx_lm.models.
        """
        model_path = get_model_path(self.model_path_str)
        config = load_config(model_path)

        num_hidden_layers = config.get("num_hidden_layers", 0)
        # Resolve start and end layers if they are None
        current_start_layer = self.start_layer if self.start_layer is not None else 0
        current_end_layer = self.end_layer if self.end_layer is not None else num_hidden_layers

        weight_files = glob.glob(str(model_path / "model*.safetensors"))
        if not weight_files:
            weight_files = glob.glob(str(model_path / "weight*.safetensors"))

        if not weight_files and strict:
            msg = f"No safetensors found in {model_path}"
            logger.error(msg)
            raise FileNotFoundError(msg)

        model_type = config.get("model_type")
        if not model_type:
            msg = "model_type not found in config.json"
            logger.error(msg)
            raise ValueError(msg)

        try:
            arch_module = importlib.import_module(f"mlx_lm.models.{model_type}")
        except ImportError as e:  # W0707
            msg = f"Model type '{model_type}' not found in mlx_lm.models."
            logger.error(msg)  # W1203: Keep f-string for error message clarity
            raise ValueError(msg) from e

        model_args_class = getattr(arch_module, "ModelArgs", None)
        if not model_args_class:
            msg = f"ModelArgs class not found in mlx_lm.models.{model_type}"
            logger.error(msg)  # W1203
            raise ValueError(msg)

        # Assuming ShardedModel __init__ is compatible with model_args_class instance or config dict
        # And that block_class is the actual nn.Module for layers, not ModelArgs
        model_shard = ShardedModel(
            config=model_args_class.from_dict(config),  # Pass ModelArgs instance
            start_layer=current_start_layer,
            end_layer=current_end_layer,
            block_class=block_class,
        )

        all_weights = {}
        for wf in weight_files:
            all_weights.update(mx.load(wf))

        shard_weights = {}
        layer_key_prefix = "model.layers"  # Common prefix

        for key, value in all_weights.items():
            if model_shard.is_first_shard and "embed_tokens" in key and key.startswith("model."):
                shard_weights[key.replace("model.", "", 1)] = value
            elif model_shard.is_last_shard and "model.norm" in key:
                shard_weights[key.replace("model.", "", 1)] = value
            elif (
                model_shard.is_last_shard and "lm_head" in key
            ):  # lm_head usually doesn't have "model."
                shard_weights[key] = value
            elif layer_key_prefix in key:
                try:
                    # e.g. model.layers.15.self_attn.q_proj.weight
                    parts = key.split(".")
                    layer_idx_str = parts[2]  # "15"
                    layer_idx = int(layer_idx_str)

                    if current_start_layer <= layer_idx < current_end_layer:
                        local_layer_idx = layer_idx - current_start_layer
                        # Reconstruct: layers.{local_idx}.self_attn.q_proj.weight
                        remapped_key = f"layers.{local_layer_idx}.{'.'.join(parts[3:])}"
                        shard_weights[remapped_key] = value
                except (ValueError, IndexError):
                    logger.warning("Could not parse layer index from key: %s", key)  # W1203
                    continue

        model_shard.load_weights(list(shard_weights.items()), strict=strict)

        if not lazy:
            mx.eval(model_shard.parameters())

        model_shard.eval()
        logger.info(
            "Successfully loaded model shard (layers %d-%d)",  # W1203
            current_start_layer,
            current_end_layer - 1,
        )
        return model_shard, config
