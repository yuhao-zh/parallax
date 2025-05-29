"""
Minimum viable executor for MLX model shards.
"""

from typing import Type

from mlx import nn

from parallax.logging_config import get_logger
from parallax.server.shard_loader import MLXModelLoader

logger = get_logger(__name__)


class ModelExecutor:
    """Executor for loading and managing MLX model shards."""

    # pylint: disable=too-few-public-methods
    def __init__(
        self,
        model_repo: str,
        start_layer: int,
        end_layer: int,
        block_class: Type[nn.Module],
    ):
        self.model_repo = model_repo

        logger.info(
            f"Executor: Loading model shard: Layers {start_layer} to {end_layer-1} of {model_repo}"
        )
        loader = MLXModelLoader(self.model_repo, start_layer=start_layer, end_layer=end_layer)
        self.model_shard, self.config = loader.load(block_class=block_class)

        logger.info("Executor: Model shard loaded successfully.")
