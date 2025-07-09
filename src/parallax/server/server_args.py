"""
CLI argument parser for Parallax server.

This module provides argument parsing functionality for the Parallax executor,
supporting various configuration options for model loading, layer sharding,
and performance tuning.
"""

import argparse

import mlx.core as mx

from parallax.utils.logging_config import get_logger

logger = get_logger(__name__)


def parse_dtype(dtype_str: str) -> mx.Dtype:
    """
    Parse dtype string to MLX dtype.

    Args:
        dtype_str: String representation of dtype (e.g., 'float16', 'bfloat16', 'float32')

    Returns:
        MLX dtype object

    Raises:
        ValueError: If dtype string is not supported
    """
    dtype_map = {
        "float16": mx.float16,
        "bfloat16": mx.bfloat16,
        "float32": mx.float32,
    }

    if dtype_str not in dtype_map:
        raise ValueError(
            f"Unsupported dtype: {dtype_str}. Supported dtypes: {list(dtype_map.keys())}"
        )

    return dtype_map[dtype_str]


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments for the Parallax executor.

    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        description="Parallax Executor - Distributed LLM Inference",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # P2P configuration
    parser.add_argument("--initial-peers", nargs="+", default=[], help="List of initial DHT peers")

    parser.add_argument(
        "--announce-maddrs",
        type=str,
        default=None,
        help="Comma-separated list of multiaddresses to announce",
    )

    parser.add_argument("--public-ip", type=str, default=None, help="Public IP address to announce")

    parser.add_argument("--dht-port", type=int, default=None, help="Port for DHT communication")

    parser.add_argument("--host-maddrs", type=str, default=None, help="Multiaddress to host")

    parser.add_argument("--dht-prefix", type=str, default="gradient", help="Prefix for DHT keys")

    parser.add_argument(
        "--notify-url", type=str, default=None, help="URL to notify when a request is finished"
    )

    # Model configuration
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to the model repository or model name (e.g., 'mlx-community/Qwen3-0.6B-bf16')",
    )

    parser.add_argument(
        "--start-layer",
        type=int,
        required=True,
        help="Starting layer index for this shard (inclusive)",
    )

    parser.add_argument(
        "--end-layer", type=int, required=True, help="Ending layer index for this shard (exclusive)"
    )

    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float16", "bfloat16", "float32"],
        help="Data type for model weights and computations",
    )

    # KV Cache configuration
    parser.add_argument(
        "--kv-cache-memory-fraction",
        type=float,
        default=0.8,
        help="Fraction of available memory to use for KV cache (0.0 to 1.0)",
    )

    parser.add_argument(
        "--kv-max-tokens-in-cache",
        type=int,
        default=None,
        help="Maximum number of tokens to store in KV cache (None for auto)",
    )

    parser.add_argument(
        "--kv-block-size", type=int, default=64, help="Block size for KV cache management"
    )

    # Scheduler configuration
    parser.add_argument(
        "--max-batch-size", type=int, default=16, help="Maximum batch size for processing requests"
    )

    parser.add_argument(
        "--max-num-tokens-in-batch",
        type=int,
        default=1024,
        help="Maximum number of tokens in a batch",
    )

    parser.add_argument(
        "--prefill-priority",
        type=int,
        default=0,
        choices=[0, 1],
        help="Priority for prefill requests (0 or 1)",
    )

    parser.add_argument(
        "--micro-batch-ratio", type=int, default=2, help="Micro batch ratio for scheduling"
    )

    parser.add_argument(
        "--scheduler-wait-ms", type=int, default=500, help="Scheduler wait time in milliseconds"
    )

    # Logging and debugging
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )

    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    # Parse dtype directly
    args.dtype = parse_dtype(args.dtype)

    # Validate arguments
    validate_args(args)

    return args


def validate_args(args: argparse.Namespace) -> None:
    """
    Validate parsed arguments.

    Args:
        args: Parsed arguments namespace

    Raises:
        ValueError: If arguments are invalid
    """
    # Validate layer indices
    if args.start_layer < 0:
        raise ValueError("start_layer must be non-negative")

    if args.end_layer <= args.start_layer:
        raise ValueError("end_layer must be greater than start_layer")

    # Validate memory fraction
    if not 0.0 <= args.kv_cache_memory_fraction <= 1.0:
        raise ValueError("kv_cache_memory_fraction must be between 0.0 and 1.0")

    # Validate batch sizes
    if args.max_batch_size <= 0:
        raise ValueError("max_batch_size must be positive")

    if args.max_num_tokens_in_batch <= 0:
        raise ValueError("max_num_tokens_in_batch must be positive")

    if args.kv_block_size <= 0:
        raise ValueError("kv_block_size must be positive")

    if args.micro_batch_ratio <= 0:
        raise ValueError("micro_batch_ratio must be positive")

    if args.scheduler_wait_ms < 0:
        raise ValueError("scheduler_wait_ms must be non-negative")

    # Validate KV cache tokens
    if args.kv_max_tokens_in_cache is not None and args.kv_max_tokens_in_cache <= 0:
        raise ValueError("kv_max_tokens_in_cache must be positive if specified")
