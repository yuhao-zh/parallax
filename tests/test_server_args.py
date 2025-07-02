"""
Tests for the server_args module.
"""

import pytest
import sys
import os
import argparse
from unittest.mock import patch

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from parallax.server.server_args import (
    parse_dtype,
    validate_args,
    parse_args
)
from parallax.server.executor import get_block_class, create_executor_config
from parallax.models.qwen3 import ParallaxQwen3Block
import mlx.core as mx


class TestParseDtype:
    """Test dtype parsing functionality."""
    
    def test_valid_dtypes(self):
        """Test parsing of valid dtypes."""
        assert parse_dtype("float16") == mx.float16
        assert parse_dtype("bfloat16") == mx.bfloat16
        assert parse_dtype("float32") == mx.float32
    
    def test_invalid_dtype(self):
        """Test parsing of invalid dtype."""
        with pytest.raises(ValueError, match="Unsupported dtype"):
            parse_dtype("invalid_dtype")


class TestGetBlockClass:
    """Test block class selection functionality."""
    
    def test_qwen3_model(self):
        """Test Qwen3 model detection."""
        block_class = get_block_class("mlx-community/Qwen3-0.6B-bf16")
        assert block_class == ParallaxQwen3Block
    
    def test_qwen3_model_case_insensitive(self):
        """Test Qwen3 model detection case insensitive."""
        block_class = get_block_class("QWEN3-model")
        assert block_class == ParallaxQwen3Block
    
    def test_unknown_model_defaults_to_qwen3(self):
        """Test that unknown models default to Qwen3."""
        block_class = get_block_class("unknown-model")
        assert block_class == ParallaxQwen3Block


class TestValidateArgs:
    """Test argument validation functionality."""
    
    def test_valid_layer_indices(self):
        """Test valid layer indices."""
        args = argparse.Namespace(
            start_layer=0,
            end_layer=10,
            kv_cache_memory_fraction=0.5,
            max_batch_size=16,
            max_num_tokens_in_batch=1024,
            kv_block_size=16,
            micro_batch_ratio=2,
            scheduler_wait_ms=500,
            kv_max_tokens_in_cache=None
        )
        
        # Should not raise any exception
        validate_args(args)
    
    def test_invalid_start_layer(self):
        """Test invalid start layer."""
        args = argparse.Namespace(
            start_layer=-1,
            end_layer=10,
            kv_cache_memory_fraction=0.5,
            max_batch_size=16,
            max_num_tokens_in_batch=1024,
            kv_block_size=16,
            micro_batch_ratio=2,
            scheduler_wait_ms=500,
            kv_max_tokens_in_cache=None
        )
        
        with pytest.raises(ValueError, match="start_layer must be non-negative"):
            validate_args(args)
    
    def test_invalid_end_layer(self):
        """Test invalid end layer."""
        args = argparse.Namespace(
            start_layer=10,
            end_layer=5,
            kv_cache_memory_fraction=0.5,
            max_batch_size=16,
            max_num_tokens_in_batch=1024,
            kv_block_size=16,
            micro_batch_ratio=2,
            scheduler_wait_ms=500,
            kv_max_tokens_in_cache=None
        )
        
        with pytest.raises(ValueError, match="end_layer must be greater than start_layer"):
            validate_args(args)


class TestCreateExecutorConfig:
    """Test executor configuration creation."""
    
    def test_create_config(self):
        """Test creating executor configuration."""
        args = argparse.Namespace(
            model_path='mlx-community/Qwen3-0.6B-bf16',
            start_layer=0,
            end_layer=14,
            dtype=mx.bfloat16,  # Now it's already a MLX dtype object
            max_batch_size=16,
            kv_block_size=16,
            kv_cache_memory_fraction=0.8,
            kv_max_tokens_in_cache=None,
            max_num_tokens_in_batch=1024,
            prefill_priority=0,
            micro_batch_ratio=2,
            scheduler_wait_ms=500
        )
        
        config = create_executor_config(args)
        
        assert config['model_repo'] == 'mlx-community/Qwen3-0.6B-bf16'
        assert config['start_layer'] == 0
        assert config['end_layer'] == 14
        assert config['dtype'] == mx.bfloat16
        assert config['kv_cache_memory_fraction'] == 0.8


class TestParseArgs:
    """Test argument parsing with mocked sys.argv."""
    
    @patch('sys.argv', [
        'test_server_args.py',
        '--model-path', 'mlx-community/Qwen3-0.6B-bf16',
        '--start-layer', '0',
        '--end-layer', '14',
        '--dtype', 'bfloat16',
        '--kv-cache-memory-fraction', '0.8'
    ])
    def test_parse_valid_args(self):
        """Test parsing valid arguments."""
        args = parse_args()
        
        assert args.model_path == 'mlx-community/Qwen3-0.6B-bf16'
        assert args.start_layer == 0
        assert args.end_layer == 14
        assert args.dtype == mx.bfloat16  # Now it's a MLX dtype object
        assert args.kv_cache_memory_fraction == 0.8
    
    @patch('sys.argv', [
        'test_server_args.py',
        '--model-path', 'test',
        '--start-layer', '10',
        '--end-layer', '5'
    ])
    def test_parse_invalid_args(self):
        """Test parsing invalid arguments."""
        with pytest.raises(ValueError, match="end_layer must be greater than start_layer"):
            parse_args()
