"""
Paged KV Cache Manager for Parallax Server

This module implements a paged key-value (KV) cache system designed for efficient
management of attention caches in large language models. It allows for dynamic
allocation and deallocation of memory blocks for individual requests, minimizing
memory fragmentation and maximizing utilization.

Core Components:

PagedKVCache:
    - Initializes and manages the main KV cache memory pools (_k_cache_pool, _v_cache_pool).
    - Handles allocation of cache space for new inference requests (`add_request`).
    - Supports extending cache space for ongoing requests (`extend_for_request`).
    - Releases cache space when requests are completed (`release_request`).
    - Provides methods to retrieve KV tensors for specified token positions within a
      sequence (`gather_kv_cache`). Currently, this uses a table lookup approach.
      This may be updated in the future with a custom Paged Flash Attention kernel
      for improved performance.
    - Allows updating KV tensors in the cache (`update_kv_cache`).
    - Utilizes a `BlockManager` to handle the underlying block allocations.

BlockManager:
    - Manages a collection of fixed-size memory `Block` objects.
    - Maintains a list of free blocks and handles allocation (`allocate_block`)
      and deallocation (`free_block`) requests.
    - Tracks reference counts for blocks to support potential future features like
      prefix caching (though not explicitly used for eviction in the current design).

SequenceKVCache:
    - Represents the KV cache state for a single inference request.
    - Stores the `request_id`, a list of `block_ids` allocated to it, and the
      `num_tokens` it currently holds.

Block:
    - A dataclass representing a fixed-size memory block.
    - Contains an `id` and a `ref_count`.

Current Design Considerations:
    - No LRU or complex eviction strategy is implemented to keep the system simpler
      and more deterministic for now. Blocks are freed when requests are explicitly released.
    - Assumes separate K and V cache pools.

Future TODOs:
    1. Support block reuse - prefix caching (ref_counts in Block lay groundwork).
    2. Support chunked prefill (current design allows allocation per request).
    3. Implement a Metal kernel for Paged Flash Attention.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import mlx.core as mx

from parallax.server.request import Request, RequestStatus
from parallax.server.server_info import HardwareInfo
from parallax.utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class Block:
    """A fixed-size memory block for storing key-value pairs.

    Attributes:
        id: The ID of the block.
        ref_count: The number of references to this block.
    """

    id: int
    ref_count: int = 0

    def reset(self):
        """Resets the reference count of the block."""
        self.ref_count = 0


@dataclass
class SequenceKVCache:
    """A sequence of key-value pairs for a specific request.

    Attributes:
        request_id: The ID of the request.
        block_ids: The list of block IDs allocated to this sequence.
        num_tokens: The total number of tokens stored for this sequence.
    """

    request_id: str
    block_ids: List[int] = field(default_factory=list)
    num_tokens: int = 0

    def add_block(self, block_id: int):
        """Adds a block to this sequence's cache."""
        self.block_ids.append(block_id)

    def get_block_ids(self) -> List[int]:
        """Returns the list of block IDs allocated to this sequence."""
        return self.block_ids

    def update_token_count(self, num_new_tokens: int):
        """Updates the total number of tokens stored for this sequence."""
        self.num_tokens += num_new_tokens

    def get_token_count(self) -> int:
        """Returns the total number of tokens stored for this sequence."""
        return self.num_tokens


class BlockManager:
    """Manages a collection of memory blocks for efficient allocation."""

    def __init__(self, num_total_blocks: int, block_size: int):
        """
        Args:
            num_total_blocks: The total number of blocks to allocate.
            block_size: The size of each block.
        """
        self.block_size = block_size
        self._blocks: List[Block] = [Block(id=i) for i in range(num_total_blocks)]
        self._free_block_ids: List[int] = list(range(num_total_blocks))

    def allocate_block(self) -> Optional[int]:
        """
        Allocates a free block.
        Increments its reference count and removes it from the free list.
        Returns the block ID if allocation is successful, otherwise None.
        """
        if not self._free_block_ids:
            return None
        block_id = self._free_block_ids.pop()
        self._blocks[block_id].ref_count += 1
        return block_id

    def free_block(self, block_id: int):
        """
        Frees an allocated block.
        Decrements its reference count. If the reference count becomes zero,
        the block is added back to the free list.
        """
        if block_id < 0 or block_id > len(self._blocks):
            logger.error(f"Attempted to free invalid block_id: {block_id}")
            return

        block = self._blocks[block_id]
        block.ref_count -= 1
        if block.ref_count == 0:
            block.reset()
            self._free_block_ids.append(block_id)
        elif block.ref_count < 0:
            raise ValueError(f"Block {block_id} ref_count went below zero.")

    def get_num_free_blocks(self) -> int:
        """Returns the number of currently free blocks."""
        return len(self._free_block_ids)

    def can_allocate(self, num_blocks_needed: int) -> bool:
        """Checks if a specified number of blocks can be allocated."""
        return self.get_num_free_blocks() >= num_blocks_needed


class PagedKVCache:
    """High-level abstraction of paged key-value cache."""

    def __init__(
        self,
        block_size: int,
        num_kv_heads: int,
        head_dim: int,
        num_layers: int,
        dtype: mx.Dtype = mx.float16,
        kv_cache_memory_fraction: float = 0.8,
        max_tokens: Optional[int] = None,
        kv_pool_size: Optional[int] = None,
    ):
        """
        Args:
            block_size: The size of each block.
            num_kv_heads: The number of key-value heads.
            head_dim: The dimension of each head.
            num_layers: The number of layers.
            dtype: The data type of the cache.
            kv_cache_memory_fraction: The fraction of total memory to use for the KV cache.
            max_tokens: The maximum number of tokens the cache can support.
            kv_pool_size: The size of the KV cache pool.
        """
        self.dtype = dtype
        self.block_size = block_size
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.num_layers = num_layers

        self.hw_info = HardwareInfo.detect()
        free_memory_in_bytes = self.hw_info.total_ram_gb * 1024**3 - mx.get_active_memory()

        self.kv_pool_size = round(free_memory_in_bytes * kv_cache_memory_fraction)
        if kv_pool_size:
            self.kv_pool_size = min(self.kv_pool_size, kv_pool_size)
        per_token_cache_size = num_layers * num_kv_heads * 2 * head_dim * dtype.size
        # Maximum number of tokens that can be stored in the cache pool.
        self.max_tokens: int = self.kv_pool_size // per_token_cache_size
        if max_tokens:
            self.max_tokens = min(self.max_tokens, max_tokens)
        self.kv_pool_size = self.max_tokens * per_token_cache_size
        # A single block stores (num_layers, num_kv_heads, block_size, head_dim)
        self.num_blocks: int = self.max_tokens // block_size

        self._block_manager = BlockManager(
            num_total_blocks=self.num_blocks, block_size=self.block_size
        )
        self._sequences: Dict[str, SequenceKVCache] = {}

        # Initialize KV Cache Pools
        self._k_cache_pool = mx.zeros(
            (num_layers, self.num_blocks, num_kv_heads, block_size, head_dim),
            dtype=self.dtype,
        )
        self._v_cache_pool = mx.zeros(
            (num_layers, self.num_blocks, num_kv_heads, block_size, head_dim),
            dtype=self.dtype,
        )

        logger.info(
            "KV Cache Memory takes %.2fGB, for %d blocks, supporting %d tokens",
            self.kv_pool_size / 1024**3,
            self.num_blocks,
            self.max_tokens,
        )
        logger.info(f"KV Cache Pool Shape (K/V): {self._k_cache_pool.shape}")

    def _num_blocks_needed_for_tokens(self, num_tokens: int) -> int:
        """Calculates how many blocks are needed for a given number of tokens."""
        return (num_tokens + self.block_size - 1) // self.block_size

    def has_request(self, request_id: str) -> bool:
        """Checks if a request is in the cache."""
        return request_id in self._sequences

    def add_request(self, request: Request, num_initial_tokens: int) -> bool:
        """
        Allocates initial KV cache blocks for a new request in prefill state.

        Args:
            request: The request to allocate for.
            num_initial_tokens: The number of tokens to allocate for.

        Returns:
            True if allocation was successful, False otherwise.
        """
        assert (
            request.status == RequestStatus.PREFILLING
        ), "add_request can only be called in prefill state."
        if request.request_id in self._sequences:
            logger.warning(f"Request {request.request_id} already has allocated cache.")
            # For now we don't handle re-allocation.
            return True

        num_blocks_needed = self._num_blocks_needed_for_tokens(num_initial_tokens)
        if not self._block_manager.can_allocate(num_blocks_needed):
            logger.error(
                f"Cannot allocate {num_blocks_needed} blocks for request {request.request_id}. "
                f"Not enough free blocks. Free blocks: {self._block_manager.get_num_free_blocks()}"
            )
            return False

        sequence_cache = SequenceKVCache(request_id=request.request_id)
        for _ in range(num_blocks_needed):
            block_id = self._block_manager.allocate_block()
            if block_id is None:
                logger.error(
                    f"Block allocation failed mid-process for request"
                    f"{request.request_id}. Rolling back."
                )
                for bid in sequence_cache.get_block_ids():
                    self._block_manager.free_block(bid)
                return False
            sequence_cache.add_block(block_id)

        sequence_cache.update_token_count(num_initial_tokens)
        self._sequences[request.request_id] = sequence_cache
        logger.info(
            f"Allocated {num_blocks_needed} blocks for "
            f"{num_initial_tokens} tokens for request {request.request_id}."
        )
        return True

    def get_num_tokens_for_request(self, request_id: str) -> int:
        """Returns the number of tokens stored for a request."""
        if request_id not in self._sequences:
            logger.error(f"Request {request_id} not found for getting token count.")
            return 0
        return self._sequences[request_id].get_token_count()

    def extend_for_request(self, request_id: str, num_additional_tokens: int) -> bool:
        """
        Extends KV cache for an existing request to accommodate more tokens.

        Args:
            request_id: The ID of the request to extend.
            num_additional_tokens: The number of additional tokens to allocate.

        Returns:
            True if extension was successful, False otherwise.
        """
        if request_id not in self._sequences:
            logger.error(f"Request {request_id} not found for extending cache.")
            return False

        sequence_cache = self._sequences[request_id]
        current_tokens = sequence_cache.get_token_count()
        new_total_tokens = current_tokens + num_additional_tokens

        current_blocks_allocated = len(sequence_cache.get_block_ids())
        required_total_blocks = self._num_blocks_needed_for_tokens(new_total_tokens)
        additional_blocks_needed = required_total_blocks - current_blocks_allocated

        if additional_blocks_needed <= 0:  # No new blocks needed
            sequence_cache.update_token_count(num_additional_tokens)
            return True

        if not self._block_manager.can_allocate(additional_blocks_needed):
            logger.error(
                f"Cannot allocate {additional_blocks_needed} additional blocks "
                f"for request {request_id}. "
                f"Not enough free blocks. Free blocks: {self._block_manager.get_num_free_blocks()}"
            )
            return False

        newly_allocated_block_ids = []
        for _ in range(additional_blocks_needed):
            block_id = self._block_manager.allocate_block()
            if block_id is None:
                logger.error(
                    f"Block allocation failed mid-extension for request "
                    f" {request_id}. Rolling back extension."
                )
                for bid in newly_allocated_block_ids:
                    self._block_manager.free_block(bid)
                return False
            sequence_cache.add_block(block_id)
            newly_allocated_block_ids.append(block_id)

        sequence_cache.update_token_count(num_additional_tokens)
        logger.info(
            f"Extended cache for request {request_id} "
            f" by {additional_blocks_needed} blocks for {num_additional_tokens} new tokens."
        )
        return True

    def release_request(self, request_id: str):
        """Frees all KV cache blocks associated with a completed or cancelled request."""
        if request_id not in self._sequences:
            logger.warning(f"Attempted to release non-existent request {request_id}.")
            return

        sequence_cache = self._sequences.pop(request_id)
        for block_id in sequence_cache.get_block_ids():
            self._block_manager.free_block(block_id)
        logger.info(
            f"Released {len(sequence_cache.get_block_ids())} blocks for request {request_id}."
        )

    def has_capacity_for_tokens(self, num_tokens: int) -> bool:
        """
        Checks if the cache can accommodate a certain number of new tokens
        (potentially for a new request or a small extension).
        This is a more general check than can_allocate in BlockManager as it considers block_size.
        """
        num_blocks_needed = self._num_blocks_needed_for_tokens(num_tokens)
        return self._block_manager.can_allocate(num_blocks_needed)

    def _get_physical_locations(
        self, sequence: SequenceKVCache, token_indices_in_sequence: List[int]
    ) -> List[Tuple[int, int]]:
        """Maps logical token indices within a sequence to physical (block_id, offset_in_block)

        Args:
            sequence: The sequence to get the physical locations for.
            token_indices_in_sequence: The list of token indices to get the physical locations for.

        Returns:
            A list of tuples, each containing a block_id and an offset_in_block.
        """
        locations = []
        block_ids = sequence.get_block_ids()
        for token_idx in token_indices_in_sequence:
            if token_idx < 0 or token_idx >= sequence.get_token_count():
                raise IndexError(
                    f"Token index {token_idx} out of range for sequence "
                    f"{sequence.request_id} with {sequence.get_token_count()} tokens."
                )

            block_num_in_sequence = token_idx // self.block_size
            offset_in_block = token_idx % self.block_size

            if block_num_in_sequence >= len(block_ids):
                raise ValueError(
                    f"Token index {token_idx} requires block"
                    f" {block_num_in_sequence} but sequence {sequence.request_id} "
                    f"only has {len(block_ids)} blocks allocated. "
                    f"Total tokens: {sequence.get_token_count()}"
                )

            physical_block_id = block_ids[block_num_in_sequence]
            locations.append((physical_block_id, offset_in_block))
        return locations

    def gather_kv_cache(
        self, request_id: str, token_indices_in_sequence: Optional[List[int]] = None
    ) -> Tuple[Optional[mx.array], Optional[mx.array]]:
        # pylint: disable=too-many-locals
        """
        Gathers KV tensors for specified token indices within a sequence.

        If token_indices_in_sequence is None, it gathers all tokens for the request.
        If token_indices_in_sequence is an empty list, it returns empty tensors.

        Args:
            request_id: The ID of the request.
            token_indices_in_sequence: A list of token indices to gather.

        Returns:
            A tuple (k_gathered, v_gathered) of the gathered KV tensors.
            The shape of the returned tensors is
            (num_layers, num_tokens_retrieved, num_kv_heads, head_dim).
            Returns (None, None) if the request is not found.
        """
        if request_id not in self._sequences:
            logger.error(f"Request {request_id} not found in KV cache.")
            return None, None

        sequence = self._sequences[request_id]

        if token_indices_in_sequence == []:
            k_empty = mx.zeros(
                (self.num_layers, 0, self.num_kv_heads, self.head_dim), dtype=self.dtype
            )
            v_empty = mx.zeros(
                (self.num_layers, 0, self.num_kv_heads, self.head_dim), dtype=self.dtype
            )
            return k_empty, v_empty

        # If token_indices_in_sequence is None, gather all tokens.
        if token_indices_in_sequence is None:
            num_tokens = sequence.get_token_count()
            indices_to_gather = list(range(num_tokens))
        else:
            indices_to_gather = token_indices_in_sequence

        if not indices_to_gather:
            raise ValueError("token_indices_in_sequence cannot be empty for gather.")

        logger.debug(
            f"Gathering KV cache for request {request_id} with indices: {indices_to_gather}"
        )
        physical_locations = self._get_physical_locations(sequence, indices_to_gather)

        unique_block_ids = sorted(list({loc[0] for loc in physical_locations}))

        # Pre-load required blocks from the main pool to avoid repeated large-scale indexing.
        # This is more efficient as it performs fewer, larger data movements.
        k_blocks_loaded = {
            block_id: self._k_cache_pool[:, block_id, :, :, :] for block_id in unique_block_ids
        }
        v_blocks_loaded = {
            block_id: self._v_cache_pool[:, block_id, :, :, :] for block_id in unique_block_ids
        }

        k_gathered_slices = []
        v_gathered_slices = []

        for block_id, offset_in_block in physical_locations:
            # Shape of k_blocks_loaded[block_id] is (num_layers, num_kv_heads, block_size, head_dim)
            # Each slice has shape (num_layers, num_kv_heads, head_dim)
            k_slice = k_blocks_loaded[block_id][:, :, offset_in_block, :]
            v_slice = v_blocks_loaded[block_id][:, :, offset_in_block, :]

            k_gathered_slices.append(k_slice)
            v_gathered_slices.append(v_slice)

        if not k_gathered_slices:
            k_empty = mx.zeros(
                (self.num_layers, 0, self.num_kv_heads, self.head_dim), dtype=self.dtype
            )
            return k_empty, k_empty.copy()

        k_stacked = mx.stack(k_gathered_slices, axis=0)
        v_stacked = mx.stack(v_gathered_slices, axis=0)

        # (num_layers, num_tokens_retrieved, num_kv_heads, head_dim)
        k_final = k_stacked.transpose(1, 0, 2, 3)
        v_final = v_stacked.transpose(1, 0, 2, 3)

        return k_final, v_final

    def update_kv_cache(
        self,
        request_id: str,
        token_indices_in_sequence: List[int],
        k_values: mx.array,
        v_values: mx.array,
    ):
        """
        Writes new K and V values into the cache for specified token positions.

        Args:
            request_id: The ID of the request to update.
            token_indices_in_sequence: The list of token indices in the sequence to update.
            k_values: The new K values to write into the cache.
                        (num_layers, num_tokens_to_update, num_kv_heads, head_dim)
            v_values: The new V values to write into the cache.
                        (num_layers, num_tokens_to_update, num_kv_heads, head_dim)
        """
        if request_id not in self._sequences:
            raise ValueError(f"Request {request_id} not found for updating KV cache.")
        if not token_indices_in_sequence:
            raise ValueError("token_indices_in_sequence cannot be empty for update.")

        sequence = self._sequences[request_id]
        if k_values.shape[1] != len(token_indices_in_sequence) or v_values.shape[1] != len(
            token_indices_in_sequence
        ):
            raise ValueError(
                f"Mismatch between number of token indices ({len(token_indices_in_sequence)}) and "
                f"k_values/v_values shape ({k_values.shape[1]}/{v_values.shape[1]}) "
                f"for request {request_id}."
            )

        if k_values.shape[0] != self.num_layers or v_values.shape[0] != self.num_layers:
            raise ValueError(
                f"Mismatch in num_layers. Expected {self.num_layers}, "
                f"got k: {k_values.shape[0]}, v: {v_values.shape[0]}"
            )

        try:
            logger.debug(
                f"Updating KV cache for request {request_id} with ind: {token_indices_in_sequence}"
            )
            physical_locations = self._get_physical_locations(sequence, token_indices_in_sequence)
        except (IndexError, ValueError) as e:
            logger.error(
                f"Error getting physical locations for request {request_id} during update: {e}"
            )
            return

        for i, (physical_block_id, offset_in_block) in enumerate(physical_locations):
            # k_values_for_token has shape (num_layers, num_kv_heads, head_dim)
            k_values_for_token = k_values[:, i, :, :]
            v_values_for_token = v_values[:, i, :, :]

            # Update the cache pools directly
            # The slices _k_cache_pool[:, physical_block_id, :, offset_in_block, :]
            # also have shape (num_layers, num_kv_heads, head_dim)
            self._k_cache_pool[:, physical_block_id, :, offset_in_block, :] = k_values_for_token
            self._v_cache_pool[:, physical_block_id, :, offset_in_block, :] = v_values_for_token

        logger.debug(
            f"Updated KV cache for {len(token_indices_in_sequence)} "
            f"tokens for request {request_id}."
        )
