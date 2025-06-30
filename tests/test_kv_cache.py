# pylint: disable=missing-function-docstring,missing-module-docstring,protected-access, too-many-locals,redefined-outer-name
# TODO: add more tests
"""Tests for the Parallax server's key-value cache system."""
import mlx.core as mx
import pytest

from parallax.server.kv_cache import BlockManager, PagedKVCache, SequenceKVCache
from parallax.server.request import Request


@pytest.fixture(autouse=True)
def mock_hardware_info(mocker):
    mock_hw = mocker.patch("parallax.server.kv_cache.HardwareInfo.detect")
    mock_hw.return_value.total_ram_gb = 16

    mock_mem = mocker.patch("parallax.server.kv_cache.mx.get_active_memory")
    mock_mem.return_value = 1 * 1024**3
    return mock_hw, mock_mem


def test_sequence_kv_cache_add_block():
    seq_cache = SequenceKVCache(request_id="req1")
    seq_cache.add_block(10)
    seq_cache.add_block(20)
    assert seq_cache.get_block_ids() == [10, 20]
    seq_cache.update_token_count(15)
    assert seq_cache.get_token_count() == 15
    seq_cache.update_token_count(10)
    assert seq_cache.get_token_count() == 25


# --- BlockManager Tests ---
@pytest.fixture
def block_manager_fixture():
    return BlockManager(num_total_blocks=10, block_size=4)


def test_block_manager_allocate_block(block_manager_fixture):
    bm = block_manager_fixture
    block_id = bm.allocate_block()
    assert block_id is not None
    assert 0 <= block_id < 10
    assert bm._blocks[block_id].ref_count == 1
    assert bm.get_num_free_blocks() == 9
    assert block_id not in bm._free_block_ids

    # Allocate all blocks
    for _ in range(9):
        assert bm.allocate_block() is not None
    assert bm.get_num_free_blocks() == 0
    assert bm.allocate_block() is None  # No more blocks


def test_block_manager_free_block(block_manager_fixture):
    bm = block_manager_fixture
    b_id1 = bm.allocate_block()  # ref_count = 1
    b_id2 = bm.allocate_block()  # ref_count = 1
    bm._blocks[b_id2].ref_count = 2  # Simulate shared block for testing decrement

    bm.free_block(b_id1)
    assert bm._blocks[b_id1].ref_count == 0
    assert b_id1 in bm._free_block_ids
    assert (
        bm.get_num_free_blocks() == 9
    )  # 10 total - 1 (b_id2 still allocated) + 0 (b_id1 now free)

    bm.free_block(b_id2)  # ref_count becomes 1
    assert bm._blocks[b_id2].ref_count == 1
    assert b_id2 not in bm._free_block_ids
    assert bm.get_num_free_blocks() == 9

    bm.free_block(b_id2)  # ref_count becomes 0
    assert bm._blocks[b_id2].ref_count == 0
    assert b_id2 in bm._free_block_ids
    assert bm.get_num_free_blocks() == 10

    # Test freeing invalid block id (should log error, not raise)
    bm.free_block(100)
    with pytest.raises(ValueError):
        bm.free_block(b_id1)
    assert b_id1 in bm._free_block_ids


def test_block_manager_can_allocate(block_manager_fixture):
    bm = block_manager_fixture
    assert bm.can_allocate(5)
    assert bm.can_allocate(10)
    assert not bm.can_allocate(11)

    bm.allocate_block()
    assert bm.can_allocate(9)
    assert not bm.can_allocate(10)


# --- PagedKVCache Tests ---
@pytest.fixture
def paged_kv_cache_params():
    # Using small values for easier testing
    return {
        "block_size": 4,
        "num_kv_heads": 2,
        "head_dim": 8,  # multiple of 8 for float16 (2 bytes)
        "num_layers": 1,
        "dtype": mx.float16,
        # Force a small, deterministic pool size for testing limits
        # 1 layer * 2 heads * 2 (K/V) * 8 head_dim * 2 bytes/float16 = 64 bytes per token
        # Max tokens = (1024 bytes / 64 bytes/token) = 16 tokens
        # Num blocks = 16 tokens / 4 tokens/block = 4 blocks
        "kv_pool_size": 1024,  # 1KB for cache pool
    }


@pytest.fixture
def paged_kv_cache(paged_kv_cache_params):
    return PagedKVCache(**paged_kv_cache_params)


def test_paged_kv_cache_initialization(paged_kv_cache, paged_kv_cache_params):
    kv = paged_kv_cache
    params = paged_kv_cache_params

    # Based on kv_pool_size = 1024 bytes
    # per_token_cache_size = 1 * 2 * 2 * 8 * 2 = 64 bytes
    # max_tokens = 1024 // 64 = 16
    # num_blocks = 16 // 4 = 4
    assert kv.max_tokens == 16
    assert kv.num_blocks == 4
    assert kv._block_manager.get_num_free_blocks() == 4
    assert len(kv._block_manager._blocks) == 4

    expected_shape = (
        params["num_layers"],
        kv.num_blocks,
        params["num_kv_heads"],
        params["head_dim"],
        params["block_size"],
    )
    assert kv._k_cache_pool.shape == expected_shape
    assert kv._v_cache_pool.shape == expected_shape
    assert kv._k_cache_pool.dtype == params["dtype"]
    assert kv._v_cache_pool.dtype == params["dtype"]
    assert len(kv._sequences) == 0


def test_paged_kv_cache_add_request(paged_kv_cache):
    kv = paged_kv_cache  # 4 blocks total, block_size 4
    req1 = Request(request_id="req1")

    # Allocate 3 tokens (needs 1 block)
    assert kv.add_request(req1, num_initial_tokens=3)
    assert "req1" in kv._sequences
    assert kv._sequences["req1"].get_token_count() == 3
    assert len(kv._sequences["req1"].get_block_ids()) == 1
    assert kv._block_manager.get_num_free_blocks() == 3  # 4 - 1

    # Attempt to re-allocate (should warn and return True/False based on impl.)
    # Current impl logs warning and returns True
    assert kv.add_request(req1, num_initial_tokens=1)
    assert kv._block_manager.get_num_free_blocks() == 3  # No change

    req2 = Request(request_id="req2")
    # Allocate 8 tokens (needs 2 blocks)
    assert kv.add_request(req2, num_initial_tokens=8)
    assert "req2" in kv._sequences
    assert kv._sequences["req2"].get_token_count() == 8
    assert len(kv._sequences["req2"].get_block_ids()) == 2
    assert kv._block_manager.get_num_free_blocks() == 1  # 3 - 2

    req3 = Request(request_id="req3")
    # Allocate 4 tokens (needs 1 block, which is available)
    assert kv.add_request(req3, num_initial_tokens=4)
    assert "req3" in kv._sequences
    assert len(kv._sequences["req3"].get_block_ids()) == 1
    assert kv._block_manager.get_num_free_blocks() == 0  # 1 - 1

    req4 = Request(request_id="req4")
    # Attempt to allocate 1 token (needs 1 block, none available)
    assert not kv.add_request(req4, num_initial_tokens=1)
    assert "req4" not in kv._sequences
    assert kv._block_manager.get_num_free_blocks() == 0


def test_paged_kv_cache_extend_for_request(paged_kv_cache):
    kv = paged_kv_cache  # 4 blocks total, block_size 4
    req1_id = "req-extend-1"
    req1 = Request(request_id=req1_id)
    kv.add_request(req1, num_initial_tokens=2)  # Needs 1 block, count=2. 3 blocks free.

    # Extend by 1 token, still fits in 1 block
    assert kv.extend_for_request(req1_id, num_additional_tokens=1)
    assert kv._sequences[req1_id].get_token_count() == 3
    assert len(kv._sequences[req1_id].get_block_ids()) == 1
    assert kv._block_manager.get_num_free_blocks() == 3

    # Extend by 2 tokens, needs 1 more block (total 3+2=5 tokens -> 2 blocks)
    assert kv.extend_for_request(req1_id, num_additional_tokens=2)
    assert kv._sequences[req1_id].get_token_count() == 5
    assert len(kv._sequences[req1_id].get_block_ids()) == 2
    assert kv._block_manager.get_num_free_blocks() == 2  # 3 - 1 new block

    # Extend non-existent request
    assert not kv.extend_for_request("non-existent-req", num_additional_tokens=1)

    # Extend to fill all blocks
    # req1 has 5 tokens, 2 blocks. 2 blocks free.
    # Extend by 7 tokens (total 5+7=12 tokens -> 3 blocks). Needs 1 more block.
    assert kv.extend_for_request(req1_id, num_additional_tokens=7)
    assert kv._sequences[req1_id].get_token_count() == 12
    assert len(kv._sequences[req1_id].get_block_ids()) == 3
    assert kv._block_manager.get_num_free_blocks() == 1  # 2 - 1 new block

    # Attempt to extend beyond capacity
    # req1 has 12 tokens, 3 blocks. 1 block free. Max tokens = 16.
    # Try to add 5 more tokens (total 17, needs 5 blocks, current 3, need 2 new blocks, only 1 free)
    assert not kv.extend_for_request(req1_id, num_additional_tokens=5)
    assert kv._sequences[req1_id].get_token_count() == 12  # Should not change
    assert len(kv._sequences[req1_id].get_block_ids()) == 3  # Should not change
    assert kv._block_manager.get_num_free_blocks() == 1  # Should not change due to rollback

    # Successfully use the last block
    # Extend by 4 tokens (total 12+4=16 tokens -> 4 blocks). Needs 1 more block.
    assert kv.extend_for_request(req1_id, num_additional_tokens=4)
    assert kv._sequences[req1_id].get_token_count() == 16
    assert len(kv._sequences[req1_id].get_block_ids()) == 4
    assert kv._block_manager.get_num_free_blocks() == 0


def test_paged_kv_cache_release_request(paged_kv_cache):
    kv = paged_kv_cache
    req1_id = "req-release-1"
    req1 = Request(request_id=req1_id)
    kv.add_request(req1, num_initial_tokens=6)  # Needs 2 blocks
    assert kv._block_manager.get_num_free_blocks() == 2  # 4 - 2

    kv.release_request(req1_id)
    assert req1_id not in kv._sequences
    assert kv._block_manager.get_num_free_blocks() == 4  # All blocks freed

    # Release non-existent request (should not error, just warn)
    kv.release_request("non-existent-req")
    assert kv._block_manager.get_num_free_blocks() == 4


def test_paged_kv_cache_has_capacity_for_tokens(paged_kv_cache):
    kv = paged_kv_cache  # 4 blocks total, block_size 4
    assert kv.has_capacity_for_tokens(1)  # Needs 1 block
    assert kv.has_capacity_for_tokens(4)  # Needs 1 block
    assert kv.has_capacity_for_tokens(5)  # Needs 2 blocks
    assert kv.has_capacity_for_tokens(16)  # Needs 4 blocks
    assert not kv.has_capacity_for_tokens(17)  # Needs 5 blocks

    req1 = Request(request_id="req-cap-1")
    kv.add_request(req1, num_initial_tokens=10)  # Needs 3 blocks. 1 block free.
    assert kv.has_capacity_for_tokens(1)  # Needs 1 block, available
    assert kv.has_capacity_for_tokens(4)  # Needs 1 block, available
    assert not kv.has_capacity_for_tokens(5)  # Needs 2 blocks, only 1 available


def test_paged_kv_cache_get_physical_locations(paged_kv_cache):
    kv = paged_kv_cache  # block_size 4
    req1_id = "req-loc-1"
    req1 = Request(request_id=req1_id)
    kv.add_request(
        req1, num_initial_tokens=10
    )  # Needs 3 blocks. Tokens 0-3 (b0), 4-7 (b1), 8-9 (b2)
    sequence = kv._sequences[req1_id]

    # Assume block IDs allocated are 0, 1, 2 in order for simplicity of testing logic here
    # (In reality, they could be any free block IDs)
    # For robust testing, we should get the actual block IDs
    actual_block_ids = sequence.get_block_ids()
    assert len(actual_block_ids) == 3

    locs = kv._get_physical_locations(sequence, [0, 3, 4, 7, 8, 9])
    expected_locs = [
        (actual_block_ids[0], 0),
        (actual_block_ids[0], 3),  # Block 0
        (actual_block_ids[1], 0),
        (actual_block_ids[1], 3),  # Block 1
        (actual_block_ids[2], 0),
        (actual_block_ids[2], 1),  # Block 2
    ]
    assert locs == expected_locs

    with pytest.raises(IndexError):  # Token index out of range
        kv._get_physical_locations(sequence, [10])

    with pytest.raises(IndexError):
        kv._get_physical_locations(sequence, [-1])

    # Test empty list
    assert kv._get_physical_locations(sequence, []) == []
