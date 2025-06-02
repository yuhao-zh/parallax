"""Tests for the Parallax server's key-value cache system."""

# pylint: disable=missing-function-docstring,missing-module-docstring,protected-access, too-many-locals,redefined-outer-name
import mlx.core as mx
import numpy as np
import pytest

from parallax.server.kv_cache import BlockManager, PagedKVCache, SequenceKVCache
from parallax.server.request import Request


# Mock HardwareInfo and get_active_memory to avoid environment dependencies in tests
@pytest.fixture(autouse=True)
def mock_hardware_info(mocker):
    mock_hw = mocker.patch("parallax.server.kv_cache.HardwareInfo.detect")
    # Simulate 16GB total RAM for consistent calculations if not overridden by kv_pool_size
    mock_hw.return_value.total_ram_gb = 16

    mock_mem = mocker.patch("parallax.server.kv_cache.mx.get_active_memory")
    # Simulate 1GB already active memory
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
        params["block_size"],
        params["head_dim"],
    )
    assert kv._k_cache_pool.shape == expected_shape
    assert kv._v_cache_pool.shape == expected_shape
    assert kv._k_cache_pool.dtype == params["dtype"]
    assert kv._v_cache_pool.dtype == params["dtype"]
    assert len(kv._sequences) == 0


def test_paged_kv_cache_allocate_for_request(paged_kv_cache):
    kv = paged_kv_cache  # 4 blocks total, block_size 4
    req1 = Request(request_id="req1")

    # Allocate 3 tokens (needs 1 block)
    assert kv.allocate_for_request(req1, num_initial_tokens=3)
    assert "req1" in kv._sequences
    assert kv._sequences["req1"].get_token_count() == 3
    assert len(kv._sequences["req1"].get_block_ids()) == 1
    assert kv._block_manager.get_num_free_blocks() == 3  # 4 - 1

    # Attempt to re-allocate (should warn and return True/False based on impl.)
    # Current impl logs warning and returns True
    assert kv.allocate_for_request(req1, num_initial_tokens=1)
    assert kv._block_manager.get_num_free_blocks() == 3  # No change

    req2 = Request(request_id="req2")
    # Allocate 8 tokens (needs 2 blocks)
    assert kv.allocate_for_request(req2, num_initial_tokens=8)
    assert "req2" in kv._sequences
    assert kv._sequences["req2"].get_token_count() == 8
    assert len(kv._sequences["req2"].get_block_ids()) == 2
    assert kv._block_manager.get_num_free_blocks() == 1  # 3 - 2

    req3 = Request(request_id="req3")
    # Allocate 4 tokens (needs 1 block, which is available)
    assert kv.allocate_for_request(req3, num_initial_tokens=4)
    assert "req3" in kv._sequences
    assert len(kv._sequences["req3"].get_block_ids()) == 1
    assert kv._block_manager.get_num_free_blocks() == 0  # 1 - 1

    req4 = Request(request_id="req4")
    # Attempt to allocate 1 token (needs 1 block, none available)
    assert not kv.allocate_for_request(req4, num_initial_tokens=1)
    assert "req4" not in kv._sequences
    assert kv._block_manager.get_num_free_blocks() == 0


def test_paged_kv_cache_extend_for_request(paged_kv_cache):
    kv = paged_kv_cache  # 4 blocks total, block_size 4
    req1_id = "req-extend-1"
    req1 = Request(request_id=req1_id)
    kv.allocate_for_request(req1, num_initial_tokens=2)  # Needs 1 block, count=2. 3 blocks free.

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
    kv.allocate_for_request(req1, num_initial_tokens=6)  # Needs 2 blocks
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
    kv.allocate_for_request(req1, num_initial_tokens=10)  # Needs 3 blocks. 1 block free.
    assert kv.has_capacity_for_tokens(1)  # Needs 1 block, available
    assert kv.has_capacity_for_tokens(4)  # Needs 1 block, available
    assert not kv.has_capacity_for_tokens(5)  # Needs 2 blocks, only 1 available


def test_paged_kv_cache_get_physical_locations(paged_kv_cache):
    kv = paged_kv_cache  # block_size 4
    req1_id = "req-loc-1"
    req1 = Request(request_id=req1_id)
    kv.allocate_for_request(
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


def test_paged_kv_cache_gather_update_kv_cache(paged_kv_cache, paged_kv_cache_params):
    kv = paged_kv_cache
    params = paged_kv_cache_params
    req_id = "req-gather-update"
    req = Request(request_id=req_id)

    num_tokens_to_alloc = 7  # Needs 2 blocks (block_size 4)
    assert kv.allocate_for_request(req, num_initial_tokens=num_tokens_to_alloc)

    # Create some dummy K/V values to update
    # Shape: (num_layers, num_tokens_to_update, num_kv_heads, head_dim)
    num_tokens_to_update = 3
    token_indices_to_update = [1, 3, 5]

    k_update_data_np = np.random.rand(
        params["num_layers"], num_tokens_to_update, params["num_kv_heads"], params["head_dim"]
    ).astype(np.float16)
    v_update_data_np = np.random.rand(
        params["num_layers"], num_tokens_to_update, params["num_kv_heads"], params["head_dim"]
    ).astype(np.float16)
    k_update_values = mx.array(k_update_data_np)
    v_update_values = mx.array(v_update_data_np)

    kv.update_kv_cache(req_id, token_indices_to_update, k_update_values, v_update_values)
    mx.eval(kv._k_cache_pool, kv._v_cache_pool)  # Ensure update is processed

    # Gather the updated tokens and some others
    token_indices_to_gather = [0, 1, 2, 3, 4, 5, 6]  # Gather all allocated tokens
    gathered_k, gathered_v = kv.gather_kv_cache(req_id, token_indices_to_gather)
    mx.eval(gathered_k, gathered_v)

    assert gathered_k is not None and gathered_v is not None
    expected_gathered_shape = (
        params["num_layers"],
        len(token_indices_to_gather),
        params["num_kv_heads"],
        params["head_dim"],
    )
    assert gathered_k.shape == expected_gathered_shape
    assert gathered_v.shape == expected_gathered_shape
    assert gathered_k.dtype == params["dtype"]
    assert gathered_v.dtype == params["dtype"]

    # Verify that the updated values are correct
    # token_indices_to_update = [1, 3, 5]
    # k_update_data_np was (L, 3, H, D), where 3 corresponds to [1,3,5]
    # gathered_k is (L, 7, H, D), where 7 corresponds to [0,1,2,3,4,5,6]

    # Check token at original index 1 (which is at gathered index 1)
    assert np.array_equal(np.array(gathered_k[:, 1, :, :]), k_update_data_np[:, 0, :, :])
    assert np.array_equal(np.array(gathered_v[:, 1, :, :]), v_update_data_np[:, 0, :, :])

    # Check token at original index 3 (which is at gathered index 3)
    assert np.array_equal(np.array(gathered_k[:, 3, :, :]), k_update_data_np[:, 1, :, :])
    assert np.array_equal(np.array(gathered_v[:, 3, :, :]), v_update_data_np[:, 1, :, :])

    # Check token at original index 5 (which is at gathered index 5)
    assert np.array_equal(np.array(gathered_k[:, 5, :, :]), k_update_data_np[:, 2, :, :])
    assert np.array_equal(np.array(gathered_v[:, 5, :, :]), v_update_data_np[:, 2, :, :])

    # Check a non-updated token (e.g., index 0) - should be zeros
    assert np.all(np.array(gathered_k[:, 0, :, :]) == 0)
    assert np.all(np.array(gathered_v[:, 0, :, :]) == 0)

    # Test gather with empty token list
    empty_k, empty_v = kv.gather_kv_cache(req_id, [])
    assert empty_k.shape == (params["num_layers"], 0, params["num_kv_heads"], params["head_dim"])
    assert empty_v.shape == (params["num_layers"], 0, params["num_kv_heads"], params["head_dim"])

    # Test gather with non-existent request_id
    none_k, none_v = kv.gather_kv_cache("non-existent", [0, 1])
    assert none_k is None
    assert none_v is None

    # Test update with mismatched number of tokens
    k_mismatch = mx.array(
        np.random.rand(params["num_layers"], 1, params["num_kv_heads"], params["head_dim"]).astype(
            np.float16
        )
    )
    v_mismatch = mx.array(
        np.random.rand(params["num_layers"], 1, params["num_kv_heads"], params["head_dim"]).astype(
            np.float16
        )
    )
    # logger.error should be called, but function returns None
    kv.update_kv_cache(req_id, [0, 1], k_mismatch, v_mismatch)  # 2 indices, 1 token data

    # Test update with non-existent request
    kv.update_kv_cache("non-existent", [0], k_mismatch, v_mismatch)


def test_paged_kv_cache_update_all_tokens_in_block():
    # Modify params for this specific test: 1 layer, 1 head, small head_dim, 1 block cache
    params = {
        "block_size": 2,
        "num_kv_heads": 1,
        "head_dim": 4,
        "num_layers": 1,
        "dtype": mx.float16,
        "kv_pool_size": 1
        * 1
        * 2
        * 4
        * 2
        * 2,  # L=1,H=1,V=2,D=4,B=2byte * 2tokens = 32 bytes => 1 block of 2 tokens
    }
    kv = PagedKVCache(**params)
    req_id = "req-full-block"
    req = Request(request_id=req_id)

    assert kv.num_blocks == 1
    assert kv.block_size == 2

    assert kv.allocate_for_request(
        req, num_initial_tokens=2
    )  # Allocate all 2 tokens in the single block

    k_update_data = (
        mx.arange(
            params["num_layers"] * 2 * params["num_kv_heads"] * params["head_dim"], dtype=mx.float32
        )
        .reshape(params["num_layers"], 2, params["num_kv_heads"], params["head_dim"])
        .astype(params["dtype"])
    )
    v_update_data = k_update_data + 100  # Make V different

    kv.update_kv_cache(req_id, [0, 1], k_update_data, v_update_data)
    mx.eval(kv._k_cache_pool, kv._v_cache_pool)

    gathered_k, gathered_v = kv.gather_kv_cache(req_id, [0, 1])
    mx.eval(gathered_k, gathered_v)

    assert np.array_equal(np.array(gathered_k), np.array(k_update_data))
    assert np.array_equal(np.array(gathered_v), np.array(v_update_data))

    # Verify raw cache pool content for the first (and only) block
    # k_update_data is (L, num_tokens, H, D)
    # _k_cache_pool is (L, num_blocks, H, block_size, D)
    # For token 0 (offset 0 in block 0):
    # _k_cache_pool[0, block_id, :, 0, :] should be k_update_data[0, 0, :, :]
    # For token 1 (offset 1 in block 0):
    # _k_cache_pool[0, block_id, :, 1, :] should be k_update_data[0, 1, :, :]

    physical_block_id = kv._sequences[req_id].get_block_ids()[0]

    expected_k_in_pool_block_token0 = np.array(k_update_data[0, 0, :, :])  # (H, D)
    actual_k_in_pool_block_token0 = np.array(kv._k_cache_pool[0, physical_block_id, :, 0, :])
    assert np.array_equal(actual_k_in_pool_block_token0, expected_k_in_pool_block_token0)

    expected_k_in_pool_block_token1 = np.array(k_update_data[0, 1, :, :])  # (H, D)
    actual_k_in_pool_block_token1 = np.array(kv._k_cache_pool[0, physical_block_id, :, 1, :])
    assert np.array_equal(actual_k_in_pool_block_token1, expected_k_in_pool_block_token1)

    expected_v_in_pool_block_token0 = np.array(v_update_data[0, 0, :, :])  # (H, D)
    actual_v_in_pool_block_token0 = np.array(kv._v_cache_pool[0, physical_block_id, :, 0, :])
    assert np.array_equal(actual_v_in_pool_block_token0, expected_v_in_pool_block_token0)

    expected_v_in_pool_block_token1 = np.array(v_update_data[0, 1, :, :])  # (H, D)
    actual_v_in_pool_block_token1 = np.array(kv._v_cache_pool[0, physical_block_id, :, 1, :])
    assert np.array_equal(actual_v_in_pool_block_token1, expected_v_in_pool_block_token1)
