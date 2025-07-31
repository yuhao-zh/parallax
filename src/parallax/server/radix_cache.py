"""
Prefix Cache class for KV Cache reuse.
This module is implemented using radix tree, which retains the
same as SGLang.
"""

from collections import defaultdict
from functools import partial
from typing import List, Optional, Tuple, Dict
import time
import heapq

import mlx.core as mx

from parallax.server.kv_cache import KVCache
from parallax.server.request import Request

class TreeNode:
    """
    Radix tree node data structure.
    Key: token id list. It should be an empty list for the root node.
    Value: kv cache positions.
    """
    counter = 0

    def __init__(self, id: Optional[int] = None):
        self.children = defaultdict(TreeNode)
        self.parent: TreeNode = None
        self.key: List[int] = None
        self.value: Optional[List[int]] = None
        self.kv_cache = None
        self.lock_ref = 0
        self.last_access_time = time.monotonic()

        self.hit_count = 0

        self.id = TreeNode.counter if id is None else id
        TreeNode.counter += 1

    @property
    def evicted(self):
        return self.value is None

    def __lt__(self, other: "TreeNode"):
        return self.last_access_time < other.last_access_time

def _key_match_page_size1(key0: List, key1: List):
    i = 0
    for k0, k1 in zip(key0, key1):
        if k0 != k1:
            break
        i += 1
    return i

def _key_match_paged(key0: List, key1: List, page_size: int):
    min_len = min(len(key0), len(key1))

    i = 0
    while i < min_len:
        if key0[i : i + page_size] != key1[i : i + page_size]:
            break
        i += page_size

    return i


class RadixCache:
    """
    Manages Radix Cache for the running executor.
    Note: Currently only support page_size=1.
    """
    def __init__(
        self,
        num_kv_heads: int,
        head_dim: int,
        num_layers: int,
        dtype: mx.Dtype,
        page_size: int = 1,
        cache_size: int = 1024,
        disable: bool = False,
    ):
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.num_layers = num_layers
        self.dtype = dtype
        self.page_size = page_size
        self.disable = disable
        self.req_to_token: Dict[str, List[int]] = {}
        self.max_token_num = self._calc_max_token_cache(cache_size)

        if self.page_size == 1:
            self.key_match_fn = _key_match_page_size1
            self.get_child_key_fn = lambda key: key[0]
        else:
            self.key_match_fn = partial(_key_match_paged, page_size=page_size)
            self.get_child_key_fn = lambda key: tuple(key[:page_size])
        self.reset()

    def reset(self):
        self.root_node = TreeNode()
        self.root_node.key = []
        self.root_node.value = []
        self.root_node.lock_ref = 1
        self.evictable_size_ = 0
        self.protected_size_ = 0
        self.req_to_token = {}

    def update_req_to_token(self, req_id: str, token_ids: List[int]):
        value = self.req_to_token.get(req_id)
        if value:
            self.req_to_token[req_id] = self.req_to_token[req_id] + token_ids
        else:
            self.req_to_token[req_id] = token_ids

    def evict_request(self, req_id: str):
        del self.req_to_token[req_id]

    def match_prefix(
            self,
            key: List[int],
        ) -> Tuple[mx.array, mx.array, int]:
        """Find the matching prefix from the radix tree.
        Args:
            key: A list of token IDs to find a matching prefix.
        Returns:
            A tuple of (key, value, matched last node)
        Note that this API can modify the internal state of the Radix tree.
        The last node creates a new child if the prefix is shorter than
        the last node's value.
        """
        if self.disable or len(key) == 0:
            return (
                [],
                self.root_node,
            )

        if self.page_size != 1:
            page_aligned_len = len(key) // self.page_size * self.page_size
            key = key[:page_aligned_len]

        value, last_node = self._match_prefix_helper(self.root_node, key)
        return value, last_node

    def insert(self, key: List, value, keys: mx.array, values: mx.array):
        if self.disable:
            return 0

        if value is None:
            value = [x for x in key]
        return self._insert_helper(self.root_node, key, value, keys, values)

    def evict(self, num_tokens: int):
        if self.disable:
            return

        leaves = self._collect_leaves()
        heapq.heapify(leaves)

        num_evicted = 0
        while num_evicted < num_tokens and len(leaves):
            x = heapq.heappop(leaves)

            if x == self.root_node:
                break
            if x.lock_ref > 0:
                continue

            # self.token_to_kv_pool_allocator.free(x.value) TODO
            num_evicted += len(x.value)
            self._delete_leaf(x)

            if len(x.parent.children) == 0:
                heapq.heappush(leaves, x.parent)

    def pretty_print(self):
        self._print_helper(self.root_node, 0)
        print(f"#tokens: {self.total_size()}")

    def total_size(self):
        return self._total_size_helper()

    def increase_lock_ref(self, node: TreeNode):
        if self.disable:
            return 0

        delta = 0
        while node != self.root_node:
            if node.lock_ref == 0:
                self.evictable_size_ -= len(node.value)
                self.protected_size_ += len(node.value)
                delta -= len(node.value)
            node.lock_ref += 1
            node = node.parent
        return delta

    def decrease_lock_ref(self, node: TreeNode):
        if self.disable:
            return 0

        delta = 0
        while node != self.root_node:
            if node.lock_ref == 1:
                self.evictable_size_ += len(node.value)
                self.protected_size_ -= len(node.value)
                delta += len(node.value)
            if node.lock_ref > 0:
                node.lock_ref -= 1
            node = node.parent
        return delta

    def cache_finished_request(
            self,
            req: Request,
            keys: mx.array,
            values: mx.array
        ):
        """Cache request when it finishes."""
        if self.disable:
            return

        token_ids = self.req_to_token[req.request_id]
        _, node = self.insert(
            key=token_ids,
            value=None,
            keys=keys,
            values=values
        )
        self.decrease_lock_ref(node)

        if self.protected_size_ > self.max_token_num:
            self.evict(self.protected_size_)
        elif self.protected_size_ + self.evictable_size_ > self.max_token_num:
            self.evict(self.protected_size_ + self.evictable_size_ - self.max_token_num)

    def cache_unfinished_request(
            self,
            req: Request,
            keys: mx.array,
            values: mx.array
        ):
        """Cache request when it is unfinished."""
        if self.disable:
            return

        token_ids = self.req_to_token[req.request_id]
        _, node = self.insert(
            key=token_ids,
            value=None,
            keys=keys,
            values=values
        )
        self.increase_lock_ref(node)
        if self.protected_size_ > self.max_token_num:
            self.evict(self.protected_size_)
        elif self.protected_size_ + self.evictable_size_ > self.max_token_num:
            self.evict(self.protected_size_ + self.evictable_size_ - self.max_token_num)

    """Internal Helper Functions"""
    def _calc_max_token_cache(self, cache_size: int):
        bpe = mx.zeros([1], dtype=self.dtype).nbytes
        bytes_per_token = self.num_layers * self.num_kv_heads * self.head_dim * bpe * 2
        max_token_num = cache_size * 1024 * 1024 // bytes_per_token
        return max_token_num

    def _match_prefix_helper(self, node: TreeNode, key: List):
        node.last_access_time = time.monotonic()

        child_key = self.get_child_key_fn(key)

        value = []
        while len(key) > 0 and child_key in node.children.keys():
            child = node.children[child_key]
            child.last_access_time = time.monotonic()
            prefix_len = self.key_match_fn(child.key, key)
            if prefix_len < len(child.key):
                new_node = self._split_node(child.key, child, prefix_len)
                value.append(new_node.value)
                node = new_node
                break
            else:
                value.append(child.value)
                node = child
                key = key[prefix_len:]

                if len(key):
                    child_key = self.get_child_key_fn(key)

        return value, node

    def _fetch_kv_cache(self, value):
        # assert value, "Fetching KV indicies should not be empty."
        # # initialize
        # init_kv_cache = self.kv_cache_dict[value[0]]
        # init_keys, _ = init_kv_cache.fetch()
        # num_layers, num_kv_heads, _, head_dim = init_keys.shape
        # init_dtype = init_keys.dtype
        # keys = mx.zeros((num_layers, num_kv_heads, 0, head_dim), dtype=init_dtype)
        # values = mx.zeros((num_layers, num_kv_heads, 0, head_dim), dtype=init_dtype)

        # # concatenate kv cache
        # for i in value:
        #     kv_cache = self.kv_cache_dict[i]
        #     cur_keys, cur_values = kv_cache.fetch()
        #     keys = mx.concatenate([keys, cur_keys], axis=2)
        #     values = mx.concatenate([values, cur_values], axis=2)
        # return keys, values
        pass

    def _split_node(self, key, child: TreeNode, split_len: int):
        # new_node -> child
        new_node = TreeNode()
        new_node.children = {self.get_child_key_fn(key[split_len:]): child}
        new_node.parent = child.parent
        new_node.lock_ref = child.lock_ref
        new_node.key = child.key[:split_len]
        new_node.value = child.value[:split_len]
        child.parent = new_node
        child.key = child.key[split_len:]
        child.value = child.value[split_len:]
        new_node.parent.children[self.get_child_key_fn(key)] = new_node

        child_keys, child_values = child.kv_cache.fetch()
        # create kv cache for new_node
        new_keys = child_keys[..., :split_len, :]
        new_values = child_values[..., :split_len, :]
        new_node.kv_cache = KVCache(
                num_kv_heads = self.num_kv_heads,
                head_dim = self.head_dim,
                num_layers = self.num_layers,
                dtype = self.dtype,
                block_size = self.page_size,
                num_initial_tokens = self.page_size,
            )
        new_node.kv_cache.update(new_keys, new_values)
        # update kv cache for child
        child_keys = child_keys[..., split_len:, :]
        child_values = child_values[..., split_len:, :]
        child.kv_cache = KVCache(
                num_kv_heads = self.num_kv_heads,
                head_dim = self.head_dim,
                num_layers = self.num_layers,
                dtype = self.dtype,
                block_size = self.page_size,
                num_initial_tokens = self.page_size,
            )
        child.kv_cache.update(child_keys, child_values)

        return new_node

    def _collect_leaves(self):
        ret_list = []
        stack = [self.root_node]

        while stack:
            cur_node = stack.pop()
            if len(cur_node.children) == 0:
                ret_list.append(cur_node)
            else:
                stack.extend(cur_node.children.values())

        return ret_list

    def _insert_helper(
            self,
            node: TreeNode,
            key: List,
            value: List,
            keys: mx.array,
            values: mx.array):
        node.last_access_time = time.monotonic()
        if len(key) == 0:
            return 0

        child_key = self.get_child_key_fn(key)

        total_prefix_length = 0
        while len(key) > 0 and child_key in node.children.keys():
            node = node.children[child_key]
            node.last_access_time = time.monotonic()
            prefix_len = self.key_match_fn(node.key, key)
            total_prefix_length += prefix_len
            key = key[prefix_len:]
            value = value[prefix_len:]

            if prefix_len < len(node.key):
                new_node = self._split_node(node.key, node, prefix_len)
                node = new_node

            if len(key):
                child_key = self.get_child_key_fn(key)

        if len(key):
            new_node = TreeNode()
            new_node.parent = node
            new_node.key = key
            new_node.value = value
            node.children[child_key] = new_node
            self.evictable_size_ += len(value)

            # create kvcache for new_node
            kv_cache = KVCache(
                num_kv_heads = self.num_kv_heads,
                head_dim = self.head_dim,
                num_layers = self.num_layers,
                dtype = self.dtype,
                block_size = self.page_size,
                num_initial_tokens = self.page_size,
            )
            keys = keys[..., total_prefix_length:, :]
            values = values[..., total_prefix_length:, :]
            kv_cache.update(keys, values)
            new_node.kv_cache = kv_cache

            node = new_node

        return total_prefix_length, node

    def _delete_leaf(self, node):
        for k, v in node.parent.children.items():
            if v == node:
                break
        del node.parent.children[k]
        self.evictable_size_ -= len(node.key)

    def _print_helper(self, node: TreeNode, indent: int):
        """Prints the radix tree in a human-readable format."""
        stack = [(node, indent)]
        while stack:
            current_node, current_indent = stack.pop()
            print(
                " " * current_indent,
                len(current_node.key),
                current_node.key[:10],
                f"r={current_node.lock_ref}",
                current_node.kv_cache,
            )
            for key, child in current_node.children.items():
                stack.append((child, current_indent + 2))

                assert key == self.get_child_key_fn(
                    child.key
                ), f"{key=}, {self.get_child_key_fn(child.key)=}"

    def _total_size_helper(self):
        total_size = 0
        stack = [self.root_node]
        while stack:
            current_node = stack.pop()
            total_size += len(current_node.value)
            for child in current_node.children.values():
                if child.evicted:
                    continue
                stack.append(child)
        return total_size


