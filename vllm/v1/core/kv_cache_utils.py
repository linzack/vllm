# SPDX-License-Identifier: Apache-2.0
"""KV-Cache Utilities."""
<<<<<<< HEAD
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, List, NamedTuple, Optional, Tuple

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.v1.kv_cache_interface import (KVCacheConfig, KVCacheSpec,
                                        KVCacheTensor)
=======
import os
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Callable, NamedTuple, Optional

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.utils import GiB_bytes, sha256
from vllm.v1.kv_cache_interface import (FullAttentionSpec, KVCacheConfig,
                                        KVCacheGroupSpec, KVCacheSpec,
                                        KVCacheTensor, SlidingWindowSpec)
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
from vllm.v1.metrics.stats import PrefixCacheStats
from vllm.v1.request import Request

logger = init_logger(__name__)


class BlockHashType(NamedTuple):
    """Hash value of a block (int), the token IDs in the block, and extra keys.
    We keep a tuple of token IDs and extra keys to reduce the likelihood of
<<<<<<< HEAD
    hash collisions when the hash value is the same. But please note that 
    hash collisions can still theoretically occur, albeit with an extremely 
    low probability.
=======
    hash collisions when the hash value is the same. By using SHA256 however,
    hash collisions are practically impossible.
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
    """
    # Hash value of the block in an integer.
    hash_value: int
    # Token IDs in the block.
<<<<<<< HEAD
    token_ids: Tuple[int, ...]
=======
    token_ids: tuple[int, ...]
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
    # Extra keys for the block.
    extra_keys: Optional[Any] = None


<<<<<<< HEAD
class PrefixCachingMetrics:
    """Metrics for prefix caching with a hit rate of the most recent N requests.

    Args:
        interval: The number of the most recent requests to aggregate.
            Defaults to 1000.
    """

    def __init__(self, interval: int = 1000):
        self.interval = interval
=======
# The hash seed for the first block of the prefix block sequence.
#
# Even if the hash function is the builtin hash(), we use sha256 to generate
# the initial hash to simplify the code. This is not performance critical
# as it is done one per process.
#
# We use a random value to avoid hash collisions or PYTHONHASHSEED environment
# variable if set such that processes can share the seed if needed.
# This aligns with the behavior of Python's hash() function, which also uses
# a random seed if PYTHONHASHSEED is not set.
NONE_HASH = int.from_bytes(os.urandom(32), byteorder="big") if os.getenv(
    'PYTHONHASHSEED') is None else sha256(os.getenv('PYTHONHASHSEED'))


class PrefixCachingMetrics:
    """Metrics for prefix caching with a hit rate of the max recent N requests.

    Args:
        max_recent_requests: The number of the max recent requests to aggregate.
            Defaults to 1000.
    """

    def __init__(self, max_recent_requests: int = 1000):
        self.max_recent_requests = max_recent_requests
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
        # The current aggregated values.
        self.aggregated_requests = 0
        self.aggregated_query_total = 0
        self.aggregated_query_hit = 0
        # A deque of (requests, queries, hits) for the most recent requests.
<<<<<<< HEAD
        self.query_queue: deque[Tuple[int, int, int]] = deque()
=======
        self.query_queue: deque[tuple[int, int, int]] = deque()
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea

    def observe(self, stats: PrefixCacheStats):
        """Observe the prefix caching for a set of requests.

        This function is called with information gathered when new requests
        are being scheduled and are looking for computed blocks.

        When there are more than `interval` requests, the oldest set of
<<<<<<< HEAD
        requestsare removed from the metrics.
=======
        requests are removed from the metrics.
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea

        Args:
            stats: The prefix cache stats.
        """
        # reset_prefix_cache was invoked before the current update.
        # Reset the metrics before aggregating the current stats.
        if stats.reset:
            self.reset()

        # Update the metrics.
        self.query_queue.append((stats.requests, stats.queries, stats.hits))
        self.aggregated_requests += stats.requests
        self.aggregated_query_total += stats.queries
        self.aggregated_query_hit += stats.hits

        # Remove the oldest stats if the number of requests exceeds.
<<<<<<< HEAD
        if self.aggregated_requests > self.interval:
=======
        if self.aggregated_requests > self.max_recent_requests:
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
            old_requests, old_queries, old_hits = self.query_queue.popleft()
            self.aggregated_requests -= old_requests
            self.aggregated_query_total -= old_queries
            self.aggregated_query_hit -= old_hits

    def reset(self):
        """Reset the metrics."""
        self.aggregated_requests = 0
        self.aggregated_query_total = 0
        self.aggregated_query_hit = 0
        self.query_queue.clear()

    @property
    def hit_rate(self) -> float:
        """Calculate the hit rate for the past N requests."""
        if self.aggregated_query_total == 0:
            return 0.0
        return self.aggregated_query_hit / self.aggregated_query_total


@dataclass
class KVCacheBlock:
    """KV-cache block metadata."""
    # Block ID, ranging from 0 to num_gpu_blocks - 1.
    block_id: int
    # Reference count.
    ref_cnt: int = 0
    # The hash of the block composed of (block hash, tuple of token IDs).
    # It is only available when the block is full.
    _block_hash: Optional[BlockHashType] = None

    # Used to construct a doubly linked list for free blocks.
    # These two attributes should only be manipulated by FreeKVCacheBlockQueue.
    prev_free_block: Optional["KVCacheBlock"] = None
    next_free_block: Optional["KVCacheBlock"] = None

    def incr_ref(self):
        self.ref_cnt += 1

    def decr_ref(self):
        self.ref_cnt -= 1

    @property
    def block_hash(self) -> Optional[BlockHashType]:
        return self._block_hash

    @block_hash.setter
    def block_hash(self, block_hash: BlockHashType):
        assert self.block_hash is None, (
            "The block already has a hash. This should not happen.")
        self._block_hash = block_hash

    def reset_hash(self):
        """Reset the block hash when the block is evicted."""
        self._block_hash = None

<<<<<<< HEAD
=======
    def __repr__(self) -> str:
        # Use block_id instead of KVCacheBlock object to avoid calling __repr__
        # on KVCacheBlock object recursively.
        prev_block_id = self.prev_free_block.block_id \
            if self.prev_free_block else None
        next_block_id = self.next_free_block.block_id \
            if self.next_free_block else None
        return (f"KVCacheBlock(block_id={self.block_id}, "
                f"ref_cnt={self.ref_cnt}, "
                f"_block_hash={self._block_hash}, "
                f"prev_free_block={prev_block_id}, "
                f"next_free_block={next_block_id})")

>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea

class FreeKVCacheBlockQueue:
    """This class organizes a list of KVCacheBlock objects to a doubly linked
    list of free blocks. We implement this class instead of using Python
    builtin deque to support removing a block in the middle of the queue
    in O(1) time. To close the performance gap to the builtin deque which is
    implemented in C++, this class does not allocate any Python objects when
<<<<<<< HEAD
    manipulating the linked list. Instead, this class manipulates the 
=======
    manipulating the linked list. Instead, this class manipulates the
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
    prev_free_block and next_free_block attributes of the given blocks.

    The queue is ordered by block ID in the beginning. When a block is allocated
    and then freed, it will be appended back with the eviction order:
    1. The least recent used block is at the front (LRU).
    2. If two blocks have the same last accessed time (allocated by the
       same sequence), the one with more hash tokens (the tail of a block
       chain) is at the front.
    Note that we maintain this order by reversing the block order when free
    blocks of a request. This operation is outside of this class.

    Args:
        blocks: A list of KVCacheBlock objects.
    """

<<<<<<< HEAD
    def __init__(self, blocks: List[KVCacheBlock]) -> None:
=======
    def __init__(self, blocks: list[KVCacheBlock]) -> None:
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
        self.num_free_blocks = len(blocks)

        # Initialize the doubly linked list of free blocks.
        self.free_list_head: Optional[KVCacheBlock] = blocks[0]
        self.free_list_tail: Optional[KVCacheBlock] = blocks[-1]
        for i in range(self.num_free_blocks):
            if i > 0:
                blocks[i].prev_free_block = blocks[i - 1]
            if i < self.num_free_blocks - 1:
                blocks[i].next_free_block = blocks[i + 1]

    def popleft(self) -> KVCacheBlock:
        """Pop the first free block and reduce num_free_blocks by 1.
<<<<<<< HEAD
        
=======

>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
        Returns:
            The first free block.
        """
        if not self.free_list_head:
            raise ValueError("No free blocks available")

        block = self.free_list_head
        self.remove(block)
        return block

    def remove(self, block: KVCacheBlock) -> None:
        """Remove a block in the free list and reduce num_free_blocks by 1.
<<<<<<< HEAD
        
=======

>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
        Args:
            block: The block to remove.
        """
        if block.prev_free_block is not None:
            # Link the previous block to the next block.
            block.prev_free_block.next_free_block = block.next_free_block
        if block.next_free_block is not None:
            # Link the next block to the previous block.
            block.next_free_block.prev_free_block = block.prev_free_block

        if block == self.free_list_head:
            # Update the head if the block is the head.
            self.free_list_head = block.next_free_block
        if block == self.free_list_tail:
            # Update the tail if the block is the tail.
            self.free_list_tail = block.prev_free_block

        # Remove the block from the linked list.
        block.prev_free_block = block.next_free_block = None
        self.num_free_blocks -= 1

    def append(self, block: KVCacheBlock) -> None:
        """Put a block back into the free list and increase
        num_free_blocks by 1.

        Args:
            block: The block to append.
        """
        if self.free_list_tail is not None:
            # Link the last block to the new block.
            self.free_list_tail.next_free_block = block
            block.prev_free_block = self.free_list_tail
            self.free_list_tail = block
        else:
            # The free list is empty.
            assert self.free_list_head is None
            self.free_list_head = self.free_list_tail = block

        block.next_free_block = None
        self.num_free_blocks += 1

<<<<<<< HEAD
    def get_all_free_blocks(self) -> List[KVCacheBlock]:
        """Get all free blocks in the free list. Mainly used for testing.
        
=======
    def get_all_free_blocks(self) -> list[KVCacheBlock]:
        """Get all free blocks in the free list. Mainly used for testing.

>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
        Returns:
            A list of free blocks.
        """
        ret = []
        curr_block = self.free_list_head
        while curr_block is not None:
            ret.append(curr_block)
            curr_block = curr_block.next_free_block
        return ret


def need_extra_keys(request: Request) -> bool:
    """Check whether the blocks allocated to this request need extra hash keys.

    Args:
<<<<<<< HEAD
        request (Request): The request. 

    Returns:
        bool: Whether blocks allocated to this request need extra hash keys. 
=======
        request (Request): The request.

    Returns:
        bool: Whether blocks allocated to this request need extra hash keys.
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
    """

    # Multimodal requests need to include the MM hash.
    # LoRA requests need to include the LoRA ID.
<<<<<<< HEAD
    return bool(request.mm_positions) or (request.lora_request is not None)
=======
    # Request with provided cache salt need to include the salt.
    return bool(request.mm_positions) or (request.lora_request
                                          is not None) or (request.cache_salt
                                                           is not None)
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea


def _gen_mm_extra_hash_keys(request: Request, start_token_idx: int,
                            end_token_idx: int,
<<<<<<< HEAD
                            start_mm_idx: int) -> Tuple[List[Any], int]:
=======
                            start_mm_idx: int) -> tuple[list[Any], int]:
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
    """Generate extra keys related to MultiModal request for block hash
    computation. For multi-modal inputs, the extra keys are
    (mm_hash, start_offset) that indicate a mm input contained in the
    block and its starting offset in the block tokens.
<<<<<<< HEAD
    
=======

>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
    Args:
        request: The request object.
        start_token_idx: The start token index of the block.
        end_token_idx: The end token index of the block.
        start_mm_idx: The start multi-modal index of the block.
<<<<<<< HEAD
    
    Returns:
        A tuple of extra keys and the next multi-modal index.
    """
    extra_keys: List[Any] = []
=======

    Returns:
        A tuple of extra keys and the next multi-modal index.
    """
    extra_keys: list[Any] = []
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea

    mm_positions, mm_hashes = request.mm_positions, request.mm_hashes
    if not mm_positions:
        return extra_keys, start_mm_idx

    if mm_positions and len(mm_positions) != len(mm_hashes):
        raise ValueError(
            "The number of multi-modal positions and hashes must match. This "
            "is likely because you do not enable MM preprocessor hashing. "
            "Please set disable_mm_preprocessor_cache=False.")

    # Note that we assume mm_positions is sorted by offset.
    # We do not need to check all mm inputs if the start token index is out of
    # range. This usually happens in the late prefill phase and decoding phase.
<<<<<<< HEAD
    if mm_positions[-1]["offset"] + mm_positions[-1][
            "length"] < start_token_idx:
=======
    if mm_positions[-1].offset + mm_positions[-1].length < start_token_idx:
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
        return extra_keys, start_mm_idx

    # Support start_mm_idx == -1 to indicate the last mm input.
    if start_mm_idx < 0:
        assert -start_mm_idx <= len(mm_positions)
        start_mm_idx = len(mm_positions) + start_mm_idx

    curr_mm_idx = start_mm_idx
    while mm_positions and curr_mm_idx < len(mm_positions):
        assert mm_hashes[curr_mm_idx] is not None
<<<<<<< HEAD
        offset = mm_positions[curr_mm_idx]["offset"]
        length = mm_positions[curr_mm_idx]["length"]
=======
        offset = mm_positions[curr_mm_idx].offset
        length = mm_positions[curr_mm_idx].length
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
        if end_token_idx > offset:
            if start_token_idx > offset + length:
                # This block has passed the current mm input.
                curr_mm_idx += 1
                continue

            # The block contains the current mm input.
            extra_keys.append(mm_hashes[curr_mm_idx])

            if end_token_idx >= offset + length:
                # If this block contains the end of the current mm input,
                # move to the next mm input as this block may also contain
                # the next mm input.
                curr_mm_idx += 1
            else:
                # Otherwise this block is done with mm inputs.
                break
        else:
            # This block has not reached the current mm input.
            break
    return extra_keys, curr_mm_idx


<<<<<<< HEAD
def _gen_lora_extra_hash_keys(request: Request) -> List[int]:
    """Generate extra keys related to LoRA for block hash computation.
    
    Args:
        request: The request object.
    
=======
def _gen_lora_extra_hash_keys(request: Request) -> list[int]:
    """Generate extra keys related to LoRA for block hash computation.

    Args:
        request: The request object.

>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
    Returns:
        Return LoRA id of the request if it is a LoRA request. Return empty
        list otherwise.
    """
    if not request.lora_request:
        return []
    return [request.lora_request.lora_int_id]


def generate_block_hash_extra_keys(
        request: Request, start_token_idx: int, end_token_idx: int,
<<<<<<< HEAD
        start_mm_idx: int) -> Tuple[Optional[Tuple[Any, ...]], int]:
    """Generate extra keys for the block hash. The extra keys can come from
    the multi-modal inputs and request specific metadata (e.g., LoRA ID).
    
=======
        start_mm_idx: int) -> tuple[Optional[tuple[Any, ...]], int]:
    """Generate extra keys for the block hash. The extra keys can come from
    the multi-modal inputs and request specific metadata (e.g., LoRA ID).

>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
    Args:
        request: The request object.
        start_token_idx: The start token index of the block.
        end_token_idx: The end token index of the block.
        start_mm_idx: The start multi-modal index of the block.
<<<<<<< HEAD
    
    Returns:
        A tuple of extra keys and the next multi-modal index.
    """
    mm_extra_keys: List[Any]
    mm_extra_keys, new_start_mm_idx = _gen_mm_extra_hash_keys(
        request, start_token_idx, end_token_idx, start_mm_idx)
    lora_extra_keys: List[int] = _gen_lora_extra_hash_keys(request)

    extra_keys: List[Any] = lora_extra_keys + mm_extra_keys
=======

    Returns:
        A tuple of extra keys and the next multi-modal index.
    """
    mm_extra_keys: list[Any]
    mm_extra_keys, new_start_mm_idx = _gen_mm_extra_hash_keys(
        request, start_token_idx, end_token_idx, start_mm_idx)
    lora_extra_keys: list[int] = _gen_lora_extra_hash_keys(request)
    cache_salt_keys: list[str] = [request.cache_salt] if (
        start_token_idx == 0 and request.cache_salt) else []

    extra_keys: list[Any] = lora_extra_keys + mm_extra_keys + cache_salt_keys
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea

    if not extra_keys:
        return None, new_start_mm_idx

    return tuple(extra_keys), new_start_mm_idx


def hash_block_tokens(
<<<<<<< HEAD
        parent_block_hash: Optional[int],
        curr_block_token_ids: Sequence[int],
        extra_keys: Optional[Tuple[Any, ...]] = None) -> BlockHashType:
=======
        hash_function: Callable,
        parent_block_hash: Optional[int],
        curr_block_token_ids: Sequence[int],
        extra_keys: Optional[tuple[Any, ...]] = None) -> BlockHashType:
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
    """Computes a hash value corresponding to the contents of a block and
    the contents of the preceding block(s). The hash value is used for
    prefix caching. We use LRU cache for this function to avoid recomputing
    hash values for the same block contents.

    Args:
        parent_block_hash: The hash of the parent block. None
            if this is the first block.
        curr_block_token_ids: A list of token ids in the current
            block. The current block is assumed to be full.
        extra_keys: Extra keys for the block.

    Returns:
        The hash value of the block and the token ids in the block.
        The entire tuple is used as the hash key of the block.
    """
    if not parent_block_hash:
<<<<<<< HEAD
        # Note that we use 'None' as a string here instead of None because
        # as of Python 3.12, hash(None) returns a constant predictable value.
        # This could possibly make it easier to find and exploit hash
        # collisions. 'None' as a string will be hashed differently per process,
        # but consistently within the same process. This is the same as the
        # behavior of None prior to Python 3.12.
        parent_block_hash = hash('None')

    curr_block_token_ids_tuple = tuple(curr_block_token_ids)
    return BlockHashType(
        hash((parent_block_hash, curr_block_token_ids_tuple, extra_keys)),
        curr_block_token_ids_tuple, extra_keys)


def hash_request_tokens(block_size: int,
                        request: Request) -> List[BlockHashType]:
=======
        parent_block_hash = NONE_HASH

    curr_block_token_ids_tuple = tuple(curr_block_token_ids)
    return BlockHashType(
        hash_function(
            (parent_block_hash, curr_block_token_ids_tuple, extra_keys)),
        curr_block_token_ids_tuple, extra_keys)


def hash_request_tokens(hash_function: Any, block_size: int,
                        request: Request) -> list[BlockHashType]:
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
    """Computes hash values of a chain of blocks given a sequence of
    token IDs. The hash value is used for prefix caching.

    Args:
        block_size: The size of each block.
        request: The request object.

    Returns:
        The list of computed hash values.
    """
    token_ids = request.all_token_ids

    req_need_extra_keys = need_extra_keys(request)
    req_extra_keys = None
    curr_mm_idx = 0

    ret = []
    parent_block_hash_value = None
    for start in range(0, len(token_ids), block_size):
        end = start + block_size
        block_token_ids = token_ids[start:end]
        # Do not hash the block if it is not full.
        if len(block_token_ids) < block_size:
            break

        if req_need_extra_keys:
            # MM and LoRA requests need extra keys for block-hash computation.
            req_extra_keys, curr_mm_idx = generate_block_hash_extra_keys(
                request, start, end, curr_mm_idx)

<<<<<<< HEAD
        block_hash = hash_block_tokens(parent_block_hash_value,
=======
        block_hash = hash_block_tokens(hash_function, parent_block_hash_value,
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
                                       block_token_ids, req_extra_keys)
        ret.append(block_hash)
        parent_block_hash_value = block_hash.hash_value
    return ret


<<<<<<< HEAD
def check_enough_kv_cache_memory(vllm_config: VllmConfig,
                                 kv_cache_spec: KVCacheSpec,
                                 available_memory: int):
    """
    Checks whether `available_memory` is enough for the KV cache to hold at 
=======
def estimate_max_model_len(vllm_config: VllmConfig,
                           kv_cache_spec: dict[str, KVCacheSpec],
                           available_memory: int) -> int:
    """
    Estimates the maximum model length that can fit in the available memory
    using binary search.

    Args:
        vllm_config: The global VllmConfig
        kv_cache_spec: The kv cache spec of each attention layer in the model
        available_memory: Memory available for KV cache in bytes.

    Returns:
        The estimated maximum model length that can fit in the available memory.
    """

    # Define a function to check if a given model length fits in memory
    def fits_in_memory(model_len: int) -> bool:
        # Modify the max_model_len for this calculation
        vllm_config.model_config.max_model_len = model_len
        # Calculate memory needed for the given model length
        memory_needed = sum(
            (layer_spec.max_memory_usage_bytes(vllm_config)
             for layer_spec in kv_cache_spec.values()),
            start=0,
        )
        return memory_needed <= available_memory

    # Binary search for the maximum model length
    current_max = vllm_config.model_config.max_model_len
    left, right = 1, current_max

    # If even the smallest model length doesn't fit, return 0
    if not fits_in_memory(left):
        return 0

    # Binary search for the maximum model length that fits
    result = 1
    while left <= right:
        mid = (left + right) // 2
        if fits_in_memory(mid):
            result = mid
            left = mid + 1
        else:
            right = mid - 1
    return result


def check_enough_kv_cache_memory(vllm_config: VllmConfig,
                                 kv_cache_spec: dict[str, KVCacheSpec],
                                 available_memory: int):
    """
    Checks whether `available_memory` is enough for the KV cache to hold at
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
    least one request with the model's max_model_len.

    Args:
        vllm_config: The global VllmConfig
<<<<<<< HEAD
        kv_cache_spec: The kv cache spec of the model
=======
        kv_cache_spec: The kv cache spec of each attention layer in the model
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
        available_memory: Memory available for KV cache in bytes.

    Raises:
        ValueError: If there is not enough memory available for the KV cache.
    """

    if available_memory <= 0:
        raise ValueError("No available memory for the cache blocks. "
                         "Try increasing `gpu_memory_utilization` when "
                         "initializing the engine.")

    max_model_len = vllm_config.model_config.max_model_len
    needed_memory = 0
    for layer_spec in kv_cache_spec.values():
<<<<<<< HEAD
        needed_memory += layer_spec.bytes_for_tokens(max_model_len)

    if needed_memory > available_memory:
        raise ValueError(
            f"To serve at least one request with the models's max seq len "
            f"({max_model_len}), ({needed_memory/1024/1024/1024:.2f} GB KV "
            f"cache is needed, which is larger than the available KV cache "
            f"memory ({available_memory/1024/1024/1024:.2f} GB). Try "
            f"increasing `gpu_memory_utilization` or decreasing "
            f"`max_model_len` when initializing the engine.")


def is_kv_cache_type_uniform(kv_cache_spec: KVCacheSpec) -> bool:
=======
        needed_memory += layer_spec.max_memory_usage_bytes(vllm_config)

    if needed_memory > available_memory:
        # Estimate the maximum model length that can fit in the available memory
        estimated_max_len = estimate_max_model_len(vllm_config, kv_cache_spec,
                                                   available_memory)
        estimated_msg = ""
        if estimated_max_len > 0:
            estimated_msg = " Based on the available memory,"
            f" the estimated maximum model length is {estimated_max_len}."

        raise ValueError(
            f"To serve at least one request with the models's max seq len "
            f"({max_model_len}), ({needed_memory/GiB_bytes:.2f} GiB KV "
            f"cache is needed, which is larger than the available KV cache "
            f"memory ({available_memory/GiB_bytes:.2f} GiB)."
            f"{estimated_msg} "
            f" Try increasing `gpu_memory_utilization` or decreasing "
            f"`max_model_len` when initializing the engine.")


def create_kv_cache_group_specs(
        kv_cache_spec: dict[str, KVCacheSpec],
        grouped_layer_names: list[list[str]]) -> list[KVCacheGroupSpec]:
    """
     Create KVCacheGroupSpec object for each kv cache group layer.
     The layers in the same group should share the same
     KVCacheSpec.

     Args:
         kv_cache_spec:
             A mapping from each layer name to its corresponding KVCacheSpec.
         grouped_layer_names:
             A list of kv cache groups, where each element is a list of layer
             names that belong to the same group and should share the same
             KVCacheSpec.
     Returns:
         A list of KVCacheGroupSpec objects, one for each group.
     """
    kv_cache_groups = []
    for layer_names_one_group in grouped_layer_names:
        layer_specs = [
            kv_cache_spec[layer_name] for layer_name in layer_names_one_group
        ]
        merged_layer_spec = layer_specs[0].merge(layer_specs)
        kv_cache_groups.append(
            KVCacheGroupSpec(layer_names_one_group, merged_layer_spec))
    return kv_cache_groups


def is_kv_cache_type_uniform(kv_cache_spec: dict[str, KVCacheSpec]) -> bool:
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
    """
    Whether all layers in the given KVCacheSpec have the same type of KV cache.

    Args:
<<<<<<< HEAD
        kv_cache_spec: The KVCacheSpec of the model
=======
        kv_cache_spec: The kv cache spec of each attention layer in the model
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea

    Returns:
        True if all layers have the same type, False otherwise.
    """

    layer_keys = set(layer.type_id for layer in kv_cache_spec.values())
    return len(layer_keys) == 1


def _get_kv_cache_config_uniform_type(vllm_config: VllmConfig,
<<<<<<< HEAD
                                      kv_cache_spec: KVCacheSpec,
                                      available_memory: int,
                                      num_layers: int) -> KVCacheConfig:
=======
                                      kv_cache_spec: dict[str, KVCacheSpec],
                                      available_memory: int) -> KVCacheConfig:
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
    """
    Generates the KV cache configuration for a model with one type of KV cache.
    Divide the available memory equally among all layers.

    Args:
        vllm_config: The global VllmConfig
<<<<<<< HEAD
        kv_cache_spec: The kv cache spec of the model
        available_memory: Memory available for KV cache in bytes.
        num_layers: The number of layers in the model.
=======
        kv_cache_spec: The kv cache spec of each attention layer in the model
        available_memory: Memory available for KV cache in bytes.
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea

    Returns:
        The generated KVCacheConfig
    """

    page_sizes = {layer.page_size_bytes for layer in kv_cache_spec.values()}
    assert len(page_sizes) == 1
    page_size = page_sizes.pop()

<<<<<<< HEAD
    num_blocks = int(available_memory // page_size // num_layers)
=======
    num_blocks = int(available_memory // page_size // len(kv_cache_spec))
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
    num_blocks = max(num_blocks, 0)

    if vllm_config.cache_config.num_gpu_blocks_override is not None:
        num_gpu_blocks_override = \
            vllm_config.cache_config.num_gpu_blocks_override
        logger.info(
            "Overriding num_gpu_blocks=%d with "
            "num_gpu_blocks_override=%d", num_blocks, num_gpu_blocks_override)
        num_blocks = num_gpu_blocks_override

    num_tokens = num_blocks * vllm_config.cache_config.block_size
    num_tokens_str = f"{num_tokens:,}"
    logger.info("GPU KV cache size: %s tokens", num_tokens_str)
    max_model_len_str = f"{vllm_config.model_config.max_model_len:,}"
    max_concurrency = num_tokens / vllm_config.model_config.max_model_len
    logger.info("Maximum concurrency for %s tokens per request: %.2fx",
                max_model_len_str, max_concurrency)

    per_layer_size = page_size * num_blocks
<<<<<<< HEAD
=======
    # All layers have the same KV cache spec, so we create one kv cache group
    # for all layers.
    grouped_layer_names = [list(kv_cache_spec.keys())]
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea

    kv_cache_config = KVCacheConfig(
        num_blocks=num_blocks,
        tensors={
            layer_name: KVCacheTensor(size=per_layer_size)
            for layer_name in kv_cache_spec
        },
<<<<<<< HEAD
        groups=[[layer_name for layer_name in kv_cache_spec]],
        kv_cache_spec=kv_cache_spec)
    return kv_cache_config


def get_kv_cache_configs(vllm_config: VllmConfig,
                         kv_cache_specs: List[KVCacheSpec],
                         available_memory: int) -> List[KVCacheConfig]:
=======
        kv_cache_groups=create_kv_cache_group_specs(kv_cache_spec,
                                                    grouped_layer_names),
    )
    return kv_cache_config


def unify_hybrid_kv_cache_specs(kv_cache_spec: dict[str, KVCacheSpec]):
    """
    Only models with one type of KV cache are supported yet. This function tries
    to convert the KV cache specs to one type if the model is a hybrid model
    with multiple type of KV cache. It will convert all SlidingWindowSpec to
    FullAttentionSpec if both types are present.

    Args:
        kv_cache_spec: The kv cache spec of each attention layer in the model
    """

    has_full_attention = any(
        isinstance(spec, FullAttentionSpec) for spec in kv_cache_spec.values())
    has_sliding_window = any(
        isinstance(spec, SlidingWindowSpec) for spec in kv_cache_spec.values())
    if has_full_attention and has_sliding_window:
        for layer_name, spec in kv_cache_spec.items():
            if isinstance(spec, SlidingWindowSpec):
                kv_cache_spec[layer_name] = FullAttentionSpec(
                    block_size=spec.block_size,
                    num_kv_heads=spec.num_kv_heads,
                    head_size=spec.head_size,
                    dtype=spec.dtype,
                    use_mla=spec.use_mla,
                    sliding_window=spec.sliding_window,
                )


def get_kv_cache_config(vllm_config: VllmConfig,
                        kv_cache_spec: dict[str, KVCacheSpec],
                        available_memory: int) -> KVCacheConfig:
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
    """
    Generates the KV cache configuration for a model
    TODO: support hybrid models with more than one type of KV cache.

    Args:
        vllm_config: The global VllmConfig
<<<<<<< HEAD
        kv_cache_specs: The kv cache specs of the model
=======
        kv_cache_spec: The kv cache spec of each attention layer in the model
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
        available_memory: Memory available for KV cache in bytes.

    Returns:
        The generated KVCacheConfigs
    """
<<<<<<< HEAD
    # Use the max number of layers to conservatively determine
    # the number of blocks.
    num_layers = max(len(kv_cache_spec) for kv_cache_spec in kv_cache_specs)
    kv_cache_configs = []
    for kv_cache_spec in kv_cache_specs:
        check_enough_kv_cache_memory(vllm_config, kv_cache_spec,
                                     available_memory)
        if is_kv_cache_type_uniform(kv_cache_spec):
            # KV cache of all layers are the same, which is true for
            # most models. Allocate the same amount of memory for
            # each layer.
            kv_cache_configs.append(
                _get_kv_cache_config_uniform_type(vllm_config, kv_cache_spec,
                                                  available_memory,
                                                  num_layers))
        else:
            raise NotImplementedError
=======
    check_enough_kv_cache_memory(vllm_config, kv_cache_spec, available_memory)
    unify_hybrid_kv_cache_specs(kv_cache_spec)
    if is_kv_cache_type_uniform(kv_cache_spec):
        # KV cache of all layers are the same, which is true for
        # most models. Allocate the same amount of memory for
        # each layer.
        return _get_kv_cache_config_uniform_type(vllm_config, kv_cache_spec,
                                                 available_memory)

    raise NotImplementedError


def unify_kv_cache_configs(kv_cache_configs: list[KVCacheConfig]):
    """
    Make the KV cache configurations for each worker consistent, so that all
    workers can be controlled by the same KVCacheManager.
    This function verifies that the layer group of each worker are the same,
    and changes the num_blocks of each worker to the smallest among all workers.

    Args:
        kv_cache_configs: The KV cache configurations for each worker. Will be
            in-place modified to make them consistent.
    """

    # Sort the kv cache groups by the type_id of their KV cache spec.
    # This can avoid the inconsistency caused by the order of groups.
    for kv_cache_config in kv_cache_configs:
        kv_cache_config.kv_cache_groups.sort(
            key=lambda x: x.kv_cache_spec.type_id)

    # Verify that the groups of each rank are the same.
    for kv_cache_config in kv_cache_configs[1:]:
        for group_rank_0, group_rank_i in zip(
                kv_cache_configs[0].kv_cache_groups,
                kv_cache_config.kv_cache_groups):
            assert group_rank_0.kv_cache_spec == group_rank_i.kv_cache_spec

    # Change the num_blocks of each rank to the smallest among all ranks. We
    # do not need to shrink the tensor size because it is valid to only use the
    # first `num_blocks` blocks of the tensor.
    min_num_blocks = min(kv_cache_config.num_blocks
                         for kv_cache_config in kv_cache_configs)
    for kv_cache_config in kv_cache_configs:
        kv_cache_config.num_blocks = min_num_blocks

>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
    return kv_cache_configs
