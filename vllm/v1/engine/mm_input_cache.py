# SPDX-License-Identifier: Apache-2.0
<<<<<<< HEAD

from typing import Any, Dict, List, Optional

from vllm.config import ModelConfig
from vllm.envs import VLLM_MM_INPUT_CACHE_SIZE
from vllm.logger import init_logger
from vllm.multimodal import (MULTIMODAL_REGISTRY, MultiModalDataDict,
                             MultiModalKwargs, MultiModalRegistry)
from vllm.utils import LRUCache

logger = init_logger(__name__)
=======
from collections.abc import Sequence
from typing import Optional

from vllm.envs import VLLM_MM_INPUT_CACHE_GIB
from vllm.multimodal import MultiModalKwargs
from vllm.multimodal.processing import ProcessingCache
from vllm.utils import is_list_of
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea

# The idea of multimodal preprocessing caching is based on having a client and
# a server, where the client executes in the frontend process (=P0) and the
# server in the core process (=P1).
#
# -- Client:
<<<<<<< HEAD
#  - Apply legacy input_mapper (if one exists) to generate MultiModalKwargs.
#  - Perform caching of the generated MultiModalKwargs.
#  - This client can be deprecated once all mutimodal models migrate to use
#    merged preprocessor with built-in caching functionality.
#
# -- Server:
#  - Perform caching of the received MultiModalKwargs.
#
# The caching for both client and server is mirrored/similar, and this allows us
# to avoid the serialization of "mm_inputs" (like pixel values) between
# client (=P0) and server (=P1) processes.

# Both Client and Server must use the same cache size
# (to perform mirrored caching). This cache size is set by the environment
# variable VLLM_MM_INPUT_CACHE_SIZE.


# TODO(ywang96): Deprecate this class once all multimodal models migrate to use
# merged preprocessor with built-in caching functionality.
class MMInputCacheClient:

    def __init__(
        self,
        model_config: ModelConfig,
        mm_registry: MultiModalRegistry = MULTIMODAL_REGISTRY,
    ):
        self.model_config = model_config
        self.mm_registry = mm_registry
        self.multi_modal_input_mapper = mm_registry.create_input_mapper(
            model_config)
        self.mm_registry.init_mm_limits_per_prompt(model_config)

        # Init cache
        self.use_cache = not model_config.disable_mm_preprocessor_cache
        self.mm_cache = LRUCache[str,
                                 MultiModalKwargs](VLLM_MM_INPUT_CACHE_SIZE)

        # DEBUG: Set to None to disable
        self.mm_debug_cache_hit_ratio_steps = None
        self.mm_cache_hits = 0
        self.mm_cache_total = 0

    def cache_hit_ratio(self, steps):
        if self.mm_cache_total > 0 and self.mm_cache_total % steps == 0:
            logger.debug("MMInputMapper: cache_hit_ratio = %.2f ",
                         self.mm_cache_hits / self.mm_cache_total)

    # NOTE: process_inputs only supports image inputs since all multimodal
    # models with other modalities have migrated to use merged preprocessor.
    def process_inputs(
        self,
        mm_data: MultiModalDataDict,
        mm_hashes: Optional[List[str]],
        mm_processor_kwargs: Optional[Dict[str, Any]],
        precomputed_mm_inputs: Optional[List[MultiModalKwargs]],
    ) -> List[MultiModalKwargs]:
        if precomputed_mm_inputs is None:
            image_inputs = mm_data["image"]
            if not isinstance(image_inputs, list):
                image_inputs = [image_inputs]
            num_inputs = len(image_inputs)
        else:
            num_inputs = len(precomputed_mm_inputs)

        # Sanity
        if self.use_cache:
            assert mm_hashes is not None
            assert num_inputs == len(mm_hashes)

        # Process each image input separately, so that later we can schedule
        # them in a fine-grained manner.
        # Apply caching (if enabled) and reuse precomputed inputs (if provided)
        ret_inputs: List[MultiModalKwargs] = []
        for input_id in range(num_inputs):
            if self.mm_debug_cache_hit_ratio_steps is not None:
                self.cache_hit_ratio(self.mm_debug_cache_hit_ratio_steps)

            mm_input = None
            if self.use_cache:
                assert mm_hashes is not None
                mm_hash = mm_hashes[input_id]
                mm_input = self.mm_cache.get(mm_hash)

            self.mm_cache_total += 1
            if mm_input is None:
                if precomputed_mm_inputs is not None:
                    # Reuse precomputed input (for merged preprocessor)
                    mm_input = precomputed_mm_inputs[input_id]
                else:
                    # Apply legacy input_mapper
                    mm_input = self.multi_modal_input_mapper(
                        {"image": [image_inputs[input_id]]},
                        mm_processor_kwargs=mm_processor_kwargs,
                    )

                if self.use_cache:
                    # Add to cache
                    assert mm_hash is not None
                    self.mm_cache.put(mm_hash, mm_input)
            else:
                self.mm_cache_hits += 1
                mm_input = None  # Avoids sending mm_input to Server

            ret_inputs.append(mm_input)

        return ret_inputs


class MMInputCacheServer:

    def __init__(self, model_config):
        self.use_cache = not model_config.disable_mm_preprocessor_cache
        self.mm_cache = LRUCache[str,
                                 MultiModalKwargs](VLLM_MM_INPUT_CACHE_SIZE)

    def get_and_update(
        self,
        mm_inputs: List[Optional[MultiModalKwargs]],
        mm_hashes: List[str],
    ) -> List[MultiModalKwargs]:
        assert len(mm_inputs) == len(mm_hashes)

        if not self.use_cache:
            return mm_inputs

        full_mm_inputs = []
        for mm_input, mm_hash in zip(mm_inputs, mm_hashes):
            assert mm_hash is not None
            if mm_input is None:
                mm_input = self.mm_cache.get(mm_hash)
                assert mm_input is not None
            else:
                self.mm_cache.put(mm_hash, mm_input)
=======
#  - BaseMultiModalProcessor to process MultiModalData into MultiModalKwargs
#    with built-in caching functionality, with mm_hash as its identifier.
#  - MirroredProcessingCache to keep track of the cached entries and
#    determine whether to send the MultiModalKwargs to P1.
#
# -- Server:
#  - MirroredProcessingCache to store the MultiModalKwargs from P0.
#
# The caching for both client and server is mirrored, and this allows us
# to avoid the serialization of "mm_inputs" (like pixel values) between
# client (=P0) and server (=P1) processes if the mm_hash is found in the client
# cache.

# Both Client and Server must use the same cache size
# (to perform mirrored caching). This cache size is set by the environment
# variable VLLM_MM_INPUT_CACHE_GIB.


class MirroredProcessingCache:

    def __init__(self, model_config):
        mm_config = model_config.multimodal_config
        disable_mm_preprocessor_cache = mm_config is not None and \
            not mm_config.disable_mm_preprocessor_cache
        self.use_cache = not disable_mm_preprocessor_cache
        self.mm_cache = ProcessingCache.get_lru_cache(VLLM_MM_INPUT_CACHE_GIB,
                                                      MultiModalKwargs)

    def get_and_update_p0(
        self,
        mm_inputs: Sequence[MultiModalKwargs],
        mm_hashes: list[str],
    ) -> Sequence[Optional[MultiModalKwargs]]:
        assert len(mm_inputs) == len(mm_hashes)

        if not self.use_cache:
            assert is_list_of(mm_inputs, MultiModalKwargs)
            return mm_inputs

        full_mm_inputs = list[Optional[MultiModalKwargs]]()
        for mm_input, mm_hash in zip(mm_inputs, mm_hashes):
            if self.mm_cache.get(mm_hash) is not None:
                mm_input = None
            else:
                self.mm_cache[mm_hash] = mm_input
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea

            full_mm_inputs.append(mm_input)

        return full_mm_inputs
<<<<<<< HEAD
=======

    def get_and_update_p1(
        self,
        mm_inputs: Sequence[Optional[MultiModalKwargs]],
        mm_hashes: list[str],
    ) -> Sequence[MultiModalKwargs]:
        assert len(mm_inputs) == len(mm_hashes)

        if not self.use_cache:
            assert is_list_of(mm_inputs, MultiModalKwargs)
            return mm_inputs

        full_mm_inputs = list[MultiModalKwargs]()
        for mm_input, mm_hash in zip(mm_inputs, mm_hashes):
            if mm_input is None:
                mm_input = self.mm_cache[mm_hash]
            else:
                self.mm_cache[mm_hash] = mm_input

            full_mm_inputs.append(mm_input)

        return full_mm_inputs

    def reset(self) -> bool:
        self.mm_cache.clear()

        return True
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
