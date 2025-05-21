# SPDX-License-Identifier: Apache-2.0

<<<<<<< HEAD
from vllm.triton_utils.importing import HAS_TRITON

__all__ = ["HAS_TRITON"]

if HAS_TRITON:

    from vllm.triton_utils.custom_cache_manager import (
        maybe_set_triton_cache_manager)

    __all__ += ["maybe_set_triton_cache_manager"]
=======
from vllm.triton_utils.importing import (HAS_TRITON, TritonLanguagePlaceholder,
                                         TritonPlaceholder)

if HAS_TRITON:
    import triton
    import triton.language as tl
else:
    triton = TritonPlaceholder()
    tl = TritonLanguagePlaceholder()

__all__ = ["HAS_TRITON", "triton", "tl"]
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
