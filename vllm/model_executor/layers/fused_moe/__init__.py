# SPDX-License-Identifier: Apache-2.0

from contextlib import contextmanager
<<<<<<< HEAD
from typing import Any, Dict, Optional
=======
from typing import Any, Optional
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea

from vllm.model_executor.layers.fused_moe.layer import (
    FusedMoE, FusedMoEMethodBase, FusedMoeWeightScaleSupported)
from vllm.triton_utils import HAS_TRITON

<<<<<<< HEAD
_config: Optional[Dict[str, Any]] = None
=======
_config: Optional[dict[str, Any]] = None
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea


@contextmanager
def override_config(config):
    global _config
    old_config = _config
    _config = config
    yield
    _config = old_config


<<<<<<< HEAD
def get_config() -> Optional[Dict[str, Any]]:
=======
def get_config() -> Optional[dict[str, Any]]:
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
    return _config


__all__ = [
    "FusedMoE",
    "FusedMoEMethodBase",
    "FusedMoeWeightScaleSupported",
    "override_config",
    "get_config",
]

if HAS_TRITON:
    # import to register the custom ops
    import vllm.model_executor.layers.fused_moe.fused_marlin_moe  # noqa
    import vllm.model_executor.layers.fused_moe.fused_moe  # noqa
<<<<<<< HEAD
    from vllm.model_executor.layers.fused_moe.fused_moe import (
        fused_experts, fused_moe, fused_topk, get_config_file_name,
        grouped_topk)
=======
    from vllm.model_executor.layers.fused_moe.cutlass_moe import (
        cutlass_moe_fp4, cutlass_moe_fp8)
    from vllm.model_executor.layers.fused_moe.fused_moe import (
        TritonExperts, fused_experts, fused_moe, fused_topk,
        get_config_file_name, grouped_topk)
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea

    __all__ += [
        "fused_moe",
        "fused_topk",
        "fused_experts",
        "get_config_file_name",
        "grouped_topk",
<<<<<<< HEAD
=======
        "cutlass_moe_fp8",
        "cutlass_moe_fp4",
        "TritonExperts",
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
    ]
