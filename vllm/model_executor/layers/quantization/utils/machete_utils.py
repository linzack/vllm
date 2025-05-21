# SPDX-License-Identifier: Apache-2.0

<<<<<<< HEAD
from typing import List, Optional, Tuple
=======
from typing import Optional
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea

import torch

from vllm.scalar_type import ScalarType, scalar_types

MACHETE_SUPPORTED_GROUP_SIZES = [-1, 128]
MACHETE_PREPACKED_BLOCK_SHAPE = [64, 128]


<<<<<<< HEAD
def query_machete_supported_quant_types(zero_points: bool) -> List[ScalarType]:
=======
def query_machete_supported_quant_types(zero_points: bool) -> list[ScalarType]:
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
    if zero_points:
        return [scalar_types.uint4, scalar_types.uint8]
    else:
        return [scalar_types.uint4b8, scalar_types.uint8b128]


<<<<<<< HEAD
def query_machete_supported_act_types(zero_points: bool) -> List[ScalarType]:
=======
def query_machete_supported_act_types(zero_points: bool) -> list[ScalarType]:
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
    return [torch.float16, torch.bfloat16]


def check_machete_supports_shape(in_features: int, out_featrues: int) \
<<<<<<< HEAD
    -> Tuple[bool, Optional[str]]:
=======
    -> tuple[bool, Optional[str]]:
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
    if in_features % MACHETE_PREPACKED_BLOCK_SHAPE[0] != 0:
        return False, "Input features size must be divisible by "\
            f"{MACHETE_PREPACKED_BLOCK_SHAPE[0]}"
    if out_featrues % MACHETE_PREPACKED_BLOCK_SHAPE[1] != 0:
        return False, "Output features size must be divisible by "\
            f"{MACHETE_PREPACKED_BLOCK_SHAPE[1]}"
    return True, None
