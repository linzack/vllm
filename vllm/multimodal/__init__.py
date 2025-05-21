# SPDX-License-Identifier: Apache-2.0
<<<<<<< HEAD

from .base import MultiModalPlaceholderMap, MultiModalPlugin
=======
from .base import MultiModalPlaceholderMap
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
from .hasher import MultiModalHashDict, MultiModalHasher
from .inputs import (BatchedTensorInputs, ModalityData, MultiModalDataBuiltins,
                     MultiModalDataDict, MultiModalKwargs,
                     MultiModalPlaceholderDict, NestedTensors)
from .registry import MultiModalRegistry

MULTIMODAL_REGISTRY = MultiModalRegistry()
"""
<<<<<<< HEAD
The global :class:`~MultiModalRegistry` is used by model runners to
dispatch data processing according to the target model.

See also:
    :ref:`mm-processing`
=======
The global {class}`~MultiModalRegistry` is used by model runners to
dispatch data processing according to the target model.

:::{seealso}
{ref}`mm-processing`
:::
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
"""

__all__ = [
    "BatchedTensorInputs",
    "ModalityData",
    "MultiModalDataBuiltins",
    "MultiModalDataDict",
    "MultiModalHashDict",
    "MultiModalHasher",
    "MultiModalKwargs",
    "MultiModalPlaceholderDict",
    "MultiModalPlaceholderMap",
<<<<<<< HEAD
    "MultiModalPlugin",
=======
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
    "NestedTensors",
    "MULTIMODAL_REGISTRY",
    "MultiModalRegistry",
]
