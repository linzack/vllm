# SPDX-License-Identifier: Apache-2.0

<<<<<<< HEAD
from typing import (TYPE_CHECKING, List, Optional, Protocol, Type, Union,
                    overload, runtime_checkable)
=======
from typing import (TYPE_CHECKING, Optional, Protocol, Union, overload,
                    runtime_checkable)
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea

import torch
import torch.nn as nn
from typing_extensions import TypeIs, TypeVar

from vllm.logger import init_logger
from vllm.utils import supports_kw

if TYPE_CHECKING:
<<<<<<< HEAD
    from vllm.attention import AttentionMetadata
    from vllm.config import VllmConfig
    from vllm.model_executor.layers.pooler import PoolerOutput
    from vllm.model_executor.layers.sampler import SamplerOutput
=======
    from vllm.config import VllmConfig
    from vllm.model_executor.layers.pooler import PoolerOutput
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
    from vllm.model_executor.pooling_metadata import PoolingMetadata
    from vllm.model_executor.sampling_metadata import SamplingMetadata

logger = init_logger(__name__)

# The type of hidden states
# Currently, T = torch.Tensor for all models except for Medusa
<<<<<<< HEAD
# which has T = List[torch.Tensor]
=======
# which has T = list[torch.Tensor]
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
T = TypeVar("T", default=torch.Tensor)
T_co = TypeVar("T_co", default=torch.Tensor, covariant=True)

# NOTE: Unlike those in `interfaces.py`, we don't define `ClassVar` tags
# for the base interfaces to avoid breaking OOT registration for existing models
# that don't inherit from the base interface classes


@runtime_checkable
class VllmModel(Protocol[T_co]):
    """The interface required for all models in vLLM."""

    def __init__(
        self,
        vllm_config: "VllmConfig",
        prefix: str = "",
    ) -> None:
        ...

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
<<<<<<< HEAD
        kv_caches: List[torch.Tensor],
        attn_metadata: "AttentionMetadata",
=======
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
    ) -> T_co:
        ...


<<<<<<< HEAD
def _check_vllm_model_init(model: Union[Type[object], object]) -> bool:
=======
def _check_vllm_model_init(model: Union[type[object], object]) -> bool:
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
    model_init = model.__init__
    return supports_kw(model_init, "vllm_config")


<<<<<<< HEAD
def _check_vllm_model_forward(model: Union[Type[object], object]) -> bool:
=======
def _check_vllm_model_forward(model: Union[type[object], object]) -> bool:
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
    model_forward = getattr(model, "forward", None)
    if not callable(model_forward):
        return False

<<<<<<< HEAD
    vllm_kws = ("input_ids", "positions", "kv_caches", "attn_metadata")
=======
    vllm_kws = ("input_ids", "positions")
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
    missing_kws = tuple(kw for kw in vllm_kws
                        if not supports_kw(model_forward, kw))

    if missing_kws and (isinstance(model, type)
                        and issubclass(model, nn.Module)):
        logger.warning(
            "The model (%s) is missing "
            "vLLM-specific keywords from its `forward` method: %s",
            model,
            missing_kws,
        )

    return len(missing_kws) == 0


@overload
<<<<<<< HEAD
def is_vllm_model(model: Type[object]) -> TypeIs[Type[VllmModel]]:
=======
def is_vllm_model(model: type[object]) -> TypeIs[type[VllmModel]]:
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
    ...


@overload
def is_vllm_model(model: object) -> TypeIs[VllmModel]:
    ...


def is_vllm_model(
<<<<<<< HEAD
    model: Union[Type[object], object],
) -> Union[TypeIs[Type[VllmModel]], TypeIs[VllmModel]]:
=======
    model: Union[type[object], object],
) -> Union[TypeIs[type[VllmModel]], TypeIs[VllmModel]]:
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
    return _check_vllm_model_init(model) and _check_vllm_model_forward(model)


@runtime_checkable
class VllmModelForTextGeneration(VllmModel[T], Protocol[T]):
    """The interface required for all generative models in vLLM."""

    def compute_logits(
        self,
        hidden_states: T,
        sampling_metadata: "SamplingMetadata",
    ) -> Optional[T]:
        """Return `None` if TP rank > 0."""
        ...

<<<<<<< HEAD
    def sample(
        self,
        logits: T,
        sampling_metadata: "SamplingMetadata",
    ) -> "SamplerOutput":
        """Only called on TP rank 0."""
        ...


@overload
def is_text_generation_model(
        model: Type[object]) -> TypeIs[Type[VllmModelForTextGeneration]]:
=======

@overload
def is_text_generation_model(
        model: type[object]) -> TypeIs[type[VllmModelForTextGeneration]]:
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
    ...


@overload
def is_text_generation_model(
        model: object) -> TypeIs[VllmModelForTextGeneration]:
    ...


def is_text_generation_model(
<<<<<<< HEAD
    model: Union[Type[object], object],
) -> Union[TypeIs[Type[VllmModelForTextGeneration]],
=======
    model: Union[type[object], object],
) -> Union[TypeIs[type[VllmModelForTextGeneration]],
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
           TypeIs[VllmModelForTextGeneration]]:
    if not is_vllm_model(model):
        return False

    if isinstance(model, type):
        return isinstance(model, VllmModelForTextGeneration)

    return isinstance(model, VllmModelForTextGeneration)


@runtime_checkable
class VllmModelForPooling(VllmModel[T], Protocol[T]):
    """The interface required for all pooling models in vLLM."""

    def pooler(
        self,
        hidden_states: T,
        pooling_metadata: "PoolingMetadata",
    ) -> "PoolerOutput":
        """Only called on TP rank 0."""
        ...


@overload
<<<<<<< HEAD
def is_pooling_model(model: Type[object]) -> TypeIs[Type[VllmModelForPooling]]:
=======
def is_pooling_model(model: type[object]) -> TypeIs[type[VllmModelForPooling]]:
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
    ...


@overload
def is_pooling_model(model: object) -> TypeIs[VllmModelForPooling]:
    ...


def is_pooling_model(
<<<<<<< HEAD
    model: Union[Type[object], object],
) -> Union[TypeIs[Type[VllmModelForPooling]], TypeIs[VllmModelForPooling]]:
=======
    model: Union[type[object], object],
) -> Union[TypeIs[type[VllmModelForPooling]], TypeIs[VllmModelForPooling]]:
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
    if not is_vllm_model(model):
        return False

    if isinstance(model, type):
        return isinstance(model, VllmModelForPooling)

    return isinstance(model, VllmModelForPooling)
