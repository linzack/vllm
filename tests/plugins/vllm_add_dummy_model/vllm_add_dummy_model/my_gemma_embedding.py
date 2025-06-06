# SPDX-License-Identifier: Apache-2.0

<<<<<<< HEAD
from typing import Iterable, List, Optional, Tuple, Union
=======
from collections.abc import Iterable
from typing import Optional, Union
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea

import torch
import torch.nn as nn

<<<<<<< HEAD
from vllm.attention import AttentionMetadata
=======
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
from vllm.config import VllmConfig
from vllm.model_executor.layers.pooler import Pooler, PoolingType
from vllm.model_executor.models.gemma2 import Gemma2Model
from vllm.model_executor.models.utils import WeightsMapper, maybe_prefix
from vllm.model_executor.pooling_metadata import PoolingMetadata
from vllm.sequence import IntermediateTensors, PoolerOutput


class MyGemma2Embedding(nn.Module):
    hf_to_vllm_mapper = WeightsMapper(orig_to_new_prefix={"model.": ""})

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        self.model = Gemma2Model(vllm_config=vllm_config,
                                 prefix=maybe_prefix(prefix, "model"))

        self._pooler = Pooler.from_config_with_defaults(
            vllm_config.model_config.pooler_config,
            pooling_type=PoolingType.LAST,
            normalize=True,
            softmax=False,
        )

        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
<<<<<<< HEAD
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
=======
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        hidden_states = self.model(
            input_ids,
            positions,
<<<<<<< HEAD
            kv_caches,
            attn_metadata,
=======
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )

        if isinstance(hidden_states, IntermediateTensors):
            return hidden_states

        # Return all-zero embeddings
        return torch.zeros_like(hidden_states)

    def pooler(
        self,
        hidden_states: torch.Tensor,
        pooling_metadata: PoolingMetadata,
    ) -> Optional[PoolerOutput]:
        return self._pooler(hidden_states, pooling_metadata)

<<<<<<< HEAD
    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
=======
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea

        weights = self.hf_to_vllm_mapper.apply(weights)
        weights = ((name, data) for name, data in weights
                   if not name.startswith("lm_head."))
        return self.model.load_weights(weights)
