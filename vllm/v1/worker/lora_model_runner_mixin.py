# SPDX-License-Identifier: Apache-2.0
"""
Define LoRA functionality mixin for model runners.
"""

from contextlib import contextmanager
<<<<<<< HEAD
from typing import Set, Tuple
=======
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea

import numpy as np
import torch.nn as nn

from vllm.config import LoRAConfig, ModelConfig, SchedulerConfig
from vllm.logger import init_logger
from vllm.lora.layers import LoRAMapping
from vllm.lora.request import LoRARequest
from vllm.lora.worker_manager import LRUCacheWorkerLoRAManager
from vllm.model_executor.models import supports_lora, supports_multimodal
from vllm.v1.worker.gpu_input_batch import InputBatch

logger = init_logger(__name__)


# Defined as a mixin for GPUModelRunner
class LoRAModelRunnerMixin:

    LORA_WARMUP_RANK = 8

    def load_lora_model(self, model: nn.Module, model_config: ModelConfig,
                        scheduler_config: SchedulerConfig,
                        lora_config: LoRAConfig, device: str) -> nn.Module:

<<<<<<< HEAD
        assert supports_lora(
            model), f"{model.__class__.__name__} does not support LoRA yet."
=======
        if not supports_lora(model):
            raise ValueError(
                f"{model.__class__.__name__} does not support LoRA yet.")
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea

        if supports_multimodal(model):
            logger.warning("Regarding multimodal models, vLLM currently "
                           "only supports adding LoRA to language model.")

<<<<<<< HEAD
        # It's necessary to distinguish between the max_position_embeddings
        # of VLMs and LLMs.
        if hasattr(model.config, "max_position_embeddings"):
            max_pos_embeddings = model.config.max_position_embeddings
        else:
            max_pos_embeddings = (
                model.config.text_config.max_position_embeddings)
=======
        # Use get_text_config() in case of multimodal models
        text_config = model_config.hf_config.get_text_config()
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea

        # Add LoRA Manager to the Model Runner
        self.lora_manager = LRUCacheWorkerLoRAManager(
            scheduler_config.max_num_seqs,
            scheduler_config.max_num_batched_tokens,
            model_config.get_vocab_size(),
            lora_config,
            device,
            model.embedding_modules,
            model.embedding_padding_modules,
<<<<<<< HEAD
            max_position_embeddings=max_pos_embeddings,
        )
        return self.lora_manager.create_lora_manager(model)

    def _set_active_loras(self, prompt_lora_mapping: Tuple[int, ...],
                          token_lora_mapping: Tuple[int, ...],
                          lora_requests: Set[LoRARequest]) -> None:
        if not self.lora_manager:
            raise RuntimeError("LoRA is not enabled.")

        # We dont make any distinction between prefills and decodes in the
        # scheduler. To that effect, set is_prefill to True so we use the
        # sgmv punica kernels always.
=======
            max_position_embeddings=text_config.max_position_embeddings,
        )
        return self.lora_manager.create_lora_manager(model)

    def _set_active_loras(self, prompt_lora_mapping: tuple[int, ...],
                          token_lora_mapping: tuple[int, ...],
                          lora_requests: set[LoRARequest]) -> None:
        if not self.lora_manager:
            raise RuntimeError("LoRA is not enabled.")

        # Set is_prefill to True, so we always use the SGMV kernels on
        # non-cuda platforms.
        # On cuda platforms we use the same kernels for prefill and
        # decode and this flag is generally ignored.
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
        lora_mapping = LoRAMapping(token_lora_mapping,
                                   prompt_lora_mapping,
                                   is_prefill=True)
        self.lora_manager.set_active_adapters(lora_requests, lora_mapping)

    def set_active_loras(self, input_batch: InputBatch,
                         num_scheduled_tokens: np.ndarray) -> None:

<<<<<<< HEAD
        prompt_lora_mapping: Tuple[int, ...]  # of size input_batch.num_reqs
        token_lora_mapping: Tuple[int,
                                  ...]  # of size np.sum(num_scheduled_tokens)
        lora_requests: Set[LoRARequest]
=======
        prompt_lora_mapping: tuple[int, ...]  # of size input_batch.num_reqs
        token_lora_mapping: tuple[int,
                                  ...]  # of size np.sum(num_scheduled_tokens)
        lora_requests: set[LoRARequest]
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
        prompt_lora_mapping, token_lora_mapping, lora_requests = \
                            input_batch.make_lora_inputs(num_scheduled_tokens)
        return self._set_active_loras(prompt_lora_mapping, token_lora_mapping,
                                      lora_requests)

    @contextmanager
<<<<<<< HEAD
    def maybe_profile_with_lora(self, lora_config: LoRAConfig,
                                num_scheduled_tokens: np.ndarray):
=======
    def maybe_dummy_run_with_lora(self, lora_config: LoRAConfig,
                                  num_scheduled_tokens: np.ndarray):
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
        if lora_config is None:
            yield
        else:
            # __enter__ code
            assert self.lora_manager is not None, "LoRA is not enabled"

            num_reqs = len(num_scheduled_tokens)
            num_loras = lora_config.max_loras

            # Make prompt lora mapping
            # Assign LoRA IDs cyclically to simulate a worst-case scenario.
            prompt_lora_mapping = (np.arange(num_reqs, dtype=np.int32) %
                                   num_loras) + 1

            # Make token lora mapping
            token_lora_mapping = np.repeat(prompt_lora_mapping,
                                           num_scheduled_tokens)

            # Make dummy lora requests
<<<<<<< HEAD
            lora_requests: Set[LoRARequest] = {
=======
            lora_requests: set[LoRARequest] = {
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
                LoRARequest(lora_name=f"warmup_{lora_id}",
                            lora_int_id=lora_id,
                            lora_path="/not/a/real/path")
                for lora_id in range(1, num_loras + 1)
            }

            with self.lora_manager.dummy_lora_cache():
                # Add the dummy LoRAs here so _set_active_loras doesn't try to
                # load from disk.
                for lr in lora_requests:
                    self.lora_manager.add_dummy_lora(
                        lr, rank=self.LORA_WARMUP_RANK)

                self._set_active_loras(tuple(prompt_lora_mapping),
                                       tuple(token_lora_mapping),
                                       lora_requests)

                yield

            # __exit__ code
            self.lora_manager.remove_all_adapters()

    def add_lora(self, lora_request: LoRARequest) -> bool:
        if not self.lora_manager:
            raise RuntimeError("LoRA is not enabled.")
<<<<<<< HEAD
        return self.lora_manager.add_adapter(lora_request)
=======
        return self.lora_manager.add_adapter(lora_request)

    def remove_lora(self, lora_id: int) -> bool:
        if not self.lora_manager:
            raise RuntimeError("LoRA is not enabled.")
        return self.lora_manager.remove_adapter(lora_id)

    def pin_lora(self, lora_id: int) -> bool:
        if not self.lora_manager:
            raise RuntimeError("LoRA is not enabled.")
        return self.lora_manager.pin_adapter(lora_id)

    def list_loras(self) -> set[int]:
        if not self.lora_manager:
            raise RuntimeError("LoRA is not enabled.")
        return self.lora_manager.list_adapters()
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
