# SPDX-License-Identifier: Apache-2.0

from unittest.mock import patch

import pytest
from transformers import PretrainedConfig

from vllm import LLM
<<<<<<< HEAD

from ..conftest import MODELS_ON_S3
=======
from vllm.engine.llm_engine import LLMEngine as V0LLMEngine
from vllm.utils import GiB_bytes
from vllm.v1.core.kv_cache_utils import get_kv_cache_config
from vllm.v1.engine.core import EngineCore as V1EngineCore

>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
from .registry import HF_EXAMPLE_MODELS


@pytest.mark.parametrize("model_arch", HF_EXAMPLE_MODELS.get_supported_archs())
def test_can_initialize(model_arch):
    model_info = HF_EXAMPLE_MODELS.get_hf_info(model_arch)
    model_info.check_available_online(on_fail="skip")
    model_info.check_transformers_version(on_fail="skip")

    # Avoid OOM
    def hf_overrides(hf_config: PretrainedConfig) -> PretrainedConfig:
        hf_config.update(model_info.hf_overrides)

<<<<<<< HEAD
        if hasattr(hf_config, "text_config"):
            text_config: PretrainedConfig = hf_config.text_config
        else:
            text_config = hf_config
=======
        text_config = hf_config.get_text_config()
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea

        text_config.update({
            "num_layers": 1,
            "num_hidden_layers": 1,
            "num_experts": 2,
            "num_experts_per_tok": 2,
            "num_local_experts": 2,
        })

        return hf_config

    # Avoid calling model.forward()
<<<<<<< HEAD
    def _initialize_kv_caches(self) -> None:
        self.cache_config.num_gpu_blocks = 0
        self.cache_config.num_cpu_blocks = 0

    with patch.object(LLM.get_engine_class(), "_initialize_kv_caches",
                      _initialize_kv_caches):
        model_name = model_info.default
        if model_name in MODELS_ON_S3:
            model_name = f"s3://vllm-ci-model-weights/{model_name.split('/')[-1]}"
        LLM(
            model_name,
            tokenizer=model_info.tokenizer,
            tokenizer_mode=model_info.tokenizer_mode,
            speculative_model=model_info.speculative_model,
            num_speculative_tokens=1 if model_info.speculative_model else None,
=======
    def _initialize_kv_caches_v0(self) -> None:
        self.cache_config.num_gpu_blocks = 0
        self.cache_config.num_cpu_blocks = 0

    def _initialize_kv_caches_v1(self, vllm_config):
        kv_cache_specs = self.model_executor.get_kv_cache_specs()
        scheduler_kv_cache_config = get_kv_cache_config(
            vllm_config,
            kv_cache_specs[0],
            20 * GiB_bytes,
        )

        # gpu_blocks (> 0), cpu_blocks, scheduler_kv_cache_config
        return 1, 0, scheduler_kv_cache_config

    with (patch.object(V0LLMEngine, "_initialize_kv_caches",
                       _initialize_kv_caches_v0),
          patch.object(V1EngineCore, "_initialize_kv_caches",
                       _initialize_kv_caches_v1)):
        LLM(
            model_info.default,
            tokenizer=model_info.tokenizer,
            tokenizer_mode=model_info.tokenizer_mode,
            speculative_config={
                "model": model_info.speculative_model,
                "num_speculative_tokens": 1,
            } if model_info.speculative_model else None,
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
            trust_remote_code=model_info.trust_remote_code,
            load_format="dummy",
            hf_overrides=hf_overrides,
        )
