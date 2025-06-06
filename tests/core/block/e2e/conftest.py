# SPDX-License-Identifier: Apache-2.0

<<<<<<< HEAD
from typing import Callable, Iterable, Optional
=======
from collections.abc import Iterable
from typing import Callable, Optional
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea

import pytest

from vllm import LLM
from vllm.distributed import cleanup_dist_env_and_memory
from vllm.model_executor.utils import set_random_seed


@pytest.fixture
def baseline_llm_generator(common_llm_kwargs, per_test_common_llm_kwargs,
                           baseline_llm_kwargs, seed):
    return create_llm_generator(common_llm_kwargs, per_test_common_llm_kwargs,
                                baseline_llm_kwargs, seed)


@pytest.fixture
def test_llm_generator(common_llm_kwargs, per_test_common_llm_kwargs,
                       test_llm_kwargs, seed):
    return create_llm_generator(common_llm_kwargs, per_test_common_llm_kwargs,
                                test_llm_kwargs, seed)


def create_llm_generator(common_llm_kwargs, per_test_common_llm_kwargs,
                         distinct_llm_kwargs, seed):
    kwargs = {
        **common_llm_kwargs,
        **per_test_common_llm_kwargs,
        **distinct_llm_kwargs,
    }

    def generator_inner():
        llm = LLM(**kwargs)

        set_random_seed(seed)

        yield llm
        del llm
        cleanup_dist_env_and_memory()

    for llm in generator_inner():
        yield llm
        del llm


def get_text_from_llm_generator(llm_generator: Iterable[LLM],
                                prompts,
                                sampling_params,
                                llm_cb: Optional[Callable[[LLM],
                                                          None]] = None):
    for llm in llm_generator:
        if llm_cb:
            llm_cb(llm)
        outputs = llm.generate(prompts, sampling_params, use_tqdm=True)
        text = [output.outputs[0].text for output in outputs]
        del llm

    return text


def get_token_ids_from_llm_generator(llm_generator, prompts, sampling_params):
    for llm in llm_generator:
        outputs = llm.generate(prompts, sampling_params, use_tqdm=True)
        token_ids = [output.outputs[0].token_ids for output in outputs]
        del llm

    return token_ids
