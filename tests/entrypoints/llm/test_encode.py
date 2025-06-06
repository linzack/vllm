# SPDX-License-Identifier: Apache-2.0

import weakref
<<<<<<< HEAD
from typing import List
=======
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea

import pytest

from vllm import LLM, PoolingParams, PoolingRequestOutput
<<<<<<< HEAD
from vllm.config import LoadFormat
from vllm.distributed import cleanup_dist_env_and_memory

MODEL_NAME = "s3://vllm-ci-model-weights/e5-mistral-7b-instruct"
=======
from vllm.distributed import cleanup_dist_env_and_memory

MODEL_NAME = "intfloat/multilingual-e5-small"
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea

PROMPTS = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]

TOKEN_IDS = [
    # Using ID={0, 1, 2, 3} results in NaN values,
    # so we add this offset of 1000
    [1000],
    [1000, 1001],
    [1000, 1002, 1001],
    [1000, 1003, 1001, 1002],
]


@pytest.fixture(scope="module")
def llm():
    # pytest caches the fixture so we use weakref.proxy to
    # enable garbage collection
    llm = LLM(model=MODEL_NAME,
<<<<<<< HEAD
              load_format=LoadFormat.RUNAI_STREAMER,
              max_num_batched_tokens=32768,
              tensor_parallel_size=1,
              gpu_memory_utilization=0.75,
              enforce_eager=True)
=======
              max_num_batched_tokens=32768,
              tensor_parallel_size=1,
              gpu_memory_utilization=0.75,
              enforce_eager=True,
              seed=0)
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea

    with llm.deprecate_legacy_api():
        yield weakref.proxy(llm)

        del llm

    cleanup_dist_env_and_memory()


<<<<<<< HEAD
def assert_outputs_equal(o1: List[PoolingRequestOutput],
                         o2: List[PoolingRequestOutput]):
=======
def assert_outputs_equal(o1: list[PoolingRequestOutput],
                         o2: list[PoolingRequestOutput]):
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
    assert [o.outputs for o in o1] == [o.outputs for o in o2]


@pytest.mark.skip_global_cleanup
@pytest.mark.parametrize('prompt_token_ids', TOKEN_IDS)
def test_v1_v2_api_consistency_single_prompt_tokens(llm: LLM,
                                                    prompt_token_ids):
    pooling_params = PoolingParams()

    with pytest.warns(DeprecationWarning, match="'prompt_token_ids'"):
        v1_output = llm.encode(prompt_token_ids=prompt_token_ids,
                               pooling_params=pooling_params)

    v2_output = llm.encode({"prompt_token_ids": prompt_token_ids},
                           pooling_params=pooling_params)
    assert_outputs_equal(v1_output, v2_output)


@pytest.mark.skip_global_cleanup
def test_v1_v2_api_consistency_multi_prompt_tokens(llm: LLM):
    pooling_params = PoolingParams()

    with pytest.warns(DeprecationWarning, match="'prompt_token_ids'"):
        v1_output = llm.encode(prompt_token_ids=TOKEN_IDS,
                               pooling_params=pooling_params)

    v2_output = llm.encode(
        [{
            "prompt_token_ids": p
        } for p in TOKEN_IDS],
        pooling_params=pooling_params,
    )
    assert_outputs_equal(v1_output, v2_output)


@pytest.mark.skip_global_cleanup
def test_multiple_pooling_params(llm: LLM):
    pooling_params = [
        PoolingParams(),
        PoolingParams(),
        PoolingParams(),
        PoolingParams(),
    ]

    # Multiple PoolingParams should be matched with each prompt
    outputs = llm.encode(PROMPTS, pooling_params=pooling_params)
    assert len(PROMPTS) == len(outputs)

    # Exception raised, if the size of params does not match the size of prompts
    with pytest.raises(ValueError):
        outputs = llm.encode(PROMPTS, pooling_params=pooling_params[:3])

    # Single PoolingParams should be applied to every prompt
    single_pooling_params = PoolingParams()
    outputs = llm.encode(PROMPTS, pooling_params=single_pooling_params)
    assert len(PROMPTS) == len(outputs)

    # pooling_params is None, default params should be applied
    outputs = llm.encode(PROMPTS, pooling_params=None)
    assert len(PROMPTS) == len(outputs)


@pytest.mark.skip_global_cleanup
def test_right_side_truncation(llm: LLM):
    # Embeddings models should truncate the end of the prompt
    tokenizer = llm.get_tokenizer()
    assert tokenizer.truncation_side == "right"
