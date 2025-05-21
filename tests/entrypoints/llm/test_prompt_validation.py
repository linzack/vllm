# SPDX-License-Identifier: Apache-2.0

import pytest

from vllm import LLM
<<<<<<< HEAD
from vllm.config import LoadFormat
=======
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea


@pytest.fixture(autouse=True)
def v1(run_with_both_engines):
    # Simple autouse wrapper to run both engines for each test
    # This can be promoted up to conftest.py to run for every
    # test in a package
    pass


def test_empty_prompt():
<<<<<<< HEAD
    llm = LLM(model="s3://vllm-ci-model-weights/gpt2",
              load_format=LoadFormat.RUNAI_STREAMER,
              enforce_eager=True)
    with pytest.raises(ValueError, match='Prompt cannot be empty'):
=======
    llm = LLM(model="openai-community/gpt2", enforce_eager=True)
    with pytest.raises(ValueError, match='decoder prompt cannot be empty'):
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
        llm.generate([""])


@pytest.mark.skip_v1
def test_out_of_vocab_token():
<<<<<<< HEAD
    llm = LLM(model="s3://vllm-ci-model-weights/gpt2",
              load_format=LoadFormat.RUNAI_STREAMER,
              enforce_eager=True)
=======
    llm = LLM(model="openai-community/gpt2", enforce_eager=True)
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
    with pytest.raises(ValueError, match='out of vocabulary'):
        llm.generate({"prompt_token_ids": [999999]})
