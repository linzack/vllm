# SPDX-License-Identifier: Apache-2.0
"""
This test file includes some cases where it is inappropriate to
only get the `eos_token_id` from the tokenizer as defined by
<<<<<<< HEAD
:meth:`vllm.LLMEngine._get_eos_token_id`.
=======
{meth}`vllm.LLMEngine._get_eos_token_id`.
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
"""
from vllm.transformers_utils.config import try_get_generation_config
from vllm.transformers_utils.tokenizer import get_tokenizer


def test_get_llama3_eos_token():
    model_name = "meta-llama/Llama-3.2-1B-Instruct"

    tokenizer = get_tokenizer(model_name)
    assert tokenizer.eos_token_id == 128009

    generation_config = try_get_generation_config(model_name,
                                                  trust_remote_code=False)
    assert generation_config is not None
    assert generation_config.eos_token_id == [128001, 128008, 128009]


def test_get_blip2_eos_token():
    model_name = "Salesforce/blip2-opt-2.7b"

    tokenizer = get_tokenizer(model_name)
    assert tokenizer.eos_token_id == 2

    generation_config = try_get_generation_config(model_name,
                                                  trust_remote_code=False)
    assert generation_config is not None
    assert generation_config.eos_token_id == 50118
