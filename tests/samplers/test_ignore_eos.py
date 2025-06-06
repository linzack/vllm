# SPDX-License-Identifier: Apache-2.0
"""Make sure ignore_eos works.

Run `pytest tests/samplers/test_ignore_eos.py`.
"""

import pytest

from vllm import SamplingParams

<<<<<<< HEAD
=======

@pytest.fixture(autouse=True)
def v1(run_with_both_engines):
    """We can run both engines for this test."""
    pass


>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
# We also test with llama because it has generation_config to specify EOS
# (past regression).
MODELS = ["distilbert/distilgpt2", "meta-llama/Llama-3.2-1B"]


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("dtype", ["half"])
@pytest.mark.parametrize("max_tokens", [512])
def test_ignore_eos(
    vllm_runner,
    example_prompts,
    model: str,
    dtype: str,
    max_tokens: int,
) -> None:
    with vllm_runner(model, dtype=dtype) as vllm_model:
        sampling_params = SamplingParams(max_tokens=max_tokens,
                                         ignore_eos=True)

        for prompt in example_prompts:
            ignore_eos_output = vllm_model.model.generate(
                prompt, sampling_params=sampling_params)
            output_length = len(ignore_eos_output[0].outputs[0].token_ids)
            assert output_length == max_tokens
