# SPDX-License-Identifier: Apache-2.0

import glob
import os
import tempfile

import depyf

<<<<<<< HEAD
from vllm.config import CompilationLevel

temp_dir = tempfile.mkdtemp()
with depyf.prepare_debug(temp_dir):
    from vllm import LLM, SamplingParams

    prompts = [
        "A robot may not injure a human being",
        "It is only with the heart that one can see rightly;",
        "The greatest glory in living lies not in never falling,",
    ]
    answers = [
        " or, through inaction, allow a human being to come to harm.",
        " what is essential is invisible to the eye.",
        " but in rising every time we fall.",
    ]
    N = 1
    # Currently, top-p sampling is disabled. `top_p` should be 1.0.
    sampling_params = SamplingParams(temperature=0.7,
                                     top_p=1.0,
                                     n=N,
                                     max_tokens=16)

    # Set `enforce_eager=True` to avoid ahead-of-time compilation.
    # In real workloads, `enforace_eager` should be `False`.

    # disable custom dispatcher, let Dynamo takes over
    # all the control
    llm = LLM(model="google/gemma-2b",
              enforce_eager=True,
              compilation_config={"level": CompilationLevel.DYNAMO_AS_IS})
    outputs = llm.generate(prompts, sampling_params)
    for output, answer in zip(outputs, answers):
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
        assert generated_text.startswith(answer)

compiled_code = sorted(
    glob.glob(os.path.join(temp_dir, "__transformed_code*.py")))

# we should only trigger Dynamo compilation three times:
# one for the profiling phase without kv cache
# one for the prefill phase with symbolic shapes
# one for the decode phase with symbolic shapes
# and later calls should not trigger Dynamo compilation again.
# NOTE: it might still trigger XLA compilation.

# check we have three compiled code
# this is the assumption when we use the custom dispatcher
assert len(compiled_code) == 3

# check all the compilations are as expected
compiled_fn = sorted(
    glob.glob(os.path.join(temp_dir, "__compiled_fn*Captured*.py")))

# the first compilation is the profiling phase,
# it should not have any kv cache
with open(compiled_fn[0]) as f:
    content = f.read()
    assert "kv_caches" not in content

# the second compilation is the prefill phase,
# it should have kv cache and the flash_attention op
with open(compiled_fn[1]) as f:
    content = f.read()
    assert "kv_caches" in content and "torch.ops.xla.flash_attention" in content

# the third compilation is the decode phase,
# it should have kv cache and the paged_attention op
with open(compiled_fn[2]) as f:
    content = f.read()
    assert "kv_caches" in content and "torch.ops.xla.paged_attention" in content
=======

def test_tpu_compilation():
    temp_dir = tempfile.mkdtemp()
    with depyf.prepare_debug(temp_dir):
        from vllm import LLM, SamplingParams

        prompts = [
            "A robot may not injure a human being",
            "It is only with the heart that one can see rightly;",
            "The greatest glory in living lies not in never falling,",
        ]
        answers = [
            " or, through inaction",
            " what is essential ",
            " but in rising ",
        ]

        # Currently, top-p sampling is disabled. `top_p` should be 1.0.
        N = 1
        sampling_params = SamplingParams(temperature=0.7,
                                         top_p=1.0,
                                         n=N,
                                         max_tokens=16)

        llm = LLM(model="Qwen/Qwen2-1.5B-Instruct",
                  max_num_batched_tokens=256,
                  max_model_len=256,
                  max_num_seqs=32,
                  enforce_eager=False)

        outputs = llm.generate(prompts, sampling_params)
        for output, answer in zip(outputs, answers):
            prompt = output.prompt
            generated_text = output.outputs[0].text
            print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
            assert generated_text.startswith(answer)

    compiled_codes = sorted(
        glob.glob(os.path.join(temp_dir, "__transformed_code*for_forward.py")))

    for i, compiled_code in enumerate(compiled_codes):
        print("{} file: {}".format(i + 1, compiled_code))

    # We should only trigger Dynamo compilation 2 times:
    # 1. Forward pass without kv_caches
    # 2. Forward pass with kv_caches
    # Check we have 2 compiled codes
    assert len(compiled_codes) == 2

    kv_cache_prefix = "kv_cache"
    attn_prefix = "ragged_paged_attention"

    def extract_compiled_index(s):
        parts = s.replace(".", "_").split("_")
        numbers = [int(part) for part in parts if part.isdigit()]
        return numbers[0]

    # Check all the compilations are as expected
    compiled_fns = sorted(glob.glob(
        os.path.join(temp_dir, "__compiled_fn*Captured*.py")),
                          key=lambda s: extract_compiled_index(s))

    for i, compiled_fn in enumerate(compiled_fns):
        print("{} file: {}".format(i + 1, compiled_fn))

    # The first compilation should not have any kv_caches
    with open(compiled_fns[0]) as f:
        content = f.read()
        assert kv_cache_prefix not in content

    # The second compilation should have kv_caches and the
    # ragged_paged_attention
    with open(compiled_fns[1]) as f:
        content = f.read()
        assert (kv_cache_prefix in content and attn_prefix in content)
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
