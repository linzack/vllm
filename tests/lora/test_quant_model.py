# SPDX-License-Identifier: Apache-2.0

# Adapted from
# https://github.com/fmmoret/vllm/blob/fm-support-lora-on-quantized-models/tests/lora/test_llama.py
from dataclasses import dataclass
<<<<<<< HEAD
from typing import List
=======
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea

import pytest

import vllm
from vllm.distributed import cleanup_dist_env_and_memory
from vllm.lora.request import LoRARequest
from vllm.platforms import current_platform


@dataclass
class ModelWithQuantization:
    model_path: str
    quantization: str


<<<<<<< HEAD
MODELS: List[ModelWithQuantization]
=======
MODELS: list[ModelWithQuantization]
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
#AWQ quantization is currently not supported in ROCm.
if current_platform.is_rocm():
    MODELS = [
        ModelWithQuantization(
            model_path="TheBloke/TinyLlama-1.1B-Chat-v0.3-GPTQ",
            quantization="GPTQ"),
    ]
else:
    MODELS = [
        ModelWithQuantization(
            model_path="TheBloke/TinyLlama-1.1B-Chat-v0.3-AWQ",
            quantization="AWQ"),
        ModelWithQuantization(
            model_path="TheBloke/TinyLlama-1.1B-Chat-v0.3-GPTQ",
            quantization="GPTQ"),
    ]


<<<<<<< HEAD
def do_sample(llm: vllm.LLM,
              lora_path: str,
              lora_id: int,
              max_tokens: int = 256) -> List[str]:
=======
@pytest.fixture(autouse=True)
def v1(run_with_both_engines_lora):
    # Simple autouse wrapper to run both engines for each test
    # This can be promoted up to conftest.py to run for every
    # test in a package
    pass


def do_sample(llm: vllm.LLM,
              lora_path: str,
              lora_id: int,
              max_tokens: int = 256) -> list[str]:
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
    raw_prompts = [
        "Give me an orange-ish brown color",
        "Give me a neon pink color",
    ]

    def format_prompt_tuples(prompt):
        return f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"

    prompts = [format_prompt_tuples(p) for p in raw_prompts]

    sampling_params = vllm.SamplingParams(temperature=0,
                                          max_tokens=max_tokens,
                                          stop=["<|im_end|>"])
    outputs = llm.generate(
        prompts,
        sampling_params,
        lora_request=LoRARequest(str(lora_id), lora_id, lora_path)
        if lora_id else None)
    # Print the outputs.
<<<<<<< HEAD
    generated_texts: List[str] = []
=======
    generated_texts: list[str] = []
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        generated_texts.append(generated_text)
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
    return generated_texts


<<<<<<< HEAD
@pytest.fixture(autouse=True)
def v1(run_with_both_engines_lora):
    # Simple autouse wrapper to run both engines for each test
    # This can be promoted up to conftest.py to run for every
    # test in a package
    pass


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("tp_size", [1])
def test_quant_model_lora(tinyllama_lora_files, num_gpus_available, model,
                          tp_size):
    if num_gpus_available < tp_size and \
        tp_size > 1 and current_platform.is_cuda_alike():
        pytest.skip(f"Not enough GPUs for tensor parallelism {tp_size}")
=======
@pytest.mark.parametrize("model", MODELS)
def test_quant_model_lora(tinyllama_lora_files, model):
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea

    llm = vllm.LLM(
        model=model.model_path,
        enable_lora=True,
        max_num_seqs=16,
        max_loras=4,
        max_model_len=400,
<<<<<<< HEAD
        tensor_parallel_size=tp_size,
=======
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
        gpu_memory_utilization=0.2,  #avoid OOM
        quantization=model.quantization,
        trust_remote_code=True,
        enable_chunked_prefill=True)

    if model.quantization is None:
        expected_no_lora_output = [
            "Here are some examples of orange-brown colors",
            "I'm sorry, I don't have"
        ]
        expected_lora_output = [
            "#ff8050",
            "#ff8080",
        ]
    elif model.quantization == "AWQ":
        expected_no_lora_output = [
            "I'm sorry, I don't understand",
            "I'm sorry, I don't understand",
        ]
        expected_lora_output = [
            "#f07700: A v",
            "#f00000: A v",
        ]
    elif model.quantization == "GPTQ":
        expected_no_lora_output = [
            "I'm sorry, I don't have",
            "I'm sorry, I don't have",
        ]
        expected_lora_output = [
            "#f08800: This is",
            "#f07788 \n#",
        ]

    def expect_match(output, expected_output):
        # HACK: GPTQ lora outputs are just incredibly unstable.
        # Assert that the outputs changed.
        if (model.quantization == "GPTQ"
                and expected_output is expected_lora_output):
            assert output != expected_no_lora_output
            for i, o in enumerate(output):
                assert o.startswith(
                    '#'), f"Expected example {i} to start with # but got {o}"
            return
        assert output == expected_output

    max_tokens = 10

    print("lora adapter created")
    output = do_sample(llm,
                       tinyllama_lora_files,
                       lora_id=0,
                       max_tokens=max_tokens)
    expect_match(output, expected_no_lora_output)

    print("lora 1")
    output = do_sample(llm,
                       tinyllama_lora_files,
                       lora_id=1,
                       max_tokens=max_tokens)
    expect_match(output, expected_lora_output)

    print("no lora")
    output = do_sample(llm,
                       tinyllama_lora_files,
                       lora_id=0,
                       max_tokens=max_tokens)
    expect_match(output, expected_no_lora_output)

    print("lora 2")
    output = do_sample(llm,
                       tinyllama_lora_files,
                       lora_id=2,
                       max_tokens=max_tokens)
    expect_match(output, expected_lora_output)

    print("removing lora")

    del llm
    cleanup_dist_env_and_memory()


@pytest.mark.parametrize("model", MODELS)
def test_quant_model_tp_equality(tinyllama_lora_files, num_gpus_available,
                                 model):
    if num_gpus_available < 2:
        pytest.skip(f"Not enough GPUs for tensor parallelism {2}")
<<<<<<< HEAD

=======
    if model.quantization == "GPTQ":
        pytest.skip("GPTQ lora outputs are just incredibly unstable")
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
    llm_tp1 = vllm.LLM(
        model=model.model_path,
        enable_lora=True,
        max_num_seqs=16,
        max_loras=4,
<<<<<<< HEAD
        tensor_parallel_size=1,
=======
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
        gpu_memory_utilization=0.2,  #avoid OOM
        quantization=model.quantization,
        trust_remote_code=True,
        enable_chunked_prefill=True)
    output_tp1 = do_sample(llm_tp1, tinyllama_lora_files, lora_id=1)

    del llm_tp1
    cleanup_dist_env_and_memory()

    llm_tp2 = vllm.LLM(
        model=model.model_path,
        enable_lora=True,
        max_num_seqs=16,
        max_loras=4,
        tensor_parallel_size=2,
        gpu_memory_utilization=0.2,  #avoid OOM
        quantization=model.quantization,
        enable_chunked_prefill=True)
    output_tp2 = do_sample(llm_tp2, tinyllama_lora_files, lora_id=1)

    del llm_tp2
    cleanup_dist_env_and_memory()

    assert output_tp1 == output_tp2
