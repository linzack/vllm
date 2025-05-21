# SPDX-License-Identifier: Apache-2.0
<<<<<<< HEAD

import gc
import time
from typing import List
=======
"""
This file demonstrates the usage of text generation with an LLM model,
comparing the performance with and without speculative decoding.

Note that still not support `v1`:
VLLM_USE_V1=0 python examples/offline_inference/mlpspeculator.py
"""

import gc
import time
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea

from vllm import LLM, SamplingParams


<<<<<<< HEAD
def time_generation(llm: LLM, prompts: List[str],
                    sampling_params: SamplingParams):
=======
def time_generation(llm: LLM, prompts: list[str],
                    sampling_params: SamplingParams, title: str):
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
    # Generate texts from the prompts. The output is a list of RequestOutput
    # objects that contain the prompt, generated text, and other information.
    # Warmup first
    llm.generate(prompts, sampling_params)
    llm.generate(prompts, sampling_params)
    start = time.time()
    outputs = llm.generate(prompts, sampling_params)
    end = time.time()
<<<<<<< HEAD
    print((end - start) / sum([len(o.outputs[0].token_ids) for o in outputs]))
=======
    print("-" * 50)
    print(title)
    print("time: ",
          (end - start) / sum(len(o.outputs[0].token_ids) for o in outputs))
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
    # Print the outputs.
    for output in outputs:
        generated_text = output.outputs[0].text
        print(f"text: {generated_text!r}")
<<<<<<< HEAD


if __name__ == "__main__":

=======
        print("-" * 50)


def main():
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
    template = (
        "Below is an instruction that describes a task. Write a response "
        "that appropriately completes the request.\n\n### Instruction:\n{}"
        "\n\n### Response:\n")

    # Sample prompts.
    prompts = [
        "Write about the president of the United States.",
    ]
    prompts = [template.format(prompt) for prompt in prompts]
    # Create a sampling params object.
    sampling_params = SamplingParams(temperature=0.0, max_tokens=200)

    # Create an LLM without spec decoding
    llm = LLM(model="meta-llama/Llama-2-13b-chat-hf")

<<<<<<< HEAD
    print("Without speculation")
    time_generation(llm, prompts, sampling_params)
=======
    time_generation(llm, prompts, sampling_params, "Without speculation")
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea

    del llm
    gc.collect()

    # Create an LLM with spec decoding
    llm = LLM(
        model="meta-llama/Llama-2-13b-chat-hf",
<<<<<<< HEAD
        speculative_model="ibm-ai-platform/llama-13b-accelerator",
    )

    print("With speculation")
    time_generation(llm, prompts, sampling_params)
=======
        speculative_config={
            "model": "ibm-ai-platform/llama-13b-accelerator",
        },
    )

    time_generation(llm, prompts, sampling_params, "With speculation")


if __name__ == "__main__":
    main()
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
