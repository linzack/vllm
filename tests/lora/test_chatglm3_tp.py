# SPDX-License-Identifier: Apache-2.0

<<<<<<< HEAD
from typing import List

import pytest

import vllm
from tests.utils import fork_new_process_for_each_test
from vllm.lora.request import LoRARequest

from ..utils import multi_gpu_test
=======
import pytest

import vllm
from vllm.lora.request import LoRARequest

from ..utils import create_new_process_for_each_test, multi_gpu_test
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea

MODEL_PATH = "THUDM/chatglm3-6b"

PROMPT_TEMPLATE = """I want you to act as a SQL terminal in front of an example database, you need only to return the sql command to me.Below is an instruction that describes a task, Write a response that appropriately completes the request.\n"\n##Instruction:\nconcert_singer contains tables such as stadium, singer, concert, singer_in_concert. Table stadium has columns such as Stadium_ID, Location, Name, Capacity, Highest, Lowest, Average. Stadium_ID is the primary key.\nTable singer has columns such as Singer_ID, Name, Country, Song_Name, Song_release_year, Age, Is_male. Singer_ID is the primary key.\nTable concert has columns such as concert_ID, concert_Name, Theme, Stadium_ID, Year. concert_ID is the primary key.\nTable singer_in_concert has columns such as concert_ID, Singer_ID. concert_ID is the primary key.\nThe Stadium_ID of concert is the foreign key of Stadium_ID of stadium.\nThe Singer_ID of singer_in_concert is the foreign key of Singer_ID of singer.\nThe concert_ID of singer_in_concert is the foreign key of concert_ID of concert.\n\n###Input:\n{query}\n\n###Response:"""  # noqa: E501

EXPECTED_LORA_OUTPUT = [
    "SELECT count(*) FROM singer",
    "SELECT avg(age) ,  min(age) ,  max(age) FROM singer WHERE country  =  'France'",  # noqa: E501
    "SELECT name ,  country ,  age FROM singer ORDER BY age",
]


<<<<<<< HEAD
def do_sample(llm: vllm.LLM, lora_path: str, lora_id: int) -> List[str]:
=======
@pytest.fixture(autouse=True)
def v1(run_with_both_engines_lora):
    # Simple autouse wrapper to run both engines for each test
    # This can be promoted up to conftest.py to run for every
    # test in a package
    pass


def do_sample(llm: vllm.LLM, lora_path: str, lora_id: int) -> list[str]:
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
    prompts = [
        PROMPT_TEMPLATE.format(query="How many singers do we have?"),
        PROMPT_TEMPLATE.format(
            query=
            "What is the average, minimum, and maximum age of all singers from France?"  # noqa: E501
        ),
        PROMPT_TEMPLATE.format(
            query=
            "Show name, country, age for all singers ordered by age from the oldest to the youngest."  # noqa: E501
        ),
    ]
    sampling_params = vllm.SamplingParams(temperature=0, max_tokens=32)
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
        generated_text = output.outputs[0].text.strip()
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


@pytest.mark.skip_v1
@fork_new_process_for_each_test
=======
@create_new_process_for_each_test()
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
def test_chatglm3_lora(chatglm3_lora_files):
    llm = vllm.LLM(MODEL_PATH,
                   max_model_len=1024,
                   enable_lora=True,
                   max_loras=4,
                   max_lora_rank=64,
<<<<<<< HEAD
                   tensor_parallel_size=1,
=======
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
                   trust_remote_code=True,
                   enable_chunked_prefill=True)

    output1 = do_sample(llm, chatglm3_lora_files, lora_id=1)
    for i in range(len(EXPECTED_LORA_OUTPUT)):
        assert output1[i] == EXPECTED_LORA_OUTPUT[i]
    output2 = do_sample(llm, chatglm3_lora_files, lora_id=2)
    for i in range(len(EXPECTED_LORA_OUTPUT)):
        assert output2[i] == EXPECTED_LORA_OUTPUT[i]


<<<<<<< HEAD
@pytest.mark.skip_v1
@multi_gpu_test(num_gpus=4)
@fork_new_process_for_each_test
=======
@multi_gpu_test(num_gpus=4)
@create_new_process_for_each_test()
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
def test_chatglm3_lora_tp4(chatglm3_lora_files):
    llm = vllm.LLM(MODEL_PATH,
                   max_model_len=1024,
                   enable_lora=True,
                   max_loras=4,
                   max_lora_rank=64,
                   tensor_parallel_size=4,
                   trust_remote_code=True,
                   fully_sharded_loras=False,
                   enable_chunked_prefill=True)

    output1 = do_sample(llm, chatglm3_lora_files, lora_id=1)
    for i in range(len(EXPECTED_LORA_OUTPUT)):
        assert output1[i] == EXPECTED_LORA_OUTPUT[i]
    output2 = do_sample(llm, chatglm3_lora_files, lora_id=2)
    for i in range(len(EXPECTED_LORA_OUTPUT)):
        assert output2[i] == EXPECTED_LORA_OUTPUT[i]


<<<<<<< HEAD
@pytest.mark.skip_v1
@multi_gpu_test(num_gpus=4)
@fork_new_process_for_each_test
=======
@multi_gpu_test(num_gpus=4)
@create_new_process_for_each_test()
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
def test_chatglm3_lora_tp4_fully_sharded_loras(chatglm3_lora_files):
    llm = vllm.LLM(MODEL_PATH,
                   max_model_len=1024,
                   enable_lora=True,
                   max_loras=4,
                   max_lora_rank=64,
                   tensor_parallel_size=4,
                   trust_remote_code=True,
                   fully_sharded_loras=True,
                   enable_chunked_prefill=True)
    output1 = do_sample(llm, chatglm3_lora_files, lora_id=1)
    for i in range(len(EXPECTED_LORA_OUTPUT)):
        assert output1[i] == EXPECTED_LORA_OUTPUT[i]
    output2 = do_sample(llm, chatglm3_lora_files, lora_id=2)
    for i in range(len(EXPECTED_LORA_OUTPUT)):
        assert output2[i] == EXPECTED_LORA_OUTPUT[i]
