# SPDX-License-Identifier: Apache-2.0
<<<<<<< HEAD

import argparse
from typing import List, Tuple
=======
"""
This file demonstrates using the `LLMEngine`
for processing prompts with various sampling parameters.
"""
import argparse
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea

from vllm import EngineArgs, LLMEngine, RequestOutput, SamplingParams
from vllm.utils import FlexibleArgumentParser


<<<<<<< HEAD
def create_test_prompts() -> List[Tuple[str, SamplingParams]]:
=======
def create_test_prompts() -> list[tuple[str, SamplingParams]]:
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
    """Create a list of test prompts with their sampling parameters."""
    return [
        ("A robot may not injure a human being",
         SamplingParams(temperature=0.0, logprobs=1, prompt_logprobs=1)),
        ("To be or not to be,",
         SamplingParams(temperature=0.8, top_k=5, presence_penalty=0.2)),
        ("What is the meaning of life?",
         SamplingParams(n=2,
<<<<<<< HEAD
                        best_of=5,
=======
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
                        temperature=0.8,
                        top_p=0.95,
                        frequency_penalty=0.1)),
    ]


def process_requests(engine: LLMEngine,
<<<<<<< HEAD
                     test_prompts: List[Tuple[str, SamplingParams]]):
    """Continuously process a list of prompts and handle the outputs."""
    request_id = 0

=======
                     test_prompts: list[tuple[str, SamplingParams]]):
    """Continuously process a list of prompts and handle the outputs."""
    request_id = 0

    print('-' * 50)
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
    while test_prompts or engine.has_unfinished_requests():
        if test_prompts:
            prompt, sampling_params = test_prompts.pop(0)
            engine.add_request(str(request_id), prompt, sampling_params)
            request_id += 1

<<<<<<< HEAD
        request_outputs: List[RequestOutput] = engine.step()
=======
        request_outputs: list[RequestOutput] = engine.step()
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea

        for request_output in request_outputs:
            if request_output.finished:
                print(request_output)
<<<<<<< HEAD
=======
                print('-' * 50)
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea


def initialize_engine(args: argparse.Namespace) -> LLMEngine:
    """Initialize the LLMEngine from the command line arguments."""
    engine_args = EngineArgs.from_cli_args(args)
    return LLMEngine.from_engine_args(engine_args)


<<<<<<< HEAD
=======
def parse_args():
    parser = FlexibleArgumentParser(
        description='Demo on using the LLMEngine class directly')
    parser = EngineArgs.add_cli_args(parser)
    return parser.parse_args()


>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
def main(args: argparse.Namespace):
    """Main function that sets up and runs the prompt processing."""
    engine = initialize_engine(args)
    test_prompts = create_test_prompts()
    process_requests(engine, test_prompts)


if __name__ == '__main__':
<<<<<<< HEAD
    parser = FlexibleArgumentParser(
        description='Demo on using the LLMEngine class directly')
    parser = EngineArgs.add_cli_args(parser)
    args = parser.parse_args()
=======
    args = parse_args()
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
    main(args)
