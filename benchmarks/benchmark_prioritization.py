# SPDX-License-Identifier: Apache-2.0
"""Benchmark offline prioritization."""
<<<<<<< HEAD
=======

>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
import argparse
import dataclasses
import json
import random
import time
<<<<<<< HEAD
from typing import List, Optional, Tuple
=======
from typing import Optional
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea

from transformers import AutoTokenizer, PreTrainedTokenizerBase

from vllm.engine.arg_utils import EngineArgs
from vllm.utils import FlexibleArgumentParser


<<<<<<< HEAD
#Select a equi-probable random priority
=======
# Select a equi-probable random priority
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
def get_random_flag():
    return 0 if random.random() < 0.5 else 1


def sample_requests(
    dataset_path: str,
    num_requests: int,
    tokenizer: PreTrainedTokenizerBase,
    fixed_output_len: Optional[int],
<<<<<<< HEAD
) -> List[Tuple[str, int, int]]:
=======
) -> list[tuple[str, int, int, int]]:
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
    if fixed_output_len is not None and fixed_output_len < 4:
        raise ValueError("output_len too small")

    # Load the dataset.
    with open(dataset_path) as f:
        dataset = json.load(f)
    # Filter out the conversations with less than 2 turns.
    dataset = [data for data in dataset if len(data["conversations"]) >= 2]
    # Only keep the first two turns of each conversation.
<<<<<<< HEAD
    dataset = [(data["conversations"][0]["value"],
                data["conversations"][1]["value"]) for data in dataset]
=======
    dataset = [
        (data["conversations"][0]["value"], data["conversations"][1]["value"])
        for data in dataset
    ]
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea

    # Shuffle the dataset.
    random.shuffle(dataset)

    # Filter out sequences that are too long or too short
<<<<<<< HEAD
    filtered_dataset: List[Tuple[str, int, int]] = []
=======
    filtered_dataset: list[tuple[str, int, int]] = []
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
    for i in range(len(dataset)):
        if len(filtered_dataset) == num_requests:
            break

        # Tokenize the prompts and completions.
        prompt = dataset[i][0]
        prompt_token_ids = tokenizer(prompt).input_ids
        completion = dataset[i][1]
        completion_token_ids = tokenizer(completion).input_ids
        prompt_len = len(prompt_token_ids)
<<<<<<< HEAD
        output_len = len(completion_token_ids
                         ) if fixed_output_len is None else fixed_output_len
=======
        output_len = (
            len(completion_token_ids) if fixed_output_len is None else fixed_output_len
        )
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
        if prompt_len < 4 or output_len < 4:
            # Prune too short sequences.
            continue
        if prompt_len > 1024 or prompt_len + output_len > 2048:
            # Prune too long sequences.
            continue

        priority = get_random_flag()

        filtered_dataset.append((prompt, prompt_len, output_len, priority))

    return filtered_dataset


def run_vllm(
<<<<<<< HEAD
    requests: List[Tuple[str, int, int]],
    n: int,
    engine_args: EngineArgs,
) -> float:
    from vllm import LLM, SamplingParams
=======
    requests: list[tuple[str, int, int]],
    n: int,
    engine_args: EngineArgs,
    disable_detokenize: bool = False,
) -> float:
    from vllm import LLM, SamplingParams

>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
    llm = LLM(**dataclasses.asdict(engine_args))

    assert all(
        llm.llm_engine.model_config.max_model_len >= (request[1] + request[2])
<<<<<<< HEAD
        for request in requests), (
            "Please ensure that max_model_len is greater than the sum of"
            " input_len and output_len for all requests.")
=======
        for request in requests
    ), (
        "Please ensure that max_model_len is greater than the sum of"
        " input_len and output_len for all requests."
    )
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea

    # Add the requests to the engine.
    prompts = []
    sampling_params = []
    priority = []
    for prompt, _, output_len, _priority in requests:
        prompts.append(prompt)
        priority.append(_priority)
        sampling_params.append(
            SamplingParams(
                n=n,
                temperature=1.0,
                top_p=1.0,
                ignore_eos=True,
                max_tokens=output_len,
<<<<<<< HEAD
            ))
=======
                detokenize=not disable_detokenize,
            )
        )
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea

    start = time.perf_counter()
    llm.generate(prompts, sampling_params, priority=priority, use_tqdm=True)
    end = time.perf_counter()
    return end - start


def main(args: argparse.Namespace):
    print(args)
    random.seed(args.seed)

    # Sample the requests.
    tokenizer = AutoTokenizer.from_pretrained(
<<<<<<< HEAD
        args.tokenizer, trust_remote_code=args.trust_remote_code)
    if args.dataset is None:
        # Synthesize a prompt with the given input length.
        prompt = "hi" * (args.input_len - 1)
        requests = [(prompt, args.input_len, args.output_len,
                     get_random_flag()) for _ in range(args.num_prompts)]
    else:
        requests = sample_requests(args.dataset, args.num_prompts, tokenizer,
                                   args.output_len)

    if args.backend == "vllm":
        elapsed_time = run_vllm(requests, args.n,
                                EngineArgs.from_cli_args(args))
    else:
        raise ValueError(f"Unknown backend: {args.backend}")
    total_num_tokens = sum(prompt_len + output_len
                           for _, prompt_len, output_len, priority in requests)
    print(f"Throughput: {len(requests) / elapsed_time:.2f} requests/s, "
          f"{total_num_tokens / elapsed_time:.2f} tokens/s")
=======
        args.tokenizer, trust_remote_code=args.trust_remote_code
    )
    if args.dataset is None:
        # Synthesize a prompt with the given input length.
        prompt = "hi" * (args.input_len - 1)
        requests = [
            (prompt, args.input_len, args.output_len, get_random_flag())
            for _ in range(args.num_prompts)
        ]
    else:
        requests = sample_requests(
            args.dataset, args.num_prompts, tokenizer, args.output_len
        )

    if args.backend == "vllm":
        elapsed_time = run_vllm(
            requests, args.n, EngineArgs.from_cli_args(args), args.disable_detokenize
        )
    else:
        raise ValueError(f"Unknown backend: {args.backend}")
    total_num_tokens = sum(
        prompt_len + output_len for _, prompt_len, output_len, priority in requests
    )
    print(
        f"Throughput: {len(requests) / elapsed_time:.2f} requests/s, "
        f"{total_num_tokens / elapsed_time:.2f} tokens/s"
    )
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea

    # Output JSON results if specified
    if args.output_json:
        results = {
            "elapsed_time": elapsed_time,
            "num_requests": len(requests),
            "total_num_tokens": total_num_tokens,
            "requests_per_second": len(requests) / elapsed_time,
            "tokens_per_second": total_num_tokens / elapsed_time,
        }
        with open(args.output_json, "w") as f:
            json.dump(results, f, indent=4)


if __name__ == "__main__":
    parser = FlexibleArgumentParser(description="Benchmark the throughput.")
<<<<<<< HEAD
    parser.add_argument("--backend",
                        type=str,
                        choices=["vllm", "hf", "mii"],
                        default="vllm")
    parser.add_argument("--dataset",
                        type=str,
                        default=None,
                        help="Path to the dataset.")
    parser.add_argument("--input-len",
                        type=int,
                        default=None,
                        help="Input prompt length for each request")
    parser.add_argument("--output-len",
                        type=int,
                        default=None,
                        help="Output length for each request. Overrides the "
                        "output length from the dataset.")
    parser.add_argument("--n",
                        type=int,
                        default=1,
                        help="Number of generated sequences per prompt.")
    parser.add_argument("--num-prompts",
                        type=int,
                        default=200,
                        help="Number of prompts to process.")
    parser.add_argument(
        '--output-json',
        type=str,
        default=None,
        help='Path to save the throughput results in JSON format.')
=======
    parser.add_argument(
        "--backend", type=str, choices=["vllm", "hf", "mii"], default="vllm"
    )
    parser.add_argument(
        "--dataset", type=str, default=None, help="Path to the dataset."
    )
    parser.add_argument(
        "--input-len",
        type=int,
        default=None,
        help="Input prompt length for each request",
    )
    parser.add_argument(
        "--output-len",
        type=int,
        default=None,
        help="Output length for each request. Overrides the "
        "output length from the dataset.",
    )
    parser.add_argument(
        "--n", type=int, default=1, help="Number of generated sequences per prompt."
    )
    parser.add_argument(
        "--num-prompts", type=int, default=200, help="Number of prompts to process."
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Path to save the throughput results in JSON format.",
    )
    parser.add_argument(
        "--disable-detokenize",
        action="store_true",
        help=(
            "Do not detokenize responses (i.e. do not include "
            "detokenization time in the latency measurement)"
        ),
    )
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea

    parser = EngineArgs.add_cli_args(parser)
    args = parser.parse_args()
    if args.tokenizer is None:
        args.tokenizer = args.model
    if args.dataset is None:
        assert args.input_len is not None
        assert args.output_len is not None
    else:
        assert args.input_len is None

    main(args)
