# SPDX-License-Identifier: Apache-2.0
"""Example Python client for `vllm.entrypoints.api_server`
<<<<<<< HEAD
=======
Start the demo server:
    python -m vllm.entrypoints.api_server --model <model_name>

>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
NOTE: The API server is used only for demonstration and simple performance
benchmarks. It is not intended for production use.
For production use, we recommend `vllm serve` and the OpenAI client API.
"""

import argparse
import json
<<<<<<< HEAD
from typing import Iterable, List
=======
from argparse import Namespace
from collections.abc import Iterable
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea

import requests


def clear_line(n: int = 1) -> None:
    LINE_UP = '\033[1A'
    LINE_CLEAR = '\x1b[2K'
    for _ in range(n):
        print(LINE_UP, end=LINE_CLEAR, flush=True)


def post_http_request(prompt: str,
                      api_url: str,
                      n: int = 1,
                      stream: bool = False) -> requests.Response:
    headers = {"User-Agent": "Test Client"}
    pload = {
        "prompt": prompt,
        "n": n,
<<<<<<< HEAD
        "use_beam_search": True,
=======
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
        "temperature": 0.0,
        "max_tokens": 16,
        "stream": stream,
    }
    response = requests.post(api_url,
                             headers=headers,
                             json=pload,
                             stream=stream)
    return response


<<<<<<< HEAD
def get_streaming_response(response: requests.Response) -> Iterable[List[str]]:
    for chunk in response.iter_lines(chunk_size=8192,
                                     decode_unicode=False,
                                     delimiter=b"\0"):
=======
def get_streaming_response(response: requests.Response) -> Iterable[list[str]]:
    for chunk in response.iter_lines(chunk_size=8192,
                                     decode_unicode=False,
                                     delimiter=b"\n"):
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
        if chunk:
            data = json.loads(chunk.decode("utf-8"))
            output = data["text"]
            yield output


<<<<<<< HEAD
def get_response(response: requests.Response) -> List[str]:
=======
def get_response(response: requests.Response) -> list[str]:
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
    data = json.loads(response.content)
    output = data["text"]
    return output


<<<<<<< HEAD
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--n", type=int, default=4)
    parser.add_argument("--prompt", type=str, default="San Francisco is a")
    parser.add_argument("--stream", action="store_true")
    args = parser.parse_args()
=======
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--n", type=int, default=1)
    parser.add_argument("--prompt", type=str, default="San Francisco is a")
    parser.add_argument("--stream", action="store_true")
    return parser.parse_args()


def main(args: Namespace):
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
    prompt = args.prompt
    api_url = f"http://{args.host}:{args.port}/generate"
    n = args.n
    stream = args.stream

    print(f"Prompt: {prompt!r}\n", flush=True)
    response = post_http_request(prompt, api_url, n, stream)

    if stream:
        num_printed_lines = 0
        for h in get_streaming_response(response):
            clear_line(num_printed_lines)
            num_printed_lines = 0
            for i, line in enumerate(h):
                num_printed_lines += 1
                print(f"Beam candidate {i}: {line!r}", flush=True)
    else:
        output = get_response(response)
        for i, line in enumerate(output):
            print(f"Beam candidate {i}: {line!r}", flush=True)
<<<<<<< HEAD
=======


if __name__ == "__main__":
    args = parse_args()
    main(args)
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
