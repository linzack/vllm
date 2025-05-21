# SPDX-License-Identifier: Apache-2.0
"""
Example of using the OpenAI entrypoint's rerank API which is compatible with
the Cohere SDK: https://github.com/cohere-ai/cohere-python
<<<<<<< HEAD

run: vllm serve BAAI/bge-reranker-base
"""
import cohere

# cohere v1 client
co = cohere.Client(base_url="http://localhost:8000", api_key="sk-fake-key")
rerank_v1_result = co.rerank(
    model="BAAI/bge-reranker-base",
    query="What is the capital of France?",
    documents=[
        "The capital of France is Paris", "Reranking is fun!",
        "vLLM is an open-source framework for fast AI serving"
    ])

print(rerank_v1_result)

# or the v2
co2 = cohere.ClientV2("sk-fake-key", base_url="http://localhost:8000")

v2_rerank_result = co2.rerank(
    model="BAAI/bge-reranker-base",
    query="What is the capital of France?",
    documents=[
        "The capital of France is Paris", "Reranking is fun!",
        "vLLM is an open-source framework for fast AI serving"
    ])

print(v2_rerank_result)
=======
Note that `pip install cohere` is needed to run this example.

run: vllm serve BAAI/bge-reranker-base
"""
from typing import Union

import cohere
from cohere import Client, ClientV2

model = "BAAI/bge-reranker-base"

query = "What is the capital of France?"

documents = [
    "The capital of France is Paris", "Reranking is fun!",
    "vLLM is an open-source framework for fast AI serving"
]


def cohere_rerank(client: Union[Client, ClientV2], model: str, query: str,
                  documents: list[str]) -> dict:
    return client.rerank(model=model, query=query, documents=documents)


def main():
    # cohere v1 client
    cohere_v1 = cohere.Client(base_url="http://localhost:8000",
                              api_key="sk-fake-key")
    rerank_v1_result = cohere_rerank(cohere_v1, model, query, documents)
    print("-" * 50)
    print("rerank_v1_result:\n", rerank_v1_result)
    print("-" * 50)

    # or the v2
    cohere_v2 = cohere.ClientV2("sk-fake-key",
                                base_url="http://localhost:8000")
    rerank_v2_result = cohere_rerank(cohere_v2, model, query, documents)
    print("rerank_v2_result:\n", rerank_v2_result)
    print("-" * 50)


if __name__ == "__main__":
    main()
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
