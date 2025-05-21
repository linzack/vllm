# SPDX-License-Identifier: Apache-2.0

<<<<<<< HEAD
from typing import Dict, List, Optional
=======
from typing import Optional
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea

from typing_extensions import TypedDict


class ServerConfig(TypedDict, total=False):
    model: str
<<<<<<< HEAD
    arguments: List[str]
=======
    arguments: list[str]
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
    system_prompt: Optional[str]
    supports_parallel: Optional[bool]
    supports_rocm: Optional[bool]


<<<<<<< HEAD
ARGS: List[str] = ["--max-model-len", "1024"]

CONFIGS: Dict[str, ServerConfig] = {
=======
ARGS: list[str] = ["--max-model-len", "1024"]

CONFIGS: dict[str, ServerConfig] = {
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
    "mistral": {
        "model":
        "mistralai/Mistral-7B-Instruct-v0.3",
        "arguments": [
            "--tokenizer-mode", "mistral",
            "--ignore-patterns=\"consolidated.safetensors\""
        ],
        "system_prompt":
        "You are a helpful assistant with access to tools. If a tool"
        " that you have would be helpful to answer a user query, "
        "call the tool. Otherwise, answer the user's query directly "
        "without calling a tool. DO NOT CALL A TOOL THAT IS IRRELEVANT "
        "to the user's question - just respond to it normally."
    },
}
