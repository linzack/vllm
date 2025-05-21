# SPDX-License-Identifier: Apache-2.0

<<<<<<< HEAD
from typing import Dict

import pytest
import requests
=======
import json

import pytest
import requests
from PIL import Image
from transformers import AutoProcessor
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea

from vllm.entrypoints.openai.protocol import EmbeddingResponse
from vllm.multimodal.utils import encode_image_base64, fetch_image

from ...utils import VLLM_PATH, RemoteOpenAIServer

MODEL_NAME = "TIGER-Lab/VLM2Vec-Full"
MAXIMUM_IMAGES = 2

vlm2vec_jinja_path = VLLM_PATH / "examples/template_vlm2vec.jinja"
assert vlm2vec_jinja_path.exists()

# Test different image extensions (JPG/PNG) and formats (gray/RGB/RGBA)
TEST_IMAGE_URLS = [
    "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/f/fa/Grayscale_8bits_palette_sample_image.png",
    "https://upload.wikimedia.org/wikipedia/commons/thumb/9/91/Venn_diagram_rgb.svg/1280px-Venn_diagram_rgb.svg.png",
    "https://upload.wikimedia.org/wikipedia/commons/0/0b/RGBA_comp.png",
]


@pytest.fixture(scope="module")
def server():
    args = [
        "--task",
        "embed",
<<<<<<< HEAD
        "--dtype",
        "bfloat16",
=======
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
        "--max-model-len",
        "2048",
        "--max-num-seqs",
        "5",
        "--enforce-eager",
        "--trust-remote-code",
        "--limit-mm-per-prompt",
<<<<<<< HEAD
        f"image={MAXIMUM_IMAGES}",
=======
        json.dumps({"image": MAXIMUM_IMAGES}),
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
        "--chat-template",
        str(vlm2vec_jinja_path),
    ]

    with RemoteOpenAIServer(MODEL_NAME, args) as remote_server:
        yield remote_server


@pytest.fixture(scope="session")
<<<<<<< HEAD
def base64_encoded_image() -> Dict[str, str]:
=======
def base64_encoded_image() -> dict[str, str]:
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
    return {
        image_url: encode_image_base64(fetch_image(image_url))
        for image_url in TEST_IMAGE_URLS
    }


<<<<<<< HEAD
=======
def get_hf_prompt_tokens(model_name, content, image_url):
    processor = AutoProcessor.from_pretrained(model_name,
                                              trust_remote_code=True,
                                              num_crops=4)

    placeholder = "<|image_1|> "
    prompt = f"{placeholder}{content}"
    images = [Image.open(requests.get(image_url, stream=True).raw)]
    inputs = processor(prompt, images, return_tensors="pt")
    return inputs.input_ids.shape[1]


>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", [MODEL_NAME])
@pytest.mark.parametrize("image_url", TEST_IMAGE_URLS)
async def test_image_embedding(server: RemoteOpenAIServer, model_name: str,
                               image_url: str):
<<<<<<< HEAD
=======
    content_text = "Represent the given image."
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
    messages = [{
        "role":
        "user",
        "content": [
            {
                "type": "image_url",
                "image_url": {
                    "url": image_url
                }
            },
            {
                "type": "text",
<<<<<<< HEAD
                "text": "Represent the given image."
=======
                "text": content_text
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
            },
        ],
    }]

    response = requests.post(
        server.url_for("v1/embeddings"),
        json={
            "model": model_name,
            "messages": messages,
            "encoding_format": "float"
        },
    )
    response.raise_for_status()
    embeddings = EmbeddingResponse.model_validate(response.json())

<<<<<<< HEAD
=======
    hf_prompt_tokens = get_hf_prompt_tokens(model_name, content_text,
                                            image_url)

>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
    assert embeddings.id is not None
    assert len(embeddings.data) == 1
    assert len(embeddings.data[0].embedding) == 3072
    assert embeddings.usage.completion_tokens == 0
<<<<<<< HEAD
    assert embeddings.usage.prompt_tokens == 763
    assert embeddings.usage.total_tokens == 763
=======
    assert embeddings.usage.prompt_tokens == hf_prompt_tokens
    assert embeddings.usage.total_tokens == hf_prompt_tokens
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
