# SPDX-License-Identifier: Apache-2.0
<<<<<<< HEAD

import json
import os
import tempfile
from collections import UserList
from enum import Enum
from typing import (Any, Callable, Dict, List, Optional, Tuple, Type,
                    TypedDict, TypeVar, Union)
=======
import json
import os
import tempfile
from enum import Enum
from typing import Any, Callable, Optional, TypedDict, TypeVar, Union
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import snapshot_download
from PIL import Image
<<<<<<< HEAD
from transformers import (AutoModelForCausalLM, AutoTokenizer, BatchEncoding,
                          BatchFeature)
=======
from transformers import (AutoConfig, AutoModelForCausalLM, AutoTokenizer,
                          BatchEncoding, BatchFeature)
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
from transformers.models.auto.auto_factory import _BaseAutoModelClass

from tests.models.utils import (TokensTextLogprobs,
                                TokensTextLogprobsPromptLogprobs)
from vllm import LLM, SamplingParams
<<<<<<< HEAD
from vllm.assets.image import ImageAsset
from vllm.assets.video import VideoAsset
from vllm.config import LoadFormat, TaskOption, TokenizerPoolConfig
=======
from vllm.assets.audio import AudioAsset
from vllm.assets.image import ImageAsset
from vllm.assets.video import VideoAsset
from vllm.config import TaskOption, _get_and_verify_dtype
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
from vllm.connections import global_http_connection
from vllm.distributed import (cleanup_dist_env_and_memory,
                              init_distributed_environment,
                              initialize_model_parallel)
from vllm.inputs import (ExplicitEncoderDecoderPrompt, TextPrompt,
<<<<<<< HEAD
                         TokensPrompt, to_enc_dec_tuple_list,
                         zip_enc_dec_prompts)
from vllm.logger import init_logger
from vllm.outputs import RequestOutput
from vllm.sampling_params import BeamSearchParams
from vllm.utils import (STR_DTYPE_TO_TORCH_DTYPE, cuda_device_count_stateless,
                        identity, is_list_of)
=======
                         to_enc_dec_tuple_list, zip_enc_dec_prompts)
from vllm.logger import init_logger
from vllm.outputs import RequestOutput
from vllm.sampling_params import BeamSearchParams
from vllm.utils import cuda_device_count_stateless
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea

logger = init_logger(__name__)

_TEST_DIR = os.path.dirname(__file__)
_TEST_PROMPTS = [os.path.join(_TEST_DIR, "prompts", "example.txt")]
_LONG_PROMPTS = [os.path.join(_TEST_DIR, "prompts", "summary.txt")]
_SYS_MSG = os.path.join(_TEST_DIR, "system_messages", "sonnet3.5_nov2024.txt")

_M = TypeVar("_M")

<<<<<<< HEAD
MODELS_ON_S3 = [
    "distilbert/distilgpt2",
    "meta-llama/Llama-2-7b-hf",
    "meta-llama/Meta-Llama-3-8B",
    "meta-llama/Llama-3.2-1B",
    "meta-llama/Llama-3.2-1B-Instruct",
    "openai-community/gpt2",
    "ArthurZ/Ilama-3.2-1B",
    "llava-hf/llava-1.5-7b-hf",
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "ai21labs/Jamba-tiny-random",
    "neuralmagic/Meta-Llama-3-8B-Instruct-FP8-KV",
    "nm-testing/Phi-3-mini-128k-instruct-FP8",
    "nm-testing/Qwen2-0.5B-Instruct-FP8-SkipQKV",
    "neuralmagic/Meta-Llama-3-8B-Instruct-FP8-KV",
    "nm-testing/Qwen2-1.5B-Instruct-FP8-K-V",
    "ModelCloud/Qwen1.5-1.8B-Chat-GPTQ-4bits-dynamic-cfg-with-lm_head-symTrue",
    "ModelCloud/Qwen1.5-1.8B-Chat-GPTQ-4bits-dynamic-cfg-with-lm_head-symFalse",
    "AMead10/Llama-3.2-1B-Instruct-AWQ",
    "shuyuej/Llama-3.2-1B-Instruct-GPTQ",
    "ModelCloud/Qwen1.5-1.8B-Chat-GPTQ-4bits-dynamic-cfg-with-lm_head",
    "ModelCloud/TinyLlama-1.1B-Chat-v1.0-GPTQ-4bit-10-25-2024",
    "TheBloke/TinyLlama-1.1B-Chat-v1.0-GPTQ",
    "neuralmagic/Meta-Llama-3-8B-Instruct-FP8",
    "amd/Llama-3.1-8B-Instruct-FP8-KV-Quark-test",
    "nm-testing/tinyllama-oneshot-w8w8-test-static-shape-change",
    "nm-testing/tinyllama-oneshot-w8-channel-a8-tensor",
    "nm-testing/asym-w8w8-int8-static-per-tensor-tiny-llama",
    "neuralmagic/Llama-3.2-1B-quantized.w8a8",
    "nm-testing/Meta-Llama-3-8B-Instruct-W8A8-Dynamic-Asym",
    "nm-testing/Meta-Llama-3-8B-Instruct-W8A8-Static-Per-Tensor-Sym",
    "nm-testing/Meta-Llama-3-8B-Instruct-W8A8-Static-Per-Tensor-Asym",
    "nm-testing/tinyllama-oneshot-w8w8-test-static-shape-change",
    "nm-testing/tinyllama-oneshot-w8a8-dynamic-token-v2",
    "nm-testing/tinyllama-oneshot-w8a8-dynamic-token-v2-asym",
    "nm-testing/tinyllama-oneshot-w8a8-channel-dynamic-token-v2",
    "nm-testing/tinyllama-oneshot-w8a8-channel-dynamic-token-v2-asym",
    "nm-testing/tinyllama-oneshot-w4a16-channel-v2",
    "nm-testing/tinyllama-oneshot-w4a16-group128-v2",
    "nm-testing/tinyllama-oneshot-w8a16-per-channel",
    "nm-testing/llama7b-one-shot-2_4-w4a16-marlin24-t",
    "nm-testing/Meta-Llama-3-8B-FP8-compressed-tensors-test",
    "nm-testing/TinyLlama-1.1B-compressed-tensors-kv-cache-scheme",
    "nm-testing/Meta-Llama-3-8B-Instruct-FP8-Dynamic-2of4-testing",
    "nm-testing/Meta-Llama-3-8B-Instruct-FP8-Static-Per-Tensor-testing",
    "nm-testing/Meta-Llama-3-8B-Instruct-FP8-Static-testing",
    "nm-testing/Meta-Llama-3-8B-Instruct-FP8-Dynamic-IA-Per-Tensor-Weight-testing",
    "nm-testing/TinyLlama-1.1B-Chat-v1.0-gsm8k-pruned.2of4-chnl_wts_per_tok_dyn_act_fp8-BitM",
    "nm-testing/TinyLlama-1.1B-Chat-v1.0-gsm8k-pruned.2of4-chnl_wts_tensor_act_fp8-BitM",
    "nm-testing/TinyLlama-1.1B-Chat-v1.0-gsm8k-pruned.2of4-tensor_wts_per_tok_dyn_act_fp8-BitM",
    "nm-testing/TinyLlama-1.1B-Chat-v1.0-gsm8k-pruned.2of4-tensor_wts_tensor_act_fp8-BitM",
    "nm-testing/TinyLlama-1.1B-Chat-v1.0-gsm8k-pruned.2of4-chnl_wts_per_tok_dyn_act_int8-BitM",
    "nm-testing/TinyLlama-1.1B-Chat-v1.0-gsm8k-pruned.2of4-chnl_wts_tensor_act_int8-BitM",
    "nm-testing/TinyLlama-1.1B-Chat-v1.0-gsm8k-pruned.2of4-tensor_wts_per_tok_dyn_act_int8-BitM",
    "nm-testing/TinyLlama-1.1B-Chat-v1.0-gsm8k-pruned.2of4-tensor_wts_tensor_act_int8-BitM",
    "nm-testing/TinyLlama-1.1B-Chat-v1.0-INT8-Dynamic-IA-Per-Channel-Weight-testing",
    "nm-testing/TinyLlama-1.1B-Chat-v1.0-INT8-Static-testing",
    "nm-testing/TinyLlama-1.1B-Chat-v1.0-INT8-Dynamic-IA-Per-Tensor-Weight-testing",
    "nm-testing/TinyLlama-1.1B-Chat-v1.0-2of4-Sparse-Dense-Compressor",
    "nm-testing/llama2.c-stories42M-pruned2.4-compressed",
]

MODEL_WEIGHTS_S3_BUCKET = "s3://vllm-ci-model-weights"

_PromptMultiModalInput = Union[List[_M], List[List[_M]]]

PromptImageInput = _PromptMultiModalInput[Image.Image]
PromptAudioInput = _PromptMultiModalInput[Tuple[np.ndarray, int]]
PromptVideoInput = _PromptMultiModalInput[np.ndarray]


def _read_prompts(filename: str) -> List[str]:
=======
_PromptMultiModalInput = Union[list[_M], list[list[_M]]]

PromptImageInput = _PromptMultiModalInput[Image.Image]
PromptAudioInput = _PromptMultiModalInput[tuple[np.ndarray, int]]
PromptVideoInput = _PromptMultiModalInput[np.ndarray]


def _read_prompts(filename: str) -> list[str]:
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
    with open(filename) as f:
        prompts = f.readlines()
        return prompts


<<<<<<< HEAD
class _ImageAssetPrompts(TypedDict):
=======
class ImageAssetPrompts(TypedDict):
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
    stop_sign: str
    cherry_blossom: str


<<<<<<< HEAD
class _ImageAssetsBase(UserList[ImageAsset]):
    pass


class _ImageAssets(_ImageAssetsBase):
=======
class ImageTestAssets(list[ImageAsset]):
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea

    def __init__(self) -> None:
        super().__init__([
            ImageAsset("stop_sign"),
            ImageAsset("cherry_blossom"),
        ])

<<<<<<< HEAD
    def prompts(self, prompts: _ImageAssetPrompts) -> List[str]:
=======
    def prompts(self, prompts: ImageAssetPrompts) -> list[str]:
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
        """
        Convenience method to define the prompt for each test image.

        The order of the returned prompts matches the order of the
        assets when iterating through this object.
        """
        return [prompts["stop_sign"], prompts["cherry_blossom"]]


<<<<<<< HEAD
class _VideoAssetPrompts(TypedDict):
    sample_demo_1: str


class _VideoAssetsBase(UserList[VideoAsset]):
    pass


class _VideoAssets(_VideoAssetsBase):

    def __init__(self) -> None:
        super().__init__([
            VideoAsset("sample_demo_1.mp4"),
        ])

    def prompts(self, prompts: _VideoAssetPrompts) -> List[str]:
        return [prompts["sample_demo_1"]]


IMAGE_ASSETS = _ImageAssets()
"""Singleton instance of :class:`_ImageAssets`."""
VIDEO_ASSETS = _VideoAssets()
"""Singleton instance of :class:`_VideoAssets`."""
=======
class VideoAssetPrompts(TypedDict):
    baby_reading: str


class VideoTestAssets(list[VideoAsset]):

    def __init__(self) -> None:
        super().__init__([
            VideoAsset("baby_reading"),
        ])

    def prompts(self, prompts: VideoAssetPrompts) -> list[str]:
        return [prompts["baby_reading"]]


class AudioAssetPrompts(TypedDict):
    mary_had_lamb: str
    winning_call: str


class AudioTestAssets(list[AudioAsset]):

    def __init__(self) -> None:
        super().__init__([
            AudioAsset("mary_had_lamb"),
            AudioAsset("winning_call"),
        ])

    def prompts(self, prompts: AudioAssetPrompts) -> list[str]:
        return [prompts["mary_had_lamb"], prompts["winning_call"]]


IMAGE_ASSETS = ImageTestAssets()
"""Singleton instance of {class}`ImageTestAssets`."""
VIDEO_ASSETS = VideoTestAssets()
"""Singleton instance of {class}`VideoTestAssets`."""
AUDIO_ASSETS = AudioTestAssets()
"""Singleton instance of {class}`AudioTestAssets`."""


@pytest.fixture(scope="function", autouse=True)
def cleanup_VLLM_USE_V1(monkeypatch):
    """
    The V1 oracle sets "VLLM_USE_V1" during loading. This means
    that each invocation of a test change the env variable.

    If we touch "VLLM_USE_V1" with monkeypatch, then any changes
    made during the test run by vLLM will be cleaned up.

    This fixture is used by every test.
    """

    # If VLLM_USE_V1 is not set, set then delete. This will
    # cause monkeypatch to clean up VLLM_USE_V1 upon exit
    # if VLLM modifies the value of envs.VLLM_USE_V1.
    if "VLLM_USE_V1" not in os.environ:
        monkeypatch.setenv("VLLM_USE_V1", "")
        monkeypatch.delenv("VLLM_USE_V1")
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea


@pytest.fixture(params=[True, False])
def run_with_both_engines(request, monkeypatch):
    # Automatically runs tests twice, once with V1 and once without
    use_v1 = request.param
    # Tests decorated with `@skip_v1` are only run without v1
    skip_v1 = request.node.get_closest_marker("skip_v1")

    if use_v1:
        if skip_v1:
            pytest.skip("Skipping test on vllm V1")
        monkeypatch.setenv('VLLM_USE_V1', '1')
    else:
        monkeypatch.setenv('VLLM_USE_V1', '0')

    yield


@pytest.fixture(autouse=True)
def init_test_http_connection():
    # pytest_asyncio may use a different event loop per test
    # so we need to make sure the async client is created anew
    global_http_connection.reuse_client = False


@pytest.fixture
def dist_init():
    temp_file = tempfile.mkstemp()[1]
    init_distributed_environment(
        world_size=1,
        rank=0,
        distributed_init_method=f"file://{temp_file}",
        local_rank=0,
        backend="nccl",
    )
    initialize_model_parallel(1, 1)
    yield
    cleanup_dist_env_and_memory()


@pytest.fixture()
def should_do_global_cleanup_after_test(request) -> bool:
    """Allow subdirectories to skip global cleanup by overriding this fixture.
    This can provide a ~10x speedup for non-GPU unit tests since they don't need
    to initialize torch.
    """

    return not request.node.get_closest_marker("skip_global_cleanup")


@pytest.fixture(autouse=True)
def cleanup_fixture(should_do_global_cleanup_after_test: bool):
    yield
    if should_do_global_cleanup_after_test:
        cleanup_dist_env_and_memory()


@pytest.fixture(autouse=True)
def dynamo_reset():
    yield
    torch._dynamo.reset()


@pytest.fixture
<<<<<<< HEAD
def example_prompts() -> List[str]:
=======
def example_prompts() -> list[str]:
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
    prompts = []
    for filename in _TEST_PROMPTS:
        prompts += _read_prompts(filename)
    return prompts


@pytest.fixture
def example_system_message() -> str:
    with open(_SYS_MSG) as f:
        return f.read()


class DecoderPromptType(Enum):
    """For encoder/decoder models only."""
    CUSTOM = 1
    NONE = 2
    EMPTY_STR = 3


@pytest.fixture
def example_encoder_decoder_prompts(
<<<<<<< HEAD
) -> Dict[DecoderPromptType, List[ExplicitEncoderDecoderPrompt]]:
=======
) -> dict[DecoderPromptType, list[ExplicitEncoderDecoderPrompt]]:
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
    '''
    Returns an encoder prompt list and a decoder prompt list, wherein each pair
    of same-index entries in both lists corresponds to an (encoder prompt,
    decoder prompt) tuple.

    Returns:

    * Encoder prompt list
    * Decoder prompt list (reverse of encoder prompt list)
    '''

    encoder_prompts = []
    for filename in _TEST_PROMPTS:
        encoder_prompts += _read_prompts(filename)

    custom_decoder_prompts = encoder_prompts[::-1]
    empty_str_decoder_prompts = [""] * len(encoder_prompts)
    none_decoder_prompts = [None] * len(encoder_prompts)

    # NONE decoder prompt type
    return {
        DecoderPromptType.NONE:
        zip_enc_dec_prompts(encoder_prompts, none_decoder_prompts),
        DecoderPromptType.EMPTY_STR:
        zip_enc_dec_prompts(encoder_prompts, empty_str_decoder_prompts),
        DecoderPromptType.CUSTOM:
        zip_enc_dec_prompts(encoder_prompts, custom_decoder_prompts),
    }


@pytest.fixture
<<<<<<< HEAD
def example_long_prompts() -> List[str]:
=======
def example_long_prompts() -> list[str]:
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
    prompts = []
    for filename in _LONG_PROMPTS:
        prompts += _read_prompts(filename)
    return prompts


@pytest.fixture(scope="session")
<<<<<<< HEAD
def image_assets() -> _ImageAssets:
=======
def image_assets() -> ImageTestAssets:
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
    return IMAGE_ASSETS


@pytest.fixture(scope="session")
<<<<<<< HEAD
def video_assets() -> _VideoAssets:
    return VIDEO_ASSETS


=======
def video_assets() -> VideoTestAssets:
    return VIDEO_ASSETS


@pytest.fixture(scope="session")
def audio_assets() -> AudioTestAssets:
    return AUDIO_ASSETS


>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
_T = TypeVar("_T", nn.Module, torch.Tensor, BatchEncoding, BatchFeature, dict)
_R = TypeVar("_R")


class HfRunner:

<<<<<<< HEAD
    def wrap_device(self, x: _T, device: Optional[str] = None) -> _T:
        from vllm.platforms import current_platform
=======
    def get_default_device(self):
        from vllm.platforms import current_platform

        return ("cpu"
                if current_platform.is_cpu() else current_platform.device_type)

    def wrap_device(self, x: _T, device: Optional[str] = None) -> _T:
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
        if x is None or isinstance(x, (bool, )):
            return x

        if device is None:
<<<<<<< HEAD
            device = "cpu" if current_platform.is_cpu() else "cuda"
=======
            device = self.device
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea

        if isinstance(x, dict):
            return {k: self.wrap_device(v, device) for k, v in x.items()}

        if hasattr(x, "device") and x.device.type == device:
            return x

        return x.to(device)

    def __init__(
        self,
        model_name: str,
<<<<<<< HEAD
        dtype: str = "half",
        *,
        model_kwargs: Optional[Dict[str, Any]] = None,
        is_sentence_transformer: bool = False,
        is_cross_encoder: bool = False,
        skip_tokenizer_init: bool = False,
        auto_cls: Type[_BaseAutoModelClass] = AutoModelForCausalLM,
        postprocess_inputs: Callable[..., BatchEncoding] = identity,
    ) -> None:
        torch_dtype = STR_DTYPE_TO_TORCH_DTYPE[dtype]

        self.model_name = model_name

        if is_sentence_transformer:
            # Lazy init required for AMD CI
            from sentence_transformers import SentenceTransformer
            self.model = self.wrap_device(
                SentenceTransformer(
                    model_name,
                    device="cpu",
                    trust_remote_code=True,
                ).to(dtype=torch_dtype))
        elif is_cross_encoder:
            # Lazy init required for AMD CI
            from sentence_transformers import CrossEncoder
            self.model = CrossEncoder(model_name,
                                      device="cpu",
                                      trust_remote_code=True)
            self.model.model = self.wrap_device(self.model.model)\
                .to(dtype=torch_dtype)
        else:
            model_kwargs = model_kwargs if model_kwargs is not None else {}
            self.model = self.wrap_device(
                auto_cls.from_pretrained(
                    model_name,
                    torch_dtype=torch_dtype,
                    trust_remote_code=True,
                    **model_kwargs,
                ))
=======
        dtype: str = "auto",
        *,
        model_kwargs: Optional[dict[str, Any]] = None,
        is_sentence_transformer: bool = False,
        is_cross_encoder: bool = False,
        skip_tokenizer_init: bool = False,
        auto_cls: type[_BaseAutoModelClass] = AutoModelForCausalLM,
    ) -> None:
        self.model_name = model_name

        self.config = AutoConfig.from_pretrained(
            model_name,
            trust_remote_code=True,
        )
        self.device = self.get_default_device()
        self.dtype = torch_dtype = _get_and_verify_dtype(self.config, dtype)

        model_kwargs = model_kwargs if model_kwargs is not None else {}
        model_kwargs.setdefault("torch_dtype", torch_dtype)

        if is_sentence_transformer:
            # Lazy init required for AMD CI
            from sentence_transformers import SentenceTransformer

            self.model = SentenceTransformer(
                model_name,
                device=self.device,
                model_kwargs=model_kwargs,
                trust_remote_code=True,
            )
        elif is_cross_encoder:
            # Lazy init required for AMD CI
            from sentence_transformers import CrossEncoder

            self.model = CrossEncoder(
                model_name,
                device=self.device,
                automodel_args=model_kwargs,
                trust_remote_code=True,
            )
        else:
            model = auto_cls.from_pretrained(
                model_name,
                trust_remote_code=True,
                **model_kwargs,
            )

            # in case some unquantized custom models are not in same dtype
            if (getattr(model, "quantization_method", None) is None
                    and any(p.dtype != self.dtype
                            for p in model.parameters())):
                model = model.to(dtype=self.dtype)

            if (getattr(model, "quantization_method", None) != "bitsandbytes"
                    and len({p.device
                             for p in model.parameters()}) < 2):
                model = model.to(device=self.device)

            self.model = model
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea

        if not skip_tokenizer_init:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                torch_dtype=torch_dtype,
                trust_remote_code=True,
            )

        # don't put this import at the top level
        # it will call torch.cuda.device_count()
        from transformers import AutoProcessor  # noqa: F401
        self.processor = AutoProcessor.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
        )
        if skip_tokenizer_init:
            self.tokenizer = self.processor.tokenizer

<<<<<<< HEAD
        self.dtype = dtype
        self.postprocess_inputs = postprocess_inputs

    def get_inputs(
        self,
        prompts: List[str],
        images: Optional[PromptImageInput] = None,
        videos: Optional[PromptVideoInput] = None,
        audios: Optional[PromptAudioInput] = None,
    ) -> List[BatchEncoding]:
=======
    def get_inputs(
        self,
        prompts: list[str],
        images: Optional[PromptImageInput] = None,
        videos: Optional[PromptVideoInput] = None,
        audios: Optional[PromptAudioInput] = None,
    ) -> list[Union[BatchFeature, BatchEncoding]]:
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
        if images is not None:
            assert len(prompts) == len(images)

        if videos is not None:
            assert len(prompts) == len(videos)

        if audios is not None:
            assert len(prompts) == len(audios)

<<<<<<< HEAD
        all_inputs: List[BatchEncoding] = []
        for i, prompt in enumerate(prompts):
            processor_kwargs: Dict[str, Any] = {
=======
        all_inputs: list[Union[BatchFeature, BatchEncoding]] = []
        for i, prompt in enumerate(prompts):
            processor_kwargs: dict[str, Any] = {
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
                "text": prompt,
                "return_tensors": "pt",
            }
            if images is not None and (image := images[i]) is not None:
                processor_kwargs["images"] = image
            if videos is not None and (video := videos[i]) is not None:
                processor_kwargs["videos"] = video
<<<<<<< HEAD
            if audios is not None and (audio_tuple := audios[i]) is not None:
                audio, sr = audio_tuple
                processor_kwargs["audio"] = audio
                processor_kwargs["sampling_rate"] = sr

            inputs = self.processor(**processor_kwargs)
            inputs = self.postprocess_inputs(inputs, dtype=self.dtype)
=======
            if audios is not None and (audio_inputs := audios[i]) is not None:
                # HACK - not all processors take sampling_rate; we should
                # clean this up in the future.
                if len(audio_inputs) == 2:
                    audio, sr = audio_inputs
                    processor_kwargs["audio"] = audio
                    processor_kwargs["sampling_rate"] = sr
                else:
                    processor_kwargs["audio"] = audio_inputs

            inputs = self.processor(**processor_kwargs)
            if isinstance(inputs, BatchFeature):
                inputs = inputs.to(dtype=self.dtype)
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea

            all_inputs.append(inputs)

        return all_inputs

<<<<<<< HEAD
    def classify(self, prompts: List[str]) -> List[str]:
=======
    def get_prompt_embeddings(self, prompts: list[str]) -> list[torch.Tensor]:
        all_inputs = self.get_inputs(prompts)
        embeddings = []
        for inputs in all_inputs:
            input_ids = self.wrap_device(inputs)["input_ids"]
            embedding = self.model.get_input_embeddings()(input_ids).squeeze(0)
            embeddings.append(embedding)
        return embeddings

    def classify(self, prompts: list[str]) -> list[str]:
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
        # output is final logits
        all_inputs = self.get_inputs(prompts)
        outputs = []
        for inputs in all_inputs:
            output = self.model(**self.wrap_device(inputs))
            logits = output.logits.softmax(dim=-1)[0].tolist()
            outputs.append(logits)

        return outputs

    def generate(
        self,
<<<<<<< HEAD
        prompts: List[str],
=======
        prompts: list[str],
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
        images: Optional[PromptImageInput] = None,
        videos: Optional[PromptVideoInput] = None,
        audios: Optional[PromptAudioInput] = None,
        **kwargs: Any,
<<<<<<< HEAD
    ) -> List[Tuple[List[List[int]], List[str]]]:
=======
    ) -> list[tuple[list[list[int]], list[str]]]:
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
        all_inputs = self.get_inputs(prompts,
                                     images=images,
                                     videos=videos,
                                     audios=audios)

<<<<<<< HEAD
        outputs: List[Tuple[List[List[int]], List[str]]] = []
        for inputs in all_inputs:
            output_ids = self.model.generate(
                **self.wrap_device(inputs, device=self.model.device.type),
=======
        outputs: list[tuple[list[list[int]], list[str]]] = []
        for inputs in all_inputs:
            output_ids = self.model.generate(
                **self.wrap_device(inputs),
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
                use_cache=True,
                **kwargs,
            )
            output_str = self.processor.batch_decode(
                output_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            output_ids = output_ids.cpu().tolist()
            outputs.append((output_ids, output_str))
        return outputs

    def generate_greedy(
        self,
<<<<<<< HEAD
        prompts: List[str],
=======
        prompts: list[str],
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
        max_tokens: int,
        images: Optional[PromptImageInput] = None,
        videos: Optional[PromptVideoInput] = None,
        audios: Optional[PromptAudioInput] = None,
        **kwargs: Any,
<<<<<<< HEAD
    ) -> List[Tuple[List[int], str]]:
=======
    ) -> list[tuple[list[int], str]]:
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
        outputs = self.generate(prompts,
                                do_sample=False,
                                max_new_tokens=max_tokens,
                                images=images,
                                videos=videos,
                                audios=audios,
                                **kwargs)

        return [(output_ids[0], output_str[0])
                for output_ids, output_str in outputs]

    def generate_beam_search(
        self,
<<<<<<< HEAD
        prompts: List[str],
        beam_width: int,
        max_tokens: int,
    ) -> List[Tuple[List[List[int]], List[str]]]:
=======
        prompts: list[str],
        beam_width: int,
        max_tokens: int,
        images: Optional[PromptImageInput] = None,
        videos: Optional[PromptVideoInput] = None,
        audios: Optional[PromptAudioInput] = None,
    ) -> list[tuple[list[list[int]], list[str]]]:
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
        outputs = self.generate(prompts,
                                do_sample=False,
                                max_new_tokens=max_tokens,
                                num_beams=beam_width,
<<<<<<< HEAD
                                num_return_sequences=beam_width)
=======
                                num_return_sequences=beam_width,
                                images=images,
                                videos=videos,
                                audios=audios)

>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
        for i in range(len(outputs)):
            output_ids, output_str = outputs[i]
            for j in range(len(output_ids)):
                output_ids[j] = [
                    x for x in output_ids[j]
                    if x != self.tokenizer.pad_token_id
                ]
            outputs[i] = (output_ids, output_str)
        return outputs

    def generate_greedy_logprobs(
        self,
<<<<<<< HEAD
        prompts: List[str],
=======
        prompts: list[str],
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
        max_tokens: int,
        images: Optional[PromptImageInput] = None,
        videos: Optional[PromptVideoInput] = None,
        audios: Optional[PromptAudioInput] = None,
        **kwargs: Any,
<<<<<<< HEAD
    ) -> List[List[torch.Tensor]]:
=======
    ) -> list[list[torch.Tensor]]:
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
        all_inputs = self.get_inputs(prompts,
                                     images=images,
                                     videos=videos,
                                     audios=audios)

<<<<<<< HEAD
        all_logprobs: List[List[torch.Tensor]] = []
        for inputs in all_inputs:
            output = self.model.generate(
                **self.wrap_device(inputs, device=self.model.device.type),
=======
        all_logprobs: list[list[torch.Tensor]] = []
        for inputs in all_inputs:
            output = self.model.generate(
                **self.wrap_device(inputs),
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
                use_cache=True,
                do_sample=False,
                max_new_tokens=max_tokens,
                output_hidden_states=True,
                return_dict_in_generate=True,
                **kwargs,
            )
            seq_logprobs = self._hidden_states_to_seq_logprobs(
                output.hidden_states)
            all_logprobs.append(seq_logprobs)
        return all_logprobs

    def _hidden_states_to_seq_logprobs(
        self,
<<<<<<< HEAD
        hidden_states: Tuple[Tuple[torch.Tensor, ...], ...],
    ) -> List[torch.Tensor]:
        output_embeddings = self.model.get_output_embeddings()

        seq_logprobs: List[torch.Tensor] = []
        for _, hidden_state in enumerate(hidden_states):
            last_hidden_states = hidden_state[-1][0]
            logits = torch.matmul(
                last_hidden_states.to(output_embeddings.weight.device),
=======
        hidden_states: tuple[tuple[torch.Tensor, ...], ...],
    ) -> list[torch.Tensor]:
        output_embeddings = self.model.get_output_embeddings()

        seq_logprobs: list[torch.Tensor] = []
        for _, hidden_state in enumerate(hidden_states):
            last_hidden_states = hidden_state[-1][0]
            logits = torch.matmul(
                last_hidden_states.to(
                    device=output_embeddings.weight.device,
                    dtype=output_embeddings.weight.dtype,
                ),
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
                output_embeddings.weight.t(),
            )
            if getattr(output_embeddings, "bias", None) is not None:
                logits += output_embeddings.bias.unsqueeze(0)
            logprobs = F.log_softmax(logits, dim=-1, dtype=torch.float32)
            seq_logprobs.append(logprobs)

        return seq_logprobs

    def _hidden_states_to_logprobs(
        self,
<<<<<<< HEAD
        hidden_states: Tuple[Tuple[torch.Tensor, ...], ...],
        num_logprobs: int,
    ) -> Tuple[List[Dict[int, float]], int]:
=======
        hidden_states: tuple[tuple[torch.Tensor, ...], ...],
        num_logprobs: int,
    ) -> tuple[list[dict[int, float]], int]:
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
        seq_logprobs = self._hidden_states_to_seq_logprobs(hidden_states)
        output_len = len(hidden_states)

        # convert to dict
<<<<<<< HEAD
        seq_logprobs_lst: List[Dict[int, float]] = []
=======
        seq_logprobs_lst: list[dict[int, float]] = []
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
        for tok_idx, tok_logprobs in enumerate(seq_logprobs):
            # drop prompt logprobs
            if tok_idx == 0:
                tok_logprobs = tok_logprobs[-1, :].reshape(1, -1)
            topk = tok_logprobs.topk(num_logprobs)

            tok_logprobs_dct = {}
            for token_id, logprob in zip(topk.indices[0], topk.values[0]):
                tok_logprobs_dct[token_id.item()] = logprob.item()

            seq_logprobs_lst.append(tok_logprobs_dct)

        return (
            seq_logprobs_lst,
            output_len,
        )

    def generate_greedy_logprobs_limit(
        self,
<<<<<<< HEAD
        prompts: List[str],
=======
        prompts: list[str],
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
        max_tokens: int,
        num_logprobs: int,
        images: Optional[PromptImageInput] = None,
        audios: Optional[PromptAudioInput] = None,
        videos: Optional[PromptVideoInput] = None,
        **kwargs: Any,
<<<<<<< HEAD
    ) -> List[TokensTextLogprobs]:
=======
    ) -> list[TokensTextLogprobs]:
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
        all_inputs = self.get_inputs(prompts,
                                     images=images,
                                     videos=videos,
                                     audios=audios)

<<<<<<< HEAD
        all_logprobs: List[List[Dict[int, float]]] = []
        all_output_ids: List[List[int]] = []
        all_output_strs: List[str] = []

        for inputs in all_inputs:
            output = self.model.generate(
                **self.wrap_device(inputs, device=self.model.device.type),
=======
        all_logprobs: list[list[dict[int, float]]] = []
        all_output_ids: list[list[int]] = []
        all_output_strs: list[str] = []

        for inputs in all_inputs:
            output = self.model.generate(
                **self.wrap_device(inputs),
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
                use_cache=True,
                do_sample=False,
                max_new_tokens=max_tokens,
                output_hidden_states=True,
                return_dict_in_generate=True,
                **kwargs,
            )

            (
                seq_logprobs_lst,
                output_len,
            ) = self._hidden_states_to_logprobs(output.hidden_states,
                                                num_logprobs)

            all_logprobs.append(seq_logprobs_lst)
            seq_ids = output.sequences[0]
            output_len = len(seq_logprobs_lst)
            output_ids = seq_ids[-output_len:]
            all_output_ids.append(output_ids.tolist())
            all_output_strs.append(self.tokenizer.decode(output_ids))

        outputs = zip(all_output_ids, all_output_strs, all_logprobs)
        return [(output_ids, output_str, output_logprobs)
                for output_ids, output_str, output_logprobs in outputs]

    def generate_encoder_decoder_greedy_logprobs_limit(
        self,
<<<<<<< HEAD
        encoder_decoder_prompts: List[ExplicitEncoderDecoderPrompt[str, str]],
=======
        encoder_decoder_prompts: list[ExplicitEncoderDecoderPrompt[str, str]],
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
        max_tokens: int,
        num_logprobs: int,
        images: Optional[PromptImageInput] = None,
        **kwargs: Any,
<<<<<<< HEAD
    ) -> List[TokensTextLogprobs]:
=======
    ) -> list[TokensTextLogprobs]:
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
        '''
        Greedy logprobs generation for vLLM encoder/decoder models
        '''

<<<<<<< HEAD
        all_logprobs: List[List[Dict[int, float]]] = []
        all_output_ids: List[List[int]] = []
        all_output_strs: List[str] = []

        for i, (encoder_prompt, decoder_prompt) in enumerate(
                to_enc_dec_tuple_list(encoder_decoder_prompts)):
            processor_kwargs: Dict[str, Any] = {
=======
        all_logprobs: list[list[dict[int, float]]] = []
        all_output_ids: list[list[int]] = []
        all_output_strs: list[str] = []

        for i, (encoder_prompt, decoder_prompt) in enumerate(
                to_enc_dec_tuple_list(encoder_decoder_prompts)):
            processor_kwargs: dict[str, Any] = {
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
                "text": encoder_prompt,
                "return_tensors": "pt",
            }
            if images is not None and images[i] is not None:
                processor_kwargs["images"] = images[i]

<<<<<<< HEAD
            encoder_input_ids = self.wrap_device(
                self.processor(**processor_kwargs).input_ids,
                device=self.model.device.type,
            )
=======
            encoder_inputs = self.processor(**processor_kwargs)
            encoder_inputs = self.wrap_device(encoder_inputs)
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea

            if decoder_prompt is None:
                decoder_input_ids = None
            else:
<<<<<<< HEAD
                decoder_input_ids = self.wrap_device(
                    self.tokenizer(decoder_prompt,
                                   return_tensors="pt").input_ids,
                    device=self.model.device.type,
                )

            output = self.model.generate(
                encoder_input_ids,
=======
                decoder_inputs = self.tokenizer(decoder_prompt,
                                                return_tensors="pt")
                decoder_input_ids = self.wrap_device(decoder_inputs.input_ids)

            output = self.model.generate(
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
                decoder_input_ids=decoder_input_ids,
                use_cache=True,
                do_sample=False,
                max_new_tokens=max_tokens,
                output_hidden_states=True,
                return_dict_in_generate=True,
<<<<<<< HEAD
=======
                **encoder_inputs,
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
                **kwargs,
            )

            (
                seq_logprobs_lst,
                output_len,
            ) = self._hidden_states_to_logprobs(output.decoder_hidden_states,
                                                num_logprobs)

            all_logprobs.append(seq_logprobs_lst)
            seq_ids = output.sequences[0]
            output_ids = seq_ids[-output_len:]
            all_output_ids.append(output_ids.tolist())
            all_output_strs.append(self.tokenizer.decode(output_ids))

        outputs = zip(all_output_ids, all_output_strs, all_logprobs)
        return [(output_ids, output_str, output_logprobs)
                for output_ids, output_str, output_logprobs in outputs]

<<<<<<< HEAD
    def encode(self, prompts: List[str]) -> List[List[torch.Tensor]]:
        return self.model.encode(prompts)

    def predict(self, prompts: List[List[str]]) -> torch.Tensor:
=======
    def encode(self, prompts: list[str], *args,
               **kwargs) -> list[list[torch.Tensor]]:
        return self.model.encode(prompts, *args, **kwargs)

    def predict(self, prompts: list[list[str]]) -> torch.Tensor:
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
        return self.model.predict(prompts, convert_to_tensor=True)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        del self.model
        cleanup_dist_env_and_memory()


@pytest.fixture(scope="session")
def hf_runner():
    return HfRunner


class VllmRunner:
<<<<<<< HEAD
=======
    """
    The default value of some arguments have been modified from
    {class}`~vllm.LLM` as follows:

    - `trust_remote_code`: Set to `True` instead of `False` for convenience.
    - `seed`: Set to `0` instead of `None` for test reproducibility.
    - `max_model_len`: Set to `1024` instead of `None` to reduce memory usage.
    - `block_size`: Set to `16` instead of `None` to reduce memory usage.
    - `enable_chunked_prefill`: Set to `False` instead of `None` for
      test reproducibility.
    - `enforce_eager`: Set to `False` to test CUDA graph.
    """
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea

    def __init__(
        self,
        model_name: str,
        task: TaskOption = "auto",
        tokenizer_name: Optional[str] = None,
        tokenizer_mode: str = "auto",
<<<<<<< HEAD
        # Use smaller max model length, otherwise bigger model cannot run due
        # to kv cache size limit.
        max_model_len: int = 1024,
        dtype: str = "half",
        disable_log_stats: bool = True,
        tensor_parallel_size: int = 1,
        block_size: int = 16,
        enable_chunked_prefill: bool = False,
        swap_space: int = 4,
        enforce_eager: Optional[bool] = False,
        load_format: Optional[LoadFormat] = None,
        **kwargs,
    ) -> None:
        if model_name in MODELS_ON_S3 and not load_format:
            model_name = (f"{MODEL_WEIGHTS_S3_BUCKET}/{model_name}")
            load_format = LoadFormat.RUNAI_STREAMER
        if not load_format:
            load_format = LoadFormat.AUTO
=======
        trust_remote_code: bool = True,
        seed: Optional[int] = 0,
        max_model_len: int = 1024,
        dtype: str = "auto",
        disable_log_stats: bool = True,
        tensor_parallel_size: int = 1,
        block_size: int = 16,
        enable_chunked_prefill: Optional[bool] = False,
        swap_space: int = 4,
        enforce_eager: Optional[bool] = False,
        **kwargs,
    ) -> None:
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
        self.model = LLM(
            model=model_name,
            task=task,
            tokenizer=tokenizer_name,
            tokenizer_mode=tokenizer_mode,
<<<<<<< HEAD
            trust_remote_code=True,
            dtype=dtype,
=======
            trust_remote_code=trust_remote_code,
            dtype=dtype,
            seed=seed,
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
            swap_space=swap_space,
            enforce_eager=enforce_eager,
            disable_log_stats=disable_log_stats,
            tensor_parallel_size=tensor_parallel_size,
            max_model_len=max_model_len,
            block_size=block_size,
            enable_chunked_prefill=enable_chunked_prefill,
<<<<<<< HEAD
            load_format=load_format,
=======
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
            **kwargs,
        )

    def get_inputs(
        self,
<<<<<<< HEAD
        prompts: List[str],
        images: Optional[PromptImageInput] = None,
        videos: Optional[PromptVideoInput] = None,
        audios: Optional[PromptAudioInput] = None,
    ) -> List[TextPrompt]:
        if images is not None:
            assert len(prompts) == len(images)

        if videos is not None:
            assert len(prompts) == len(videos)

        if audios is not None:
            assert len(prompts) == len(audios)

        inputs = [TextPrompt(prompt=prompt) for prompt in prompts]
        if images is not None:
            for i, image in enumerate(images):
                if image is not None:
                    inputs[i]["multi_modal_data"] = {"image": image}

        if videos is not None:
            for i, video in enumerate(videos):
                if video is not None:
                    inputs[i]["multi_modal_data"] = {"video": video}

        if audios is not None:
            for i, audio in enumerate(audios):
                if audio is not None:
                    inputs[i]["multi_modal_data"] = {"audio": audio}
=======
        prompts: Union[list[str], list[torch.Tensor]],
        images: Optional[PromptImageInput] = None,
        videos: Optional[PromptVideoInput] = None,
        audios: Optional[PromptAudioInput] = None,
    ) -> list[TextPrompt]:

        if any(x is not None and len(x) != len(prompts)
               for x in [images, videos, audios]):
            raise ValueError(
                "All non-None multimodal inputs must have the same length as "
                "prompts")

        inputs = []
        for i, prompt in enumerate(prompts):
            multi_modal_data = {}
            if images is not None and (image := images[i]) is not None:
                multi_modal_data["image"] = image
            if videos is not None and (video := videos[i]) is not None:
                multi_modal_data["video"] = video
            if audios is not None and (audio := audios[i]) is not None:
                multi_modal_data["audio"] = audio

            text_prompt_kwargs = {
                ("prompt" if isinstance(prompt, str) else "prompt_embeds"):
                prompt,
                "multi_modal_data": multi_modal_data or None
            }
            inputs.append(TextPrompt(**text_prompt_kwargs))
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea

        return inputs

    def generate(
        self,
<<<<<<< HEAD
        prompts: List[str],
=======
        prompts: Union[list[str], list[torch.Tensor]],
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
        sampling_params: SamplingParams,
        images: Optional[PromptImageInput] = None,
        videos: Optional[PromptVideoInput] = None,
        audios: Optional[PromptAudioInput] = None,
        **kwargs: Any,
<<<<<<< HEAD
    ) -> List[Tuple[List[List[int]], List[str]]]:
=======
    ) -> list[tuple[list[list[int]], list[str]]]:
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
        inputs = self.get_inputs(prompts,
                                 images=images,
                                 videos=videos,
                                 audios=audios)

        req_outputs = self.model.generate(inputs,
                                          sampling_params=sampling_params,
                                          **kwargs)

<<<<<<< HEAD
        outputs: List[Tuple[List[List[int]], List[str]]] = []
        for req_output in req_outputs:
            prompt_str = req_output.prompt
            prompt_ids = req_output.prompt_token_ids
            req_sample_output_ids: List[List[int]] = []
            req_sample_output_strs: List[str] = []
=======
        outputs: list[tuple[list[list[int]], list[str]]] = []
        for req_output in req_outputs:
            prompt_str = req_output.prompt
            prompt_ids = req_output.prompt_token_ids
            req_sample_output_ids: list[list[int]] = []
            req_sample_output_strs: list[str] = []
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
            for sample in req_output.outputs:
                output_str = sample.text
                output_ids = list(sample.token_ids)
                req_sample_output_ids.append(prompt_ids + output_ids)
<<<<<<< HEAD
                req_sample_output_strs.append(prompt_str + output_str)
=======
                req_sample_output_strs.append((prompt_str or "") + output_str)
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
            outputs.append((req_sample_output_ids, req_sample_output_strs))
        return outputs

    @staticmethod
    def _final_steps_generate_w_logprobs(
<<<<<<< HEAD
        req_outputs: List[RequestOutput],
    ) -> List[TokensTextLogprobsPromptLogprobs]:
        outputs: List[TokensTextLogprobsPromptLogprobs] = []
=======
        req_outputs: list[RequestOutput],
    ) -> list[TokensTextLogprobsPromptLogprobs]:
        outputs: list[TokensTextLogprobsPromptLogprobs] = []
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
        for req_output in req_outputs:
            assert len(req_output.outputs) > 0
            for sample in req_output.outputs:
                output_str = sample.text
                output_ids = list(sample.token_ids)
                output_logprobs = sample.logprobs
            outputs.append((output_ids, output_str, output_logprobs,
                            req_output.prompt_logprobs))
        return outputs

    def generate_w_logprobs(
        self,
<<<<<<< HEAD
        prompts: List[str],
=======
        prompts: list[str],
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
        sampling_params: SamplingParams,
        images: Optional[PromptImageInput] = None,
        audios: Optional[PromptAudioInput] = None,
        videos: Optional[PromptVideoInput] = None,
        **kwargs: Any,
<<<<<<< HEAD
    ) -> Union[List[TokensTextLogprobs],
               List[TokensTextLogprobsPromptLogprobs]]:
=======
    ) -> Union[list[TokensTextLogprobs],
               list[TokensTextLogprobsPromptLogprobs]]:
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
        inputs = self.get_inputs(prompts,
                                 images=images,
                                 videos=videos,
                                 audios=audios)

        req_outputs = self.model.generate(inputs,
                                          sampling_params=sampling_params,
                                          **kwargs)

        toks_str_logsprobs_prompt_logprobs = (
            self._final_steps_generate_w_logprobs(req_outputs))
        # Omit prompt logprobs if not required by sampling params
        return ([x[0:-1] for x in toks_str_logsprobs_prompt_logprobs]
                if sampling_params.prompt_logprobs is None else
                toks_str_logsprobs_prompt_logprobs)

    def generate_encoder_decoder_w_logprobs(
        self,
<<<<<<< HEAD
        encoder_decoder_prompts: List[ExplicitEncoderDecoderPrompt[str, str]],
        sampling_params: SamplingParams,
    ) -> Union[List[TokensTextLogprobs],
               List[TokensTextLogprobsPromptLogprobs]]:
=======
        encoder_decoder_prompts: list[ExplicitEncoderDecoderPrompt[str, str]],
        sampling_params: SamplingParams,
    ) -> Union[list[TokensTextLogprobs],
               list[TokensTextLogprobsPromptLogprobs]]:
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
        '''
        Logprobs generation for vLLM encoder/decoder models
        '''

        assert sampling_params.logprobs is not None
        req_outputs = self.model.generate(encoder_decoder_prompts,
                                          sampling_params=sampling_params)
        toks_str_logsprobs_prompt_logprobs = (
            self._final_steps_generate_w_logprobs(req_outputs))
        # Omit prompt logprobs if not required by sampling params
        return ([x[0:-1] for x in toks_str_logsprobs_prompt_logprobs]
                if sampling_params.prompt_logprobs is None else
                toks_str_logsprobs_prompt_logprobs)

    def generate_greedy(
        self,
<<<<<<< HEAD
        prompts: List[str],
=======
        prompts: Union[list[str], list[torch.Tensor]],
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
        max_tokens: int,
        images: Optional[PromptImageInput] = None,
        videos: Optional[PromptVideoInput] = None,
        audios: Optional[PromptAudioInput] = None,
        **kwargs: Any,
<<<<<<< HEAD
    ) -> List[Tuple[List[int], str]]:
=======
    ) -> list[tuple[list[int], str]]:
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
        greedy_params = SamplingParams(temperature=0.0, max_tokens=max_tokens)
        outputs = self.generate(prompts,
                                greedy_params,
                                images=images,
                                videos=videos,
                                audios=audios,
                                **kwargs)
        return [(output_ids[0], output_str[0])
                for output_ids, output_str in outputs]

    def generate_greedy_logprobs(
        self,
<<<<<<< HEAD
        prompts: List[str],
=======
        prompts: list[str],
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
        max_tokens: int,
        num_logprobs: int,
        num_prompt_logprobs: Optional[int] = None,
        images: Optional[PromptImageInput] = None,
        audios: Optional[PromptAudioInput] = None,
        videos: Optional[PromptVideoInput] = None,
<<<<<<< HEAD
        stop_token_ids: Optional[List[int]] = None,
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Union[List[TokensTextLogprobs],
               List[TokensTextLogprobsPromptLogprobs]]:
=======
        stop_token_ids: Optional[list[int]] = None,
        stop: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> Union[list[TokensTextLogprobs],
               list[TokensTextLogprobsPromptLogprobs]]:
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
        greedy_logprobs_params = SamplingParams(
            temperature=0.0,
            max_tokens=max_tokens,
            logprobs=num_logprobs,
            prompt_logprobs=num_prompt_logprobs,
            stop_token_ids=stop_token_ids,
            stop=stop)

        return self.generate_w_logprobs(prompts,
                                        greedy_logprobs_params,
                                        images=images,
                                        audios=audios,
                                        videos=videos,
                                        **kwargs)

    def generate_encoder_decoder_greedy_logprobs(
        self,
<<<<<<< HEAD
        encoder_decoder_prompts: List[ExplicitEncoderDecoderPrompt[str, str]],
        max_tokens: int,
        num_logprobs: int,
        num_prompt_logprobs: Optional[int] = None,
    ) -> Union[List[TokensTextLogprobs],
               List[TokensTextLogprobsPromptLogprobs]]:
=======
        encoder_decoder_prompts: list[ExplicitEncoderDecoderPrompt[str, str]],
        max_tokens: int,
        num_logprobs: int,
        num_prompt_logprobs: Optional[int] = None,
        skip_special_tokens: bool = True,
    ) -> Union[list[TokensTextLogprobs],
               list[TokensTextLogprobsPromptLogprobs]]:
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
        greedy_logprobs_params = SamplingParams(
            temperature=0.0,
            max_tokens=max_tokens,
            logprobs=num_logprobs,
            prompt_logprobs=(num_prompt_logprobs),
<<<<<<< HEAD
=======
            skip_special_tokens=skip_special_tokens,
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
        )
        '''
        Greedy logprobs generation for vLLM encoder/decoder models
        '''

        return self.generate_encoder_decoder_w_logprobs(
            encoder_decoder_prompts, greedy_logprobs_params)

    def generate_beam_search(
        self,
<<<<<<< HEAD
        prompts: Union[List[str], List[List[int]]],
        beam_width: int,
        max_tokens: int,
    ) -> List[Tuple[List[List[int]], List[str]]]:
        if is_list_of(prompts, str, check="all"):
            prompts = [TextPrompt(prompt=prompt) for prompt in prompts]
        else:
            prompts = [
                TokensPrompt(prompt_token_ids=tokens) for tokens in prompts
            ]
        outputs = self.model.beam_search(
            prompts,
=======
        prompts: list[str],
        beam_width: int,
        max_tokens: int,
        images: Optional[PromptImageInput] = None,
        videos: Optional[PromptVideoInput] = None,
        audios: Optional[PromptAudioInput] = None,
    ) -> list[tuple[list[list[int]], list[str]]]:
        inputs = self.get_inputs(prompts,
                                 images=images,
                                 videos=videos,
                                 audios=audios)

        outputs = self.model.beam_search(
            inputs,
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
            BeamSearchParams(beam_width=beam_width, max_tokens=max_tokens))
        returned_outputs = []
        for output in outputs:
            token_ids = [x.tokens for x in output.sequences]
            texts = [x.text for x in output.sequences]
            returned_outputs.append((token_ids, texts))
        return returned_outputs

<<<<<<< HEAD
    def classify(self, prompts: List[str]) -> List[List[float]]:
        req_outputs = self.model.classify(prompts)
        return [req_output.outputs.probs for req_output in req_outputs]

    def encode(
        self,
        prompts: List[str],
        images: Optional[PromptImageInput] = None,
        videos: Optional[PromptVideoInput] = None,
        audios: Optional[PromptAudioInput] = None,
    ) -> List[List[float]]:
=======
    def classify(self, prompts: list[str]) -> list[list[float]]:
        req_outputs = self.model.classify(prompts)
        return [req_output.outputs.probs for req_output in req_outputs]

    def encode(self,
               prompts: list[str],
               images: Optional[PromptImageInput] = None,
               videos: Optional[PromptVideoInput] = None,
               audios: Optional[PromptAudioInput] = None,
               *args,
               **kwargs) -> list[list[float]]:
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
        inputs = self.get_inputs(prompts,
                                 images=images,
                                 videos=videos,
                                 audios=audios)

<<<<<<< HEAD
        req_outputs = self.model.embed(inputs)
=======
        req_outputs = self.model.embed(inputs, *args, **kwargs)
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
        return [req_output.outputs.embedding for req_output in req_outputs]

    def score(
        self,
<<<<<<< HEAD
        text_1: Union[str, List[str]],
        text_2: Union[str, List[str]],
    ) -> List[float]:
=======
        text_1: Union[str, list[str]],
        text_2: Union[str, list[str]],
    ) -> list[float]:
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
        req_outputs = self.model.score(text_1, text_2)
        return [req_output.outputs.score for req_output in req_outputs]

    def apply_model(self, func: Callable[[nn.Module], _R]) -> list[_R]:
        executor = self.model.llm_engine.model_executor
        return executor.apply_model(func)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        del self.model
        cleanup_dist_env_and_memory()


@pytest.fixture(scope="session")
def vllm_runner():
    return VllmRunner


<<<<<<< HEAD
def get_tokenizer_pool_config(tokenizer_group_type):
    if tokenizer_group_type is None:
        return None
    if tokenizer_group_type == "ray":
        return TokenizerPoolConfig(pool_size=1,
                                   pool_type="ray",
                                   extra_config={})
    if isinstance(tokenizer_group_type, type):
        return TokenizerPoolConfig(pool_size=1,
                                   pool_type=tokenizer_group_type,
                                   extra_config={})
    raise ValueError(f"Unknown tokenizer_group_type: {tokenizer_group_type}")


=======
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
@pytest.fixture()
def temporary_enable_log_propagate():
    import logging
    logger = logging.getLogger("vllm")
    logger.propagate = True
    yield
    logger.propagate = False


@pytest.fixture()
def caplog_vllm(temporary_enable_log_propagate, caplog):
    # To capture vllm log, we should enable propagate=True temporarily
    # because caplog depends on logs propagated to the root logger.
    yield caplog


@pytest.fixture(scope="session")
def num_gpus_available():
    """Get number of GPUs without initializing the CUDA context
    in current process."""

    return cuda_device_count_stateless()


temp_dir = tempfile.gettempdir()
_dummy_opt_path = os.path.join(temp_dir, "dummy_opt")
_dummy_llava_path = os.path.join(temp_dir, "dummy_llava")
_dummy_gemma2_embedding_path = os.path.join(temp_dir, "dummy_gemma2_embedding")


@pytest.fixture
def dummy_opt_path():
    json_path = os.path.join(_dummy_opt_path, "config.json")
    if not os.path.exists(_dummy_opt_path):
        snapshot_download(repo_id="facebook/opt-125m",
                          local_dir=_dummy_opt_path,
                          ignore_patterns=[
                              "*.bin", "*.bin.index.json", "*.pt", "*.h5",
                              "*.msgpack"
                          ])
        assert os.path.exists(json_path)
        with open(json_path) as f:
            config = json.load(f)
        config["architectures"] = ["MyOPTForCausalLM"]
        with open(json_path, "w") as f:
            json.dump(config, f)
    return _dummy_opt_path


@pytest.fixture
def dummy_llava_path():
    json_path = os.path.join(_dummy_llava_path, "config.json")
    if not os.path.exists(_dummy_llava_path):
        snapshot_download(repo_id="llava-hf/llava-1.5-7b-hf",
                          local_dir=_dummy_llava_path,
                          ignore_patterns=[
                              "*.bin", "*.bin.index.json", "*.pt", "*.h5",
                              "*.msgpack"
                          ])
        assert os.path.exists(json_path)
        with open(json_path) as f:
            config = json.load(f)
        config["architectures"] = ["MyLlava"]
        with open(json_path, "w") as f:
            json.dump(config, f)
    return _dummy_llava_path


@pytest.fixture
def dummy_gemma2_embedding_path():
    json_path = os.path.join(_dummy_gemma2_embedding_path, "config.json")
    if not os.path.exists(_dummy_gemma2_embedding_path):
        snapshot_download(repo_id="BAAI/bge-multilingual-gemma2",
                          local_dir=_dummy_gemma2_embedding_path,
                          ignore_patterns=[
                              "*.bin", "*.bin.index.json", "*.pt", "*.h5",
                              "*.msgpack"
                          ])
        assert os.path.exists(json_path)
        with open(json_path) as f:
            config = json.load(f)
        config["architectures"] = ["MyGemma2Embedding"]
        with open(json_path, "w") as f:
            json.dump(config, f)
    return _dummy_gemma2_embedding_path


# Add the flag `--optional` to allow run tests
# that are marked with @pytest.mark.optional
def pytest_addoption(parser):
    parser.addoption("--optional",
                     action="store_true",
                     default=False,
                     help="run optional test")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--optional"):
        # --optional given in cli: do not skip optional tests
        return
    skip_optional = pytest.mark.skip(reason="need --optional option to run")
    for item in items:
        if "optional" in item.keywords:
            item.add_marker(skip_optional)
<<<<<<< HEAD
=======


@pytest.fixture(scope="session")
def cli_config_file():
    """Return the path to the CLI config file."""
    return os.path.join(_TEST_DIR, "config", "test_config.yaml")


@pytest.fixture(scope="session")
def cli_config_file_with_model():
    """Return the path to the CLI config file with model."""
    return os.path.join(_TEST_DIR, "config", "test_config_with_model.yaml")
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
