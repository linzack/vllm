# SPDX-License-Identifier: Apache-2.0

# Adapted from
# https://github.com/huggingface/transformers/blob/v4.28.0/src/transformers/models/llama/modeling_llama.py
# Copyright 2023 The vLLM team.
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Inference-only MiniCPM-V model compatible with HuggingFace weights."""
import math
<<<<<<< HEAD
import re
from collections import Counter
from functools import cached_property, partial
from typing import (Any, Callable, Dict, Iterable, List, Literal, Mapping,
                    Optional, Set, Tuple, TypedDict, Union)
=======
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from functools import partial
from typing import Any, Callable, Literal, Optional, TypedDict, Union
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea

import numpy as np
import torch
import torch.types
<<<<<<< HEAD
from PIL import Image
=======
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
from torch import nn
from transformers import BatchFeature, PretrainedConfig
from typing_extensions import TypeVar

<<<<<<< HEAD
from vllm.attention import AttentionMetadata
=======
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
from vllm.config import VllmConfig
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.resampler import (BaseResampler, Resampler2,
                                                  get_2d_sincos_pos_embed)
<<<<<<< HEAD
from vllm.model_executor.layers.sampler import SamplerOutput, get_sampler
=======
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
from vllm.model_executor.model_loader.utils import set_default_torch_dtype
from vllm.model_executor.models.llama import LlamaForCausalLM
from vllm.model_executor.models.minicpm import MiniCPMForCausalLM
from vllm.model_executor.models.module_mapping import MultiModelKeys
from vllm.model_executor.models.qwen2 import Qwen2ForCausalLM
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.multimodal import MULTIMODAL_REGISTRY, MultiModalKwargs
from vllm.multimodal.inputs import (MultiModalDataDict, MultiModalFieldConfig,
<<<<<<< HEAD
                                    MultiModalInputs, PlaceholderRange)
from vllm.multimodal.parse import (DictEmbeddingItems, ImageItem, ImageSize,
                                   ModalityData, ModalityDataItems,
                                   MultiModalDataItems, MultiModalDataParser,
                                   VideoItem)
from vllm.multimodal.processing import (BaseMultiModalProcessor,
                                        BaseProcessingInfo, PromptReplacement)
from vllm.multimodal.profiling import BaseDummyInputsBuilder, ProcessorInputs
from vllm.sequence import IntermediateTensors

from .idefics2_vision_model import Idefics2VisionTransformer
from .interfaces import SupportsLoRA, SupportsMultiModal, SupportsPP
from .utils import AutoWeightsLoader, maybe_prefix

CPU_DEVICE = torch.device("cpu")

RawImageType = Union[Image.Image, torch.Tensor]
=======
                                    NestedTensors)
from vllm.multimodal.parse import (DictEmbeddingItems, ImageItem,
                                   ImageProcessorItems, ImageSize,
                                   ModalityData, ModalityDataItems,
                                   MultiModalDataItems, MultiModalDataParser,
                                   VideoItem, VideoProcessorItems)
from vllm.multimodal.processing import (BaseMultiModalProcessor,
                                        BaseProcessingInfo, PromptReplacement,
                                        PromptUpdate, PromptUpdateDetails)
from vllm.multimodal.profiling import BaseDummyInputsBuilder
from vllm.platforms import current_platform
from vllm.sequence import IntermediateTensors
from vllm.utils import flatten_2d_lists

from .idefics2_vision_model import Idefics2VisionTransformer
from .interfaces import (MultiModalEmbeddings, SupportsLoRA,
                         SupportsMultiModal, SupportsPP)
from .utils import (AutoWeightsLoader, flatten_bn, maybe_prefix,
                    merge_multimodal_embeddings)

# For profile run
_MAX_FRAMES_PER_VIDEO = 16
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea


class MiniCPMVImagePixelInputs(TypedDict):
    type: Literal["pixel_values"]
<<<<<<< HEAD
    data: List[torch.Tensor]
=======
    pixel_values: list[torch.Tensor]
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
    """
    Shape: `(batch_size * num_images * num_slices, num_channels, height, width)`

    Note that the image size may vary, so we pass it as a list
    instead of a batched tensor.
    """

<<<<<<< HEAD
    image_bounds: torch.Tensor
    """
    Shape: `(batch_size * num_images * num_slices, 2)`

    This should be in `(start, stop)` format.
    """

=======
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
    tgt_sizes: torch.Tensor
    """
    Shape: `(batch_size * num_images * num_slices, 2)`

    This should be in `(height, width)` format.
    """

<<<<<<< HEAD

class MiniCPMVImageEmbeddingInputs(TypedDict):
    type: Literal["image_embeds"]
    data: torch.Tensor
    """
    Shape: `(batch_size * num_images * num_slices, 
             image_feature_size, hidden_size)`
=======
    num_slices: torch.Tensor
    """Shape: `(batch_size * num_images)`"""


class MiniCPMVImageEmbeddingInputs(TypedDict):
    type: Literal["image_embeds"]
    image_embeds: Union[torch.Tensor, list[torch.Tensor]]
    """
    Shape: `(batch_size * num_images, num_slices, hidden_size)`
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea

    `hidden_size` must match the hidden size of language model backbone.
    instead of a batched tensor.
    """

<<<<<<< HEAD
    image_bounds: torch.Tensor
    """
    Shape: `(batch_size * num_images * num_slices, 2)`

    This should be in `(start, stop)` format.
    """

=======
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea

MiniCPMVImageInputs = Union[MiniCPMVImagePixelInputs,
                            MiniCPMVImageEmbeddingInputs]

DEFAULT_LN = partial(nn.LayerNorm, eps=1e-6)


class Resampler2_5(BaseResampler):

    def __init__(self,
                 num_queries: int,
                 embed_dim: int,
                 num_heads: int,
                 kv_dim: Optional[int] = None,
                 norm_layer: Callable[[int], nn.LayerNorm] = DEFAULT_LN,
<<<<<<< HEAD
                 max_size: Tuple[int, int] = (70, 70),
=======
                 max_size: tuple[int, int] = (70, 70),
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
                 quant_config: Optional[QuantizationConfig] = None,
                 prefix: str = "") -> None:
        super().__init__(num_queries,
                         embed_dim,
                         num_heads,
                         kv_dim,
                         norm_layer,
                         quant_config=quant_config,
                         prefix=prefix)

        self.max_size = max_size
        self._set_2d_pos_cache(self.max_size)

    def _set_2d_pos_cache(self,
<<<<<<< HEAD
                          max_size: Tuple[int, int],
=======
                          max_size: tuple[int, int],
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
                          device: torch.types.Device = "cpu") -> None:
        pos_embed_arr = get_2d_sincos_pos_embed(self.embed_dim,
                                                max_size,
                                                version=(2, 5))
        pos_embed = torch.from_numpy(pos_embed_arr).float().to(device)
        self.register_buffer("pos_embed", pos_embed, persistent=False)

    def _adjust_pos_cache(self, tgt_sizes: torch.Tensor,
                          device: torch.types.Device) -> None:
        max_h = tgt_sizes[:, 0].max().item()
        max_w = tgt_sizes[:, 1].max().item()
        assert isinstance(max_h, int) and isinstance(max_w, int)

        if max_h > self.max_size[0] or max_w > self.max_size[1]:
            self.max_size = (
                max(max_h, self.max_size[0]),
                max(max_w, self.max_size[1]),
            )
            self._set_2d_pos_cache(self.max_size, device)

    def forward(self, x: torch.Tensor,
                tgt_sizes: torch.Tensor) -> torch.Tensor:
        assert x.shape[0] == tgt_sizes.shape[0]
        bs = x.shape[0]

        device = x.device
        dtype = x.dtype

        patch_len = tgt_sizes[:, 0] * tgt_sizes[:, 1]

        self._adjust_pos_cache(tgt_sizes, device=device)

        max_patch_len = patch_len.max().item()
        assert isinstance(max_patch_len, int)

        key_padding_mask = torch.zeros((bs, max_patch_len),
                                       dtype=torch.bool,
                                       device=device)

        pos_embed = []
        for i in range(bs):
            tgt_h, tgt_w = tgt_sizes[i].tolist()
            pos_embed.append(self.pos_embed[:tgt_h, :tgt_w, :].reshape(
                (tgt_h * tgt_w, -1)).to(dtype))  # patches * D
            key_padding_mask[i, patch_len[i]:] = True
        pos_embed = torch.nn.utils.rnn.pad_sequence(pos_embed,
                                                    batch_first=True,
                                                    padding_value=0.0).permute(
                                                        1, 0,
                                                        2)  # BLD => L * B * D
        x, _ = self.kv_proj(x)  # B * L * D
        x = self.ln_kv(x).permute(1, 0, 2)  # L * B * D

        q = self.ln_q(self.query)  # Q * D

        out = self.attn(
            self._repeat(q, bs),  # Q * B * D
            x + pos_embed,  # L * B * D +  L * B * D
            x,
            key_padding_mask=key_padding_mask,
        )[0]
        #  out: Q * B * D
        x = out.permute(1, 0, 2)  # B * Q * D

        x = self.ln_post(x)
        x = x @ self.proj
        return x


<<<<<<< HEAD
def get_version_by_config(config: PretrainedConfig) -> Tuple[int, ...]:
=======
def get_version_by_config(config: PretrainedConfig) -> tuple[int, ...]:
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
    version_float = getattr(config, "version", None)

    # The old configs do not include version number
    # TODO: Remove this after the HF repos are updated
    if version_float is None:
        if config.hidden_size == 2304 and config.query_num == 64:
            return (2, 0)
        return (2, 5)
    version_str = str(version_float)
    return tuple(int(x) for x in version_str.split("."))


def _minicpmv_field_config(hf_inputs: Mapping[str, torch.Tensor]):
<<<<<<< HEAD
    image_num_slices = hf_inputs.get("image_num_slices", torch.empty(0))
    video_num_slices = hf_inputs.get("video_num_slices", torch.empty(0))

    return dict(
        pixel_values=MultiModalFieldConfig.flat_from_sizes(
            "image", image_num_slices),
        image_sizes=MultiModalFieldConfig.batched("image"),
        tgt_sizes=MultiModalFieldConfig.flat_from_sizes(
            "image", image_num_slices),
        image_num_slices=MultiModalFieldConfig.batched("image"),
        image_embeds=MultiModalFieldConfig.flat_from_sizes(
            "image", image_num_slices),
        video_pixel_values=MultiModalFieldConfig.flat_from_sizes(
            "video", video_num_slices),
        video_image_sizes=MultiModalFieldConfig.batched("video"),
        video_tgt_sizes=MultiModalFieldConfig.flat_from_sizes(
            "video", video_num_slices),
        video_embeds=MultiModalFieldConfig.flat_from_sizes(
            "video", video_num_slices),
        video_num_slices=MultiModalFieldConfig.batched("video"),
=======
    pixel_values = hf_inputs.get("pixel_values", torch.empty(0))
    num_images = len(pixel_values)

    video_pixel_values = hf_inputs.get("video_pixel_values", torch.empty(0))
    num_videos = len(video_pixel_values)

    return dict(
        pixel_values=MultiModalFieldConfig.batched("image"),
        image_sizes=MultiModalFieldConfig.batched("image"),
        tgt_sizes=MultiModalFieldConfig.batched("image"),
        image_embeds=MultiModalFieldConfig.batched("image"),
        video_pixel_values=MultiModalFieldConfig.batched("video"),
        video_image_sizes=MultiModalFieldConfig.batched("video"),
        video_tgt_sizes=MultiModalFieldConfig.batched("video"),
        video_embeds=MultiModalFieldConfig.batched("video"),
        image_token_id=MultiModalFieldConfig.shared("image", num_images),
        video_token_id=MultiModalFieldConfig.shared("video", num_videos),
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
    )


class MiniCPMVImageEmbeddingItems(DictEmbeddingItems):

    def __init__(
        self,
        data: Mapping[str, torch.Tensor],
        fields_factory: Callable[
            [Mapping[str, torch.Tensor]],
            Mapping[str, MultiModalFieldConfig],
        ],
    ) -> None:
        super().__init__(
            data,
            modality="image",
            required_fields={"image_embeds", "image_sizes"},
            fields_factory=fields_factory,
        )

    def get_image_size(self, index: int) -> ImageSize:
        image_size = self.get(index)["image_sizes"].tolist()
        return ImageSize(width=image_size[0], height=image_size[1])


class MiniCPMVVideoEmbeddingItems(DictEmbeddingItems):

    def __init__(
        self,
        data: Mapping[str, torch.Tensor],
        fields_factory: Callable[
            [Mapping[str, torch.Tensor]],
            Mapping[str, MultiModalFieldConfig],
        ],
    ) -> None:
        super().__init__(
            data,
            modality="video",
            required_fields={"video_embeds", "video_image_sizes"},
            fields_factory=fields_factory,
        )

    def get_frame_size(self, index: int) -> ImageSize:
        frame_size = self.get(index)["video_image_sizes"].tolist()
        return ImageSize(width=frame_size[0], height=frame_size[1])

    def get_num_frames(self, index: int) -> int:
        return len(self.get(index)["video_image_sizes"])


class MiniCPMVMultiModalDataParser(MultiModalDataParser):

    def _parse_image_data(
        self,
        data: Union[dict[str, torch.Tensor], ModalityData[ImageItem]],
<<<<<<< HEAD
    ) -> ModalityDataItems[Any, Any]:
=======
    ) -> Optional[ModalityDataItems[Any, Any]]:
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
        if isinstance(data, dict):
            return MiniCPMVImageEmbeddingItems(
                data,
                fields_factory=_minicpmv_field_config,
            )

        return super()._parse_image_data(data)

    def _parse_video_data(
        self,
        data: Union[dict[str, torch.Tensor], ModalityData[VideoItem]],
<<<<<<< HEAD
    ) -> ModalityDataItems[Any, Any]:
=======
    ) -> Optional[ModalityDataItems[Any, Any]]:
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
        if isinstance(data, dict):
            return MiniCPMVVideoEmbeddingItems(
                data,
                fields_factory=_minicpmv_field_config,
            )

        return super()._parse_video_data(data)


class MiniCPMVProcessingInfo(BaseProcessingInfo):
    image_pattern = "(<image>./</image>)"
    video_pattern = "(<video>./</video>)"

    def get_hf_config(self):
        return self.ctx.get_hf_config()

    def get_hf_processor(self, **kwargs: object):
        hf_processor = self.ctx.get_hf_processor(**kwargs)

        # NumPy arrays are considered as Iterable but not Sequence in
        # https://github.com/huggingface/transformers/blob/main/src/transformers/image_transforms.py#L428
        image_processor = hf_processor.image_processor  # type: ignore
        for attr in ("mean", "std"):
            val = getattr(image_processor, attr)
            if isinstance(val, np.ndarray):
                setattr(image_processor, attr, val.tolist())

        return hf_processor

    def get_image_processor(self):
        hf_processor = self.get_hf_processor()
        image_processor = hf_processor.image_processor  # type: ignore
        return image_processor

    def get_model_version(self):
        return get_version_by_config(self.get_hf_config())

<<<<<<< HEAD
    def get_supported_mm_modalities(self) -> List[str]:
        if self.get_model_version() == (2, 6):
            return ["image", "video"]
        else:
            return ["image"]

    def get_supported_mm_limits(self) -> Mapping[str, Optional[int]]:
        if self.get_model_version() == (2, 6):
            return {"image": None, "video": None}
        else:
            return {"image": None}

    def get_mm_max_tokens_per_item(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> Mapping[str, int]:
        mm_max_tokens = {"image": self.get_max_image_tokens()}
        if self.get_model_version() == (2, 6):
            mm_max_tokens["video"] = self.get_max_video_tokens(seq_len)
        return mm_max_tokens

    def get_max_video_frame_tokens(self) -> int:
        frame_size = self.get_video_frame_size_with_most_features()
        return self.get_num_image_tokens(frame_size,
                                         self.get_video_max_slice_num())

    def get_max_video_tokens(self, seq_len: int) -> int:
        return self.get_max_video_frame_tokens(
        ) * self.get_num_frames_with_most_features(seq_len)

    def get_slice_query_num(self) -> int:
        hf_config = self.get_hf_config()
        query_num = getattr(hf_config, "query_num", 64)
        return query_num

    def get_max_slice_num(self) -> int:
        hf_config = self.get_hf_config()
        max_slice_num = getattr(hf_config, "max_slice_num", 9)
        return max_slice_num

    def get_sliced_grid(self, image_size: ImageSize,
                        max_slice_num: int) -> Tuple[int, int]:
        if self.get_model_version() == (2, 6):
            slice_grid = self.get_image_processor().get_sliced_grid(
                image_size, max_slice_num)
        else:
            slice_grid = self.get_image_processor().get_sliced_grid(image_size)
        return slice_grid

    def get_num_image_tokens(self, image_size: ImageSize,
                             max_slice_num: int) -> int:
        slice_grid = self.get_sliced_grid(image_size, max_slice_num)
        num_tokens = self.get_slice_query_num(
        ) + 2  # <image>(<unk> * query_num)</image>
        if slice_grid is not None:
            if self.get_model_version() == (2, 6):
                num_additional_tokens = 0
            else:
                # <slice><image>(<unk> * query_num)</image></slice>
                num_additional_tokens = 2
            num_tokens += ((self.get_slice_query_num() + 2) \
                            * slice_grid[0] * slice_grid[1]) \
                            + slice_grid[1] - 1 + num_additional_tokens
        return num_tokens

    def get_image_slice_nums(self, image_size: torch.Tensor,
                             max_slice_nums: int) -> int:
        grid = self.get_sliced_grid(image_size, max_slice_nums)
        return 1 if grid is None else grid[0] * grid[1] + 1

    def get_max_image_tokens(self) -> int:
        image_size = self.get_image_size_with_most_features()
        return self.get_num_image_tokens(image_size, self.get_max_slice_num())

    def get_image_size_with_most_features(self) -> ImageSize:
        # Result in the max possible feature size (h:w = 9:1)
        return self.get_default_image_sizes(self.get_max_slice_num())
=======
    def get_supported_mm_limits(self) -> Mapping[str, Optional[int]]:
        mm_limits = {"image": None}
        if self.get_model_version() == (2, 6):
            mm_limits["video"] = None

        return mm_limits

    def get_slice_image_placeholder(
        self,
        image_size: ImageSize,
        # For MiniCPM V/O 2.6
        image_idx: int = 0,
        max_slice_nums: Optional[int] = None,
        use_image_id: bool = True,
    ) -> str:
        image_processor = self.get_image_processor()
        version = self.get_model_version()

        if version == (2, 0) or version == (2, 5):
            return image_processor.get_slice_image_placeholder(image_size)

        return image_processor.get_slice_image_placeholder(
            image_size,
            image_idx=image_idx,
            max_slice_nums=max_slice_nums,
            use_image_id=use_image_id,
        )

    def get_sliced_grid(
        self,
        image_size: ImageSize,
        # For MiniCPM V/O 2.6
        max_slice_nums: Optional[int] = None,
    ) -> Optional[tuple[int, int]]:
        image_processor = self.get_image_processor()
        version = self.get_model_version()

        if version == (2, 0) or version == (2, 5):
            return image_processor.get_sliced_grid(image_size)

        if max_slice_nums is None:
            max_slice_nums = image_processor.max_slice_nums

        return image_processor.get_sliced_grid(
            image_size,
            max_slice_nums=max_slice_nums,
        )

    def get_num_image_tokens(
        self,
        image_size: ImageSize,
        max_slice_nums: Optional[int] = None,
    ) -> int:
        image_processor = self.get_image_processor()

        grid = self.get_sliced_grid(
            image_size,
            max_slice_nums=max_slice_nums,
        )
        if grid is None:
            ncols = nrows = 0
        else:
            ncols, nrows = grid

        return (ncols * nrows + 1) * image_processor.image_feature_size

    def get_max_image_tokens(self) -> int:
        image_size = self.get_image_size_with_most_features()
        return self.get_num_image_tokens(image_size)

    def get_image_max_slice_num(self) -> int:
        return getattr(self.get_hf_config(), "max_slice_num", 9)

    def get_image_size_with_most_features(self) -> ImageSize:
        image_size = getattr(self.get_hf_config(), "image_size", 448)
        max_slice_num = self.get_image_max_slice_num()
        return ImageSize(width=image_size, height=image_size * max_slice_num)

    def get_max_video_frame_tokens(self) -> int:
        frame_size = self.get_video_frame_size_with_most_features()

        return self.get_num_image_tokens(
            frame_size,
            max_slice_nums=self.get_video_max_slice_num(),
        )

    def get_max_video_tokens(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> int:
        num_frames = self.get_num_frames_with_most_features(seq_len, mm_counts)
        num_video_tokens_total = self.get_max_video_frame_tokens() * num_frames
        return num_video_tokens_total
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea

    def get_video_max_slice_num(self) -> int:
        return 1

    def get_video_frame_size_with_most_features(self) -> ImageSize:
<<<<<<< HEAD
        return self.get_default_image_sizes(self.get_video_max_slice_num())
=======
        image_size = getattr(self.get_hf_config(), "image_size", 448)
        max_slice_num = self.get_video_max_slice_num()
        return ImageSize(width=image_size, height=image_size * max_slice_num)
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea

    def get_max_video_frames(self, max_tokens: int) -> int:
        num_frame_tokens = self.get_max_video_frame_tokens()
        num_frames = max_tokens // num_frame_tokens
        return num_frames

<<<<<<< HEAD
    def get_num_frames_with_most_features(self, seq_len: int) -> int:
        mm_config = self.ctx.get_mm_config()
        max_images = mm_config.limit_per_prompt.get("image", 1)
        max_videos = mm_config.limit_per_prompt.get("video", 1)

        # count <image_idx></image_idx> tokens
        # which are not in get_max_image_tokens
        max_image_tokens = self.get_max_image_tokens(
        ) * max_images + 4 * max_images
        max_total_frames = self.get_max_video_frames(seq_len -
                                                     max_image_tokens)

        num_frames = max(max_total_frames // max(max_videos, 1), 1)

        return num_frames

    def get_default_image_sizes(self, num_slices: int) -> ImageSize:
        image_size = getattr(self.get_hf_config(), "image_size", 448)
        return ImageSize(width=image_size, height=image_size * num_slices)
=======
    def get_num_frames_with_most_features(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> int:
        max_images = mm_counts.get("image", 0)
        max_videos = mm_counts.get("video", 0)

        max_image_tokens = self.get_max_image_tokens() * max_images
        max_total_frames = self.get_max_video_frames(seq_len -
                                                     max_image_tokens)
        max_frames_per_video = min(max_total_frames // max(max_videos, 1),
                                   _MAX_FRAMES_PER_VIDEO)

        return max(max_frames_per_video, 1)
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea


_I = TypeVar("_I",
             bound=MiniCPMVProcessingInfo,
             default=MiniCPMVProcessingInfo)


class MiniCPMVDummyInputsBuilder(BaseDummyInputsBuilder[_I]):

<<<<<<< HEAD
    def get_dummy_processor_inputs(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> ProcessorInputs:
=======
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        num_images = mm_counts.get("image", 0)
        num_videos = mm_counts.get("video", 0)

        image_prompt_texts = self.info.image_pattern * num_images
        video_prompt_texts = self.info.video_pattern * num_videos

        return image_prompt_texts + video_prompt_texts

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> MultiModalDataDict:
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
        num_images = mm_counts.get("image", 0)
        num_videos = mm_counts.get("video", 0)

        image_width, image_height = \
            self.info.get_image_size_with_most_features()
        video_width, video_height = \
            self.info.get_video_frame_size_with_most_features()
        num_video_frames = \
<<<<<<< HEAD
            self.info.get_num_frames_with_most_features(seq_len)

        mm_data = {
=======
            self.info.get_num_frames_with_most_features(seq_len, mm_counts)

        return {
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
            "image":
            self._get_dummy_images(width=image_width,
                                   height=image_height,
                                   num_images=num_images),
            "video": [
                self._get_dummy_images(width=video_width,
                                       height=video_height,
                                       num_images=num_video_frames)
            ] * num_videos,
        }

<<<<<<< HEAD
        image_prompt_texts = self.info.image_pattern * num_images
        video_prompt_texts = self.info.video_pattern * num_videos

        return ProcessorInputs(prompt_text=image_prompt_texts +
                               video_prompt_texts,
                               mm_data=mm_data)

=======
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea

class MiniCPMVMultiModalProcessor(BaseMultiModalProcessor[_I]):

    def _get_data_parser(self) -> MultiModalDataParser:
        return MiniCPMVMultiModalDataParser()

<<<<<<< HEAD
    def get_slice_image_placeholder(self, image_size: ImageSize,
                                    **kwargs) -> str:
        image_processor = self.info.get_image_processor()
        version = self.info.get_model_version()
        if version == (2, 0) or version == (2, 5):
            return image_processor.get_slice_image_placeholder(image_size)
        return image_processor.get_slice_image_placeholder(
            image_size, **kwargs)

    def get_image_prompt_texts(self,
                               image_size: ImageSize,
                               image_idx: int = 0) -> str:
        prompt_texts = self.get_slice_image_placeholder(image_size,
                                                        image_idx=image_idx)
        return prompt_texts

    def get_video_prompt_texts(self, image_size: ImageSize,
                               num_frames: int) -> str:
        prompt_texts = "".join(
            self.get_slice_image_placeholder(
                image_size=image_size,
                image_idx=0,
                max_slice_nums=self.info.get_video_max_slice_num(),
                use_image_id=False) for image_idx in range(num_frames))
        return prompt_texts

    def get_special_tokens(self) -> Dict[str, torch.Tensor]:
        tokenizer = self.info.get_tokenizer()
        special_tokens = {
            "im_start_id": torch.tensor(tokenizer.im_start_id),
            "im_end_id": torch.tensor(tokenizer.im_end_id)
        }
        if hasattr(tokenizer, "slice_start_id"):
            special_tokens["slice_start_id"] = torch.tensor(
                tokenizer.slice_start_id)
            special_tokens["slice_end_id"] = torch.tensor(
                tokenizer.slice_end_id)
        return special_tokens

    @staticmethod
    def repack_processor_outputs(outputs: Any) -> BatchFeature:
        valid_keys = ["pixel_values", "image_sizes", "tgt_sizes"]
        outputs = {key: outputs[key][0] for key in valid_keys}
        return outputs

    def process_images(self, mm_data: Mapping[str, object],
                       mm_kwargs: Mapping[str, object]) -> Dict[str, object]:
        images = mm_data.pop("images", [])
        image_embeds = mm_data.pop("image_embeds", [])
        if isinstance(images, Image.Image):
            images = [images]
        if isinstance(images, (list, torch.Tensor)) and len(images) > 0:
            image_outputs = super()._call_hf_processor(
                prompt=self.info.image_pattern * len(images),
                mm_data={"images": images},
                mm_kwargs=mm_kwargs)
            image_outputs = MiniCPMVMultiModalProcessor.\
                repack_processor_outputs(image_outputs)
        elif len(image_embeds) > 0:
            image_sizes = mm_data.pop("image_sizes", None)
            image_outputs = {
                "image_embeds": torch.cat(image_embeds),
                "image_sizes": image_sizes
            }
        else:
            image_outputs = {}
        return image_outputs

    def process_videos(self, mm_data: Mapping[str, object],
                       mm_kwargs: Mapping[str, object]) -> Dict[str, object]:
        videos = mm_data.pop("videos", [])
        video_embeds = mm_data.pop("video_embeds", [])
        if len(videos) > 0 and isinstance(videos[0], Image.Image):
            videos = [videos]
        if isinstance(videos, list) and len(videos) > 0:
            video_outputs = {
                "video_pixel_values": [],
                "video_image_sizes": [],
                "video_tgt_sizes": [],
                "num_frames": []
            }
            for video in videos:
                parsed_video = []
                for frame in video:
                    if isinstance(frame, np.ndarray):
                        parsed_video.append(Image.fromarray(frame))
                    else:
                        parsed_video.append(frame)
                video = parsed_video
                single_video_outputs = super()._call_hf_processor(
                    prompt=self.info.image_pattern * len(video),
                    mm_data={"images": video},
                    mm_kwargs={
                        **mm_kwargs, "max_slice_nums":
                        self.info.get_video_max_slice_num()
                    })
                video_outputs["num_frames"].append(len(video))
                for key in single_video_outputs:
                    if "video_" + key in video_outputs:
                        if key == "image_sizes":
                            video_outputs["video_" + key].append(
                                single_video_outputs[key][0][0])
                        else:
                            video_outputs["video_" +
                                          key] += single_video_outputs[key][0]
        elif len(video_embeds):
            image_sizes = mm_data.pop("image_sizes", None)
            num_frames = mm_data.pop("num_frames", None)
            video_outputs = {
                "video_embeds": torch.cat(video_embeds),
                "video_image_sizes": image_sizes,
                "num_frames": num_frames
            }
        else:
            video_outputs = {}
        return video_outputs

    def get_placeholder_match_pattern(self) -> str:
        return r"\(<(image|video)>./</\1>\)"

    def get_placeholder_split_pattern(self) -> str:
        return r"\(<(?:image|video)>./</(?:image|video)>\)"

    def process_mm_inputs(self, mm_data, mm_kwargs) -> object:
        return {
            "image": self.process_images(mm_data, mm_kwargs),
            "video": self.process_videos(mm_data, mm_kwargs)
        }

    def get_input_modalities(self, mm_data) -> List[str]:
        supported_mm_modalities = self.info.get_supported_mm_modalities()
        input_modalities = []
        for modality in supported_mm_modalities:
            if modality in mm_data and mm_data[modality] != {}:
                input_modalities.append(modality)
        return input_modalities

    def get_modality_num_counter(self, modality: str) -> str:
        if modality == "image":
            return "image_sizes"
        elif modality == "video":
            return "video_image_sizes"

    def get_num_slices_by_modality(self, inputs: Dict[str, object],
                                   modality: str, index: int) -> int:
        if modality == "image":
            return self.info.get_image_slice_nums(
                inputs[modality]["image_sizes"][index],
                self.info.get_max_slice_num())
        elif modality == "video":
            return self.info.get_image_slice_nums(
                inputs[modality]["video_image_sizes"][index],
                self.info.get_video_max_slice_num()
            ) * inputs[modality]["num_frames"][index]
        else:
            raise ValueError(f"Unexpected modality: {modality}")

    def check_mm_inputs(self, inputs: Dict[str, object],
                        matches: List[str]) -> None:
        counts = Counter(matches)
        for modality, count in counts.items():
            if modality not in inputs or not inputs[modality]:
                raise ValueError(f"None input data of {modality}."
                                 "But prompt requires.")
            counter_key = self.get_modality_num_counter(modality)
            if len(inputs[modality][counter_key]) != count:
                raise ValueError(f"The prompt requires {count} "
                                 f"{modality} inputs while you pass "
                                 f"{len(inputs[modality][counter_key])}")

    def get_prompt_texts_by_modality(self, inputs: Dict[str, object],
                                     modality: str, index: int) -> str:
        if modality == "image":
            return self.get_image_prompt_texts(
                inputs["image"]["image_sizes"][index], index)
        elif modality == "video":
            return self.get_video_prompt_texts(
                inputs["video"]["video_image_sizes"][index],
                inputs["video"]["num_frames"][index])
        else:
            raise ValueError(f"Unexpected modality: {modality}")

    def call_base_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        return super()._call_hf_processor(prompt=prompt,
                                          mm_data=mm_data,
                                          mm_kwargs=mm_kwargs)
=======
    def get_image_prompt_texts(self,
                               image_size: ImageSize,
                               image_idx: int = 0) -> str:
        return self.info.get_slice_image_placeholder(
            image_size,
            image_idx=image_idx,
        )

    def get_video_prompt_texts(self, image_size: ImageSize,
                               num_frames: int) -> str:
        return self.info.get_slice_image_placeholder(
            image_size=image_size,
            image_idx=0,
            max_slice_nums=self.info.get_video_max_slice_num(),
            use_image_id=False,
        ) * num_frames

    def process_images(
        self,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, NestedTensors]:
        if (images := mm_data.get("images")) is None:
            return {}

        parsed_images = (self._get_data_parser().parse_mm_data({
            "image": images
        }).get_items("image",
                     (MiniCPMVImageEmbeddingItems, ImageProcessorItems)))

        if isinstance(parsed_images, MiniCPMVImageEmbeddingItems):
            image_inputs = {}
        else:
            image_inputs = self._base_call_hf_processor(
                prompts=[self.info.image_pattern] * len(parsed_images),
                mm_data={"images": [[image] for image in parsed_images]},
                mm_kwargs=mm_kwargs,
                out_keys={"pixel_values", "image_sizes", "tgt_sizes"},
            )

        tokenizer = self.info.get_tokenizer()
        unk_token_id = tokenizer.get_vocab()["<unk>"]
        image_inputs["image_token_id"] = torch.tensor(unk_token_id)

        return image_inputs

    def process_videos(
        self,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, NestedTensors]:
        if (videos := mm_data.get("videos")) is None:
            return {}

        parsed_videos = (self._get_data_parser().parse_mm_data({
            "video": videos
        }).get_items("video",
                     (MiniCPMVVideoEmbeddingItems, VideoProcessorItems)))

        if isinstance(parsed_videos, MiniCPMVVideoEmbeddingItems):
            video_inputs = {}
        else:
            video_inputs = self._base_call_hf_processor(
                prompts=[
                    self.info.image_pattern * len(video)
                    for video in parsed_videos
                ],
                mm_data={"images": list(parsed_videos)},
                mm_kwargs={
                    **mm_kwargs,
                    "max_slice_nums":
                    self.info.get_video_max_slice_num(),
                },
                out_keys={"pixel_values", "image_sizes", "tgt_sizes"},
            )

        video_inputs = {f"video_{k}": v for k, v in video_inputs.items()}

        tokenizer = self.info.get_tokenizer()
        unk_token_id = tokenizer.get_vocab()["<unk>"]
        video_inputs["video_token_id"] = torch.tensor(unk_token_id)

        return video_inputs

    def process_mm_inputs(
        self,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, NestedTensors]:
        return {
            **self.process_images(mm_data, mm_kwargs),
            **self.process_videos(mm_data, mm_kwargs),
        }

    def _base_call_hf_processor(
        self,
        prompts: list[str],
        mm_data: Mapping[str, Sequence[object]],
        mm_kwargs: Mapping[str, object],
        *,
        out_keys: set[str],
    ) -> dict[str, NestedTensors]:
        # This processor supports zipping prompt and mm_data together
        if self.info.get_model_version() == (2, 6):
            inputs = super()._call_hf_processor(
                prompt=prompts,  # type: ignore
                mm_data=mm_data,
                mm_kwargs=mm_kwargs,
            )
        else:
            inputs = defaultdict[str, list[torch.Tensor]](list)

            for i, prompt in enumerate(prompts):
                inputs_one = super()._call_hf_processor(
                    prompt=prompt,
                    mm_data={
                        k: v[i]
                        for k, v in mm_data.items()
                    },
                    mm_kwargs=mm_kwargs,
                )

                for k, v in inputs_one.items():
                    assert len(v) == 1, (k, len(v))
                    inputs[k].append(v[0])

        return {k: inputs[k] for k in out_keys}
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea

    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
    ) -> BatchFeature:
<<<<<<< HEAD
        # Do not support combination inputs of images and videos for now
        # Try to handle interleaved multimodal data
        tokenizer = self.info.get_tokenizer()
        inputs = self.process_mm_inputs(mm_data, mm_kwargs)
        mm_input_modalities = self.get_input_modalities(inputs)
        num_mm_slices = {modality: [] for modality in mm_input_modalities}
        for modality in mm_input_modalities:
            num_counter_key = self.get_modality_num_counter(modality)
            for index in range(len(inputs[modality][num_counter_key])):
                num_mm_slices[modality].append(
                    self.get_num_slices_by_modality(inputs, modality, index))
        return {
            "input_ids": np.array([tokenizer.encode(prompt)]),
            **{
                key: value
                for modality in inputs
                for key, value in inputs[modality].items()
            },
            **{
                f"{modality}_num_slices": num_mm_slices[modality]
                for modality in mm_input_modalities
            }
        }

    def _hf_processor_applies_repl(
=======
        tokenizer = self.info.get_tokenizer()

        input_ids = torch.tensor([tokenizer.encode(prompt)])
        mm_inputs = self.process_mm_inputs(mm_data, mm_kwargs)

        return BatchFeature({
            "input_ids": input_ids,
            **mm_inputs,
        })

    def _hf_processor_applies_updates(
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
        self,
        prompt_text: str,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> bool:
        return False

<<<<<<< HEAD
    def _get_prompt_replacements(
            self, mm_items: MultiModalDataItems,
            hf_processor_mm_kwargs: Mapping[str, Any],
            out_mm_kwargs: MultiModalKwargs) -> List[PromptReplacement]:
=======
    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargs,
    ) -> Sequence[PromptUpdate]:
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
        placeholder = {
            "image": self.info.image_pattern,
            "video": self.info.video_pattern,
        }

<<<<<<< HEAD
        def get_replacement_minicpmv(item_idx: int, modality: str):
            if modality == "image":
                return self.get_image_prompt_texts(
                    mm_items["image"].get_image_size(item_idx), item_idx)
            else:  # video
                return self.get_video_prompt_texts(
                    mm_items["video"].get_frame_size(item_idx),
                    mm_items["video"].get_num_frames(item_idx))
=======
        def get_image_replacement(item_idx: int):
            images = mm_items.get_items(
                "image", (MiniCPMVImageEmbeddingItems, ImageProcessorItems))

            image_size = images.get_image_size(item_idx)

            return PromptUpdateDetails.select_text(
                self.get_image_prompt_texts(image_size, item_idx),
                "<unk>",
            )

        def get_video_replacement(item_idx: int):
            videos = mm_items.get_items(
                "video", (MiniCPMVVideoEmbeddingItems, VideoProcessorItems))

            frame_size = videos.get_frame_size(item_idx)
            num_frames = videos.get_num_frames(item_idx)

            return PromptUpdateDetails.select_text(
                self.get_video_prompt_texts(frame_size, num_frames),
                "<unk>",
            )

        get_replacement = {
            "image": get_image_replacement,
            "video": get_video_replacement,
        }
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea

        return [
            PromptReplacement(modality=modality,
                              target=placeholder[modality],
<<<<<<< HEAD
                              replacement=partial(get_replacement_minicpmv,
                                                  modality=modality))
=======
                              replacement=get_replacement[modality])
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
            for modality in ("image", "video")
        ]

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        return _minicpmv_field_config(hf_inputs)

<<<<<<< HEAD
    def apply(
        self,
        prompt: Union[str, List[int]],
        mm_data: MultiModalDataDict,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> MultiModalInputs:
        supported_mm_modalities = self.info.get_supported_mm_modalities()
        if isinstance(prompt, list):
            prompt = self.info.get_tokenizer().decode(prompt)
        matches = re.findall(self.get_placeholder_match_pattern(), prompt)
        mm_orders = {
            f"{modality}_orders":
            torch.tensor(
                [index for index, m in enumerate(matches) if m == modality])
            for modality in supported_mm_modalities
        }
        result = super().apply(prompt, mm_data, hf_processor_mm_kwargs)
        # Exclude <image_id>x</image_id> from placeholders
        if "image" in result["mm_placeholders"] and \
            self.info.get_model_version() == (2, 6):
            result["mm_placeholders"]["image"] = [
                PlaceholderRange(offset=p["offset"] + 3 + idx // 10,
                                 length=p["length"] - 3 - idx // 10)
                for idx, p in enumerate(result["mm_placeholders"]["image"])
            ]
        result["mm_kwargs"].update(**mm_orders)
        result["mm_kwargs"].update(**self.get_special_tokens())
        return result

=======
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea

class MiniCPMVBaseModel(nn.Module, SupportsMultiModal, SupportsPP):
    """
    The abstract class of MiniCPMV can only be inherited, but cannot be
    instantiated.
    """

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        config = vllm_config.model_config.hf_config
        multimodal_config = vllm_config.model_config.multimodal_config
        quant_config = vllm_config.quant_config
        super().__init__()
        # All MiniCPM-V models disable `tie_word_embeddings` but
        # `PretrainedConfig.tie_word_embeddings` defaults to True; we cannot
        # check `tie_word_embeddings` until vLLM integrate MiniCPM-V model
        # and config class
        self.config = config
        self.multimodal_config = multimodal_config

        self.version = get_version_by_config(self.config)
        self.llm = self.init_llm(vllm_config=vllm_config,
                                 prefix=maybe_prefix(prefix, "llm"))
        self.vpm = self.init_vision_module(config,
                                           quant_config,
                                           prefix=maybe_prefix(prefix, "vpm"))
        self.vision_dim = (self.vpm.embed_dim if self.version == (2, 0) else
                           self.vpm.embeddings.embed_dim)
        self.embed_dim = self.config.hidden_size

        self.resampler = self.init_resampler(self.embed_dim,
                                             self.vision_dim,
                                             quant_config=quant_config,
                                             prefix=maybe_prefix(
                                                 prefix, "resampler"))

<<<<<<< HEAD
        self.make_empty_intermediate_tensors = (
            self.llm.make_empty_intermediate_tensors)

    @cached_property
    def sampler(self):
        if hasattr(self.llm, "sampler"):
            return self.llm.sampler

        return get_sampler()

    def get_embedding_with_vision(
        self,
        input_ids: torch.Tensor,
        image_inputs: Optional[MiniCPMVImageInputs],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        vlm_embedding: torch.Tensor = self.llm.get_input_embeddings(input_ids)

        if image_inputs is None:  # No image
            vision_hidden_states = torch.tensor([], device=input_ids.device)
        else:
            if image_inputs["type"] == "image_embeds":
                vision_hidden_states = (image_inputs["data"].type(
                    vlm_embedding.dtype).to(vlm_embedding.device))
            else:
                vision_hidden_states = self.get_vision_hidden_states(
                    image_inputs)

            # See NOTE in _parse_and_validate_inputs
            image_bounds = image_inputs["image_bounds"]
            if len(image_bounds) > 0:
                image_indices = torch.stack([
                    torch.arange(start, end, dtype=torch.long)
                    for start, end in image_bounds.tolist()
                ]).to(vlm_embedding.device)
                vlm_embedding.scatter_(
                    0,
                    image_indices.view(-1, 1).repeat(1,
                                                     vlm_embedding.shape[-1]),
                    vision_hidden_states.view(-1,
                                              vision_hidden_states.shape[-1]),
                )

        return vlm_embedding, vision_hidden_states

    def _get_image_bounds(
            self,
            input_ids: torch.Tensor,
            im_start_id: torch.Tensor,
            im_end_id: torch.Tensor,
            slice_start_id: Optional[torch.Tensor] = None,
            slice_end_id: Optional[torch.Tensor] = None) -> torch.Tensor:
        # All the images in the batch should share the same special image
        # bound token ids.
        start_cond = input_ids == im_start_id[0]
        end_cond = input_ids == im_end_id[0]
        if slice_start_id is not None:
            start_cond |= (input_ids == slice_start_id[0])
            end_cond |= (input_ids == slice_end_id[0])

        image_start_tokens, = torch.where(start_cond)
        image_start_tokens += 1
        image_end_tokens, = torch.where(end_cond)
        valid_image_nums = max(len(image_start_tokens), len(image_end_tokens))

        if valid_image_nums == 0:
            return torch.zeros((0, 2), device=input_ids.device)

        return torch.hstack([
            image_start_tokens[:valid_image_nums].unsqueeze(-1),
            image_end_tokens[:valid_image_nums].unsqueeze(-1),
        ])

    def _parse_and_validate_image_inputs(
        self,
        input_ids: torch.Tensor,
        **kwargs: object,
    ) -> Optional[MiniCPMVImageInputs]:
        mm_data = {
            "image": {
                key: kwargs.pop(key, [])
                for key in ["pixel_values", "tgt_sizes", "image_num_slices"]
            },
            "video": {
                "pixel_values": kwargs.pop("video_pixel_values", []),
                "tgt_sizes": kwargs.pop("video_tgt_sizes", []),
                "video_num_slices": kwargs.pop("video_num_slices", [])
            }
        }
        im_start_id = kwargs.pop("im_start_id", None)
        im_end_id = kwargs.pop("im_end_id", None)
        slice_start_id = kwargs.pop("slice_start_id", None)
        slice_end_id = kwargs.pop("slice_end_id", None)
        mm_orders = {
            f"{modality}": kwargs.pop(f"{modality}_orders", None)
            for modality in ["image", "video", "audio"]
        }
        batch_size = max(len(mm_data["image"]["pixel_values"]),
                         len(mm_data["video"]["pixel_values"]))
        image_embeds = kwargs.pop("image_embeds", None)
        video_embeds = kwargs.pop("video_embeds", None)
        if image_embeds is not None and video_embeds is not None:
            raise ValueError(
                "Incorrect inputs for vision embeddings. "
                "Image embeds and video embeds can not exist simultaneously.")
        if video_embeds is not None:
            image_embeds = video_embeds
        if image_embeds is not None:
            if not isinstance(image_embeds, (torch.Tensor, list)):
                raise ValueError(f"Incorrect type of image embeds. "
                                 f"Got type: {type(image_embeds)}")
            image_embeds = torch.concat(
                [image_embeds[i] for i in range(len(image_embeds))])

            return MiniCPMVImageEmbeddingInputs(
                image_bounds=self._get_image_bounds(input_ids, im_start_id,
                                                    im_end_id, slice_start_id,
                                                    slice_end_id),
                data=image_embeds,
                type="image_embeds",
            )
        for modality, modality_mm_data in mm_data.items():
            if not isinstance(modality_mm_data["pixel_values"],
                              (torch.Tensor, list)):
                raise ValueError(
                    "Incorrect type of pixel values. "
                    f"Got type: {type(modality_mm_data['pixel_values'])}")

            if not isinstance(modality_mm_data["tgt_sizes"],
                              (torch.Tensor, list)):
                raise ValueError(
                    "Incorrect type of target sizes. "
                    f"Got type: {type(modality_mm_data['tgt_sizes'])}")

            if len(modality_mm_data["pixel_values"]) != len(
                    modality_mm_data["tgt_sizes"]):
                raise ValueError(
                    "Inconsistent batch lengths, found: "
                    f"{len(modality_mm_data['pixel_values'])} vs. "
                    f"{len(modality_mm_data['tgt_sizes'])}")

        pixel_values_flat: List[torch.Tensor] = []
        tgt_sizes_flat: List[torch.Tensor] = []
        for b in range(batch_size):
            mm_counts = {"image": 0, "video": 0} if self.version == (2, 6) \
                        else {"image": 0}
            mm_slice_counts = {"image": 0, "video": 0} \
                               if self.version == (2, 6) else {"image": 0}
            mm_orders_b = [(index, modality) for modality in mm_counts
                           for index in mm_orders[modality][b]]
            for _, modality in sorted(mm_orders_b, key=lambda x: x[0]):
                pos = mm_counts[modality]
                num_slices = mm_data[modality][f"{modality}_num_slices"][b][
                    pos]
                slice_start_idx = mm_slice_counts[modality]
                slice_end_idx = slice_start_idx + num_slices
                pixel_values_flat += mm_data[modality]["pixel_values"][b][
                    slice_start_idx:slice_end_idx]
                tgt_sizes_flat += mm_data[modality]["tgt_sizes"][b][
                    slice_start_idx:slice_end_idx]
                mm_counts[modality] += 1
                mm_slice_counts[modality] += num_slices

        # NOTE: Input IDs does not contain image tokens during memory profiling,
        # so we allow it to be empty
=======
        self.mm_token_ids = set[int]()
        self.make_empty_intermediate_tensors = (
            self.llm.make_empty_intermediate_tensors)

    def _parse_and_validate_vision_input(
        self,
        modality: str,
        **kwargs: object,
    ) -> Optional[MiniCPMVImageInputs]:
        pixel_values = kwargs.pop("pixel_values", None)
        image_embeds = kwargs.pop("image_embeds", None)

        if pixel_values is None and image_embeds is None:
            return None

        image_token_id = kwargs.pop("image_token_id")
        if image_token_id is not None:
            assert isinstance(image_token_id, torch.Tensor)
            self.mm_token_ids.add(image_token_id.flatten().unique().item())

        if image_embeds is not None:
            if not isinstance(image_embeds, (torch.Tensor, list)):
                raise ValueError(
                    f"Incorrect type of image_embeds for {modality=}. "
                    f"Got type: {type(image_embeds)}")

            image_embeds_flat = flatten_bn(image_embeds)

            return MiniCPMVImageEmbeddingInputs(
                type="image_embeds",
                image_embeds=image_embeds_flat,
            )

        if not isinstance(pixel_values, (torch.Tensor, list)):
            raise ValueError(
                f"Incorrect type of pixel_values for {modality=}. "
                f"Got type: {type(pixel_values)}")

        tgt_sizes = kwargs.pop("tgt_sizes")
        if not isinstance(tgt_sizes, (torch.Tensor, list)):
            raise ValueError(f"Incorrect type of tgt_sizes for {modality=}. "
                             f"Got type: {type(tgt_sizes)}")

        num_slices = [[len(p) for p in ps] for ps in pixel_values]
        num_slices_flat = flatten_bn(torch.tensor(num_slices))

        pixel_values_flat = flatten_bn(flatten_2d_lists(pixel_values))
        tgt_sizes_flat = flatten_bn(flatten_2d_lists(tgt_sizes), concat=True)

>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
        if len(pixel_values_flat) != len(tgt_sizes_flat):
            raise ValueError("Inconsistent flattened lengths, found: "
                             f"{len(pixel_values_flat)} vs. "
                             f"{len(tgt_sizes_flat)}")

<<<<<<< HEAD
        if len(pixel_values_flat) == 0:
            return None

        if im_start_id is None:
            return None

        return MiniCPMVImagePixelInputs(
            image_bounds=self._get_image_bounds(input_ids, im_start_id,
                                                im_end_id, slice_start_id,
                                                slice_end_id),
            data=pixel_values_flat,
            tgt_sizes=torch.stack(tgt_sizes_flat),
            type="pixel_values",
        )

    def _parse_and_validate_inputs(self, input_ids: torch.Tensor,
                                   **kwargs: object):
        return self._parse_and_validate_image_inputs(input_ids, **kwargs)
=======
        return MiniCPMVImagePixelInputs(
            type="pixel_values",
            pixel_values=pixel_values_flat,
            tgt_sizes=tgt_sizes_flat,
            num_slices=num_slices_flat,
        )

    def _parse_and_validate_multimodal_inputs(self, **kwargs: object) -> dict:
        modalities = {}

        # Preserve the order of modalities if there are multiple of them
        # from the order of kwargs.
        for input_key in kwargs:
            if input_key in ("pixel_values",
                             "image_embeds") and "images" not in modalities:
                modalities["images"] = self._parse_and_validate_vision_input(
                    "images", **kwargs)
            if input_key in ("video_pixel_values",
                             "video_embeds") and "videos" not in modalities:

                def _image_key(video_key: str):
                    if video_key == "video_token_id":
                        return "image_token_id"

                    return video_key.removeprefix("video_")

                modalities["videos"] = self._parse_and_validate_vision_input(
                    "videos", **{
                        _image_key(k): v
                        for k, v in kwargs.items()
                    })

        return modalities

    def _process_vision_input(
        self,
        image_input: MiniCPMVImageInputs,
    ) -> Union[torch.Tensor, list[torch.Tensor], tuple[torch.Tensor, ...]]:
        if image_input["type"] == "image_embeds":
            return image_input["image_embeds"]

        image_features_flat = self.get_vision_hidden_states(image_input)

        num_slices = image_input["num_slices"]
        return [
            e.flatten(0, 1)
            for e in image_features_flat.split(num_slices.tolist())
        ]

    def _process_multimodal_inputs(self, modalities: dict):
        # The result multimodal_embeddings is tuple of tensors, with each
        # tensor correspoending to a multimodal data item (image or video).
        multimodal_embeddings: tuple[torch.Tensor, ...] = ()

        # NOTE: It is important to iterate over the keys in this dictionary
        # to preserve the order of the modalities.
        for modality in modalities:
            if modality == "images":
                image_input = modalities["images"]
                image_features = self._process_vision_input(image_input)
                multimodal_embeddings += tuple(image_features)
            if modality == "videos":
                video_input = modalities["videos"]
                video_features = self._process_vision_input(video_input)
                multimodal_embeddings += tuple(video_features)

        return multimodal_embeddings

    def get_language_model(self) -> torch.nn.Module:
        return self.llm

    def get_multimodal_embeddings(
            self, **kwargs: object) -> Optional[MultiModalEmbeddings]:
        modalities = self._parse_and_validate_multimodal_inputs(**kwargs)
        if not modalities:
            return None

        return self._process_multimodal_inputs(modalities)

    def get_input_embeddings(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: Optional[MultiModalEmbeddings] = None,
    ) -> torch.Tensor:
        inputs_embeds = self.llm.get_input_embeddings(input_ids)
        if multimodal_embeddings is not None:
            assert len(self.mm_token_ids) > 0
            inputs_embeds = merge_multimodal_embeddings(
                input_ids,
                inputs_embeds,
                multimodal_embeddings,
                list(self.mm_token_ids),
            )
        return inputs_embeds
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
<<<<<<< HEAD
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        if intermediate_tensors is not None:
            vlm_embeddings = None
        else:
            image_inputs = \
                self._parse_and_validate_inputs(input_ids, **kwargs)
            vlm_embeddings, _ = self.get_embedding_with_vision(
                input_ids, image_inputs)

        # always pass the input via `inputs_embeds`
        # to make sure the computation graph is consistent
        # for `torch.compile` integration
        input_ids = None

        output = self.llm.model(
            input_ids=input_ids,
            positions=positions,
            kv_caches=kv_caches,
            attn_metadata=attn_metadata,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=vlm_embeddings,
        )
        return output
=======
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        if intermediate_tensors is not None:
            inputs_embeds = None

        # NOTE: In v1, inputs_embeds is always generated at model runner from
        # `get_multimodal_embeddings` and `get_input_embeddings`, this
        # condition is only for v0 compatibility.
        elif inputs_embeds is None:
            vision_embeddings = self.get_multimodal_embeddings(**kwargs)

            inputs_embeds = self.get_input_embeddings(input_ids,
                                                      vision_embeddings)
            input_ids = None

        hidden_states = self.llm.model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )
        return hidden_states
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[torch.Tensor]:
        return self.llm.compute_logits(hidden_states, sampling_metadata)

<<<<<<< HEAD
    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        next_tokens = self.sampler(logits, sampling_metadata)
        return next_tokens

    def load_weights(self, weights: Iterable[Tuple[str,
                                                   torch.Tensor]]) -> Set[str]:
=======
    def load_weights(self, weights: Iterable[tuple[str,
                                                   torch.Tensor]]) -> set[str]:
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights)

    def get_mm_mapping(self) -> MultiModelKeys:
        """
        Get the module prefix in multimodal models
        """
        return MultiModelKeys.from_string_field(language_model="llm",
                                                connector="resampler",
                                                tower_model="vpm")

    def init_llm(
        self,
        vllm_config: VllmConfig,
        prefix: str = "",
    ) -> nn.Module:
        raise NotImplementedError

    def init_vision_module(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig],
        prefix: str = "",
    ) -> nn.Module:
        raise NotImplementedError

    def init_resampler(self,
                       embed_dim: int,
                       vision_dim: int,
                       quant_config: Optional[QuantizationConfig] = None,
                       prefix: str = "") -> nn.Module:
        raise NotImplementedError

<<<<<<< HEAD
    def get_vision_embedding(
        self,
        pixel_values: List[torch.Tensor],
        patch_attn_mask: Optional[torch.Tensor] = None,
        tgt_sizes: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        raise NotImplementedError

    def get_vision_hidden_states(self,
                                 data: MiniCPMVImageInputs) -> torch.Tensor:
=======
    def get_vision_hidden_states(
            self, data: MiniCPMVImagePixelInputs) -> torch.Tensor:
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
        raise NotImplementedError


class MiniCPMV2_0(MiniCPMVBaseModel):

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        assert self.version == (2, 0)

    def init_llm(
        self,
        vllm_config: VllmConfig,
        prefix: str = "",
    ) -> nn.Module:
        return MiniCPMForCausalLM(vllm_config=vllm_config, prefix=prefix)

    def init_vision_module(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig],
        prefix: str = "",
    ) -> nn.Module:
        # TODO: refactor vision model through timm wrapper from transformers
        try:
            import timm
        except ImportError:
            raise ImportError("Please install timm==0.9.10") from ImportError

        with set_default_torch_dtype(torch.float16):
            model = timm.create_model(
                "vit_so400m_patch14_siglip_384.webli",
                pretrained=False,
                num_classes=0,
                dynamic_img_size=True,
                dynamic_img_pad=True,
            )

        model = model.to(dtype=torch.get_default_dtype())

        if (isinstance(model, timm.models.VisionTransformer)
                and model.attn_pool is not None):
            model.attn_pool = torch.nn.Identity()

        if self.config.drop_vision_last_layer:
            model.blocks = model.blocks[:-1]

        return model

<<<<<<< HEAD
    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_tokens(input_ids)

=======
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
    def init_resampler(self,
                       embed_dim: int,
                       vision_dim: int,
                       quant_config: Optional[QuantizationConfig] = None,
                       prefix: str = "") -> nn.Module:
        with set_default_torch_dtype(torch.float16):
            resampler = Resampler2(embed_dim=embed_dim,
                                   num_heads=embed_dim // 128,
                                   grid_size=int(
                                       math.sqrt(self.config.query_num)),
                                   kv_dim=vision_dim,
                                   adaptive=False,
                                   do_post_projection=True,
                                   quant_config=quant_config,
                                   prefix=prefix)

<<<<<<< HEAD
        return resampler.to(device="cuda", dtype=torch.get_default_dtype())

    def get_vision_embedding(
        self,
        pixel_values: List[torch.Tensor],
        patch_attn_mask: Optional[torch.Tensor] = None,
        tgt_sizes: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        res = []
        dtype = self.vpm.pos_embed.data.dtype
        for pixel_value in pixel_values:
            H, W = pixel_value[0].shape[-2:]
            tgt_size = (
                math.ceil(H / self.vpm.patch_embed.patch_size[0]),
                math.ceil(W / self.vpm.patch_embed.patch_size[0]),
            )
            vision_embedding = self.vpm.forward_features(
                pixel_value.unsqueeze(0).type(dtype))
            if (hasattr(self.vpm, "num_prefix_tokens")
                    and self.vpm.num_prefix_tokens > 0):
                vision_embedding = vision_embedding[:, self.vpm.
                                                    num_prefix_tokens:]
            res.append(self.resampler(vision_embedding, tgt_size))
        return torch.vstack(res)

    def get_vision_hidden_states(self,
                                 data: MiniCPMVImageInputs) -> torch.Tensor:
        pixel_values = data["data"]

        return self.get_vision_embedding(pixel_values)

=======
        return resampler.to(device=current_platform.device_type,
                            dtype=torch.get_default_dtype())

    def get_vision_hidden_states(
            self, data: MiniCPMVImagePixelInputs) -> torch.Tensor:
        pixel_values = data["pixel_values"]

        P_h, P_w = self.vpm.patch_embed.patch_size
        dtype: torch.dtype = self.vpm.pos_embed.data.dtype
        num_prefix_tokens = getattr(self.vpm, "num_prefix_tokens", 0)

        res = list[torch.Tensor]()
        for pixel_value in pixel_values:
            H, W = pixel_value[0].shape[-2:]
            tgt_size = (math.ceil(H / P_h), math.ceil(W / P_w))
            vision_embedding = self.vpm.forward_features(
                pixel_value.unsqueeze(0).type(dtype))

            if num_prefix_tokens > 0:
                vision_embedding = vision_embedding[:, num_prefix_tokens:]
            res.append(self.resampler(vision_embedding, tgt_size))

        return torch.vstack(res)

>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea

class MiniCPMV2_5(MiniCPMVBaseModel, SupportsLoRA):
    packed_modules_mapping = {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ],
        "gate_up_proj": [
            "gate_proj",
            "up_proj",
        ],
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        assert self.version == (2, 5)

    def init_llm(
        self,
        vllm_config: VllmConfig,
        prefix: str = "",
    ) -> nn.Module:
        return LlamaForCausalLM(vllm_config=vllm_config, prefix=prefix)

    def init_vision_module(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig],
        prefix: str = "",
    ) -> nn.Module:
        model = Idefics2VisionTransformer(config.vision_config,
                                          quant_config=quant_config,
                                          prefix=prefix)
        if self.config.drop_vision_last_layer:
            model.encoder.layers = model.encoder.layers[:-1]
        return model

    def init_resampler(self,
                       embed_dim: int,
                       vision_dim: int,
                       quant_config: Optional[QuantizationConfig] = None,
                       prefix: str = "") -> nn.Module:
        with set_default_torch_dtype(torch.float16):
            resampler = Resampler2_5(num_queries=self.config.query_num,
                                     embed_dim=embed_dim,
                                     num_heads=embed_dim // 128,
                                     kv_dim=vision_dim,
                                     quant_config=quant_config,
                                     prefix=prefix)

<<<<<<< HEAD
        return resampler.to(device="cuda", dtype=torch.get_default_dtype())

    def get_vision_embedding(
        self,
        pixel_values: List[torch.Tensor],
        patch_attn_mask: Optional[torch.Tensor] = None,
        tgt_sizes: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        vision_embedding = self.vpm(pixel_values,
                                    patch_attention_mask=patch_attn_mask)
        vision_embedding = self.resampler(vision_embedding, tgt_sizes)
        return vision_embedding

    def get_vision_hidden_states(self,
                                 data: MiniCPMVImageInputs) -> torch.Tensor:
        pixel_values = data["data"]
        tgt_sizes = data["tgt_sizes"]

        device = self.vpm.embeddings.position_embedding.weight.device
        dtype = self.vpm.embeddings.position_embedding.weight.dtype
        all_pixel_values_lst = [
            i.flatten(end_dim=1).permute(1, 0) for i in pixel_values
        ]

        max_patches = (tgt_sizes[:, 0] * tgt_sizes[:, 1]).max().item()
        assert isinstance(max_patches, int)

        all_pixel_values = torch.nn.utils.rnn.pad_sequence(
            all_pixel_values_lst, batch_first=True, padding_value=0.0)
        B, L, _ = all_pixel_values.shape
        all_pixel_values = all_pixel_values.permute(0, 2,
                                                    1).reshape(B, 3, -1, L)

        patch_attn_mask = torch.zeros((B, 1, max_patches),
                                      dtype=torch.bool,
                                      device=device)
        for i in range(B):
            patch_attn_mask[i, :tgt_sizes[i][0] * tgt_sizes[i][1]] = True

        return self.get_vision_embedding(all_pixel_values.type(dtype),
                                         patch_attn_mask, tgt_sizes)
=======
        return resampler.to(device=current_platform.device_type,
                            dtype=torch.get_default_dtype())

    def get_vision_hidden_states(
            self, data: MiniCPMVImagePixelInputs) -> torch.Tensor:
        pixel_values = data["pixel_values"]
        tgt_sizes = data["tgt_sizes"]

        B = len(pixel_values)
        P = pixel_values[0].shape[-2]
        L = max(item.shape[-1] for item in pixel_values)
        device = pixel_values[0].device
        dtype = pixel_values[0].dtype

        all_pixel_values = torch.zeros((B, 3, P, L),
                                       dtype=dtype,
                                       device=device)
        for i, pixel_values_item in enumerate(pixel_values):
            L_item = pixel_values_item.shape[-1]
            all_pixel_values[i, ..., :L_item] = pixel_values_item

        num_patches = tgt_sizes.prod(-1)
        max_patches = num_patches.max().item()
        assert isinstance(max_patches, int)

        patch_attn_mask = torch.zeros((B, max_patches),
                                      dtype=torch.bool,
                                      device=device)
        for i, num_patches_item in enumerate(num_patches):
            patch_attn_mask[i, :num_patches_item] = True

        vision_embedding = self.vpm(
            all_pixel_values,
            patch_attention_mask=patch_attn_mask.unsqueeze(1),
            tgt_sizes=None,
        )

        return self.resampler(vision_embedding, tgt_sizes)
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea


class MiniCPMV2_6(MiniCPMVBaseModel, SupportsLoRA):
    packed_modules_mapping = {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ],
        "gate_up_proj": [
            "gate_proj",
            "up_proj",
        ],
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        assert self.version == (2, 6)

    def init_llm(
        self,
        vllm_config: VllmConfig,
        prefix: str = "",
    ) -> nn.Module:
        return Qwen2ForCausalLM(vllm_config=vllm_config, prefix=prefix)

    def init_vision_module(
        self,
        config: PretrainedConfig,
<<<<<<< HEAD
        quant_config: Optional[QuantizationConfig],
=======
        quant_config: Optional[QuantizationConfig] = None,
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
        prefix: str = "",
    ) -> nn.Module:
        model = Idefics2VisionTransformer(config.vision_config,
                                          quant_config=quant_config,
                                          prefix=prefix)
        if self.config.drop_vision_last_layer:
            model.encoder.layers = model.encoder.layers[:-1]
        return model

    def init_resampler(self,
                       embed_dim: int,
                       vision_dim: int,
                       quant_config: Optional[QuantizationConfig] = None,
                       prefix: str = "") -> nn.Module:
        with set_default_torch_dtype(torch.float16):
            # The resampler in 2.6 remains consistent with the one in 2.5.
            resampler = Resampler2_5(num_queries=self.config.query_num,
                                     embed_dim=embed_dim,
                                     num_heads=embed_dim // 128,
                                     kv_dim=vision_dim,
                                     quant_config=quant_config,
                                     prefix=prefix)

<<<<<<< HEAD
        return resampler.to(device="cuda", dtype=torch.get_default_dtype())

    def get_vision_embedding(
        self,
        pixel_values: List[torch.Tensor],
        patch_attn_mask: Optional[torch.Tensor] = None,
        tgt_sizes: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        vision_embedding = self.vpm(
            pixel_values,
            patch_attention_mask=patch_attn_mask,
            tgt_sizes=tgt_sizes,
        )
        return vision_embedding

    def get_vision_hidden_states(self,
                                 data: MiniCPMVImageInputs) -> torch.Tensor:
        pixel_values = data["data"]
        tgt_sizes = data["tgt_sizes"]

        device = self.vpm.embeddings.position_embedding.weight.device
        dtype = self.vpm.embeddings.position_embedding.weight.dtype
        all_pixel_values_lst = [
            i.flatten(end_dim=1).permute(1, 0) for i in pixel_values
        ]

        max_patches = (tgt_sizes[:, 0] * tgt_sizes[:, 1]).max().item()
        assert isinstance(max_patches, int)

        all_pixel_values = torch.nn.utils.rnn.pad_sequence(
            all_pixel_values_lst, batch_first=True, padding_value=0.0)
        B, L, _ = all_pixel_values.shape
        all_pixel_values = all_pixel_values.permute(0, 2,
                                                    1).reshape(B, 3, -1, L)

        patch_attn_mask = torch.zeros((B, 1, max_patches),
                                      dtype=torch.bool,
                                      device=device)
        for i in range(B):
            patch_attn_mask[i, 0, :tgt_sizes[i][0] * tgt_sizes[i][1]] = True
        vision_embedding = self.vpm(
            all_pixel_values.type(dtype),
            patch_attention_mask=patch_attn_mask,
=======
        return resampler.to(device=current_platform.device_type,
                            dtype=torch.get_default_dtype())

    def get_vision_hidden_states(
            self, data: MiniCPMVImagePixelInputs) -> torch.Tensor:
        pixel_values = data["pixel_values"]
        tgt_sizes = data["tgt_sizes"]

        B = len(pixel_values)
        P = pixel_values[0].shape[-2]
        L = max(item.shape[-1] for item in pixel_values)
        device = pixel_values[0].device
        dtype = pixel_values[0].dtype

        all_pixel_values = torch.zeros((B, 3, P, L),
                                       dtype=dtype,
                                       device=device)
        for i, pixel_values_item in enumerate(pixel_values):
            L_item = pixel_values_item.shape[-1]
            all_pixel_values[i, ..., :L_item] = pixel_values_item

        num_patches = tgt_sizes.prod(-1)
        max_patches = num_patches.max().item()
        assert isinstance(max_patches, int)

        patch_attn_mask = torch.zeros((B, max_patches),
                                      dtype=torch.bool,
                                      device=device)
        for i, num_patches_item in enumerate(num_patches):
            patch_attn_mask[i, :num_patches_item] = True

        vision_embedding = self.vpm(
            all_pixel_values,
            patch_attention_mask=patch_attn_mask.unsqueeze(1),
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
            tgt_sizes=tgt_sizes,
        )

        return self.resampler(vision_embedding, tgt_sizes)


_SUPPORT_VERSION = {
    (2, 0): MiniCPMV2_0,
    (2, 5): MiniCPMV2_5,
    (2, 6): MiniCPMV2_6,
}


@MULTIMODAL_REGISTRY.register_processor(
    MiniCPMVMultiModalProcessor,
    info=MiniCPMVProcessingInfo,
    dummy_inputs=MiniCPMVDummyInputsBuilder)
class MiniCPMV(MiniCPMVBaseModel, SupportsMultiModal, SupportsLoRA):
    """
    Different versions of MiniCPMV use different visual encoders and LLMs,
    which is not conducive to the current integration logic of LoRA and
    bitsandbytes in vLLM. Therefore, it is necessary to separate them.
    """

    def __new__(cls, *, vllm_config: VllmConfig, prefix: str = ""):
        config = vllm_config.model_config.hf_config
        if not hasattr(config, "version"):
            if config.hidden_size == 2304 and config.query_num == 64:
                version = (2, 0)
            else:
                version = (2, 5)
        else:
            version = str(config.version).split(".")
            version = tuple([int(x) for x in version])
        # Dispatch class based on version
        instance_cls = _SUPPORT_VERSION.get(version)
        if instance_cls is None:
            raise ValueError(
                "Currently, MiniCPMV only supports versions 2.0, 2.5, and 2.6")

        # quant_config references base class members,
        # so update values before init is called
        cls.packed_modules_mapping.update(instance_cls.packed_modules_mapping)
        cls.embedding_modules.update(instance_cls.embedding_modules)
        cls.embedding_padding_modules += instance_cls.embedding_padding_modules
        return instance_cls(vllm_config=vllm_config, prefix=prefix)
