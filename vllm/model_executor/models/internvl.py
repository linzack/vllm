# SPDX-License-Identifier: Apache-2.0

# adapted from https://huggingface.co/OpenGVLab/InternVL2-4B/blob/main/modeling_internvl_chat.py
# --------------------------------------------------------
# InternVL
# Copyright (c) 2023 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
from abc import ABC, abstractmethod
<<<<<<< HEAD
from functools import cached_property
from typing import (Iterable, List, Literal, Mapping, Optional, Set, Tuple,
                    TypedDict, TypeVar, Union)
=======
from collections.abc import Iterable, Mapping, Sequence
from typing import Literal, Optional, TypedDict, TypeVar, Union
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea

import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
<<<<<<< HEAD
from transformers import BatchFeature, PretrainedConfig, TensorType

from vllm.attention import AttentionMetadata
from vllm.config import VllmConfig
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.quantization.awq import AWQConfig
from vllm.model_executor.layers.sampler import SamplerOutput, get_sampler
=======
from transformers import BatchEncoding, PretrainedConfig, TensorType

from vllm.config import VllmConfig
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.quantization.awq import AWQConfig
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
from vllm.model_executor.models.intern_vit import (InternVisionModel,
                                                   InternVisionPatchModel)
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.multimodal import MULTIMODAL_REGISTRY
<<<<<<< HEAD
from vllm.multimodal.inputs import (MultiModalFieldConfig, MultiModalKwargs,
                                    NestedTensors)
=======
from vllm.multimodal.inputs import (MultiModalDataDict, MultiModalFieldConfig,
                                    MultiModalKwargs, NestedTensors)
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
from vllm.multimodal.parse import (ImageEmbeddingItems, ImageProcessorItems,
                                   ImageSize, MultiModalDataItems)
from vllm.multimodal.processing import (BaseMultiModalProcessor,
                                        BaseProcessingInfo, PromptReplacement,
<<<<<<< HEAD
                                        PromptReplacementDetails)
from vllm.multimodal.profiling import BaseDummyInputsBuilder, ProcessorInputs
from vllm.sequence import IntermediateTensors
from vllm.transformers_utils.tokenizer import AnyTokenizer

from .interfaces import SupportsMultiModal, SupportsPP
=======
                                        PromptUpdate, PromptUpdateDetails)
from vllm.multimodal.profiling import BaseDummyInputsBuilder
from vllm.sequence import IntermediateTensors
from vllm.transformers_utils.tokenizer import AnyTokenizer

from .interfaces import MultiModalEmbeddings, SupportsMultiModal, SupportsPP
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
from .utils import (AutoWeightsLoader, flatten_bn, init_vllm_registered_model,
                    maybe_prefix, merge_multimodal_embeddings)

IMG_START = '<img>'
IMG_END = '</img>'
IMG_CONTEXT = '<IMG_CONTEXT>'

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

<<<<<<< HEAD
DDDDDDEEEEEBBBBUUUG = False
print(f"enter model_executor/models/internvl.py")

class InternVLImagePixelInputs(TypedDict):
    type: Literal["pixel_values"]
    data: torch.Tensor
=======

class InternVLImagePixelInputs(TypedDict):
    type: Literal["pixel_values"]
    pixel_values_flat: torch.Tensor
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
    """
    Shape:
    `(batch_size * num_images * (1 + num_patches), num_channels, height, width)`
    """
<<<<<<< HEAD
    patches_per_image: List[int]
    """
    List of number of total patches for each image in the batch.
    """
=======

    num_patches: torch.Tensor
    """Shape: `(batch_size * num_images)`"""
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea


class InternVLImageEmbeddingInputs(TypedDict):
    type: Literal["image_embeds"]
<<<<<<< HEAD
    data: NestedTensors
=======
    data: Union[torch.Tensor, list[torch.Tensor]]
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
    """ 
    A tensor of shape `(num_images, total_image_feature_size, hidden_size)`
    or a list of tensors of shape `(total_image_feature_size, hidden_size)`

    `hidden_size` must match the hidden size of language model backbone.
    """


InternVLImageInputs = Union[InternVLImagePixelInputs,
                            InternVLImageEmbeddingInputs]


# adapted from https://huggingface.co/OpenGVLab/InternVL2-1B
def build_transform(input_size: int):
<<<<<<< HEAD
    print(f"internvl build_transform() input_size: {input_size}")

=======
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    return T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size),
                 interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])


# adapted from https://huggingface.co/OpenGVLab/InternVL2-1B
def find_closest_aspect_ratio(
    aspect_ratio: float,
    target_ratios: list[tuple[int, int]],
    *,
    width: int,
    height: int,
    image_size: int,
) -> tuple[int, int]:
<<<<<<< HEAD
    global DDDDDDEEEEEBBBBUUUG
    if DDDDDDEEEEEBBBBUUUG:
        print(f"internvl find_closest_aspect_ratio() aspect_ratio: {aspect_ratio}, target_ratios: {target_ratios}")
        print(f"internvl find_closest_aspect_ratio() width: {width}, height: {height}, image_size: {image_size}")

=======
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
<<<<<<< HEAD
    if DDDDDDEEEEEBBBBUUUG:
        print(f"internvl find_closest_aspect_ratio() best_ratio: {best_ratio}")
=======
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
    return best_ratio


def resolve_internvl_min_max_num(
    *,
    min_dynamic_patch: int,
    max_dynamic_patch: int,
    dynamic_image_size: bool,
    use_thumbnail: bool,
) -> tuple[int, int]:
<<<<<<< HEAD
    global DDDDDDEEEEEBBBBUUUG
    if DDDDDDEEEEEBBBBUUUG:
        print(f"internvl resolve_internvl_min_max_num() min_dynamic_patch: {min_dynamic_patch}, max_dynamic_patch: {max_dynamic_patch}")
        print(f"internvl resolve_internvl_min_max_num() dynamic_image_size: {dynamic_image_size}, use_thumbnail: {use_thumbnail}")
=======
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
    min_dynamic_patch = min_dynamic_patch if dynamic_image_size else 1
    max_dynamic_patch = max_dynamic_patch if dynamic_image_size else 1

    if use_thumbnail and max_dynamic_patch != 1:
        max_dynamic_patch += 1

    return min_dynamic_patch, max_dynamic_patch


def get_internvl_target_ratios(
    min_num: int,
    max_num: int,
) -> list[tuple[int, int]]:
<<<<<<< HEAD
    #global DDDDDDEEEEEBBBBUUUG
    #if DDDDDDEEEEEBBBBUUUG:
    #    print(f"internvl get_internvl_target_ratios() min_num: {min_num}, max_num: {max_num}")
=======
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
    target_ratios = {(i, j)
                     for n in range(min_num, max_num + 1)
                     for i in range(1, n + 1)
                     for j in range(1, n + 1) if min_num <= i * j <= max_num}
    return sorted(target_ratios, key=lambda x: x[0] * x[1])


def calculate_internvl_targets(
    *,
    orig_width: int,
    orig_height: int,
    target_ratios: list[tuple[int, int]],
    image_size: int,
    use_thumbnail: bool,
) -> tuple[int, int, int]:
<<<<<<< HEAD
    global DDDDDDEEEEEBBBBUUUG
    if DDDDDDEEEEEBBBBUUUG:
        print(f"internvl calculate_internvl_targets()")
=======
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
    aspect_ratio = orig_width / orig_height

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio,
        target_ratios,
        width=orig_width,
        height=orig_height,
        image_size=image_size,
    )

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # add thumbnail image if num_blocks != 1
    if use_thumbnail and blocks != 1:
        blocks += 1

    return blocks, target_width, target_height


# adapted from https://huggingface.co/OpenGVLab/InternVL2-1B
def dynamic_preprocess_internvl(
    image: Image.Image,
    *,
    target_ratios: list[tuple[int, int]],
    image_size: int,
    use_thumbnail: bool,
) -> list[Image.Image]:
<<<<<<< HEAD
    global DDDDDDEEEEEBBBBUUUG
    if DDDDDDEEEEEBBBBUUUG:
        print(f"internvl dynamic_preprocess_internvl() image: {image}")
=======
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
    orig_width, orig_height = image.size

    # calculate the number of blocks without thumbnail
    blocks, target_width, target_height = calculate_internvl_targets(
        orig_width=orig_width,
        orig_height=orig_height,
        target_ratios=target_ratios,
        image_size=image_size,
        use_thumbnail=False,
    )

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = ((i % (target_width // image_size)) * image_size,
               (i // (target_width // image_size)) * image_size,
               ((i % (target_width // image_size)) + 1) * image_size,
               ((i // (target_width // image_size)) + 1) * image_size)
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)

    assert len(processed_images) == blocks

    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)

<<<<<<< HEAD
    if DDDDDDEEEEEBBBBUUUG:    
        print(f"internvl dynamic_preprocess_internvl() processed_images: {processed_images}")
=======
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
    return processed_images


# adapted from https://huggingface.co/OpenGVLab/InternVL2-1B
def image_to_pixel_values_internvl(
    image: Image.Image,
    *,
    input_size: int,
    min_num: int,
    max_num: int,
    use_thumbnail: bool,
) -> torch.Tensor:
<<<<<<< HEAD
    global DDDDDDEEEEEBBBBUUUG
    if DDDDDDEEEEEBBBBUUUG:
        print(f"internvl image_to_pixel_values_internvl()")
=======
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
    target_ratios = get_internvl_target_ratios(min_num, max_num)

    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess_internvl(
        image,
        target_ratios=target_ratios,
        image_size=input_size,
        use_thumbnail=use_thumbnail,
    )

    pixel_values = torch.stack([transform(image) for image in images])
<<<<<<< HEAD
    if DDDDDDEEEEEBBBBUUUG:    
        print(f"internvl image_to_pixel_values_internvl() pixel_values: {pixel_values}")
=======
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
    return pixel_values


class BaseInternVLProcessor(ABC):
    """
    This model doesn't define its own HF processor,
    so we implement our own one here.

    The code to insert image tokens is based on:
    https://huggingface.co/OpenGVLab/InternVL2-1B/blob/main/modeling_internvl_chat.py#L252
    """

    def __init__(
        self,
        config: PretrainedConfig,
        tokenizer: AnyTokenizer,
        *,
        min_dynamic_patch: Optional[int] = None,
        max_dynamic_patch: Optional[int] = None,
        dynamic_image_size: Optional[bool] = None,
    ) -> None:
        super().__init__()

        self.config = config
        self.tokenizer = tokenizer

        image_size: int = config.vision_config.image_size
        patch_size: int = config.vision_config.patch_size

        if min_dynamic_patch is None:
            min_dynamic_patch = config.min_dynamic_patch
        assert isinstance(min_dynamic_patch, int)

        if max_dynamic_patch is None:
            max_dynamic_patch = config.max_dynamic_patch
        assert isinstance(max_dynamic_patch, int)

        if dynamic_image_size is None:
            dynamic_image_size = config.dynamic_image_size
        assert isinstance(dynamic_image_size, bool)

        self.num_image_token = int(
            (image_size // patch_size)**2 * (config.downsample_ratio**2))
        self.image_size = image_size
        self.min_dynamic_patch = min_dynamic_patch
        self.max_dynamic_patch = max_dynamic_patch
        self.dynamic_image_size = dynamic_image_size
        self.use_thumbnail: bool = config.use_thumbnail

    @property
    @abstractmethod
    def image_token_id(self) -> int:
        raise NotImplementedError

    @abstractmethod
<<<<<<< HEAD
    def get_image_repl_features(
        self,
        feature_size: int,
        num_patches: Optional[int],
    ) -> str:
        raise NotImplementedError

    @abstractmethod
    def get_image_repl_full(
        self,
        feature_size: int,
        num_patches: Optional[int],
    ) -> str:
=======
    def get_image_repl(
        self,
        feature_size: int,
        num_patches: Optional[int],
    ) -> PromptUpdateDetails[str]:
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
        raise NotImplementedError

    def resolve_min_max_num(
        self,
        *,
        min_dynamic_patch: Optional[int] = None,
        max_dynamic_patch: Optional[int] = None,
        dynamic_image_size: Optional[bool] = None,
        use_thumbnail: Optional[bool] = None,
    ) -> tuple[int, int]:
        min_dynamic_patch = (self.min_dynamic_patch if min_dynamic_patch
                             is None else min_dynamic_patch)
        max_dynamic_patch = (self.max_dynamic_patch if max_dynamic_patch
                             is None else max_dynamic_patch)
        dynamic_image_size = (self.dynamic_image_size if dynamic_image_size
                              is None else dynamic_image_size)
        use_thumbnail = (self.use_thumbnail
                         if use_thumbnail is None else use_thumbnail)

        return resolve_internvl_min_max_num(
            min_dynamic_patch=min_dynamic_patch,
            max_dynamic_patch=max_dynamic_patch,
            dynamic_image_size=dynamic_image_size,
            use_thumbnail=use_thumbnail,
        )

    def resolve_target_ratios(
        self,
        *,
        min_dynamic_patch: Optional[int] = None,
        max_dynamic_patch: Optional[int] = None,
        dynamic_image_size: Optional[bool] = None,
        use_thumbnail: Optional[bool] = None,
    ) -> list[tuple[int, int]]:
        min_num, max_num = self.resolve_min_max_num(
            min_dynamic_patch=min_dynamic_patch,
            max_dynamic_patch=max_dynamic_patch,
            dynamic_image_size=dynamic_image_size,
            use_thumbnail=use_thumbnail,
        )

        return get_internvl_target_ratios(min_num, max_num)

    def get_num_image_tokens(
        self,
        *,
        image_width: int,
        image_height: int,
    ) -> int:
        target_ratios = self.resolve_target_ratios(
            use_thumbnail=False,  # Applied in calculate_targets
        )

        num_patches, _, _ = calculate_internvl_targets(
            orig_width=image_width,
            orig_height=image_height,
            image_size=self.image_size,
            target_ratios=target_ratios,
            use_thumbnail=self.use_thumbnail,
        )

        return num_patches * self.num_image_token

    def _images_to_pixel_values_lst(
        self,
        images: list[Image.Image],
        min_dynamic_patch: Optional[int] = None,
        max_dynamic_patch: Optional[int] = None,
        dynamic_image_size: Optional[bool] = None,
    ) -> list[torch.Tensor]:
        min_num, max_num = self.resolve_min_max_num(
            min_dynamic_patch=min_dynamic_patch,
            max_dynamic_patch=max_dynamic_patch,
            dynamic_image_size=dynamic_image_size,
            use_thumbnail=False,  # Applied in image_to_pixel_values
        )

        return [
            image_to_pixel_values_internvl(
                image,
                input_size=self.image_size,
                min_num=min_num,
                max_num=max_num,
                use_thumbnail=self.use_thumbnail,
            ) for image in images
        ]

    def __call__(
        self,
        text: Optional[Union[str, list[str]]] = None,
        images: Optional[Union[Image.Image, list[Image.Image]]] = None,
        min_dynamic_patch: Optional[int] = None,
        max_dynamic_patch: Optional[int] = None,
        dynamic_image_size: Optional[bool] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
<<<<<<< HEAD
    ) -> BatchFeature:
=======
    ) -> Mapping[str, NestedTensors]:
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
        if text is None:
            text = []
        if not isinstance(text, list):
            text = [text]
        if images is None:
            images = []
        if not isinstance(images, list):
            images = [images]

        if len(images) == 0:
            image_inputs = {}
        else:
            pixel_values_lst = self._images_to_pixel_values_lst(
                images,
                min_dynamic_patch=min_dynamic_patch,
                max_dynamic_patch=max_dynamic_patch,
                dynamic_image_size=dynamic_image_size,
            )
<<<<<<< HEAD
            image_inputs = {
                "pixel_values_flat": torch.cat(pixel_values_lst),
                "image_num_patches": list(map(len, pixel_values_lst)),
=======
            image_inputs: dict[str, NestedTensors] = {
                "pixel_values_flat":
                torch.cat(pixel_values_lst),
                "image_num_patches":
                torch.tensor([len(item) for item in pixel_values_lst]),
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
            }

            for pixel_values in pixel_values_lst:
                num_patches = pixel_values.shape[0]
                feature_size = num_patches * self.num_image_token

<<<<<<< HEAD
                image_repl = self.get_image_repl_full(feature_size,
                                                      num_patches)
                text = [t.replace('<image>', image_repl, 1) for t in text]

        text_inputs = self.tokenizer(text)

        return BatchFeature(
            {
                **text_inputs,
                **image_inputs,
            },
            tensor_type=return_tensors,
        )
=======
                image_repl = self.get_image_repl(feature_size, num_patches)
                text = [t.replace('<image>', image_repl.full, 1) for t in text]

        text_inputs = self.tokenizer(text)

        return {
            **BatchEncoding(text_inputs, tensor_type=return_tensors),
            **image_inputs,
        }
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea


class InternVLProcessor(BaseInternVLProcessor):

    @property
    def image_token_id(self) -> int:
        return self.tokenizer.get_vocab()[IMG_CONTEXT]

<<<<<<< HEAD
    def get_image_repl_features(
        self,
        feature_size: int,
        num_patches: Optional[int],
    ) -> str:
        return IMG_CONTEXT * feature_size

    def get_image_repl_full(
        self,
        feature_size: int,
        num_patches: Optional[int],
    ) -> str:
        features = self.get_image_repl_features(feature_size, num_patches)
        return IMG_START + features + IMG_END
=======
    def get_image_repl(
        self,
        feature_size: int,
        num_patches: Optional[int],
    ) -> PromptUpdateDetails[str]:
        repl_features = IMG_CONTEXT * feature_size
        repl_full = IMG_START + repl_features + IMG_END

        return PromptUpdateDetails.select_text(repl_full, IMG_CONTEXT)
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea


class BaseInternVLProcessingInfo(BaseProcessingInfo):

    @abstractmethod
    def get_hf_processor(
        self,
        *,
        min_dynamic_patch: Optional[int] = None,
        max_dynamic_patch: Optional[int] = None,
        dynamic_image_size: Optional[bool] = None,
        **kwargs: object,
    ) -> BaseInternVLProcessor:
        raise NotImplementedError

    def get_supported_mm_limits(self) -> Mapping[str, Optional[int]]:
        return {"image": None}

<<<<<<< HEAD
    def get_mm_max_tokens_per_item(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> Mapping[str, int]:
        return {"image": self.get_max_image_tokens()}

=======
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
    def get_num_image_tokens(
        self,
        *,
        image_width: int,
        image_height: int,
        processor: Optional[BaseInternVLProcessor],
    ) -> int:
        if processor is None:
            processor = self.get_hf_processor()

        return processor.get_num_image_tokens(
            image_width=image_width,
            image_height=image_height,
        )

<<<<<<< HEAD
    def get_max_image_tokens(self) -> int:
        target_width, target_height = self.get_image_size_with_most_features()

        return self.get_num_image_tokens(
            image_width=target_width,
            image_height=target_height,
            processor=None,
        )

=======
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
    def get_image_size_with_most_features(self) -> ImageSize:
        processor = self.get_hf_processor()

        base_size = processor.image_size
        target_ratios = processor.resolve_target_ratios()

        largest_feature_size, largest_feature_pinpoint = 0, None
        for wr, hr in target_ratios:
            width, height = base_size * wr, base_size * hr

            feat_size = self.get_num_image_tokens(
                image_width=width,
                image_height=height,
                processor=processor,
            )
            if feat_size > largest_feature_size:
                largest_feature_size = feat_size
                largest_feature_pinpoint = ImageSize(width=width,
                                                     height=height)

        if largest_feature_size == 0 or largest_feature_pinpoint is None:
            raise ValueError("Cannot have a largest feature size of 0!")

        return largest_feature_pinpoint


_I = TypeVar("_I", bound=BaseInternVLProcessingInfo)


class InternVLDummyInputsBuilder(BaseDummyInputsBuilder[_I]):

<<<<<<< HEAD
    def get_dummy_processor_inputs(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> ProcessorInputs:
=======
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        num_images = mm_counts.get("image", 0)

        return "<image>" * num_images

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> MultiModalDataDict:
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
        target_width, target_height = \
            self.info.get_image_size_with_most_features()
        num_images = mm_counts.get("image", 0)

<<<<<<< HEAD
        mm_data = {
=======
        return {
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
            "image":
            self._get_dummy_images(width=target_width,
                                   height=target_height,
                                   num_images=num_images)
        }

<<<<<<< HEAD
        return ProcessorInputs(
            prompt_text="<image>" * num_images,
            mm_data=mm_data,
        )

=======
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea

class InternVLMultiModalProcessor(BaseMultiModalProcessor[_I]):

    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
<<<<<<< HEAD
    ) -> BatchFeature:
        global DDDDDDEEEEEBBBBUUUG
        if DDDDDDEEEEEBBBBUUUG:
            print(f"InternVLMultiModalProcessor _call_hf_processor() prompt: {prompt}, mm_data: {mm_data}, mm_kwargs: {mm_kwargs}")
=======
    ) -> Mapping[str, NestedTensors]:
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
        processed_outputs = super()._call_hf_processor(
            prompt=prompt,
            mm_data=mm_data,
            mm_kwargs=mm_kwargs,
        )

<<<<<<< HEAD
        image_token_id = self.info.get_hf_processor(**mm_kwargs).image_token_id
        image_data = mm_data.get("images", [])
        if DDDDDDEEEEEBBBBUUUG:
            print(f"InternVLMultiModalProcessor _call_hf_processor() processed_outputs: {processed_outputs}, image_token_id: {image_token_id}, image_data: {image_data}")
        assert isinstance(image_data, list)
=======
        hf_processor = self.info.get_hf_processor(**mm_kwargs)
        image_token_id = hf_processor.image_token_id
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea

        # Since there may be extra tokens in the feature placeholders,
        # we need to pass the image token ID to the model to select the
        # tokens to merge from the vision encoder outputs
        processed_outputs["image_token_id"] = torch.tensor(image_token_id)
<<<<<<< HEAD
        if DDDDDDEEEEEBBBBUUUG:
            print(f"InternVLMultiModalProcessor _call_hf_processor() processed_outputs: {processed_outputs}")
        
=======

>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
        return processed_outputs

    def _get_mm_fields_config(
        self,
<<<<<<< HEAD
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        global DDDDDDEEEEEBBBBUUUG
        if DDDDDDEEEEEBBBBUUUG:
            print(f"InternVLMultiModalProcessor _get_mm_fields_config() hf_inputs: {hf_inputs}, hf_processor_mm_kwargs: {hf_processor_mm_kwargs}")

        image_num_patches = hf_inputs.get("image_num_patches", torch.empty(0))
        num_images = len(image_num_patches)
        if DDDDDDEEEEEBBBBUUUG:
            print(f"InternVLMultiModalProcessor _get_mm_fields_config() image_num_patches: {image_num_patches}, num_images: {num_images}")
=======
        hf_inputs: Mapping[str, NestedTensors],
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        image_num_patches = hf_inputs.get("image_num_patches", torch.empty(0))
        num_images = len(image_num_patches)
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea

        return dict(
            pixel_values_flat=MultiModalFieldConfig.flat_from_sizes(
                "image", image_num_patches),
            image_num_patches=MultiModalFieldConfig.batched("image"),
            image_embeds=MultiModalFieldConfig.batched("image"),
            image_token_id=MultiModalFieldConfig.shared("image", num_images),
        )

<<<<<<< HEAD
    def _get_prompt_replacements(
=======
    def _get_prompt_updates(
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargs,
<<<<<<< HEAD
    ) -> list[PromptReplacement]:
        global DDDDDDEEEEEBBBBUUUG
        #if DDDDDDEEEEEBBBBUUUG:
        print(f"InternVLMultiModalProcessor _get_prompt_replacements() mm_items: {mm_items}, hf_processor_mm_kwargs: {hf_processor_mm_kwargs}, out_mm_kwargs: {out_mm_kwargs}")
        
=======
    ) -> Sequence[PromptUpdate]:
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
        hf_processor = self.info.get_hf_processor(**hf_processor_mm_kwargs)

        if "image_num_patches" in out_mm_kwargs:
            image_num_patches = out_mm_kwargs["image_num_patches"]
            assert isinstance(image_num_patches, torch.Tensor)
            image_num_patches = image_num_patches.tolist()
<<<<<<< HEAD
            #if DDDDDDEEEEEBBBBUUUG:
            print(f"InternVLMultiModalProcessor _get_prompt_replacements() image_num_patches: {image_num_patches}")
=======
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
        elif "image_embeds" in out_mm_kwargs:
            # TODO: Use image size information in dictionary embedding inputs
            # to compute num_patches (similar to Qwen2-VL)
            image_num_patches = [None] * len(out_mm_kwargs["image_embeds"])
<<<<<<< HEAD
            #if DDDDDDEEEEEBBBBUUUG:
            print(f"InternVLMultiModalProcessor _get_prompt_replacements() image_embeds image_num_patches: {image_num_patches}")
        else:
            image_num_patches = []
            #if DDDDDDEEEEEBBBBUUUG:
            print(f"InternVLMultiModalProcessor _get_prompt_replacements() else image_num_patches: {image_num_patches}")

        def get_replacement_internvl(item_idx: int):
            #if DDDDDDEEEEEBBBBUUUG:
            print(f"InternVLMultiModalProcessor get_replacement_internvl() item_idx: {item_idx}")
            images = mm_items.get_items(
                "image", (ImageEmbeddingItems, ImageProcessorItems))
            #if DDDDDDEEEEEBBBBUUUG:
            print(f"InternVLMultiModalProcessor get_replacement_internvl() images: {images}")
=======
        else:
            image_num_patches = []

        def get_replacement_internvl(item_idx: int):
            images = mm_items.get_items(
                "image", (ImageEmbeddingItems, ImageProcessorItems))
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea

            if isinstance(images, ImageEmbeddingItems):
                feature_size = images.get_feature_size(item_idx)
            else:
                image_size = images.get_image_size(item_idx)
                feature_size = self.info.get_num_image_tokens(
                    image_width=image_size.width,
                    image_height=image_size.height,
                    processor=hf_processor,
                )

            num_patches = image_num_patches[item_idx]
            if num_patches is not None:
                assert isinstance(num_patches, int)

<<<<<<< HEAD
            print(f"InternVLMultiModalProcessor get_replacement_internvl() feature_size: {feature_size}, num_patches: {num_patches}")
            return PromptReplacementDetails(
                full=hf_processor.get_image_repl_full(feature_size,
                                                      num_patches),
                features=hf_processor.get_image_repl_features(
                    feature_size, num_patches),
            )
=======
            return hf_processor.get_image_repl(feature_size, num_patches)
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea

        return [
            PromptReplacement(
                modality="image",
<<<<<<< HEAD
                ############zack target="<image>",
                target="<IMG_CONTEXT>", #zack
=======
                target="<image>",
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
                replacement=get_replacement_internvl,
            )
        ]


class InternVLProcessingInfo(BaseInternVLProcessingInfo):

    def get_hf_processor(
        self,
        *,
        min_dynamic_patch: Optional[int] = None,
        max_dynamic_patch: Optional[int] = None,
        dynamic_image_size: Optional[bool] = None,
        **kwargs: object,
    ) -> InternVLProcessor:
        if min_dynamic_patch is not None:
            kwargs["min_dynamic_patch"] = min_dynamic_patch
        if max_dynamic_patch is not None:
            kwargs["max_dynamic_patch"] = max_dynamic_patch
        if dynamic_image_size is not None:
            kwargs["dynamic_image_size"] = dynamic_image_size

        return self.ctx.init_processor(
            InternVLProcessor,
            config=self.get_hf_config(),
            tokenizer=self.get_tokenizer(),
            **kwargs,
        )


@MULTIMODAL_REGISTRY.register_processor(
    InternVLMultiModalProcessor,
    info=InternVLProcessingInfo,
    dummy_inputs=InternVLDummyInputsBuilder)
class InternVLChatModel(nn.Module, SupportsMultiModal, SupportsPP):

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()

        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        multimodal_config = vllm_config.model_config.multimodal_config

        self.config = config
        self.multimodal_config = multimodal_config
        self._patch_quant_config(config, quant_config)

        image_size = config.force_image_size or config.vision_config.image_size
        patch_size = config.vision_config.patch_size
        self.patch_size = patch_size
        self.num_image_token = int(
            (image_size // patch_size)**2 * (config.downsample_ratio**2))
        self.downsample_ratio = config.downsample_ratio
        self.ps_version = config.ps_version

        self.llm_arch_name = config.text_config.architectures[0]
        self.is_mono = self.llm_arch_name == 'InternLM2VEForCausalLM'
        self.vision_model = self._init_vision_model(
            config,
            quant_config=quant_config,
            is_mono=self.is_mono,
            prefix=maybe_prefix(prefix, "vision_model"),
        )

        self.language_model = init_vllm_registered_model(
            vllm_config=vllm_config,
            hf_config=config.text_config,
            prefix=maybe_prefix(prefix, "language_model"),
        )

        self.mlp1 = self._init_mlp1(config)

        self.img_context_token_id = None
        self.visual_token_mask = None
        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors)

    def _patch_quant_config(self, config: PretrainedConfig,
                            quant_config: QuantizationConfig):
        # the awq models from OpenGVLab missing `modules_to_not_convert`
        # patch the quant_config to add `modules_to_not_convert` back
        if isinstance(quant_config, AWQConfig):
            text_config = config.text_config
            llm_quant_config = getattr(text_config, "quantization_config",
                                       None)
            if (not quant_config.modules_to_not_convert) and \
                (llm_quant_config is not None):
                quant_config.modules_to_not_convert.append("vision_model")

<<<<<<< HEAD
    @cached_property
    def sampler(self):
        if hasattr(self.language_model, "sampler"):
            return self.language_model.sampler

        return get_sampler()

=======
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
    def _init_vision_model(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig],
        *,
        is_mono: bool,
        prefix: str,
    ):
        if not is_mono:
            vision_feature_layer = config.select_layer
            if vision_feature_layer < 0:
                num_hidden_layers = config.vision_config.num_hidden_layers \
                    + vision_feature_layer + 1
            else:
                num_hidden_layers = vision_feature_layer + 1

            return InternVisionModel(
                config.vision_config,
                quant_config=quant_config,
                num_hidden_layers_override=num_hidden_layers,
                prefix=prefix,
            )
        else:
            return InternVisionPatchModel(config.vision_config)

    def _init_mlp1(self, config: PretrainedConfig) -> nn.Sequential:
        vit_hidden_size = config.vision_config.hidden_size
        llm_hidden_size = config.text_config.hidden_size

        return nn.Sequential(
            nn.LayerNorm(vit_hidden_size * int(1 / self.downsample_ratio)**2),
            nn.Linear(vit_hidden_size * int(1 / self.downsample_ratio)**2,
                      llm_hidden_size),
            nn.GELU(),
            nn.Linear(llm_hidden_size, llm_hidden_size),
        )

    def pixel_shuffle(self, x, scale_factor=0.5):
        n, w, h, c = x.size()
        # N, W, H, C --> N, W, H * scale, C // scale
        x = x.view(n, w, int(h * scale_factor), int(c / scale_factor))
        # N, W, H * scale, C // scale --> N, H * scale, W, C // scale
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(n, int(h * scale_factor), int(w * scale_factor),
                   int(c / (scale_factor * scale_factor)))
        if self.ps_version == 'v1':
            pass
        else:
            x = x.permute(0, 2, 1, 3).contiguous()
        return x

    def extract_feature(self, pixel_values: torch.Tensor) -> torch.Tensor:
        vit_embeds = self.vision_model(pixel_values=pixel_values)
        vit_embeds = vit_embeds[:, 1:, :]

        h = w = int(vit_embeds.shape[1]**0.5)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], h, w, -1)
        vit_embeds = self.pixel_shuffle(vit_embeds,
                                        scale_factor=self.downsample_ratio)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], -1,
                                        vit_embeds.shape[-1])
        vit_embeds = self.mlp1(vit_embeds)
        return vit_embeds

    def _validate_pixel_values(self, data: torch.Tensor) -> torch.Tensor:

        h = w = self.config.vision_config.image_size
        expected_dims = (3, h, w)

        def _validate_shape(d: torch.Tensor):
            actual_dims = tuple(d.shape)

            if actual_dims != expected_dims:
                expected_expr = str(expected_dims)
                raise ValueError(
                    "The expected shape of pixel values per image per batch "
                    f" per patch is {expected_expr}. "
                    f"You supplied {tuple(d.shape)}.")

        for d in data:
            _validate_shape(d)

        return data

    def _parse_and_validate_image_input(
            self, **kwargs: object) -> Optional[InternVLImageInputs]:
<<<<<<< HEAD
        print(f"InternVLChatModel _parse_and_validate_image_input()")
        pixel_values_flat = kwargs.pop("pixel_values_flat", None)
        image_num_patches = kwargs.pop("image_num_patches", None)
        image_embeds = kwargs.pop("image_embeds", None)
        global DDDDDDEEEEEBBBBUUUG
        #if DDDDDDEEEEEBBBBUUUG:
        print(f"InternVLChatModel _parse_and_validate_image_input() pixel_values_flat: {pixel_values_flat}, image_num_patches: {image_num_patches}, image_embeds: {image_embeds}")
=======
        pixel_values_flat = kwargs.pop("pixel_values_flat", None)
        image_num_patches = kwargs.pop("image_num_patches", None)
        image_embeds = kwargs.pop("image_embeds", None)

>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
        if pixel_values_flat is None and image_embeds is None:
            return None

        if image_embeds is not None:
<<<<<<< HEAD
            if not isinstance(image_embeds, torch.Tensor):
=======
            if not isinstance(image_embeds, (torch.Tensor, list)):
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
                raise ValueError("Incorrect type of image embeddings. "
                                 f"Got type: {type(image_embeds)}")

            return InternVLImageEmbeddingInputs(
                type="image_embeds",
                data=flatten_bn(image_embeds),
            )

        image_token_id = kwargs["image_token_id"]
<<<<<<< HEAD
        if DDDDDDEEEEEBBBBUUUG:
            print(f"InternVLChatModel _parse_and_validate_image_input() image_token_id: {image_token_id}")
=======
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
        assert isinstance(image_token_id, torch.Tensor)
        self.img_context_token_id = image_token_id.flatten().unique().item()

        if pixel_values_flat is not None:
            if not isinstance(pixel_values_flat, (torch.Tensor, list)):
                raise ValueError("Incorrect type of pixel values. "
                                 f"Got type: {type(pixel_values_flat)}")

<<<<<<< HEAD
            assert isinstance(image_num_patches, (torch.Tensor, list))

            return InternVLImagePixelInputs(
                type="pixel_values",
                data=self._validate_pixel_values(
                    flatten_bn(pixel_values_flat, concat=True)),
                patches_per_image=flatten_bn(image_num_patches,
                                             concat=True).tolist())
=======
            if not isinstance(image_num_patches, (torch.Tensor, list)):
                raise ValueError("Incorrect type of image_num_patches. "
                                 f"Got type: {type(image_num_patches)}")

            pixel_values_flat = flatten_bn(pixel_values_flat, concat=True)
            image_num_patches = flatten_bn(image_num_patches, concat=True)

            return InternVLImagePixelInputs(
                type="pixel_values",
                pixel_values_flat=self._validate_pixel_values(
                    pixel_values_flat),
                num_patches=image_num_patches,
            )
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea

        raise AssertionError("This line should be unreachable.")

    def _process_image_input(
        self,
        image_input: InternVLImageInputs,
<<<<<<< HEAD
    ) -> tuple[torch.Tensor, ...]:
        global DDDDDDEEEEEBBBBUUUG
        if DDDDDDEEEEEBBBBUUUG:
            print(f"InternVLChatModel _process_image_input() image_input: {image_input}")
=======
    ) -> Union[torch.Tensor, list[torch.Tensor], tuple[torch.Tensor, ...]]:
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
        if image_input["type"] == "image_embeds":
            return image_input["data"]

        assert self.vision_model is not None

<<<<<<< HEAD
        image_embeds = self.extract_feature(image_input["data"])

        patches_per_image = image_input["patches_per_image"]

        # Only one image in the current batch
        if len(patches_per_image) == 1:
            image_embeds = image_embeds.view(
                -1, self.config.text_config.hidden_size).unsqueeze(0)
            return image_embeds
=======
        image_embeds = self.extract_feature(image_input["pixel_values_flat"])

        num_patches = image_input["num_patches"]

        # Only one image in the current batch
        if len(num_patches) == 1:
            return image_embeds.view(
                -1, self.config.text_config.hidden_size).unsqueeze(0)
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea

        # NOTE: Image embeddings are split into separate tensors for each image
        # by the size of each embedding.
        feature_size = image_embeds.shape[1]
        image_embeds = image_embeds.view(-1,
                                         self.config.text_config.hidden_size)
        image_feature_sizes = [
<<<<<<< HEAD
            num_patches * feature_size for num_patches in patches_per_image
        ]
        image_embeds = image_embeds.split(image_feature_sizes)
        if DDDDDDEEEEEBBBBUUUG:
            print(f"InternVLChatModel _process_image_input() image_embeds: {image_embeds}")
        return image_embeds
=======
            num_patches * feature_size for num_patches in num_patches
        ]
        return image_embeds.split(image_feature_sizes)
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea

    def _set_visual_token_mask(self, input_ids: torch.Tensor) -> None:
        if self.is_mono:
            self.visual_token_mask = (
                input_ids == self.img_context_token_id).reshape(-1, 1)
        else:
            self.visual_token_mask = None

<<<<<<< HEAD
    def get_multimodal_embeddings(self, **kwargs) -> Optional[NestedTensors]:
        image_input = self._parse_and_validate_image_input(**kwargs)
        if image_input is None:
            return None
        vision_embeddings = self._process_image_input(image_input)
        return vision_embeddings
=======
    def get_language_model(self) -> torch.nn.Module:
        return self.language_model

    def get_multimodal_embeddings(
            self, **kwargs: object) -> Optional[MultiModalEmbeddings]:
        image_input = self._parse_and_validate_image_input(**kwargs)
        if image_input is None:
            return None

        return self._process_image_input(image_input)
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea

    def get_input_embeddings(
        self,
        input_ids: torch.Tensor,
<<<<<<< HEAD
        multimodal_embeddings: Optional[NestedTensors] = None,
    ) -> torch.Tensor:
        global DDDDDDEEEEEBBBBUUUG
        if DDDDDDEEEEEBBBBUUUG:
            print(f"InternVLChatModel get_input_embeddings() input_ids: {input_ids}, multimodal_embeddings: {multimodal_embeddings}")
=======
        multimodal_embeddings: Optional[MultiModalEmbeddings] = None,
    ) -> torch.Tensor:
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
        inputs_embeds = self.language_model.get_input_embeddings(input_ids)
        if multimodal_embeddings is not None:
            assert self.img_context_token_id is not None
            self._set_visual_token_mask(input_ids)
            inputs_embeds = merge_multimodal_embeddings(
<<<<<<< HEAD
                input_ids, inputs_embeds, multimodal_embeddings,
                self.img_context_token_id)
=======
                input_ids,
                inputs_embeds,
                multimodal_embeddings,
                self.img_context_token_id,
            )
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
        return inputs_embeds

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
<<<<<<< HEAD
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs: object,
    ) -> Union[SamplerOutput, IntermediateTensors]:
        global DDDDDDEEEEEBBBBUUUG
        if DDDDDDEEEEEBBBBUUUG:
            print(f"InternVLChatModel forward() input_ids: {input_ids}, positions: {positions}")
            print(f"InternVLChatModel forward() kv_caches: {kv_caches}, attn_metadata: {attn_metadata}")
            print(f"InternVLChatModel forward() intermediate_tensors: {intermediate_tensors}, inputs_embeds: {inputs_embeds}")
=======
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs: object,
    ) -> IntermediateTensors:
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea

        if intermediate_tensors is not None:
            input_ids = None
            inputs_embeds = None

        # NOTE: In v1, inputs_embeds is always generated at model runner, this
        # condition is for v0 compatibility.
        elif inputs_embeds is None:
            vision_embeddings = self.get_multimodal_embeddings(**kwargs)
            inputs_embeds = self.get_input_embeddings(input_ids,
                                                      vision_embeddings)
            input_ids = None

        forward_kwargs = {
            "input_ids": input_ids,
            "positions": positions,
<<<<<<< HEAD
            "kv_caches": kv_caches,
            "attn_metadata": attn_metadata,
=======
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
            "intermediate_tensors": intermediate_tensors,
            "inputs_embeds": inputs_embeds,
        }

        # Only required if the model is mono-architecture
        if self.visual_token_mask is not None:
            forward_kwargs.update(
                {"visual_token_mask": self.visual_token_mask})
            self.visual_token_mask = None

        hidden_states = self.language_model.model(**forward_kwargs)
<<<<<<< HEAD
        if DDDDDDEEEEEBBBBUUUG:
            print(f"InternVLChatModel forward() hidden_states: {hidden_states}")

=======
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[torch.Tensor]:
        return self.language_model.compute_logits(hidden_states,
                                                  sampling_metadata)

<<<<<<< HEAD
    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        return self.language_model.sample(logits, sampling_metadata)

    def load_weights(self, weights: Iterable[Tuple[str,
                                                   torch.Tensor]]) -> Set[str]:
        loader = AutoWeightsLoader(self)
=======
    def load_weights(self, weights: Iterable[tuple[str,
                                                   torch.Tensor]]) -> set[str]:
        # unused modules appear in OpenGVLab/InternVideo2_5_Chat_8B
        skip_prefixes = [
            "action_embed", "temporal_embed", "track_embed",
            "track_embed_decoder", "box_token", "cg_criterion", "cg_model",
            "loc_encoder", "loc_decoder", "sam", "temporal_token",
            "track_token"
        ]
        loader = AutoWeightsLoader(self, skip_prefixes=skip_prefixes)
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
        return loader.load_weights(weights)
