# SPDX-License-Identifier: Apache-2.0

import math
<<<<<<< HEAD
from dataclasses import dataclass, fields
from functools import cached_property
from typing import Iterable, List, Mapping, Optional, Set, Tuple, Union
=======
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, fields
from functools import cached_property
from typing import Literal, Optional, TypedDict, Union
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea

import torch
import torch.nn as nn
import torch.nn.functional as F
from mistral_common.protocol.instruct.messages import ImageChunk
<<<<<<< HEAD
from PIL import Image
from transformers import PixtralVisionConfig
=======
from mistral_common.tokens.tokenizers.multimodal import ImageEncoder
from PIL import Image
from transformers import PixtralVisionConfig, TensorType
from transformers.image_utils import ImageInput
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
from transformers.models.pixtral.image_processing_pixtral import (
    _num_image_tokens as _get_pixtral_hf_num_image_tokens)
from transformers.models.pixtral.modeling_pixtral import (
    PixtralRotaryEmbedding, apply_rotary_pos_emb, position_ids_in_meshgrid)
<<<<<<< HEAD

from vllm.attention import AttentionMetadata
from vllm.config import VllmConfig
from vllm.distributed import divide, get_tensor_model_parallel_world_size
from vllm.inputs import (INPUT_REGISTRY, DecoderOnlyInputs, DummyData,
                         InputContext, token_inputs)
=======
from transformers.tokenization_utils_base import TextInput

from vllm.config import VllmConfig
from vllm.distributed import divide, get_tensor_model_parallel_world_size
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
from vllm.model_executor.layers.activation import get_act_and_mul_fn
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (MergedColumnParallelLinear,
                                               QKVParallelLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.quantization import QuantizationConfig
<<<<<<< HEAD
from vllm.model_executor.layers.sampler import SamplerOutput, get_sampler
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.multimodal import MULTIMODAL_REGISTRY, MultiModalKwargs
from vllm.multimodal.inputs import NestedTensors, PlaceholderRange
from vllm.multimodal.utils import consecutive_placeholder_ranges
from vllm.sequence import IntermediateTensors, SequenceData
from vllm.transformers_utils.tokenizer import cached_tokenizer_from_config

from .interfaces import SupportsMultiModal, SupportsPP
from .utils import (init_vllm_registered_model, maybe_prefix,
=======
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.multimodal import MULTIMODAL_REGISTRY, MultiModalKwargs
from vllm.multimodal.inputs import (MultiModalDataDict, MultiModalFieldConfig,
                                    NestedTensors)
from vllm.multimodal.parse import (ImageProcessorItems, ImageSize,
                                   MultiModalDataItems)
from vllm.multimodal.processing import (BaseMultiModalProcessor,
                                        BaseProcessingInfo, MultiModalHashes,
                                        PromptReplacement, PromptUpdate,
                                        PromptUpdateDetails)
from vllm.multimodal.profiling import BaseDummyInputsBuilder
from vllm.sequence import IntermediateTensors
from vllm.transformers_utils.tokenizer import (MistralTokenizer,
                                               cached_tokenizer_from_config)

from .interfaces import MultiModalEmbeddings, SupportsMultiModal, SupportsPP
from .utils import (flatten_bn, init_vllm_registered_model, maybe_prefix,
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
                    merge_multimodal_embeddings)
from .vision import VisionEncoderInfo, resolve_visual_encoder_outputs

try:
    from xformers import ops as xops
    USE_XFORMERS_OPS = True
except ImportError:
    USE_XFORMERS_OPS = False

<<<<<<< HEAD

def get_max_pixtral_image_tokens(ctx: InputContext):
    tokenizer = cached_tokenizer_from_config(ctx.model_config)
    mm_encoder = tokenizer.instruct.mm_encoder

    image_config = mm_encoder.mm_config if hasattr(
        mm_encoder, "mm_config") else mm_encoder.image_config

    max_image_size = image_config.max_image_size
    image_patch_size = image_config.image_patch_size

    return ((max_image_size // image_patch_size)**2)


def dummy_data_for_pixtral(ctx: InputContext, seq_len: int,
                           mm_counts: Mapping[str, int]):
    tokenizer = cached_tokenizer_from_config(ctx.model_config)

    mm_encoder = tokenizer.mistral.instruct_tokenizer.mm_encoder
    image_token_id = mm_encoder.special_ids.img

    mm_config = ctx.get_mm_config()
    num_images = mm_config.limit_per_prompt.get("image", 1)

    # dummy size
    size = 256
    image = Image.new("RGB", (size, size), color=0)

    encoding = tokenizer.instruct.mm_encoder(ImageChunk(image=image))
    image_feature_size = len(encoding.tokens)
    num_image_tokens = image_feature_size * num_images
    seq_data = SequenceData.from_prompt_token_counts(
        (image_token_id, num_image_tokens),
        (0, seq_len - num_image_tokens),
    )

    mm_data = {"image": num_images * [image]}
    mm_placeholders = {
        "image":
        consecutive_placeholder_ranges(num_items=num_images,
                                       item_size=image_feature_size)
    }
    return DummyData(seq_data, mm_data, mm_placeholders)


def input_mapper_for_pixtral(ctx: InputContext,
                             data: object) -> MultiModalKwargs:
    """Maps the input data to its MultiModalKwargs (if any).

    Args:
        ctx: Context of the loaded model.
        data: data potentially containing PIL images to be processed
            and mapped to `images`.

    Returns:
        MultiModalKwargs containing the stacked normalized images tensor or
        image embeddings.
    """
    tokenizer = cached_tokenizer_from_config(ctx.model_config)

    data_list = data if isinstance(data, list) else [data]

    images = []
    image_tokens_list = []
    for image_data in data_list:
        image = ImageChunk(image=image_data)
        encoding = tokenizer.instruct.mm_encoder(image)
        image = torch.from_numpy(encoding.image).to(dtype=torch.float16)
        images.append(image)
        image_tokens_list.append(encoding.tokens)

    image_tokens = torch.tensor([
        token_id for image_tokens in image_tokens_list
        for token_id in image_tokens
    ])
    return MultiModalKwargs({"images": images, "image_tokens": image_tokens})


def input_processor_for_pixtral(ctx: InputContext, inputs: DecoderOnlyInputs):
    multi_modal_data = inputs.get("multi_modal_data")
    if multi_modal_data is None or "image" not in multi_modal_data:
        return inputs

    prompt_token_ids = inputs.get("prompt_token_ids")
    prompt = inputs.get("prompt")
    tokenizer = cached_tokenizer_from_config(ctx.model_config)

    mm_encoder = tokenizer.mistral.instruct_tokenizer.mm_encoder
    image_token_id = mm_encoder.special_ids.img
    image_break_id = mm_encoder.special_ids.img_break
    image_end_id = mm_encoder.special_ids.img_end

    if image_token_id not in inputs['prompt_token_ids']:
        raise ValueError(
            f"You've passed {inputs=} without {image_token_id=}"
            " Make sure to process your input via mistral_common's"
            " tokenizer or pass a chat completion request. For more"
            " For more info, see: "
            "https://github.com/vllm-project/vllm/issues/8411.")

    # Get precise tracking of placeholder positions
    placeholder_ranges = []
    curr_offset = -1
    curr_length = 0
    for i in range(len(prompt_token_ids)):
        if prompt_token_ids[i] in (image_token_id, image_break_id):
            if curr_offset < 0:
                curr_offset = i
            curr_length += 1
        elif prompt_token_ids[i] == image_end_id:
            curr_length += 1
            placeholder_ranges.append(
                PlaceholderRange(offset=curr_offset, length=curr_length))
            curr_offset = -1
            curr_length = 0
        else:
            pass
    return token_inputs(prompt=prompt,
                        prompt_token_ids=prompt_token_ids,
                        multi_modal_data=multi_modal_data,
                        multi_modal_placeholders={"image": placeholder_ranges})


@MULTIMODAL_REGISTRY.register_image_input_mapper(input_mapper_for_pixtral)
@MULTIMODAL_REGISTRY.register_max_image_tokens(get_max_pixtral_image_tokens)
@INPUT_REGISTRY.register_dummy_data(dummy_data_for_pixtral)
@INPUT_REGISTRY.register_input_processor(input_processor_for_pixtral)
=======
PATCH_MERGE = "patch_merge"


class PixtralImagePixelInputs(TypedDict):
    type: Literal["pixel_values"]

    images: Union[torch.Tensor, list[torch.Tensor]]
    """
    Shape: `(batch_size * num_images, num_channels, image_width, image_height)`

    The result of stacking {attr}`ImageEncoding.tokens` from each prompt.
    """


class PixtralProcessorAdapter:
    """
    Provide a HF-compatible interface for
    {class}`mistral_common.tokens.tokenizers.multimodal.ImageEncoder`.
    """

    def __init__(self, tokenizer: MistralTokenizer) -> None:
        super().__init__()

        self.tokenizer = tokenizer

    @property
    def image_processor(self) -> ImageEncoder:
        image_encoder = self.tokenizer.instruct.mm_encoder
        assert isinstance(image_encoder, ImageEncoder)
        return image_encoder

    @cached_property
    def image_break_id(self) -> int:
        return self.image_processor.special_ids.img_break

    @cached_property
    def image_token_id(self) -> int:
        return self.image_processor.special_ids.img

    @cached_property
    def image_end_id(self) -> int:
        return self.image_processor.special_ids.img_end

    @cached_property
    def image_size(self) -> int:
        return self.image_processor.mm_config.max_image_size

    @cached_property
    def patch_size(self) -> int:
        return self.image_processor.mm_config.image_patch_size

    def __call__(
        self,
        text: Optional[Union[TextInput, list[TextInput]]] = None,
        images: Optional[Union[ImageInput, list[ImageInput]]] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        **kwargs,
    ) -> Mapping[str, NestedTensors]:
        if text is None:
            text = []
        if not isinstance(text, list):
            text = [text]
        if images is None:
            images = []
        if not isinstance(images, list):
            images = [images]

        if not images:
            input_ids = self.tokenizer(text).input_ids

            return {"input_ids": torch.tensor(input_ids)}

        # Allow dummy text, which is used for profiling as well as token inputs
        if any(len(t) > 0 for t in text):
            raise ValueError(
                "You've passed text inputs instead of token inputs. "
                "Make sure to process your input via `mistral_common`'s "
                "tokenizer or pass a chat completion request. "
                "For more info, see: "
                "https://github.com/vllm-project/vllm/issues/8411.")

        images_processed = list[torch.Tensor]()
        images_tokens = list[torch.Tensor]()

        for image in images:
            image_inputs = self.image_processor(ImageChunk(image=image))
            image_processed = torch.tensor(image_inputs.image)
            image_tokens = torch.tensor(image_inputs.tokens)

            images_processed.append(image_processed)
            images_tokens.append(image_tokens)

        return {
            "input_ids": torch.cat(images_tokens)[None].expand(len(text), -1),
            "images": images_processed,
        }


class PixtralProcessingInfo(BaseProcessingInfo):

    def get_tokenizer(self) -> MistralTokenizer:
        tokenizer = cached_tokenizer_from_config(self.ctx.model_config)
        if not isinstance(tokenizer, MistralTokenizer):
            raise ValueError("This model requires `--tokenizer-mode mistral`")

        return tokenizer

    def get_hf_processor(self) -> PixtralProcessorAdapter:
        return PixtralProcessorAdapter(self.get_tokenizer())

    def get_supported_mm_limits(self) -> Mapping[str, Optional[int]]:
        return {"image": None}

    def get_vision_config(
        self,
        processor: Optional[PixtralProcessorAdapter] = None,
    ):
        if processor is None:
            processor = self.get_hf_processor()

        return PixtralVisionConfig(
            image_size=processor.image_size,
            patch_size=processor.patch_size,
        )

    def get_num_image_tokens(
        self,
        *,
        image_width: int,
        image_height: int,
        processor: Optional[PixtralProcessorAdapter] = None,
    ) -> int:
        if processor is None:
            processor = self.get_hf_processor()

        ncols, nrows = processor.image_processor._image_to_num_tokens(
            Image.new("RGB", (image_width, image_height)))

        return ncols * nrows

    def get_image_size_with_most_features(self) -> ImageSize:
        image_processor = self.get_hf_processor().image_processor
        max_image_size = image_processor.mm_config.max_image_size

        return ImageSize(width=max_image_size, height=max_image_size)


class PixtralDummyInputsBuilder(BaseDummyInputsBuilder[PixtralProcessingInfo]):

    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        return ""

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> MultiModalDataDict:
        num_images = mm_counts.get("image", 0)

        target_width, target_height = \
            self.info.get_image_size_with_most_features()

        return {
            "image":
            self._get_dummy_images(width=target_width,
                                   height=target_height,
                                   num_images=num_images)
        }


class PixtralMultiModalProcessor(BaseMultiModalProcessor[PixtralProcessingInfo]
                                 ):

    def _get_mm_fields_config(
        self,
        hf_inputs: Mapping[str, NestedTensors],
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        return dict(images=MultiModalFieldConfig.batched("image"))

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargs,
    ) -> Sequence[PromptUpdate]:
        processor = self.info.get_hf_processor(**hf_processor_mm_kwargs)

        image_break_id = processor.image_break_id
        image_token_id = processor.image_token_id
        image_end_id = processor.image_end_id

        def get_replacement(item_idx: int):
            images = mm_items.get_items("image", ImageProcessorItems)
            image_size = images.get_image_size(item_idx)

            ncols, nrows = processor.image_processor._image_to_num_tokens(
                Image.new("RGB", (image_size.width, image_size.height)))

            tokens = ([image_token_id] * ncols + [image_break_id]) * nrows
            tokens[-1] = image_end_id

            return PromptUpdateDetails.select_token_id(tokens, image_token_id)

        return [
            PromptReplacement(
                modality="image",
                target="",  # Never match the prompt (see below note)
                replacement=get_replacement,
            ),
        ]

    def _cached_apply_hf_processor(
        self,
        prompt: Union[str, list[int]],
        mm_data_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        *,
        return_mm_hashes: bool,
    ) -> tuple[list[int], MultiModalKwargs, Optional[MultiModalHashes], bool]:
        prompt_ids, mm_kwargs, mm_hashes, _ = super(
        )._cached_apply_hf_processor(
            prompt=prompt,
            mm_data_items=mm_data_items,
            hf_processor_mm_kwargs=hf_processor_mm_kwargs,
            return_mm_hashes=return_mm_hashes,
        )

        # NOTE: The tokens are already inserted by the chat template
        return prompt_ids, mm_kwargs, mm_hashes, True


@MULTIMODAL_REGISTRY.register_processor(PixtralMultiModalProcessor,
                                        info=PixtralProcessingInfo,
                                        dummy_inputs=PixtralDummyInputsBuilder)
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
class PixtralForConditionalGeneration(nn.Module, SupportsMultiModal,
                                      SupportsPP):

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        multimodal_config = vllm_config.model_config.multimodal_config
        self.config = config
        self.multimodal_config = multimodal_config

        dataclass_fields = {field.name for field in fields(VisionEncoderArgs)}
        vision_args = {
            key: value
            for key, value in self.config.vision_config.to_dict().items()
            if key in dataclass_fields
        }

<<<<<<< HEAD
        if not ("image_break_token_id" in vision_args
                and "image_end_token_id" in vision_args):
            raise ValueError(
                "'image_break_token_id' and 'image_end_token_id' not found "
                "in the vision_encoder arguments. Please download the latest "
                "version of 'params.json' from the model repository.")

=======
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
        self.vision_args = VisionEncoderArgs(**vision_args)

        # init MistralForCausalLM
        self.language_model = init_vllm_registered_model(
            vllm_config=vllm_config,
            hf_config=config.text_config,
            prefix=maybe_prefix(prefix, "language_model"),
        )

        self.vision_encoder = VisionTransformer(self.vision_args)
<<<<<<< HEAD
=======

        if self.vision_args.add_pre_mm_projector_layer_norm:
            self.pre_mm_projector_norm = RMSNorm(self.vision_args.hidden_size,
                                                 eps=1e-5)

        if self.vision_args.mm_projector_id == PATCH_MERGE:
            self.patch_merger = PatchMerger(
                vision_encoder_dim=self.vision_args.hidden_size,
                spatial_merge_size=self.vision_args.spatial_merge_size,
                use_mlp_bias=False,
            )

>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
        self.vision_language_adapter = VisionLanguageAdapter(
            self.vision_args, dim=config.text_config.hidden_size)

        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors)

<<<<<<< HEAD
    @cached_property
    def sampler(self):
        if hasattr(self.language_model, "sampler"):
            return self.language_model.sampler

        return get_sampler()

    def get_multimodal_embeddings(self, **kwargs) -> Optional[NestedTensors]:
        image_input, image_tokens = self._parse_and_validate_image_input(
            **kwargs)
        if image_input is None:
            return None

        vision_embeddings = self._process_image_input(image_input)

        # NOTE: We patch the outputs of the vision encoder with embeddings
        # from `[IMG_BREAK]` and `[IMG_END]` tokens.
        image_embeds = self.language_model.get_input_embeddings(image_tokens)
        image_token_mask = image_tokens == self.vision_args.image_token_id
        image_embeds[image_token_mask] = vision_embeddings

        # NOTE: Image embeddings are split into separate tensors for each image
        # by the indices of `[IMG_END]` token.
        image_end_mask = image_tokens == self.vision_args.image_end_token_id
        split_indices = torch.where(image_end_mask)[0] + 1
        if len(split_indices) <= 1:
            # Do not split, return as tensor of shape [1, fs, hs]
            return image_embeds.unsqueeze(0)

        # If the last split index is the last index in image_tokens, we
        # ignore it to avoid empty split tensor
        if split_indices[-1] == len(image_tokens):
            split_indices = split_indices[:-1]

        image_embeds = image_embeds.tensor_split(split_indices.cpu())
        return image_embeds
=======
    def _parse_and_validate_image_input(
            self, **kwargs: object) -> Optional[PixtralImagePixelInputs]:
        images = kwargs.pop("images", None)
        if images is None:
            return None

        if not isinstance(images, (torch.Tensor, list)):
            raise ValueError("Incorrect type of images. "
                             f"Got type: {type(images)}")

        return PixtralImagePixelInputs(
            type="pixel_values",
            images=flatten_bn(images),
        )

    def _process_image_input(
        self,
        image_input: PixtralImagePixelInputs,
    ) -> tuple[torch.Tensor, ...]:
        images = image_input["images"]
        image_features = self.vision_encoder(images)
        feature_sizes = [
            image_feature.shape[0] for image_feature in image_features
        ]
        image_features = torch.cat(image_features)
        if self.vision_args.add_pre_mm_projector_layer_norm:
            image_features = self.pre_mm_projector_norm(image_features)
        if self.vision_args.mm_projector_id == PATCH_MERGE:
            patch_size = self.vision_args.patch_size
            spatial_merge_size_square = self.vision_args.spatial_merge_size**2
            img_patch_dims = [(img.shape[1] // patch_size,
                               img.shape[2] // patch_size) for img in images]
            feature_sizes = [
                feature_size // spatial_merge_size_square
                for feature_size in feature_sizes
            ]
            image_features = self.patch_merger(image_features,
                                               image_sizes=img_patch_dims)
        image_embeds = self.vision_language_adapter(image_features)
        image_embeds = torch.split(image_embeds, feature_sizes)
        return image_embeds

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
=======
        multimodal_embeddings: Optional[MultiModalEmbeddings] = None,
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
    ) -> torch.Tensor:
        inputs_embeds = self.language_model.get_input_embeddings(input_ids)
        if multimodal_embeddings is not None:
            inputs_embeds = merge_multimodal_embeddings(
<<<<<<< HEAD
                input_ids, inputs_embeds, multimodal_embeddings, [
                    self.vision_args.image_token_id,
                    self.vision_args.image_break_token_id,
                    self.vision_args.image_end_token_id,
                ])
=======
                input_ids,
                inputs_embeds,
                multimodal_embeddings,
                self.vision_args.image_token_id,
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
=======
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs: object,
    ) -> Union[torch.Tensor, IntermediateTensors]:
<<<<<<< HEAD
        """Run forward pass for pixtral.
        """
=======
        """Run forward pass for pixtral."""
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
        if intermediate_tensors is not None:
            inputs_embeds = None

        # NOTE: In v1, inputs_embeds is always generated at model runner, this
        # condition is for v0 compatibility.
        elif inputs_embeds is None:
            vision_embeddings = self.get_multimodal_embeddings(**kwargs)
            inputs_embeds = self.get_input_embeddings(input_ids,
                                                      vision_embeddings)
            input_ids = None

        hidden_states = self.language_model.model(input_ids,
                                                  positions,
<<<<<<< HEAD
                                                  kv_caches,
                                                  attn_metadata,
=======
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
                                                  intermediate_tensors,
                                                  inputs_embeds=inputs_embeds)

        return hidden_states

<<<<<<< HEAD
    def _parse_and_validate_image_input(
        self,
        images: Optional[Union[List[List[torch.Tensor]], List[torch.Tensor],
                               torch.Tensor]] = None,
        image_tokens: Optional[torch.Tensor] = None,
    ) -> Tuple[Optional[List[torch.Tensor]], Optional[torch.Tensor]]:
        if images is None:
            return None, None

        if isinstance(images, torch.Tensor):
            # if passed as batch take all images
            N, B, C, W, H = images.shape
            images = images.reshape(N * B, C, W, H)
            images = [images[i] for i in range(images.size(0))]
        elif isinstance(images, list):
            # if passed as list flatten lists of tensors
            flatten_images = []
            for imgs_per_req in images:
                imgs_per_req = [
                    imgs_per_req[i] for i in range(imgs_per_req.size(0))
                ] if isinstance(imgs_per_req, torch.Tensor) else imgs_per_req

                flatten_images.extend(imgs_per_req)

            images = flatten_images

        if isinstance(image_tokens, torch.Tensor):
            # image_tokens are batched
            image_tokens = image_tokens.flatten()
        elif isinstance(image_tokens, list):
            # image_tokens are of different lengths thus passed as a list
            image_tokens = torch.cat(image_tokens)

        assert image_tokens.dim() == 1

        return images, image_tokens

    def _process_image_input(self,
                             image_input: List[torch.Tensor]) -> torch.Tensor:
        return self.vision_language_adapter(self.vision_encoder(image_input))

=======
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
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

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):

        def is_vision_encoder_weights(weight: Tuple[str, torch.Tensor]):
            return weight[0].startswith("vision_encoder")

        def is_vision_lang_adapter_weights(weight: Tuple[str, torch.Tensor]):
            return weight[0].startswith("vision_language_adapter")

        # Get references to parameters for direct loading
        vision_encoder_dict = dict(self.vision_encoder.named_parameters())
=======
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):

        def is_vision_encoder_weights(weight: tuple[str, torch.Tensor]):
            return weight[0].startswith("vision_encoder")

        def is_vision_lang_adapter_weights(weight: tuple[str, torch.Tensor]):
            return weight[0].startswith("vision_language_adapter")

        def is_patch_merger(weight: tuple[str, torch.Tensor]):
            return weight[0].startswith("patch_merger")

        def is_pre_mm_projector_norm(weight: tuple[str, torch.Tensor]):
            return weight[0].startswith("pre_mm_projector_norm")

        # Get references to parameters for direct loading
        vision_encoder_dict = dict(self.vision_encoder.named_parameters())
        patch_merger_dict = dict(self.patch_merger.named_parameters(
        )) if self.vision_args.mm_projector_id == PATCH_MERGE else dict()
        pre_mm_projector_norm_dict = dict(
            self.pre_mm_projector_norm.named_parameters(
            )) if self.vision_args.add_pre_mm_projector_layer_norm else dict()
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
        vision_lang_adapter_dict = dict(
            self.vision_language_adapter.named_parameters())

        def llm_weights_generator():
            # Single pass over weights
            for name, w in weights:
                if is_vision_encoder_weights((name, w)):
                    # Load vision encoder weights directly
                    trimmed_name = '.'.join(name.split(".")[1:])
                    param = vision_encoder_dict[trimmed_name]
                    with torch.no_grad():
                        default_weight_loader(param, w)
<<<<<<< HEAD
=======
                elif is_patch_merger((name, w)):
                    # Load vision patch merger weights directly
                    trimmed_name = '.'.join(name.split(".")[1:])
                    param = patch_merger_dict[trimmed_name]
                    with torch.no_grad():
                        default_weight_loader(param, w)
                elif is_pre_mm_projector_norm((name, w)):
                    # Load vision pre_mm_projector_norm weights directly
                    trimmed_name = '.'.join(name.split(".")[1:])
                    param = pre_mm_projector_norm_dict[trimmed_name]
                    with torch.no_grad():
                        default_weight_loader(param, w)
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
                elif is_vision_lang_adapter_weights((name, w)):
                    # Load vision-language adapter weights directly
                    trimmed_name = '.'.join(name.split(".")[1:])
                    param = vision_lang_adapter_dict[trimmed_name]
                    with torch.no_grad():
                        default_weight_loader(param, w)
                else:
                    # LLM weights: yield them to be loaded
                    # by language_model.load_weights
                    yield (name, w)

        # Now we call the language model load with the generator
        self.language_model.load_weights(llm_weights_generator())


# Vision encoder
@dataclass
class VisionEncoderArgs:
    hidden_size: int
    num_channels: int
    image_size: int
    patch_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    rope_theta: float  # for rope-2D
    image_token_id: int
<<<<<<< HEAD
    image_break_token_id: int
    image_end_token_id: int
    adapter_bias: bool = True
=======
    adapter_bias: bool = True
    spatial_merge_size: int = 1
    add_pre_mm_projector_layer_norm: bool = False
    mm_projector_id: str = ""
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea


def _reshape_for_broadcast(freqs_cis: torch.Tensor,
                           x: torch.Tensor) -> torch.Tensor:
    """
    freqs_cis: complex - (seq_len, head_dim / 2)
    x: complex - (bsz, seq_len, head_dim / 2)
    """
    ndim = x.ndim
    assert ndim > 1
    assert freqs_cis.shape == (x.shape[1], x.shape[-1]), (
        freqs_cis.shape,
        (x.shape[1], x.shape[-1]),
    )
    shape = [
        d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)
    ]
    return freqs_cis.view(*shape)


def precompute_freqs_cis_2d(
    dim: int,
    height: int,
    width: int,
    theta: float,
) -> torch.Tensor:
    """
    freqs_cis: 2D complex tensor of shape (height, width, dim // 2)
        to be indexed by (height, width) position tuples
    """
    # (dim / 2) frequency bases
    freqs = 1.0 / (theta**(torch.arange(0, dim, 2).float() / dim))

    h = torch.arange(height, device=freqs.device)
    w = torch.arange(width, device=freqs.device)

    freqs_h = torch.outer(h, freqs[::2]).float()
    freqs_w = torch.outer(w, freqs[1::2]).float()
    freqs_2d = torch.cat(
        [
            freqs_h[:, None, :].repeat(1, width, 1),
            freqs_w[None, :, :].repeat(height, 1, 1),
        ],
        dim=-1,
    )
    return torch.polar(torch.ones_like(freqs_2d), freqs_2d)


def apply_rotary_emb_vit(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
<<<<<<< HEAD
) -> Tuple[torch.Tensor, torch.Tensor]:
=======
) -> tuple[torch.Tensor, torch.Tensor]:
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    assert freqs_cis.dtype == torch.complex64
    freqs_cis = _reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


class FeedForward(nn.Module):

    def __init__(self, args: VisionEncoderArgs):
        super().__init__()
        assert args.intermediate_size is not None
        self.w1 = nn.Linear(args.hidden_size,
                            args.intermediate_size,
                            bias=False)
        self.w2 = nn.Linear(args.intermediate_size,
                            args.hidden_size,
                            bias=False)
        self.w3 = nn.Linear(args.hidden_size,
                            args.intermediate_size,
                            bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Attention(nn.Module):

    def __init__(self, args: VisionEncoderArgs):
        super().__init__()
        self.args = args
        assert not args.hidden_size % args.num_attention_heads
        self.n_heads = args.num_attention_heads
        self.head_dim = args.hidden_size // args.num_attention_heads

        self.wq = nn.Linear(args.hidden_size, args.hidden_size, bias=False)
        self.wk = nn.Linear(args.hidden_size, args.hidden_size, bias=False)
        self.wv = nn.Linear(args.hidden_size, args.hidden_size, bias=False)
        self.wo = nn.Linear(args.hidden_size, args.hidden_size, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        freqs_cis: torch.Tensor,
    ) -> torch.Tensor:
        batch, patches, _ = x.shape

        q, k, v = self.wq(x), self.wk(x), self.wv(x)
        q = q.reshape(batch, patches, self.n_heads, self.head_dim)
        k = k.reshape(batch, patches, self.n_heads, self.head_dim)
        v = v.reshape(batch, patches, self.n_heads, self.head_dim)

        q, k = apply_rotary_emb_vit(q, k, freqs_cis=freqs_cis)
        out = xops.memory_efficient_attention(q, k, v, attn_bias=mask)
        out = out.reshape(batch, patches, self.n_heads * self.head_dim)
        return self.wo(out)


class TransformerBlock(nn.Module):

    def __init__(self, args: VisionEncoderArgs):
        super().__init__()
        self.attention = Attention(args)
        self.feed_forward = FeedForward(args)
        self.attention_norm = RMSNorm(args.hidden_size, eps=1e-5)
        self.ffn_norm = RMSNorm(args.hidden_size, eps=1e-5)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        freqs_cis: torch.Tensor,
    ) -> torch.Tensor:
        r = self.attention.forward(self.attention_norm(x),
                                   mask=mask,
                                   freqs_cis=freqs_cis)
        h = x + r
        r = self.feed_forward.forward(self.ffn_norm(h))
        out = h + r
        return out


class Transformer(nn.Module):

    def __init__(self, args: VisionEncoderArgs):
        super().__init__()
        self.layers = torch.nn.ModuleList()
        for _ in range(args.num_hidden_layers):
            self.layers.append(TransformerBlock(args))

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        freqs_cis: Optional[torch.Tensor],
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, mask=mask, freqs_cis=freqs_cis)
        return x


<<<<<<< HEAD
def position_meshgrid(patch_embeds_list: List[torch.Tensor], ) -> torch.Tensor:
=======
def position_meshgrid(patch_embeds_list: list[torch.Tensor], ) -> torch.Tensor:
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
    positions = torch.cat([
        torch.stack(
            torch.meshgrid(
                torch.arange(p.shape[-2]),
                torch.arange(p.shape[-1]),
                indexing="ij",
            ),
            dim=-1,
        ).reshape(-1, 2) for p in patch_embeds_list
    ])
    return positions


class VisionTransformer(nn.Module):

    def __init__(self, args: VisionEncoderArgs):
        super().__init__()
        self.args = args
        self.patch_conv = nn.Conv2d(
            in_channels=args.num_channels,
            out_channels=args.hidden_size,
            kernel_size=args.patch_size,
            stride=args.patch_size,
            bias=False,
        )
        self.ln_pre = RMSNorm(args.hidden_size, eps=1e-5)
        self.transformer = Transformer(args)

        head_dim = self.args.hidden_size // self.args.num_attention_heads
        assert head_dim % 2 == 0, "ROPE requires even head_dim"
        self._freqs_cis: Optional[torch.Tensor] = None

    @property
    def max_patches_per_side(self) -> int:
        return self.args.image_size // self.args.patch_size

    @property
    def device(self) -> torch.types.Device:
        return next(self.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    @property
    def freqs_cis(self) -> torch.Tensor:
        if self._freqs_cis is None:
            self._freqs_cis = precompute_freqs_cis_2d(
                dim=self.args.hidden_size // self.args.num_attention_heads,
                height=self.max_patches_per_side,
                width=self.max_patches_per_side,
                theta=self.args.rope_theta,
            )

        if self._freqs_cis.device != self.device:
            self._freqs_cis = self._freqs_cis.to(device=self.device)

        return self._freqs_cis

    def forward(
        self,
<<<<<<< HEAD
        images: List[torch.Tensor],
    ) -> torch.Tensor:
        """
        Args:
            images: list of N_img images of variable sizes, 
                each of shape (C, H, W)
        Returns:
            image_features: tensor of token features for 
=======
        images: list[torch.Tensor],
    ) -> torch.Tensor:
        """
        Args:
            images: list of N_img images of variable sizes,
                each of shape (C, H, W)
        Returns:
            image_features: tensor of token features for
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
                all tokens of all images of shape (N_toks, D)
        """
        # pass images through initial convolution independently
        patch_embeds_list = [
            self.patch_conv(img.unsqueeze(0).to(self.dtype)) for img in images
        ]

<<<<<<< HEAD
        # flatten to a single sequence
        patch_embeds = torch.cat(
            [p.flatten(2).permute(0, 2, 1) for p in patch_embeds_list], dim=1)
=======
        patch_embeds = [
            p.flatten(2).permute(0, 2, 1) for p in patch_embeds_list
        ]
        embed_sizes = [p.shape[1] for p in patch_embeds]

        # flatten to a single sequence
        patch_embeds = torch.cat(patch_embeds, dim=1)
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
        patch_embeds = self.ln_pre(patch_embeds)

        # positional embeddings
        positions = position_meshgrid(patch_embeds_list).to(self.device)
        freqs_cis = self.freqs_cis[positions[:, 0], positions[:, 1]]

        # pass through Transformer with a block diagonal mask delimiting images
        if USE_XFORMERS_OPS:
            mask = xops.fmha.attn_bias.BlockDiagonalMask.from_seqlens(
                [p.shape[-2] * p.shape[-1] for p in patch_embeds_list], )
        else:
            raise ImportError("Xformers is required for Pixtral inference "
                              "with the Mistral format")
        out = self.transformer(patch_embeds, mask=mask, freqs_cis=freqs_cis)

<<<<<<< HEAD
        # remove batch dimension of the single sequence
        return out.squeeze(0)
=======
        # squeeze dim 0 and split into separate tensors for each image
        return torch.split(out.squeeze(0), embed_sizes)
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea


class VisionLanguageAdapter(nn.Module):

    def __init__(self, args: VisionEncoderArgs, dim: int):
        super().__init__()
        assert isinstance(args, VisionEncoderArgs)
        self.w_in = nn.Linear(
            args.hidden_size,
            dim,
            bias=args.adapter_bias,
        )
        self.gelu = nn.GELU()
        self.w_out = nn.Linear(dim, dim, bias=args.adapter_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_out(self.gelu(self.w_in(x)))


<<<<<<< HEAD
=======
class PatchMerger(nn.Module):
    """
    Learned merging of spatial_merge_size ** 2 patches
    """

    def __init__(
        self,
        vision_encoder_dim: int,
        spatial_merge_size: int,
        use_mlp_bias: bool = False,
    ) -> None:
        super().__init__()

        mlp_input_dim = vision_encoder_dim * (spatial_merge_size**2)

        self.spatial_merge_size = spatial_merge_size
        self.mlp_input_dim = mlp_input_dim

        self.merging_layer = nn.Linear(
            mlp_input_dim,
            vision_encoder_dim,
            bias=use_mlp_bias,
        )

    def forward(self, x: torch.Tensor,
                image_sizes: list[tuple[int, int]]) -> torch.Tensor:
        # image_sizes specified in tokens
        assert sum([h * w for h, w in image_sizes]) == len(x)

        # x is (N, vision_encoder_dim)
        x = self.permute(x, image_sizes)

        # x is (N / spatial_merge_size ** 2,
        #       vision_encoder_dim * spatial_merge_size ** 2)
        x = self.merging_layer(x)

        # x is (N / spatial_merge_size ** 2, vision_encoder_dim)
        return x

    def permute(
        self,
        x: torch.Tensor,
        image_sizes: list[tuple[int, int]],
    ) -> torch.Tensor:
        """
        Args:
            x: (N, D) where N is flattened and concatenated patch tokens
                for all images
            image_sizes: list of tuple of (height, width) in tokens for
                each image
        Returns:
            image_features: reorders patch tokens so each grid of
                (spatial_merge_size, spatial_merge_size) is contiguous.
                now (N / spatial_merge_size ** 2, D * spatial_merge_size ** 2)
        """

        sub_grids = get_sub_grids(
            x=x,
            image_sizes=image_sizes,
            spatial_merge_size=self.spatial_merge_size
        )  # list of [d x sub_grid_size x sub_grid_size x n_patches]
        permuted_tensor: list[torch.Tensor] = []
        for grid in sub_grids:
            n_patches = grid.shape[-1]
            permuted_tensor.append(grid.view(-1, n_patches).t(
            ))  # n_patches x d * sub_grid_size * sub_grid_size
        return torch.cat(
            permuted_tensor, dim=0
        )  # (N / spatial_merge_size ** 2, d * spatial_merge_size ** 2)


def get_sub_grids(
    x: torch.Tensor,
    image_sizes: list[tuple[int, int]],
    spatial_merge_size: int,
) -> list[torch.Tensor]:
    # image_sizes specified in tokens
    tokens_per_image = [h * w for h, w in image_sizes]
    d = x.shape[-1]
    all_img_sub_grids: list[torch.Tensor] = []
    sub_grid_size = spatial_merge_size

    for image_index, image_tokens in enumerate(x.split(tokens_per_image)):
        # Reshape image_tokens into a 2D grid
        h, w = image_sizes[image_index]
        image_grid = image_tokens.view(h, w, d).permute(
            2, 0, 1)[None, :, :, :]  # 1 x d x h x w
        sub_grids = torch.nn.functional.unfold(image_grid,
                                               kernel_size=sub_grid_size,
                                               stride=sub_grid_size)
        sub_grids = sub_grids.view(
            1, d, sub_grid_size, sub_grid_size,
            -1)  # 1 x d x sub_grid_size x sub_grid_size x n_patches

        all_img_sub_grids.append(sub_grids[0])

    return all_img_sub_grids


>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
#### HF Transformers version of Pixtral ####
# Based off https://github.com/huggingface/transformers/blob/d7950bff82b18c823193d17d72188c5e46d06c83/src/transformers/models/pixtral/modeling_pixtral.py
# This model follows the Llava family, meaning image embeddings are placed
# instead of the `[IMG]` token placeholders.
# The model uses [`PixtralVisionModel`] for its vision encoder,
# and [`MistralForCausalLM`] for its language decoder.


<<<<<<< HEAD
def get_pixtral_hf_patch_grid_length(*, image_size: int,
                                     patch_size: int) -> int:
    # Since interpolation is applied, the image size need not be divisible
    # assert image_size % patch_size == 0
    return image_size // patch_size


def get_pixtral_hf_image_feature_size(
    *,
    image_size: int,
    patch_size: int,
) -> int:
    grid_length = get_pixtral_hf_patch_grid_length(
        image_size=image_size,
        patch_size=patch_size,
    )

    # Consider the image_break_token
    return (grid_length + 1) * grid_length


def get_max_pixtral_hf_image_tokens(hf_config: PixtralVisionConfig) -> int:
    grid_length = get_pixtral_hf_patch_grid_length(
        image_size=hf_config.image_size,
        patch_size=hf_config.patch_size,
    )

    # Consider the image_break_token
    return (grid_length + 1) * grid_length


def dummy_image_for_pixtral_hf(
    hf_config: PixtralVisionConfig,
    num_images: int,
    *,
    image_width_override: Optional[int] = None,
    image_height_override: Optional[int] = None,
):
    width = height = hf_config.image_size
    if image_width_override is not None:
        width = image_width_override
    if image_height_override is not None:
        height = image_height_override

    image = Image.new("RGB", (width, height), color=0)
    return {"image": image if num_images == 1 else [image] * num_images}


# Adapted from transformers.models.pixtral.image_processing_pixtral.get_resize_output_image_size # noqa: E501
# https://github.com/huggingface/transformers/blob/2bd4d5897dc73e8b172832070a6f9e567a0df017/src/transformers/models/pixtral/image_processing_pixtral.py#L180
def get_pixtral_hf_image_feature_grid_size(
    hf_config: PixtralVisionConfig,
    *,
    image_width: int,
    image_height: int,
) -> tuple[int, int]:
    max_width = max_height = hf_config.image_size
    patch_width = patch_height = hf_config.patch_size

    ratio = max(image_width / max_width, image_height / max_height)

    if ratio > 1:
        image_width = int(math.ceil(image_width / ratio))
        image_height = int(math.ceil(image_height / ratio))

    nrows, ncols = _get_pixtral_hf_num_image_tokens(
        (image_height, image_width),
        (patch_height, patch_width),
    )  # type: ignore

    return ncols, nrows


=======
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
class PixtralHFEncoderInfo(VisionEncoderInfo[PixtralVisionConfig]):

    def get_num_image_tokens(
        self,
        *,
        image_width: int,
        image_height: int,
    ) -> int:
<<<<<<< HEAD
        return get_pixtral_hf_image_feature_size(
            image_size=self.vision_config.image_size,
            patch_size=self.vision_config.patch_size,
        )

    def get_max_image_tokens(self) -> int:
        return get_max_pixtral_hf_image_tokens(self.vision_config)
=======
        ncols, nrows = self.get_patch_grid_size(
            image_width=image_width,
            image_height=image_height,
        )
        return ncols * nrows
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea

    def get_image_size(self) -> int:
        return self.vision_config.image_size

    def get_patch_size(self) -> int:
<<<<<<< HEAD
        return self.vision_config.patch_size

    def get_patch_grid_length(self) -> int:
        return get_pixtral_hf_patch_grid_length(
            image_size=self.vision_config.image_size,
            patch_size=self.vision_config.patch_size,
        )
=======
        # spatial_merge_size is needed for Mistral3
        spatial_merge_size = getattr(self.hf_config, "spatial_merge_size", 1)
        return self.vision_config.patch_size * spatial_merge_size

    def get_patch_grid_length(self) -> int:
        image_size, patch_size = self.get_image_size(), self.get_patch_size()

        # Since interpolation is applied, the image size need not be divisible
        # assert image_size % patch_size == 0
        return image_size // patch_size

    # Adapted from: https://github.com/huggingface/transformers/blob/v4.49.0/src/transformers/models/pixtral/image_processing_pixtral.py#L99
    def get_patch_grid_size(
        self,
        *,
        image_width: int,
        image_height: int,
    ) -> tuple[int, int]:
        max_width = max_height = self.get_image_size()
        patch_width = patch_height = self.get_patch_size()

        ratio = max(image_width / max_width, image_height / max_height)

        if ratio > 1:
            image_width = int(math.floor(image_width / ratio))
            image_height = int(math.floor(image_height / ratio))

        nrows, ncols = _get_pixtral_hf_num_image_tokens(
            (image_height, image_width),
            (patch_height, patch_width),
        )  # type: ignore

        return ncols, nrows
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea


class PixtralHFMLP(nn.Module):

    def __init__(
        self,
        config: PixtralVisionConfig,
        quant_config: Optional[QuantizationConfig] = None,
        *,
        prefix: str = "",
    ) -> None:
        super().__init__()

        assert config.intermediate_size is not None
        self.gate_up_proj = MergedColumnParallelLinear(
            input_size=config.hidden_size,
            output_sizes=[config.intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj")
        self.down_proj = RowParallelLinear(input_size=config.intermediate_size,
                                           output_size=config.hidden_size,
                                           bias=False,
                                           quant_config=quant_config,
                                           prefix=f"{prefix}.down_proj")
        self.act_and_mul = get_act_and_mul_fn(config.hidden_act)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_and_mul(gate_up)
        x, _ = self.down_proj(x)
        return x


class PixtralHFAttention(nn.Module):

    def __init__(
        self,
        config: PixtralVisionConfig,
        quant_config: Optional[QuantizationConfig] = None,
        *,
        prefix: str = "",
    ) -> None:
        super().__init__()

        self.config = config
        assert not config.hidden_size % config.num_attention_heads
        self.total_num_heads = config.num_attention_heads
        tp_size = get_tensor_model_parallel_world_size()
        self.n_heads = divide(config.num_attention_heads, tp_size)
        self.head_dim = config.hidden_size // config.num_attention_heads

        self.qkv_proj = QKVParallelLinear(
            hidden_size=config.hidden_size,
            head_size=self.head_dim,
            total_num_heads=self.total_num_heads,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        assert self.total_num_heads * self.head_dim == config.hidden_size
        self.o_proj = RowParallelLinear(
            input_size=config.hidden_size,
            output_size=config.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        position_embeddings: torch.Tensor,
<<<<<<< HEAD
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
=======
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
        batch, patches, _ = hidden_states.size()

        qkv_states, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv_states.chunk(3, dim=-1)

        # Transpose q and k to apply HF's Rotary Position Embedding
        q = q.view(batch, patches, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, patches, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, patches, self.n_heads, self.head_dim)
        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=0)

        if USE_XFORMERS_OPS:
            # Transpose q and k back for attention
            q = q.transpose(1, 2).contiguous()
            k = k.transpose(1, 2).contiguous()

            out = xops.memory_efficient_attention(q,
                                                  k,
                                                  v,
                                                  attn_bias=attention_mask)
        else:
            v = v.transpose(1, 2)
            out = nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=attention_mask)
            out = out.transpose(1, 2)

        out = out.view(batch, patches, self.n_heads * self.head_dim)
        attn_output, _ = self.o_proj(out)

        return attn_output, None


class PixtralHFTransformerBlock(nn.Module):

    def __init__(
        self,
        config: PixtralVisionConfig,
        quant_config: Optional[QuantizationConfig] = None,
        *,
        prefix: str = "",
    ) -> None:
        super().__init__()

        self.attention_norm = RMSNorm(config.hidden_size, eps=1e-5)
        self.attention = PixtralHFAttention(config,
                                            quant_config=quant_config,
                                            prefix=f"{prefix}.attention")
        self.feed_forward = PixtralHFMLP(config,
                                         quant_config=quant_config,
                                         prefix=f"{prefix}.feed_forward")
        self.ffn_norm = RMSNorm(config.hidden_size, eps=1e-5)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        position_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        r, _ = self.attention.forward(self.attention_norm(hidden_states),
                                      attention_mask=attention_mask,
                                      position_embeddings=position_embeddings)
        h = hidden_states + r
        r = self.feed_forward.forward(self.ffn_norm(h))
        out = h + r
        return out


class PixtralHFTransformer(nn.Module):

    def __init__(
        self,
        config: PixtralVisionConfig,
        quant_config: Optional[QuantizationConfig] = None,
        *,
        num_hidden_layers_override: Optional[int] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()

        if num_hidden_layers_override is None:
            num_hidden_layers = config.num_hidden_layers
        else:
            num_hidden_layers = num_hidden_layers_override

        self.layers = nn.ModuleList([
            PixtralHFTransformerBlock(config=config,
                                      quant_config=quant_config,
                                      prefix=f"{prefix}.layers.{layer_idx}")
            for layer_idx in range(num_hidden_layers)
        ])

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor,
        position_embeddings: torch.Tensor,
        return_all_hidden_states: bool,
    ) -> torch.Tensor:
        hidden_states_pool = [x]

        for layer in self.layers:
            x = layer(x, attention_mask, position_embeddings)
            if return_all_hidden_states:
                hidden_states_pool.append(x)
        # If we have multiple feature sample layers, we return all hidden
        # states in order and grab the ones we need by index.
        if return_all_hidden_states:
            return hidden_states_pool
        return x


class PixtralHFVisionModel(nn.Module):

    def __init__(
        self,
        config: PixtralVisionConfig,
        quant_config: Optional[QuantizationConfig] = None,
        *,
        num_hidden_layers_override: Optional[int] = None,
        require_post_norm: Optional[bool] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()

        self.config = config

        self.patch_conv = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=config.hidden_size,
            kernel_size=config.patch_size,
            stride=config.patch_size,
            bias=False,
        )
        self.ln_pre = RMSNorm(config.hidden_size, eps=1e-5)
        self.transformer = PixtralHFTransformer(
            config,
            quant_config,
            num_hidden_layers_override=num_hidden_layers_override,
            prefix=f"{prefix}.transformer",
        )

        num_hidden_layers = config.num_hidden_layers
        if len(self.transformer.layers) > config.num_hidden_layers:
            raise ValueError(
                f"The original encoder only has {num_hidden_layers} "
                f"layers, but you requested {len(self.transformer.layers)} "
                "layers.")

        if require_post_norm is True:
            msg = "PixtralHFVisionModel does not have post-layernorm"
            raise ValueError(msg)

        self.dtype = next(self.parameters()).dtype
        self.device = next(self.parameters()).device
        self.patch_positional_embedding = PixtralRotaryEmbedding(
            config, self.device)

    def forward(
        self,
<<<<<<< HEAD
        pixel_values: List[torch.Tensor],
        feature_sample_layers: Optional[list[int]] = None,
    ) -> torch.Tensor:
=======
        pixel_values: list[torch.Tensor],
        feature_sample_layers: Optional[list[int]] = None,
    ) -> tuple[torch.Tensor, ...]:
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
        """
        Args:
            pixel_values: Each image to be processed will be a separate tensor
                in pixel_values. This means it will be a list of tensors
                because multiple requests batched can have multiple images,
                each with their own shape potentially
            feature_sample_layers: Layer indices whose features should be
                concatenated and used as the visual encoder output. If none
                are provided, the last layer is used.

        Returns:
            image_features: tensor of token features for
                all tokens of all images of shape (N_toks, D)
        """
        # pass images through initial convolution independently
        patch_embeds_list = [
            self.patch_conv(img.unsqueeze(0).to(self.dtype))
            for img in pixel_values
        ]

<<<<<<< HEAD
        # flatten to a single sequence
        patch_embeds = torch.cat(
            [p.flatten(2).permute(0, 2, 1) for p in patch_embeds_list], dim=1)
=======
        patch_embeds = [
            p.flatten(2).permute(0, 2, 1) for p in patch_embeds_list
        ]
        embed_sizes = [p.shape[1] for p in patch_embeds]

        # flatten to a single sequence
        patch_embeds = torch.cat(patch_embeds, dim=1)
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
        patch_embeds = self.ln_pre(patch_embeds)

        # positional embeddings
        position_ids = position_ids_in_meshgrid(
            patch_embeds_list,
            max_width=self.config.image_size // self.config.patch_size).to(
                self.device)
        position_embedding = self.patch_positional_embedding(
            patch_embeds, position_ids)

        if USE_XFORMERS_OPS:
            attention_mask = xops.fmha.attn_bias.BlockDiagonalMask.from_seqlens(
                [p.shape[-2] * p.shape[-1] for p in patch_embeds_list], )
        else:
            from transformers.models.pixtral.modeling_pixtral import (
                generate_block_attention_mask)
            attention_mask = generate_block_attention_mask(
                [p.shape[-2] * p.shape[-1] for p in patch_embeds_list],
                patch_embeds)

        return_all_hidden_states = feature_sample_layers is not None
        out = self.transformer(
            patch_embeds,
            attention_mask,
            position_embedding,
            return_all_hidden_states=return_all_hidden_states)

        out = resolve_visual_encoder_outputs(out, feature_sample_layers, None,
                                             self.config.num_hidden_layers)

<<<<<<< HEAD
        return out

    # (TODO) Add prefix argument for filtering out weights to be loaded
    #        ref: https://github.com/vllm-project/vllm/pull/7186#discussion_r1734163986
    def load_weights(self, weights: Iterable[Tuple[str,
                                                   torch.Tensor]]) -> Set[str]:
=======
        # squeeze dim 0 and split into separate tensors for each image
        return torch.split(out.squeeze(0), embed_sizes)

    # (TODO) Add prefix argument for filtering out weights to be loaded
    #        ref: https://github.com/vllm-project/vllm/pull/7186#discussion_r1734163986
    def load_weights(self, weights: Iterable[tuple[str,
                                                   torch.Tensor]]) -> set[str]:
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
        ]
        params_dict = dict(self.named_parameters())
<<<<<<< HEAD
        loaded_params: Set[str] = set()
=======
        loaded_params: set[str] = set()
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
        layer_count = len(self.transformer.layers)

        for name, loaded_weight in weights:
            # omit layers when num_hidden_layers_override is set
            if name.startswith("transformer.layers"):
                layer_idx = int(name.split(".")[2])
                if layer_idx >= layer_count:
                    continue

            for (param_name, weight_name, shard_id) in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params
