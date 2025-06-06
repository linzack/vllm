# SPDX-License-Identifier: Apache-2.0
<<<<<<< HEAD

from dataclasses import dataclass
from functools import cached_property
from typing import (TYPE_CHECKING, Any, Dict, Generic, Iterable, List, Literal,
                    Optional, Tuple, Union, cast)

import torch
from typing_extensions import NotRequired, TypedDict, TypeVar, assert_never

if TYPE_CHECKING:
    from vllm.multimodal import (MultiModalDataDict, MultiModalKwargs,
                                 MultiModalPlaceholderDict)
    from vllm.multimodal.inputs import MultiModalInputs
=======
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, Generic, Literal, Optional, Union, cast

import torch
from typing_extensions import NotRequired, TypedDict, TypeIs, TypeVar

if TYPE_CHECKING:
    from vllm.multimodal.inputs import MultiModalDataDict, MultiModalInputs
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea


class TextPrompt(TypedDict):
    """Schema for a text prompt."""

    prompt: str
    """The input text to be tokenized before passing to the model."""

    multi_modal_data: NotRequired["MultiModalDataDict"]
    """
    Optional multi-modal data to pass to the model,
    if the model supports it.
    """

<<<<<<< HEAD
    mm_processor_kwargs: NotRequired[Dict[str, Any]]
=======
    mm_processor_kwargs: NotRequired[dict[str, Any]]
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
    """
    Optional multi-modal processor kwargs to be forwarded to the
    multimodal input mapper & processor. Note that if multiple modalities
    have registered mappers etc for the model being considered, we attempt
    to pass the mm_processor_kwargs to each of them.
    """

<<<<<<< HEAD
=======
    cache_salt: NotRequired[str]
    """
    Optional cache salt to be used for prefix caching.
    """

>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea

class TokensPrompt(TypedDict):
    """Schema for a tokenized prompt."""

<<<<<<< HEAD
    prompt_token_ids: List[int]
    """A list of token IDs to pass to the model."""

    token_type_ids: NotRequired[List[int]]
=======
    prompt_token_ids: list[int]
    """A list of token IDs to pass to the model."""

    token_type_ids: NotRequired[list[int]]
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
    """A list of token type IDs to pass to the cross encoder model."""

    multi_modal_data: NotRequired["MultiModalDataDict"]
    """
    Optional multi-modal data to pass to the model,
    if the model supports it.
    """

<<<<<<< HEAD
    mm_processor_kwargs: NotRequired[Dict[str, Any]]
=======
    mm_processor_kwargs: NotRequired[dict[str, Any]]
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
    """
    Optional multi-modal processor kwargs to be forwarded to the
    multimodal input mapper & processor. Note that if multiple modalities
    have registered mappers etc for the model being considered, we attempt
    to pass the mm_processor_kwargs to each of them.
    """

<<<<<<< HEAD

SingletonPrompt = Union[str, TextPrompt, TokensPrompt]
"""
Set of possible schemas for a single prompt:

- A text prompt (:class:`str` or :class:`TextPrompt`)
- A tokenized prompt (:class:`TokensPrompt`)
=======
    cache_salt: NotRequired[str]
    """
    Optional cache salt to be used for prefix caching.
    """


class EmbedsPrompt(TypedDict):
    """Schema for a prompt provided via token embeddings."""

    prompt_embeds: torch.Tensor
    """The embeddings of the prompt."""

    cache_salt: NotRequired[str]
    """
    Optional cache salt to be used for prefix caching.
    """


SingletonPrompt = Union[str, TextPrompt, TokensPrompt, EmbedsPrompt]
"""
Set of possible schemas for a single prompt:

- A text prompt ({class}`str` or {class}`TextPrompt`)
- A tokenized prompt ({class}`TokensPrompt`)
- An embeddings prompt ({class}`EmbedsPrompt`)
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea

Note that "singleton" is as opposed to a data structure
which encapsulates multiple prompts, i.e. of the sort
which may be utilized for encoder/decoder models when
the user desires to express both the encoder & decoder
<<<<<<< HEAD
prompts explicitly, i.e. :class:`ExplicitEncoderDecoderPrompt`

A prompt of type :class:`SingletonPrompt` may be employed
=======
prompts explicitly, i.e. {class}`ExplicitEncoderDecoderPrompt`

A prompt of type {class}`SingletonPrompt` may be employed
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
as (1) input to a decoder-only model, (2) input to
the encoder of an encoder/decoder model, in the scenario
where the decoder-prompt is not specified explicitly, or
(3) as a member of a larger data structure encapsulating
<<<<<<< HEAD
more than one prompt, i.e. :class:`ExplicitEncoderDecoderPrompt`
"""

=======
more than one prompt, i.e. {class}`ExplicitEncoderDecoderPrompt`
"""


def is_tokens_prompt(prompt: SingletonPrompt) -> TypeIs[TokensPrompt]:
    return (isinstance(prompt, dict) and "prompt_token_ids" in prompt
            and "prompt_embeds" not in prompt)


def is_embeds_prompt(prompt: SingletonPrompt) -> TypeIs[EmbedsPrompt]:
    return (isinstance(prompt, dict) and "prompt_token_ids" not in prompt
            and "prompt_embeds" in prompt)


>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
_T1_co = TypeVar("_T1_co",
                 bound=SingletonPrompt,
                 default=SingletonPrompt,
                 covariant=True)
_T2_co = TypeVar("_T2_co",
                 bound=SingletonPrompt,
                 default=SingletonPrompt,
                 covariant=True)


# TODO: Make fields ReadOnly once mypy supports it
class ExplicitEncoderDecoderPrompt(TypedDict, Generic[_T1_co, _T2_co]):
    """
    Represents an encoder/decoder model input prompt,
    comprising an explicit encoder prompt and a decoder prompt.

    The encoder and decoder prompts, respectively, may be formatted
<<<<<<< HEAD
    according to any of the :class:`SingletonPrompt` schemas,
=======
    according to any of the {class}`SingletonPrompt` schemas,
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
    and are not required to have the same schema.

    Only the encoder prompt may have multi-modal data. mm_processor_kwargs
    should be at the top-level, and should not be set in the encoder/decoder
    prompts, since they are agnostic to the encoder/decoder.

<<<<<<< HEAD
    Note that an :class:`ExplicitEncoderDecoderPrompt` may not
    be used as an input to a decoder-only model,
    and that the :code:`encoder_prompt` and :code:`decoder_prompt`
    fields of this data structure themselves must be
    :class:`SingletonPrompt` instances.
=======
    Note that an {class}`ExplicitEncoderDecoderPrompt` may not
    be used as an input to a decoder-only model,
    and that the `encoder_prompt` and `decoder_prompt`
    fields of this data structure themselves must be
    {class}`SingletonPrompt` instances.
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
    """

    encoder_prompt: _T1_co

    decoder_prompt: Optional[_T2_co]

<<<<<<< HEAD
    mm_processor_kwargs: NotRequired[Dict[str, Any]]
=======
    mm_processor_kwargs: NotRequired[dict[str, Any]]
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea


PromptType = Union[SingletonPrompt, ExplicitEncoderDecoderPrompt]
"""
Set of possible schemas for an LLM input, including
both decoder-only and encoder/decoder input types:

<<<<<<< HEAD
- A text prompt (:class:`str` or :class:`TextPrompt`)
- A tokenized prompt (:class:`TokensPrompt`)
- A single data structure containing both an encoder and a decoder prompt
  (:class:`ExplicitEncoderDecoderPrompt`)
=======
- A text prompt ({class}`str` or {class}`TextPrompt`)
- A tokenized prompt ({class}`TokensPrompt`)
- An embeddings prompt ({class}`EmbedsPrompt`)
- A single data structure containing both an encoder and a decoder prompt
  ({class}`ExplicitEncoderDecoderPrompt`)
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
"""


class TokenInputs(TypedDict):
    """Represents token-based inputs."""

    type: Literal["token"]
    """The type of inputs."""

<<<<<<< HEAD
    prompt_token_ids: List[int]
    """The token IDs of the prompt."""

    token_type_ids: NotRequired[List[int]]
=======
    prompt_token_ids: list[int]
    """The token IDs of the prompt."""

    token_type_ids: NotRequired[list[int]]
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
    """The token type IDs of the prompt."""

    prompt: NotRequired[str]
    """
    The original prompt text corresponding to the token IDs, if available.
    """

<<<<<<< HEAD
    multi_modal_data: NotRequired["MultiModalDataDict"]
    """
    Optional multi-modal data to pass to the model,
    if the model supports it.
    """

    multi_modal_inputs: NotRequired["MultiModalKwargs"]
    """
    Optional multi-modal inputs to pass to the model,
    if the model supports it.
    """

    multi_modal_placeholders: NotRequired["MultiModalPlaceholderDict"]
    """
    Placeholder ranges for the multi-modal data.
    """

    multi_modal_hashes: NotRequired[List[str]]
    """
    The hashes of the multi-modal data.
    """

    mm_processor_kwargs: NotRequired[Dict[str, Any]]
    """
    Optional multi-modal processor kwargs to be forwarded to the
    multimodal input mapper & processor. Note that if multiple modalities
    have registered mappers etc for the model being considered, we attempt
    to pass the mm_processor_kwargs to each of them.
=======
    cache_salt: NotRequired[str]
    """
    Optional cache salt to be used for prefix caching.
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
    """


def token_inputs(
<<<<<<< HEAD
    prompt_token_ids: List[int],
    token_type_ids: Optional[List[int]] = None,
    prompt: Optional[str] = None,
    multi_modal_data: Optional["MultiModalDataDict"] = None,
    multi_modal_inputs: Optional["MultiModalKwargs"] = None,
    multi_modal_hashes: Optional[List[str]] = None,
    multi_modal_placeholders: Optional["MultiModalPlaceholderDict"] = None,
    mm_processor_kwargs: Optional[Dict[str, Any]] = None,
) -> TokenInputs:
    """Construct :class:`TokenInputs` from optional values."""
=======
    prompt_token_ids: list[int],
    token_type_ids: Optional[list[int]] = None,
    prompt: Optional[str] = None,
    cache_salt: Optional[str] = None,
) -> TokenInputs:
    """Construct {class}`TokenInputs` from optional values."""
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
    inputs = TokenInputs(type="token", prompt_token_ids=prompt_token_ids)

    if prompt is not None:
        inputs["prompt"] = prompt
    if token_type_ids is not None:
        inputs["token_type_ids"] = token_type_ids
<<<<<<< HEAD
    if multi_modal_data is not None:
        inputs["multi_modal_data"] = multi_modal_data
    if multi_modal_inputs is not None:
        inputs["multi_modal_inputs"] = multi_modal_inputs
    if multi_modal_hashes is not None:
        inputs["multi_modal_hashes"] = multi_modal_hashes
    if multi_modal_placeholders is not None:
        inputs["multi_modal_placeholders"] = multi_modal_placeholders
    if mm_processor_kwargs is not None:
        inputs["mm_processor_kwargs"] = mm_processor_kwargs
=======
    if cache_salt is not None:
        inputs["cache_salt"] = cache_salt
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea

    return inputs


<<<<<<< HEAD
DecoderOnlyInputs = Union[TokenInputs, "MultiModalInputs"]
"""
The inputs in :class:`~vllm.LLMEngine` before they are
=======
class EmbedsInputs(TypedDict):
    """Represents embeddings-based inputs."""

    type: Literal["embeds"]
    """The type of inputs."""

    prompt_embeds: torch.Tensor
    """The embeddings of the prompt."""

    cache_salt: NotRequired[str]
    """
    Optional cache salt to be used for prefix caching.
    """


def embeds_inputs(
    prompt_embeds: torch.Tensor,
    cache_salt: Optional[str] = None,
) -> EmbedsInputs:
    """Construct :class:`EmbedsInputs` from optional values."""
    inputs = EmbedsInputs(type="embeds", prompt_embeds=prompt_embeds)

    if cache_salt is not None:
        inputs["cache_salt"] = cache_salt

    return inputs


DecoderOnlyInputs = Union[TokenInputs, EmbedsInputs, "MultiModalInputs"]
"""
The inputs in {class}`~vllm.LLMEngine` before they are
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
passed to the model executor.
This specifies the data required for decoder-only models.
"""


class EncoderDecoderInputs(TypedDict):
    """
<<<<<<< HEAD
    The inputs in :class:`~vllm.LLMEngine` before they are
=======
    The inputs in {class}`~vllm.LLMEngine` before they are
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
    passed to the model executor.

    This specifies the required data for encoder-decoder models.
    """
    encoder: Union[TokenInputs, "MultiModalInputs"]
    """The inputs for the encoder portion."""

    decoder: Union[TokenInputs, "MultiModalInputs"]
    """The inputs for the decoder portion."""


<<<<<<< HEAD
SingletonInputs = Union[TokenInputs, "MultiModalInputs"]
"""
A processed :class:`SingletonPrompt` which can be passed to
:class:`vllm.sequence.Sequence`.
"""


@dataclass
class SingletonInputsAdapter:
    """
    Unified interface to access the components of :class:`SingletonInputs`.
    """
    inputs: SingletonInputs

    @cached_property
    def prompt(self) -> Optional[str]:
        inputs = self.inputs

        if inputs["type"] == "token" or inputs["type"] == "multimodal":
            return inputs.get("prompt")

        assert_never(inputs)  # type: ignore[arg-type]

    @cached_property
    def prompt_token_ids(self) -> List[int]:
        inputs = self.inputs

        if inputs["type"] == "token" or inputs["type"] == "multimodal":
            return inputs.get("prompt_token_ids", [])

        assert_never(inputs)  # type: ignore[arg-type]

    @cached_property
    def token_type_ids(self) -> List[int]:
        inputs = self.inputs

        if inputs["type"] == "token" or inputs["type"] == "multimodal":
            return inputs.get("token_type_ids", [])

        assert_never(inputs)  # type: ignore[arg-type]

    @cached_property
    def prompt_embeds(self) -> Optional[torch.Tensor]:
        inputs = self.inputs

        if inputs["type"] == "token" or inputs["type"] == "multimodal":
            return None

        assert_never(inputs)  # type: ignore[arg-type]

    @cached_property
    def multi_modal_data(self) -> "MultiModalDataDict":
        inputs = self.inputs

        if inputs["type"] == "token":
            return inputs.get("multi_modal_data", {})

        if inputs["type"] == "multimodal":
            return inputs.get("mm_kwargs", {})

        assert_never(inputs)  # type: ignore[arg-type]

    @cached_property
    def multi_modal_inputs(self) -> Union[Dict, "MultiModalKwargs"]:
        inputs = self.inputs

        if inputs["type"] == "token":
            return inputs.get("multi_modal_inputs", {})

        if inputs["type"] == "multimodal":
            return inputs.get("mm_kwargs", {})

        assert_never(inputs)  # type: ignore[arg-type]

    @cached_property
    def multi_modal_hashes(self) -> List[str]:
        inputs = self.inputs

        if inputs["type"] == "token":
            return inputs.get("multi_modal_hashes", [])

        if inputs["type"] == "multimodal":
            # only the case when we use MultiModalInputs
            return inputs.get("mm_hashes", [])  # type: ignore[return-value]

        assert_never(inputs)  # type: ignore[arg-type]

    @cached_property
    def multi_modal_placeholders(self) -> "MultiModalPlaceholderDict":
        inputs = self.inputs

        if inputs["type"] == "token":
            return inputs.get("multi_modal_placeholders", {})

        if inputs["type"] == "multimodal":
            return inputs.get("mm_placeholders", {})

        assert_never(inputs)  # type: ignore[arg-type]

    @cached_property
    def mm_processor_kwargs(self) -> Dict[str, Any]:
        inputs = self.inputs

        if inputs["type"] == "token":
            return inputs.get("mm_processor_kwargs", {})

        if inputs["type"] == "multimodal":
            return {}

        assert_never(inputs)  # type: ignore[arg-type]


ProcessorInputs = Union[DecoderOnlyInputs, EncoderDecoderInputs]
"""
The inputs to :data:`vllm.inputs.InputProcessor`.
=======
SingletonInputs = Union[TokenInputs, EmbedsInputs, "MultiModalInputs"]
"""
A processed {class}`SingletonPrompt` which can be passed to
{class}`vllm.sequence.Sequence`.
"""

ProcessorInputs = Union[DecoderOnlyInputs, EncoderDecoderInputs]
"""
The inputs to {data}`vllm.inputs.InputProcessor`.
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
"""

_T1 = TypeVar("_T1", bound=SingletonPrompt, default=SingletonPrompt)
_T2 = TypeVar("_T2", bound=SingletonPrompt, default=SingletonPrompt)


def build_explicit_enc_dec_prompt(
    encoder_prompt: _T1,
    decoder_prompt: Optional[_T2],
<<<<<<< HEAD
    mm_processor_kwargs: Optional[Dict[str, Any]] = None,
=======
    mm_processor_kwargs: Optional[dict[str, Any]] = None,
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
) -> ExplicitEncoderDecoderPrompt[_T1, _T2]:
    if mm_processor_kwargs is None:
        mm_processor_kwargs = {}
    return ExplicitEncoderDecoderPrompt(
        encoder_prompt=encoder_prompt,
        decoder_prompt=decoder_prompt,
        mm_processor_kwargs=mm_processor_kwargs)


def zip_enc_dec_prompts(
    enc_prompts: Iterable[_T1],
    dec_prompts: Iterable[Optional[_T2]],
<<<<<<< HEAD
    mm_processor_kwargs: Optional[Union[Iterable[Dict[str, Any]],
                                        Dict[str, Any]]] = None,
) -> List[ExplicitEncoderDecoderPrompt[_T1, _T2]]:
    """
    Zip encoder and decoder prompts together into a list of
    :class:`ExplicitEncoderDecoderPrompt` instances.
    
=======
    mm_processor_kwargs: Optional[Union[Iterable[dict[str, Any]],
                                        dict[str, Any]]] = None,
) -> list[ExplicitEncoderDecoderPrompt[_T1, _T2]]:
    """
    Zip encoder and decoder prompts together into a list of
    {class}`ExplicitEncoderDecoderPrompt` instances.

>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
    ``mm_processor_kwargs`` may also be provided; if a dict is passed, the same
    dictionary will be used for every encoder/decoder prompt. If an iterable is
    provided, it will be zipped with the encoder/decoder prompts.
    """
    if mm_processor_kwargs is None:
<<<<<<< HEAD
        mm_processor_kwargs = cast(Dict[str, Any], {})
=======
        mm_processor_kwargs = cast(dict[str, Any], {})
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
    if isinstance(mm_processor_kwargs, dict):
        return [
            build_explicit_enc_dec_prompt(
                encoder_prompt, decoder_prompt,
<<<<<<< HEAD
                cast(Dict[str, Any], mm_processor_kwargs))
=======
                cast(dict[str, Any], mm_processor_kwargs))
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
            for (encoder_prompt,
                 decoder_prompt) in zip(enc_prompts, dec_prompts)
        ]
    return [
        build_explicit_enc_dec_prompt(encoder_prompt, decoder_prompt,
                                      mm_proc_kwargs)
        for (encoder_prompt, decoder_prompt, mm_proc_kwargs
             ) in zip(enc_prompts, dec_prompts, mm_processor_kwargs)
    ]


def to_enc_dec_tuple_list(
    enc_dec_prompts: Iterable[ExplicitEncoderDecoderPrompt[_T1, _T2]],
<<<<<<< HEAD
) -> List[Tuple[_T1, Optional[_T2]]]:
=======
) -> list[tuple[_T1, Optional[_T2]]]:
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
    return [(enc_dec_prompt["encoder_prompt"],
             enc_dec_prompt["decoder_prompt"])
            for enc_dec_prompt in enc_dec_prompts]
