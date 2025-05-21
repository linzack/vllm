# SPDX-License-Identifier: Apache-2.0

<<<<<<< HEAD
from .data import (DecoderOnlyInputs, EncoderDecoderInputs,
                   ExplicitEncoderDecoderPrompt, ProcessorInputs, PromptType,
                   SingletonInputs, SingletonInputsAdapter, SingletonPrompt,
                   TextPrompt, TokenInputs, TokensPrompt,
                   build_explicit_enc_dec_prompt, to_enc_dec_tuple_list,
                   token_inputs, zip_enc_dec_prompts)
=======
from .data import (DecoderOnlyInputs, EmbedsInputs, EncoderDecoderInputs,
                   ExplicitEncoderDecoderPrompt, ProcessorInputs, PromptType,
                   SingletonInputs, SingletonPrompt, TextPrompt, TokenInputs,
                   TokensPrompt, build_explicit_enc_dec_prompt, embeds_inputs,
                   to_enc_dec_tuple_list, token_inputs, zip_enc_dec_prompts)
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
from .registry import (DummyData, InputContext, InputProcessingContext,
                       InputRegistry)

INPUT_REGISTRY = InputRegistry()
"""
<<<<<<< HEAD
The global :class:`~InputRegistry` which is used by :class:`~vllm.LLMEngine`
=======
The global {class}`~InputRegistry` which is used by {class}`~vllm.LLMEngine`
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
to dispatch data processing according to the target model.
"""

__all__ = [
    "TextPrompt",
    "TokensPrompt",
    "PromptType",
    "SingletonPrompt",
    "ExplicitEncoderDecoderPrompt",
    "TokenInputs",
<<<<<<< HEAD
    "token_inputs",
=======
    "EmbedsInputs",
    "token_inputs",
    "embeds_inputs",
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
    "DecoderOnlyInputs",
    "EncoderDecoderInputs",
    "ProcessorInputs",
    "SingletonInputs",
<<<<<<< HEAD
    "SingletonInputsAdapter",
=======
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
    "build_explicit_enc_dec_prompt",
    "to_enc_dec_tuple_list",
    "zip_enc_dec_prompts",
    "INPUT_REGISTRY",
    "DummyData",
    "InputContext",
    "InputProcessingContext",
    "InputRegistry",
]
