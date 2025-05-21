# SPDX-License-Identifier: Apache-2.0

<<<<<<< HEAD
import asyncio
import base64
import time
from typing import AsyncGenerator, Final, List, Literal, Optional, Union, cast

import numpy as np
from fastapi import Request
from typing_extensions import assert_never
=======
import base64
from typing import Final, Literal, Optional, Union, cast

import numpy as np
from fastapi import Request
from typing_extensions import assert_never, override
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea

from vllm.config import ModelConfig
from vllm.engine.protocol import EngineClient
from vllm.entrypoints.chat_utils import ChatTemplateContentFormatOption
from vllm.entrypoints.logger import RequestLogger
from vllm.entrypoints.openai.protocol import (EmbeddingChatRequest,
                                              EmbeddingRequest,
                                              EmbeddingResponse,
                                              EmbeddingResponseData,
                                              ErrorResponse, UsageInfo)
<<<<<<< HEAD
from vllm.entrypoints.openai.serving_engine import OpenAIServing
=======
from vllm.entrypoints.openai.serving_engine import (EmbeddingServeContext,
                                                    OpenAIServing,
                                                    ServeContext)
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
from vllm.entrypoints.openai.serving_models import OpenAIServingModels
from vllm.logger import init_logger
from vllm.outputs import (EmbeddingOutput, EmbeddingRequestOutput,
                          PoolingRequestOutput)
<<<<<<< HEAD
from vllm.utils import merge_async_iterators
=======
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea

logger = init_logger(__name__)


def _get_embedding(
    output: EmbeddingOutput,
    encoding_format: Literal["float", "base64"],
<<<<<<< HEAD
) -> Union[List[float], str]:
=======
) -> Union[list[float], str]:
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
    if encoding_format == "float":
        return output.embedding
    elif encoding_format == "base64":
        # Force to use float32 for base64 encoding
        # to match the OpenAI python client behavior
        embedding_bytes = np.array(output.embedding, dtype="float32").tobytes()
        return base64.b64encode(embedding_bytes).decode("utf-8")

    assert_never(encoding_format)


<<<<<<< HEAD
class OpenAIServingEmbedding(OpenAIServing):
=======
class EmbeddingMixin(OpenAIServing):

    async def _preprocess(
        self,
        ctx: ServeContext,
    ) -> Optional[ErrorResponse]:
        ctx = cast(EmbeddingServeContext, ctx)
        try:
            (
                ctx.lora_request,
                ctx.prompt_adapter_request,
            ) = self._maybe_get_adapters(ctx.request)

            tokenizer = await self.engine_client.get_tokenizer(ctx.lora_request
                                                               )

            if ctx.prompt_adapter_request is not None:
                raise NotImplementedError("Prompt adapter is not supported "
                                          "for embedding models")

            if isinstance(ctx.request, EmbeddingChatRequest):
                (
                    _,
                    ctx.request_prompts,
                    ctx.engine_prompts,
                ) = await self._preprocess_chat(
                    ctx.request,
                    tokenizer,
                    ctx.request.messages,
                    chat_template=ctx.request.chat_template
                    or ctx.chat_template,
                    chat_template_content_format=ctx.
                    chat_template_content_format,
                    # In embedding requests, we are not generating tokens,
                    # so there is no need to append extra tokens to the input
                    add_generation_prompt=False,
                    continue_final_message=False,
                    truncate_prompt_tokens=ctx.truncate_prompt_tokens,
                    add_special_tokens=ctx.request.add_special_tokens,
                )
            else:
                (ctx.request_prompts,
                 ctx.engine_prompts) = await self._preprocess_completion(
                     ctx.request,
                     tokenizer,
                     ctx.request.input,
                     truncate_prompt_tokens=ctx.truncate_prompt_tokens,
                     add_special_tokens=ctx.request.add_special_tokens,
                 )
            return None
        except (ValueError, TypeError) as e:
            logger.exception("Error in preprocessing prompt inputs")
            return self.create_error_response(str(e))

    def _build_response(
        self,
        ctx: ServeContext,
    ) -> Union[EmbeddingResponse, ErrorResponse]:
        items: list[EmbeddingResponseData] = []
        num_prompt_tokens = 0

        final_res_batch_checked = cast(list[PoolingRequestOutput],
                                       ctx.final_res_batch)

        for idx, final_res in enumerate(final_res_batch_checked):
            embedding_res = EmbeddingRequestOutput.from_base(final_res)

            item = EmbeddingResponseData(
                index=idx,
                embedding=_get_embedding(embedding_res.outputs,
                                         ctx.request.encoding_format),
            )
            prompt_token_ids = final_res.prompt_token_ids

            items.append(item)
            num_prompt_tokens += len(prompt_token_ids)

        usage = UsageInfo(
            prompt_tokens=num_prompt_tokens,
            total_tokens=num_prompt_tokens,
        )

        return EmbeddingResponse(
            id=ctx.request_id,
            created=ctx.created_time,
            model=ctx.model_name,
            data=items,
            usage=usage,
        )


class OpenAIServingEmbedding(EmbeddingMixin):
    request_id_prefix = "embd"
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea

    def __init__(
        self,
        engine_client: EngineClient,
        model_config: ModelConfig,
        models: OpenAIServingModels,
        *,
        request_logger: Optional[RequestLogger],
        chat_template: Optional[str],
        chat_template_content_format: ChatTemplateContentFormatOption,
    ) -> None:
        super().__init__(engine_client=engine_client,
                         model_config=model_config,
                         models=models,
                         request_logger=request_logger)

        self.chat_template = chat_template
        self.chat_template_content_format: Final = chat_template_content_format

    async def create_embedding(
        self,
        request: EmbeddingRequest,
        raw_request: Optional[Request] = None,
    ) -> Union[EmbeddingResponse, ErrorResponse]:
        """
        Embedding API similar to OpenAI's API.

        See https://platform.openai.com/docs/api-reference/embeddings/create
        for the API specification. This API mimics the OpenAI Embedding API.
        """
<<<<<<< HEAD
        error_check_ret = await self._check_model(request)
        if error_check_ret is not None:
            return error_check_ret

        encoding_format = request.encoding_format
        if request.dimensions is not None:
            return self.create_error_response(
                "dimensions is currently not supported")

        model_name = self._get_model_name(request.model)
        request_id = f"embd-{self._base_request_id(raw_request)}"
        created_time = int(time.time())

        truncate_prompt_tokens = None

        if request.truncate_prompt_tokens is not None:
            if request.truncate_prompt_tokens <= self.max_model_len:
                truncate_prompt_tokens = request.truncate_prompt_tokens
            else:
                return self.create_error_response(
                    "truncate_prompt_tokens value is "
                    "greater than max_model_len."
                    " Please, select a smaller truncation size.")

        try:
            (
                lora_request,
                prompt_adapter_request,
            ) = self._maybe_get_adapters(request)

            tokenizer = await self.engine_client.get_tokenizer(lora_request)

            if prompt_adapter_request is not None:
                raise NotImplementedError("Prompt adapter is not supported "
                                          "for embedding models")

            if isinstance(request, EmbeddingChatRequest):
                (
                    _,
                    request_prompts,
                    engine_prompts,
                ) = await self._preprocess_chat(
                    request,
                    tokenizer,
                    request.messages,
                    chat_template=request.chat_template or self.chat_template,
                    chat_template_content_format=self.
                    chat_template_content_format,
                    # In embedding requests, we are not generating tokens,
                    # so there is no need to append extra tokens to the input
                    add_generation_prompt=False,
                    continue_final_message=False,
                    truncate_prompt_tokens=truncate_prompt_tokens,
                    add_special_tokens=request.add_special_tokens,
                )
            else:
                (request_prompts,
                 engine_prompts) = await self._preprocess_completion(
                     request,
                     tokenizer,
                     request.input,
                     truncate_prompt_tokens=truncate_prompt_tokens,
                     add_special_tokens=request.add_special_tokens,
                 )
        except ValueError as e:
            logger.exception("Error in preprocessing prompt inputs")
            return self.create_error_response(str(e))

        # Schedule the request and get the result generator.
        generators: List[AsyncGenerator[PoolingRequestOutput, None]] = []
        try:
            pooling_params = request.to_pooling_params()

            for i, engine_prompt in enumerate(engine_prompts):
                request_id_item = f"{request_id}-{i}"

                self._log_inputs(request_id_item,
                                 request_prompts[i],
                                 params=pooling_params,
                                 lora_request=lora_request,
                                 prompt_adapter_request=prompt_adapter_request)

                trace_headers = (None if raw_request is None else await
                                 self._get_trace_headers(raw_request.headers))

                generator = self.engine_client.encode(
                    engine_prompt,
                    pooling_params,
                    request_id_item,
                    lora_request=lora_request,
                    trace_headers=trace_headers,
                    priority=request.priority,
                )

                generators.append(generator)
        except ValueError as e:
            # TODO: Use a vllm-specific Validation Error
            return self.create_error_response(str(e))

        result_generator = merge_async_iterators(*generators)

        num_prompts = len(engine_prompts)

        # Non-streaming response
        final_res_batch: List[Optional[PoolingRequestOutput]]
        final_res_batch = [None] * num_prompts
        try:
            async for i, res in result_generator:
                final_res_batch[i] = res

            assert all(final_res is not None for final_res in final_res_batch)

            final_res_batch_checked = cast(List[PoolingRequestOutput],
                                           final_res_batch)

            response = self.request_output_to_embedding_response(
                final_res_batch_checked,
                request_id,
                created_time,
                model_name,
                encoding_format,
            )
        except asyncio.CancelledError:
            return self.create_error_response("Client disconnected")
        except ValueError as e:
            # TODO: Use a vllm-specific Validation Error
            return self.create_error_response(str(e))

        return response

    def request_output_to_embedding_response(
        self,
        final_res_batch: List[PoolingRequestOutput],
        request_id: str,
        created_time: int,
        model_name: str,
        encoding_format: Literal["float", "base64"],
    ) -> EmbeddingResponse:
        items: List[EmbeddingResponseData] = []
        num_prompt_tokens = 0

        for idx, final_res in enumerate(final_res_batch):
            embedding_res = EmbeddingRequestOutput.from_base(final_res)

            item = EmbeddingResponseData(
                index=idx,
                embedding=_get_embedding(embedding_res.outputs,
                                         encoding_format),
            )
            prompt_token_ids = final_res.prompt_token_ids

            items.append(item)
            num_prompt_tokens += len(prompt_token_ids)

        usage = UsageInfo(
            prompt_tokens=num_prompt_tokens,
            total_tokens=num_prompt_tokens,
        )

        return EmbeddingResponse(
            id=request_id,
            created=created_time,
            model=model_name,
            data=items,
            usage=usage,
        )
=======
        model_name = self._get_model_name(request.model)
        request_id = (f"{self.request_id_prefix}-"
                      f"{self._base_request_id(raw_request)}")

        ctx = EmbeddingServeContext(
            request=request,
            raw_request=raw_request,
            model_name=model_name,
            request_id=request_id,
            chat_template=self.chat_template,
            chat_template_content_format=self.chat_template_content_format,
        )

        return await super().handle(ctx)  # type: ignore

    @override
    def _validate_request(
        self,
        ctx: ServeContext[EmbeddingRequest],
    ) -> Optional[ErrorResponse]:
        if error := super()._validate_request(ctx):
            return error

        ctx.truncate_prompt_tokens = ctx.request.truncate_prompt_tokens

        pooling_params = ctx.request.to_pooling_params()

        try:
            pooling_params.verify(self.model_config)
        except ValueError as e:
            return self.create_error_response(str(e))

        return None
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
