# SPDX-License-Identifier: Apache-2.0
<<<<<<< HEAD
import enum
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple
=======
import bisect
import gc
import time
from typing import TYPE_CHECKING, Optional, cast
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
from unittest.mock import patch

import numpy as np
import torch
import torch.distributed
import torch.nn as nn
# TPU XLA related
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr

<<<<<<< HEAD
from vllm.attention import AttentionMetadata
from vllm.attention.backends.abstract import AttentionType
from vllm.attention.layer import Attention
from vllm.config import VllmConfig
from vllm.forward_context import set_forward_context
from vllm.logger import init_logger
from vllm.model_executor.model_loader import get_model
from vllm.sampling_params import SamplingType
from vllm.utils import LayerBlockType, cdiv, is_pin_memory_available
from vllm.v1.attention.backends.pallas import (PallasAttentionBackend,
                                               PallasMetadata)
from vllm.v1.kv_cache_interface import (FullAttentionSpec, KVCacheConfig,
                                        KVCacheSpec)
from vllm.v1.outputs import LogprobsTensors, ModelRunnerOutput
from vllm.v1.utils import bind_kv_cache
from vllm.v1.worker.gpu_input_batch import CachedRequestState, InputBatch

if TYPE_CHECKING:
    from vllm.v1.core.scheduler import SchedulerOutput
=======
import vllm.envs as envs
from vllm.attention.backends.abstract import AttentionType
from vllm.attention.layer import Attention
from vllm.compilation.wrapper import TorchCompileWrapperWithCustomDispatcher
from vllm.config import VllmConfig, get_layers_from_vllm_config
from vllm.forward_context import set_forward_context
from vllm.logger import init_logger
from vllm.model_executor.model_loader import get_model
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (BatchedTensorInputs, MultiModalKwargs,
                                    PlaceholderRange)
from vllm.multimodal.utils import group_mm_inputs_by_modality
from vllm.sequence import IntermediateTensors
from vllm.utils import LayerBlockType, cdiv, is_pin_memory_available
from vllm.v1.attention.backends.pallas import (PallasAttentionBackend,
                                               PallasMetadata)
from vllm.v1.core.encoder_cache_manager import compute_encoder_budget
from vllm.v1.kv_cache_interface import (AttentionSpec, FullAttentionSpec,
                                        KVCacheConfig, KVCacheSpec,
                                        SlidingWindowSpec)
from vllm.v1.outputs import (EMPTY_MODEL_RUNNER_OUTPUT, LogprobsTensors,
                             ModelRunnerOutput)
from vllm.v1.sample.tpu.metadata import TPUSupportedSamplingMetadata
from vllm.v1.sample.tpu.sampler import Sampler as TPUSampler
from vllm.v1.utils import bind_kv_cache
from vllm.v1.worker.gpu_input_batch import CachedRequestState, InputBatch
from vllm.v1.worker.lora_model_runner_mixin import LoRAModelRunnerMixin

from .utils import sanity_check_mm_encoder_outputs

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea

logger = init_logger(__name__)

# Here we utilize the behavior that out-of-bound index is ignored.
# FIXME(woosuk): Find a more reliable way to prevent possible bugs.
_PAD_SLOT_ID = 1_000_000_000
<<<<<<< HEAD


class ExecutionMode(enum.Enum):
    PREFILL = enum.auto()
    DECODE = enum.auto()
    PREFIX_PREFILL = enum.auto()

    def is_prefill(self) -> bool:
        return self in (ExecutionMode.PREFILL, ExecutionMode.PREFIX_PREFILL)


@dataclass
class PromptDecodeInfo:
    prompt_req_ids: List[str]
    decode_req_ids: List[str]
    prompt_scheduled_tokens: List[int]


@dataclass
class PromptData:
    input_tokens: torch.Tensor
    input_positions: torch.Tensor
    attn_metadata: PallasMetadata


@dataclass
class DecodeData:
    input_tokens: Optional[torch.Tensor] = None
    input_positions: Optional[torch.Tensor] = None
    attn_metadata: Optional[PallasMetadata] = None


class TPUModelRunner:
=======
INVALID_TOKEN_ID = -1
# Smallest output size
MIN_NUM_SEQS = 8


#########################################################
# Ways to avoid recompilation
#########################################################
#
# The model executor has two primary components:
# 1. preparing the model and sampler inputs
# 2. executing the model and sampler.
# The core idea is to avoid any TPU computation during input preparation. For
# better compilation tracking and increased flexibility, the model execution and
# sampler are divided into several distinct components.
#
# Below are the detailed steps:
#
# Step 1
# It is recommended to avoid TPU operations when preparing the model and sampler
# inputs. CPU tensors can be prepared and transferred to the XLA device using
# cpu_tensor.to(xla_device), which only triggers CPU to TPU transfers and avoids
# compilation.
#
# Step 2
# The TPU execution should be decomposed into subgraphs (4 at the moment):
# 1. the main model
# 2. selecting hidden states for each request
# 3. sampler
# 4. encoder.
# Each subgraph should be decorated in a torch.compile. This is used to make
# sure that we have the same subgraph topology in both dummy_run and
# xecute_model. The results from these subgraphs should either be passed to
# other subgraphs, or transferred from TPU to CPU using xla_tensor.cpu() for
# subsequent processing on the CPU.
#
# Step 3
# The dummy_run should be comprehensive, ensuring all potential input shapes and
# branch predictions are included as subgraph inputs to facilitate
# pre-compilation.
class TPUModelRunner(LoRAModelRunnerMixin):
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.cache_config = vllm_config.cache_config
        self.lora_config = vllm_config.lora_config
        self.load_config = vllm_config.load_config
        self.parallel_config = vllm_config.parallel_config
        self.scheduler_config = vllm_config.scheduler_config
        self.speculative_config = vllm_config.speculative_config
        self.prompt_adapter_config = vllm_config.prompt_adapter_config
        self.observability_config = vllm_config.observability_config
        self.device_config = vllm_config.device_config

        model_config = self.model_config
        cache_config = self.cache_config
        scheduler_config = self.scheduler_config
        parallel_config = self.parallel_config
        self.device = device
<<<<<<< HEAD
        self.pin_memory = is_pin_memory_available()
        self.dtype = self.model_config.dtype
=======
        self.check_recompilation = envs.VLLM_XLA_CHECK_RECOMPILATION

        self.enforce_eager = model_config.enforce_eager

        self.num_xla_graphs = 0
        self._update_num_xla_graphs("init")

        self.pin_memory = is_pin_memory_available()
        self.dtype = self.model_config.dtype
        self._hidden_states_dtype = self.dtype
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea

        self.is_multimodal_model = model_config.is_multimodal_model
        self.sliding_window = model_config.get_sliding_window()
        self.block_size = cache_config.block_size
        self.max_model_len = model_config.max_model_len
        self.max_num_blocks_per_req = cdiv(self.max_model_len, self.block_size)
<<<<<<< HEAD
        self.max_num_tokens = scheduler_config.max_num_batched_tokens
        self.max_num_reqs = scheduler_config.max_num_seqs
=======
        # InputBatch needs to work with sampling tensors greater than padding
        # to avoid dynamic shapes. Also, avoid suboptimal alignment.
        self.max_num_reqs = max(scheduler_config.max_num_seqs, MIN_NUM_SEQS)
        self.num_tokens_paddings = _get_token_paddings(
            min_token_size=16,
            max_token_size=scheduler_config.max_num_batched_tokens,
            padding_gap=envs.VLLM_TPU_BUCKET_PADDING_GAP)
        # In case `max_num_tokens < max(num_tokens_paddings)` use the actual
        # padded max value to pre-allocate data structures and pre-compile.
        self.max_num_tokens = self.num_tokens_paddings[-1]
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea

        # Model-related.
        self.num_attn_layers = model_config.get_num_layers_by_block_type(
            parallel_config, LayerBlockType.attention)
        self.num_query_heads = model_config.get_num_attention_heads(
            parallel_config)
        self.num_kv_heads = model_config.get_num_kv_heads(parallel_config)
        self.head_size = model_config.get_head_size()
        self.hidden_size = model_config.get_hidden_size()
<<<<<<< HEAD

        self.model: Optional[nn.Module] = None

        # Persistent batch.
        self.input_batch = InputBatch(
            max_num_reqs=self.max_num_reqs,
            max_model_len=self.max_model_len,
            max_num_blocks_per_req=self.max_num_blocks_per_req,
            device=self.device,
            pin_memory=self.pin_memory,
            vocab_size=self.model_config.get_vocab_size(),
        )

        # Request states.
        self.requests: Dict[str, CachedRequestState] = {}

        # req_id -> (input_id -> encoder_output)
        self.encoder_cache: Dict[str, Dict[int, torch.Tensor]] = {}

        # KV caches for forward pass
        self.kv_caches: List[Tuple[torch.Tensor, torch.Tensor]] = []

        # Cached torch/numpy tensors
        self.num_swaps = 2
        self.cur_swap_id = 0
        self.input_ids_cpu = []
        self.input_ids_np = []
        self.input_positions_cpu = []
        self.input_positions_np = []
        self.slot_mapping_cpu = []
        self.slot_mapping_np = []
        self.prompt_context_lens_cpu = []
        self.prompt_effective_query_lens_cpu = []
        self.decode_context_lens_cpu = []
        self.decode_context_lens_np = []
        for _ in range(self.num_swaps):
            self.input_ids_cpu.append(
                torch.empty(self.max_num_tokens,
                            dtype=torch.int32,
                            device="cpu"))
            self.input_ids_np.append(self.input_ids_cpu[-1].numpy())

            self.input_positions_cpu.append(
                torch.empty(self.max_num_tokens,
                            dtype=torch.int32,
                            device="cpu"))
            self.input_positions_np.append(
                self.input_positions_cpu[-1].numpy())

            self.slot_mapping_cpu.append(
                torch.empty(self.max_num_tokens,
                            dtype=torch.int64,
                            device="cpu"))
            self.slot_mapping_np.append(self.slot_mapping_cpu[-1].numpy())

            self.prompt_context_lens_cpu.append(
                torch.empty((1), dtype=torch.int32, device="cpu"))
            self.prompt_effective_query_lens_cpu.append(
                torch.empty((1), dtype=torch.int32, device="cpu"))

            self.decode_context_lens_cpu.append(
                torch.empty(self.max_num_tokens,
                            dtype=torch.int32,
                            device="cpu"))
            self.decode_context_lens_np.append(
                self.decode_context_lens_cpu[-1].numpy())

        # Range tensor with values [0 .. self.max_num_tokens - 1].
        # Used to initialize positions / context_lens / seq_lens
        self.arange_np = np.arange(self.max_num_tokens, dtype=np.int32)
=======
        self.vocab_size = model_config.get_vocab_size()

        # Multi-modal data support
        self.mm_registry = MULTIMODAL_REGISTRY
        self.uses_mrope = model_config.uses_mrope
        # TODO: Support M-RoPE (e.g, Qwen2-VL)
        assert not self.uses_mrope, "TPU does not support M-RoPE yet."

        encoder_compute_budget, encoder_cache_size = compute_encoder_budget(
            model_config=model_config,
            scheduler_config=scheduler_config,
            mm_registry=self.mm_registry,
        )
        self.max_num_encoder_input_tokens = encoder_compute_budget
        self.encoder_cache_size = encoder_cache_size

        # Lazy initialization
        # self.model: nn.Module  # Set after load_model
        self.kv_caches: list[torch.Tensor] = []
        # req_id -> (input_id -> encoder_output)
        self.encoder_cache: dict[str, dict[int, torch.Tensor]] = {}
        # self.input_batch: InputBatch  # Persistent batch.

        # Request states.
        self.requests: dict[str, CachedRequestState] = {}

        # Cached torch/numpy tensor
        # The pytorch tensor and numpy array share the same buffer.
        # Sometimes the numpy op is faster so we create both.
        self.input_ids_cpu = torch.zeros(self.max_num_tokens,
                                         dtype=torch.int32,
                                         device="cpu")

        self.positions_cpu = torch.zeros(self.max_num_tokens,
                                         dtype=torch.int32,
                                         device="cpu")
        self.positions_np = self.positions_cpu.numpy()

        self.block_table_cpu = torch.zeros(
            (self.max_num_reqs, self.max_num_blocks_per_req),
            dtype=torch.int32,
            device="cpu")

        self.query_start_loc_cpu = torch.zeros(self.max_num_tokens + 1,
                                               dtype=torch.int32,
                                               device="cpu",
                                               pin_memory=self.pin_memory)
        self.query_start_loc_np = self.query_start_loc_cpu.numpy()

        self.seq_lens_cpu = torch.zeros(self.max_num_tokens,
                                        dtype=torch.int32,
                                        device="cpu",
                                        pin_memory=self.pin_memory)
        self.seq_lens_np = self.seq_lens_cpu.numpy()

        # Range tensor with values [0 .. self.max_num_tokens - 1].
        # Used to initialize positions / context_lens / seq_lens
        # Keep in int64 to avoid overflow with long context
        self.arange_np = np.arange(self.max_num_tokens, dtype=np.int64)
        self.num_reqs_paddings = _get_req_paddings(
            min_req_size=MIN_NUM_SEQS, max_req_size=self.max_num_reqs)

        # tensors for structured decoding
        self.grammar_bitmask_cpu = torch.zeros(
            (self.max_num_reqs, cdiv(self.vocab_size, 32)),
            dtype=torch.int32,
            device="cpu",
            pin_memory=self.pin_memory)
        self.require_structured_out_cpu = torch.zeros(
            (self.max_num_reqs, 1),
            dtype=torch.bool,
            device="cpu",
            pin_memory=self.pin_memory)
        self.structured_decode_arange = torch.arange(
            0, 32, device="cpu", pin_memory=self.pin_memory)

        # Get maximum number of mm items per modality (batch size).
        self.max_num_mm_items_by_modality = dict()
        if (self.is_multimodal_model and self.max_num_encoder_input_tokens > 0
                and self.encoder_cache_size > 0):
            max_tokens_by_modality_dict = (
                MULTIMODAL_REGISTRY.
                get_max_tokens_per_item_by_nonzero_modality(self.model_config))
            for modality, max_tokens in max_tokens_by_modality_dict.items():
                # Check how many items of this modality can be supported by
                # the encoder budget.
                encoder_budget = min(self.max_num_encoder_input_tokens,
                                     self.encoder_cache_size)

                max_num_mm_items_encoder_budget = cdiv(encoder_budget,
                                                       max_tokens)

                # Check how many items of this modality can be supported by
                # the decoder budget.
                max_mm_items_per_req = self.mm_registry.\
                    get_mm_limits_per_prompt(self.model_config)[modality]

                # NOTE: We do not consider max_num_batched_tokens on purpose
                # because the multimodal embeddings can be generated in advance
                # and chunked prefilled.
                max_num_mm_items_decoder_budget = self.max_num_reqs * \
                    max_mm_items_per_req

                max_num_mm_items = min(max_num_mm_items_encoder_budget,
                                       max_num_mm_items_decoder_budget)
                self.max_num_mm_items_by_modality[modality] = max_num_mm_items

    def _update_num_xla_graphs(self, case_str):
        check_comp = self.check_recompilation and not self.enforce_eager
        if not check_comp:
            return

        total_cached_graphs = xr.get_num_cached_compilation_graph()
        new_compiled_graphs = total_cached_graphs - self.num_xla_graphs
        if new_compiled_graphs == 0:
            return

        logger.info("Add new %d compiled XLA graphs due to %s",
                    new_compiled_graphs, case_str)
        self.num_xla_graphs += new_compiled_graphs

    def _verify_num_xla_graphs(self, case_str):
        check_comp = self.check_recompilation and not self.enforce_eager
        if not check_comp:
            return

        curr_cached_graph = xr.get_num_cached_compilation_graph()
        assert self.num_xla_graphs == curr_cached_graph, (
            "Recompilation after warm up is detected during {}."
            " num_xla_graphs = {} curr_cached_graph = {}".format(
                case_str, self.num_xla_graphs, curr_cached_graph))
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea

    def _update_states(self, scheduler_output: "SchedulerOutput") -> bool:
        """Update the cached states and the persistent batch with the scheduler
        output.

        The updated states are used by the `_prepare_inputs` function to create
        the input GPU tensors for the model.

        Returns:
<<<<<<< HEAD
            True if there is a new/resumed/paused/finished request in the batch.
=======
            True if there is a new/resumed/paused/finished request.
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
            If False, we can skip copying SamplingMetadata to the GPU.
        """
        # Remove finished requests from the cached states.
        for req_id in scheduler_output.finished_req_ids:
            self.requests.pop(req_id, None)
<<<<<<< HEAD
=======
            self.encoder_cache.pop(req_id, None)
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea

        # Remove the finished requests from the persistent batch.
        # NOTE(woosuk): There could be an edge case where finished_req_ids and
        # scheduled_req_ids overlap. This happens when a request is aborted and
        # then resubmitted with the same ID. In this case, we treat them as two
        # distinct requests - clearing the cached states for the first request
        # and handling the second as a new request.
<<<<<<< HEAD
        removed_req_indices: List[int] = []
=======
        removed_req_indices: list[int] = []
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
        for req_id in scheduler_output.finished_req_ids:
            req_index = self.input_batch.remove_request(req_id)
            if req_index is not None:
                removed_req_indices.append(req_index)

<<<<<<< HEAD
=======
        # Free the cached encoder outputs.
        for req_id, input_id in scheduler_output.free_encoder_input_ids:
            encoder_outputs = self.encoder_cache.get(req_id)
            if encoder_outputs is not None:
                encoder_outputs.pop(input_id, None)
                if not encoder_outputs:
                    self.encoder_cache.pop(req_id, None)

>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
        # Remove the unscheduled requests from the persistent batch.
        # NOTE(woosuk): The unscheduled requests are either preempted requests
        # or running requests that are not scheduled in this step. We remove
        # them from the persistent batch but keep their cached states since
        # they will be scheduled again sometime in the future.
        scheduled_req_ids = scheduler_output.num_scheduled_tokens.keys()
        cached_req_ids = self.input_batch.req_id_to_index.keys()
        unscheduled_req_ids = cached_req_ids - scheduled_req_ids
        # NOTE(woosuk): The persistent batch optimization assumes that
        # consecutive batches contain mostly the same requests. If batches
        # have low request overlap (e.g., alternating between two distinct
        # sets of requests), this optimization becomes very inefficient.
        for req_id in unscheduled_req_ids:
            req_index = self.input_batch.remove_request(req_id)
            assert req_index is not None
            removed_req_indices.append(req_index)

<<<<<<< HEAD
        req_ids_to_add: List[str] = []
=======
        req_ids_to_add: list[str] = []
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
        # Add new requests to the cached states.
        for new_req_data in scheduler_output.scheduled_new_reqs:
            req_id = new_req_data.req_id
            sampling_params = new_req_data.sampling_params
<<<<<<< HEAD
            if sampling_params.sampling_type == SamplingType.RANDOM_SEED:
                generator = torch.Generator(device=self.device)
                generator.manual_seed(sampling_params.seed)
            else:
                generator = None
=======
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea

            self.requests[req_id] = CachedRequestState(
                req_id=req_id,
                prompt_token_ids=new_req_data.prompt_token_ids,
<<<<<<< HEAD
                prompt=new_req_data.prompt,
                mm_inputs=new_req_data.mm_inputs,
                mm_positions=new_req_data.mm_positions,
                sampling_params=sampling_params,
                generator=generator,
=======
                mm_inputs=new_req_data.mm_inputs,
                mm_positions=new_req_data.mm_positions,
                sampling_params=sampling_params,
                generator=None,
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
                block_ids=new_req_data.block_ids,
                num_computed_tokens=new_req_data.num_computed_tokens,
                output_token_ids=[],
                lora_request=new_req_data.lora_request,
            )

            req_ids_to_add.append(req_id)

        # Update the states of the running/resumed requests.
        for req_data in scheduler_output.scheduled_cached_reqs:
            req_id = req_data.req_id
            req_state = self.requests[req_id]

            # Update the cached states.
            req_state.num_computed_tokens = req_data.num_computed_tokens
            if not req_data.resumed_from_preemption:
                # Append the new blocks to the existing block IDs.
                req_state.block_ids.extend(req_data.new_block_ids)
            else:
                # The request is resumed from preemption.
                # Replace the existing block IDs with the new ones.
                req_state.block_ids = req_data.new_block_ids

            req_index = self.input_batch.req_id_to_index.get(req_id)
            if req_index is None:
                # The request is not in the persistent batch.
                # The request was either preempted and resumed later, or was not
                # scheduled in the previous step and needs to be added again.
                req_ids_to_add.append(req_id)
                continue

            # Update the persistent batch.
            self.input_batch.num_computed_tokens_cpu[req_index] = (
                req_data.num_computed_tokens)
<<<<<<< HEAD
            start_index = len(req_state.block_ids) - len(
                req_data.new_block_ids)
            self.input_batch.block_table.append_row(req_index, start_index,
                                                    req_data.new_block_ids)
=======
            self.input_batch.block_table.append_row(req_data.new_block_ids,
                                                    req_index)
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea

        # Add the new or resumed requests to the persistent batch.
        # The smaller empty indices are filled first.
        removed_req_indices = sorted(removed_req_indices, reverse=True)
        for req_id in req_ids_to_add:
            req_state = self.requests[req_id]
            if removed_req_indices:
                # Fill the empty index.
                req_index = removed_req_indices.pop()
            else:
                # Append to the end.
                req_index = None
            self.input_batch.add_request(req_state, req_index)

        # Condense the batched states if there are empty indices.
        if removed_req_indices:
            self.input_batch.condense(removed_req_indices)
<<<<<<< HEAD
        return len(unscheduled_req_ids) > 0 or len(req_ids_to_add) > 0

    def swap_step(self):
        self.cur_swap_id = (self.cur_swap_id + 1) % self.num_swaps
=======

        return len(unscheduled_req_ids) > 0 or len(req_ids_to_add) > 0
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea

    def get_model(self) -> nn.Module:
        assert self.model is not None
        return self.model

<<<<<<< HEAD
    def get_kv_cache_spec(self) -> KVCacheSpec:
        """
        Generates the KVCacheSpec by parsing the kv cache format from each 
        Attention module in the static forward context.
        Returns:
            KVCacheSpec: A dictionary mapping layer names to their KV cache 
            format. Layers that do not need KV cache are not included.
        """

        forward_ctx = self.vllm_config.compilation_config.static_forward_context
        block_size = self.vllm_config.cache_config.block_size
        kv_cache_spec: KVCacheSpec = {}
        for layer_name, attn_module in forward_ctx.items():
            # TODO: Support other attention modules, e.g., sliding window,
            # cross-attention, MLA.
            assert isinstance(attn_module, Attention)
            if attn_module.attn_type == AttentionType.DECODER:
                kv_cache_spec[layer_name] = FullAttentionSpec(
                    block_size=block_size,
                    num_kv_heads=attn_module.num_kv_heads,
                    head_size=attn_module.head_size,
                    dtype=attn_module.dtype,
                )
=======
    def get_kv_cache_spec(self) -> dict[str, KVCacheSpec]:
        """
        Generates the KVCacheSpec by parsing the kv cache format from each
        Attention module in the static forward context.
        Returns:
            KVCacheSpec: A dictionary mapping layer names to their KV cache
            format. Layers that do not need KV cache are not included.
        """

        layers = get_layers_from_vllm_config(self.vllm_config, Attention)
        block_size = self.vllm_config.cache_config.block_size
        kv_cache_spec: dict[str, KVCacheSpec] = {}
        for layer_name, attn_module in layers.items():
            if attn_module.attn_type == AttentionType.DECODER:
                if attn_module.sliding_window is not None:
                    kv_cache_spec[layer_name] = SlidingWindowSpec(
                        block_size=block_size,
                        num_kv_heads=attn_module.num_kv_heads,
                        head_size=attn_module.head_size,
                        dtype=attn_module.dtype,
                        sliding_window=attn_module.sliding_window,
                        use_mla=False,
                    )
                else:
                    kv_cache_spec[layer_name] = FullAttentionSpec(
                        block_size=block_size,
                        num_kv_heads=attn_module.num_kv_heads,
                        head_size=attn_module.head_size,
                        dtype=attn_module.dtype,
                        use_mla=False,
                    )
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
            elif attn_module.attn_type in (AttentionType.ENCODER,
                                           AttentionType.ENCODER_ONLY):
                # encoder-only attention does not need KV cache.
                continue
            elif attn_module.attn_type == AttentionType.ENCODER_DECODER:
                raise NotImplementedError
            else:
                raise ValueError(
                    f"Unknown attention type: {attn_module.attn_type}")

        return kv_cache_spec

<<<<<<< HEAD
    def _get_prompts_and_decodes(
        self,
        scheduler_output: "SchedulerOutput",
    ) -> PromptDecodeInfo:
=======
    def _prepare_inputs(self, scheduler_output: "SchedulerOutput"):
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
        total_num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens
        assert total_num_scheduled_tokens > 0
        num_reqs = self.input_batch.num_reqs
        assert num_reqs > 0

<<<<<<< HEAD
        # Traverse decodes first
        decode_req_ids = []
        for i in range(num_reqs):
            req_id = self.input_batch.req_ids[i]
            assert req_id is not None

            num_computed_tokens = self.input_batch.num_computed_tokens_cpu[i]
            num_prompt_tokens = self.input_batch.num_prompt_tokens[i]
            num_scheduled_tokens = scheduler_output.num_scheduled_tokens[
                req_id]

            if num_computed_tokens < num_prompt_tokens:
                # This is prompt
                break

            # This is decode
            assert num_scheduled_tokens == 1
            decode_req_ids.append(req_id)

        # Traverse prompts
        prompt_req_ids = []
        prompt_scheduled_tokens = []
        for i in range(len(decode_req_ids), num_reqs):
            req_id = self.input_batch.req_ids[i]
            assert req_id is not None

            num_computed_tokens = self.input_batch.num_computed_tokens_cpu[i]
            num_prompt_tokens = self.input_batch.num_prompt_tokens[i]
            num_scheduled_tokens = scheduler_output.num_scheduled_tokens[
                req_id]

            # Must be prompt
            assert num_computed_tokens < num_prompt_tokens

            prompt_req_ids.append(req_id)
            prompt_scheduled_tokens.append(num_scheduled_tokens)

        return PromptDecodeInfo(prompt_req_ids, decode_req_ids,
                                prompt_scheduled_tokens)

    def _prepare_prompt(self, req_index: int,
                        num_scheduled_tokens: int) -> PromptData:
        num_computed_tokens = self.input_batch.num_computed_tokens_cpu[
            req_index]
        num_prompt_tokens = self.input_batch.num_prompt_tokens[req_index]

        # Must be prompt
        assert num_computed_tokens < num_prompt_tokens

        # Prompt len
        prompt_len = num_scheduled_tokens
        padded_prompt_len = _get_padded_prompt_len(prompt_len)
        assert padded_prompt_len <= self.max_model_len

        # Seq len
        seq_len = num_computed_tokens + prompt_len
        padded_seq_len = num_computed_tokens + padded_prompt_len

        # Input tokens
        input_tokens_cpu = self.input_batch.token_ids_cpu_tensor[
            req_index, num_computed_tokens:padded_seq_len]
        input_tokens_cpu[prompt_len:] = 0

        # Input positions
        input_positions_np = self.input_positions_np[
            self.cur_swap_id][:padded_prompt_len]
        np.add(num_computed_tokens,
               self.arange_np[:padded_prompt_len],
               out=input_positions_np)
        input_positions_np[prompt_len:] = 0

        # Slot mapping
        block_table_np = \
            self.input_batch.block_table.get_numpy_array()
        block_numbers_np = block_table_np[req_index, input_positions_np //
                                          self.block_size]
        block_offsets_np = input_positions_np % self.block_size

        slot_mapping_np = self.slot_mapping_np[
            self.cur_swap_id][:padded_prompt_len]
        np.add(block_numbers_np * self.block_size,
               block_offsets_np,
               out=slot_mapping_np)
        slot_mapping_np[prompt_len:] = _PAD_SLOT_ID

        # Block table
        block_table_cpu = None
        if num_computed_tokens > 0:
            block_table_cpu = self.input_batch.block_table.get_cpu_tensor()
            block_table_cpu = block_table_cpu[req_index]

        # Context len
        self.prompt_context_lens_cpu[self.cur_swap_id][0] = 0
        if num_computed_tokens > 0:
            self.prompt_context_lens_cpu[self.cur_swap_id][0] = seq_len

        # Effective query len
        self.prompt_effective_query_lens_cpu[self.cur_swap_id][0] = prompt_len

        # Get final tensors
        input_tokens = input_tokens_cpu.reshape(1, -1).to(self.device)
        input_positions = self.input_positions_cpu[
            self.cur_swap_id][:padded_prompt_len].reshape(1,
                                                          -1).to(self.device)
        slot_mapping = self.slot_mapping_cpu[
            self.cur_swap_id][:padded_prompt_len].reshape(1,
                                                          -1).to(self.device)
        block_table = block_table_cpu.reshape(1, -1).to(
            self.device) if block_table_cpu is not None else None

        context_lens = self.prompt_context_lens_cpu[self.cur_swap_id].to(
            self.device)
        effective_query_lens = self.prompt_effective_query_lens_cpu[
            self.cur_swap_id].to(self.device)

        self.swap_step()

        # Attn metadata
        attn_metadata = PallasMetadata(
            num_prefills=1,
            num_prefill_tokens=0,  # NOTE: This is not used.
            num_decode_tokens=0,
            slot_mapping=slot_mapping,
            multi_modal_placeholder_index_maps=None,
            enable_kv_scales_calculation=True,
            block_tables=block_table,
            context_lens=context_lens,
            effective_query_lens=effective_query_lens,
        )

        return PromptData(input_tokens, input_positions, attn_metadata)

    def _prepare_decode(
        self,
        decode_req_ids: List[str],
    ) -> DecodeData:
        # Batch size
        batch_size = len(decode_req_ids)
        padded_batch_size = _get_padded_batch_size(batch_size)
        assert padded_batch_size <= self.max_model_len

        # Init [0 .. batch_size - 1]
        req_indices_np = self.arange_np[:padded_batch_size]

        # Input positions
        input_positions_np = self.input_positions_np[
            self.cur_swap_id][:padded_batch_size]
        np.add(self.input_batch.num_computed_tokens_cpu[:padded_batch_size],
               0,
               out=input_positions_np)
        input_positions_np[batch_size:] = 0
        input_positions_cpu = self.input_positions_cpu[
            self.cur_swap_id][:padded_batch_size]

        # Input tokens
        token_indices_np = (
            input_positions_np +
            req_indices_np * self.input_batch.token_ids_cpu.shape[1])
        input_tokens_cpu = self.input_ids_cpu[
            self.cur_swap_id][:padded_batch_size]
        torch.index_select(self.input_batch.token_ids_cpu_tensor.flatten(),
                           0,
                           torch.from_numpy(token_indices_np),
                           out=input_tokens_cpu)
        input_tokens_cpu[batch_size:] = 0

        # Slot mapping
        block_table_indices_np = (
            req_indices_np * self.max_num_blocks_per_req +
            input_positions_np // self.block_size)

        block_table_cpu = self.input_batch.block_table.get_cpu_tensor()

        block_numbers_np = block_table_cpu.flatten(
        )[block_table_indices_np].numpy()

        block_offsets_np = input_positions_np % self.block_size

        slot_mapping_np = self.slot_mapping_np[
            self.cur_swap_id][:padded_batch_size]
        np.add(block_numbers_np * self.block_size,
               block_offsets_np,
               out=slot_mapping_np)
        slot_mapping_np[batch_size:] = _PAD_SLOT_ID

        block_table_cpu = block_table_cpu[:padded_batch_size]

        # Context lens
        context_lens_np = self.decode_context_lens_np[
            self.cur_swap_id][:padded_batch_size]
        np.add(self.input_batch.num_computed_tokens_cpu[:padded_batch_size],
               1,
               out=context_lens_np)
        context_lens_np[batch_size:] = 0

        # Get final tensors
        input_tokens = input_tokens_cpu.reshape(-1, 1).to(self.device)
        input_positions = input_positions_cpu.reshape(-1, 1).to(self.device)
        slot_mapping = self.slot_mapping_cpu[
            self.cur_swap_id][:padded_batch_size].reshape(-1,
                                                          1).to(self.device)
        block_table = block_table_cpu.to(self.device)
        context_lens = self.decode_context_lens_cpu[
            self.cur_swap_id][:padded_batch_size].to(self.device)

        self.swap_step()

        # Attn metadata
        attn_metadata = PallasMetadata(
            num_prefills=0,
            num_prefill_tokens=0,
            num_decode_tokens=padded_batch_size,
            slot_mapping=slot_mapping,
            multi_modal_placeholder_index_maps=None,
            enable_kv_scales_calculation=True,
            block_tables=block_table,
            context_lens=context_lens,
            effective_query_lens=None,
        )

        return DecodeData(input_tokens=input_tokens,
                          input_positions=input_positions,
                          attn_metadata=attn_metadata)
=======
        # Get the number of scheduled tokens for each request.
        num_scheduled_tokens_per_req = []
        max_num_scheduled_tokens_all_reqs = 0
        for req_id in self.input_batch.req_ids[:num_reqs]:
            assert req_id is not None
            num_tokens = scheduler_output.num_scheduled_tokens[req_id]
            num_scheduled_tokens_per_req.append(num_tokens)
            max_num_scheduled_tokens_all_reqs = max(
                max_num_scheduled_tokens_all_reqs, num_tokens)
        num_scheduled_tokens_per_req = np.array(num_scheduled_tokens_per_req,
                                                dtype=np.int32)
        assert max_num_scheduled_tokens_all_reqs > 0

        # Get request indices.
        # E.g., [2, 5, 3] -> [0, 0, 1, 1, 1, 1, 1, 2, 2, 2]
        # For each scheduled token, what are the corresponding req index.
        req_indices = np.repeat(self.arange_np[:num_reqs],
                                num_scheduled_tokens_per_req)

        # Get batched arange.
        # E.g., [2, 5, 3] -> [0, 1, 0, 1, 2, 3, 4, 0, 1, 2]
        # For each scheduled token, what is its position in corresponding req.
        arange = np.concatenate(
            [self.arange_np[:n] for n in num_scheduled_tokens_per_req])

        # Get positions.
        positions_np = self.positions_np[:total_num_scheduled_tokens]
        np.add(self.input_batch.num_computed_tokens_cpu[req_indices],
               arange,
               out=positions_np)

        # Get token indices.
        # E.g., [0, 1, 0, 1, 2, 3, 4, 0, 1, 2]
        # -> [0, 1, M, M + 1, M + 2, M + 3, M + 4, 2 * M, 2 * M + 1, 2 * M + 2]
        # where M is the max_model_len.
        token_indices = (positions_np +
                         req_indices * self.input_batch.token_ids_cpu.shape[1])

        # NOTE(woosuk): We use torch.index_select instead of np.take here
        # because torch.index_select is much faster than np.take for large
        # tensors.
        torch.index_select(self.input_batch.token_ids_cpu_tensor.flatten(),
                           0,
                           torch.from_numpy(token_indices),
                           out=self.input_ids_cpu[:total_num_scheduled_tokens])

        # Calculate the slot mapping.
        # E.g., [0, 1, 0, 1, 2, 3, 4, 0, 1, 2]
        # -> [0, 0, K, K, K + 1, K + 1, K + 2, 2 * K, 2 * K, 2 * K + 1]
        # where K is the max_num_blocks_per_req and the block size is 2.
        # NOTE(woosuk): We can't simply use `token_indices // block_size` here
        # because M (max_model_len) is not necessarily divisible by block_size.
        # req_indices: # E.g., [2, 5, 3] -> [0, 0, 1, 1, 1, 1, 1, 2, 2, 2]
        block_table_indices = (req_indices * self.max_num_blocks_per_req +
                               positions_np // self.block_size)
        # NOTE(woosuk): We use torch.index_select instead of np.take here
        # because torch.index_select is much faster than np.take for large
        # tensors.
        block_table_cpu = self.input_batch.block_table[0].get_cpu_tensor()
        block_numbers = block_table_cpu.flatten()[block_table_indices].numpy()
        block_offsets = positions_np % self.block_size
        np.add(block_numbers * self.block_size,
               block_offsets,
               out=self.input_batch.block_table[0].
               slot_mapping_np[:total_num_scheduled_tokens])

        # Prepare the attention metadata.
        self.query_start_loc_np[0] = 0
        np.cumsum(num_scheduled_tokens_per_req,
                  out=self.query_start_loc_np[1:num_reqs + 1])
        self.query_start_loc_np[num_reqs + 1:] = 1

        self.seq_lens_np[:num_reqs] = (
            self.input_batch.num_computed_tokens_cpu[:num_reqs] +
            num_scheduled_tokens_per_req)

        # Do the padding and copy the tensors to the TPU.
        padded_total_num_scheduled_tokens = _get_padded_token_len(
            self.num_tokens_paddings, total_num_scheduled_tokens)
        # Zero out to avoid spurious values from prev iteration (last cp chunk)
        self.input_ids_cpu[
            total_num_scheduled_tokens:padded_total_num_scheduled_tokens] = 0
        self.input_ids = self.input_ids_cpu[:
                                            padded_total_num_scheduled_tokens].to(
                                                self.device)
        self.position_ids = self.positions_cpu[:
                                               padded_total_num_scheduled_tokens].to(
                                                   self.device)
        self.input_batch.block_table[0].slot_mapping_cpu[
            total_num_scheduled_tokens:] = _PAD_SLOT_ID
        slot_mapping = (
            self.input_batch.block_table[0].
            slot_mapping_cpu[:padded_total_num_scheduled_tokens].to(
                self.device))
        block_tables = self.block_table_cpu[:self.max_num_reqs]
        block_tables[:num_reqs, :self.max_num_blocks_per_req] = (
            self.input_batch.block_table[0].get_cpu_tensor()[:num_reqs])
        block_tables = block_tables.to(self.device)
        query_start_loc = self.query_start_loc_cpu[:self.max_num_reqs + 1].to(
            self.device)
        seq_lens = self.seq_lens_cpu[:self.max_num_reqs].to(self.device)

        if self.lora_config is not None:
            # We need to respect padding when activating LoRA adapters
            padded_num_scheduled_tokens_per_req = np.copy(
                num_scheduled_tokens_per_req
            )  # Copying to avoid accidental state corruption bugs
            padded_num_scheduled_tokens_per_req[-1] += \
                padded_total_num_scheduled_tokens - total_num_scheduled_tokens

            self.set_active_loras(self.input_batch,
                                  padded_num_scheduled_tokens_per_req)

        attn_metadata = PallasMetadata(
            slot_mapping=slot_mapping,
            block_tables=block_tables,
            context_lens=seq_lens,
            query_start_loc=query_start_loc,
            num_seqs=torch.tensor([num_reqs],
                                  dtype=torch.int32,
                                  device=self.device),
        )
        # NOTE(woosuk): Due to chunked prefills, there can be at most 1 partial
        # request in the batch. While we should not sample any token from this
        # partial request, we do so for simplicity. We will ignore the sampled
        # token from the partial request.
        # TODO: Support prompt logprobs.
        padded_num_reqs = _get_padded_num_reqs_with_upper_limit(
            num_reqs, self.max_num_reqs)
        # Indices at which we sample (positions of last token in the sequence).
        # Padded to avoid recompiling when `num_reqs` varies.
        logits_indices = self.query_start_loc_cpu[1:padded_num_reqs + 1] - 1
        logits_indices = logits_indices.to(self.device)

        layer_names = get_layers_from_vllm_config(self.vllm_config,
                                                  Attention).keys()
        per_layer_attn_metadata = {
            layer_name: attn_metadata
            for layer_name in layer_names
        }
        return per_layer_attn_metadata, logits_indices, padded_num_reqs

    def _scatter_placeholders(
        self,
        embeds: torch.Tensor,
        is_embed: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if is_embed is None:
            return embeds

        placeholders = embeds.new_full(
            (is_embed.shape[0], embeds.shape[-1]),
            fill_value=torch.nan,
        )
        placeholders[is_embed] = embeds
        return placeholders

    def _gather_placeholders(
        self,
        placeholders: torch.Tensor,
        is_embed: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if is_embed is None:
            return placeholders

        return placeholders[is_embed]

    def _execute_mm_encoder(self, scheduler_output: "SchedulerOutput"):
        scheduled_encoder_inputs = scheduler_output.scheduled_encoder_inputs
        if not scheduled_encoder_inputs:
            return

        # Batch the multi-modal inputs.
        mm_inputs = list[MultiModalKwargs]()
        req_ids_pos = list[tuple[str, int, PlaceholderRange]]()
        for req_id, encoder_input_ids in scheduled_encoder_inputs.items():
            req_state = self.requests[req_id]

            for mm_input_id in encoder_input_ids:
                mm_inputs.append(req_state.mm_inputs[mm_input_id])
                req_ids_pos.append(
                    (req_id, mm_input_id, req_state.mm_positions[mm_input_id]))

        # Batch mm inputs as much as we can: if a request in the batch has
        # multiple modalities or a different modality than the previous one,
        # we process it separately to preserve item order.
        # FIXME(ywang96): This is a hacky way to deal with multiple modalities
        # in the same batch while still being able to benefit from batching
        # multimodal inputs. The proper solution should be reordering the
        # encoder outputs.
        grouped_mm_inputs_list = group_mm_inputs_by_modality(mm_inputs)

        encoder_outputs = []
        for grouped_mm_inputs in grouped_mm_inputs_list:
            batched_mm_inputs = MultiModalKwargs.batch(grouped_mm_inputs)
            batched_mm_inputs = MultiModalKwargs.as_kwargs(batched_mm_inputs,
                                                           device=self.device)

            # Run the encoder.
            # `curr_group_outputs` is either of the following:
            # 1. A tensor of shape (num_items, feature_size, hidden_size)
            # in case feature_size is fixed across all multimodal items.
            # 2. A list or tuple (length: num_items) of tensors, each of shape
            # (feature_size, hidden_size) in case the feature size is dynamic
            # depending on the input multimodal items.
            xm.mark_step()
            curr_group_outputs = self.model.get_multimodal_embeddings(
                **batched_mm_inputs)
            xm.mark_step()

            sanity_check_mm_encoder_outputs(
                curr_group_outputs,
                expected_num_items=len(grouped_mm_inputs),
            )

            if isinstance(curr_group_outputs, torch.Tensor):
                encoder_outputs.append(curr_group_outputs)
            else:
                assert isinstance(curr_group_outputs, (list, tuple))
                for output in curr_group_outputs:
                    encoder_outputs.append(output)

        # Cache the encoder outputs.
        # NOTE (NickLucche) here we diverge from logic in other runners, as we
        # assume to only have whole mm items to process. Hence we avoid the
        # intrinsic dynamism that `scatter_mm_placeholders` introduces.
        for (req_id, input_id, pos_info), output in zip(
                req_ids_pos,
                encoder_outputs,
        ):
            if req_id not in self.encoder_cache:
                self.encoder_cache[req_id] = {}
            assert pos_info.is_embed is None, "Expected all positions to be"\
                " contiguous and embeddings."
            self.encoder_cache[req_id][input_id] = output

    def _gather_mm_embeddings(
        self,
        scheduler_output: "SchedulerOutput",
    ) -> list[torch.Tensor]:
        mm_embeds: list[torch.Tensor] = []
        for req_id in self.input_batch.req_ids:
            num_scheduled_tokens = scheduler_output.num_scheduled_tokens[
                req_id]
            req_state = self.requests[req_id]
            num_computed_tokens = req_state.num_computed_tokens
            mm_positions = req_state.mm_positions
            # TODO unroll loop and assume/enforce --disable_chunked_mm_input
            # NOTE (NickLucche) here we diverge from logic in other runners, as
            # we assume to only have whole mm items to process. Hence we avoid
            # the intrinsic dynamism that `gather_mm_placeholders` introduces.
            for i, pos_info in enumerate(mm_positions):
                start_pos = pos_info.offset
                num_encoder_tokens = pos_info.length

                # The encoder output is needed if the two ranges overlap:
                # [num_computed_tokens,
                #  num_computed_tokens + num_scheduled_tokens) and
                # [start_pos, start_pos + num_encoder_tokens)
                if start_pos >= num_computed_tokens + num_scheduled_tokens:
                    # The encoder output is not needed in this step.
                    break
                if start_pos + num_encoder_tokens <= num_computed_tokens:
                    # The encoder output is already processed and stored
                    # in the decoder's KV cache.
                    continue

                assert req_id in self.encoder_cache
                assert i in self.encoder_cache[req_id]
                assert pos_info.is_embed is None, "Expected all positions to"\
                " be contiguous and embeddings."
                encoder_output = self.encoder_cache[req_id][i]
                mm_embeds.append(encoder_output)
        return mm_embeds

    def _get_model_inputs(self, input_ids: torch.Tensor,
                          mm_embeds: list[torch.Tensor]):
        if self.is_multimodal_model:
            # NOTE(woosuk): To unify token ids and soft tokens (vision
            # embeddings), we always use embeddings (rather than token ids)
            # as input to the multimodal model, even when the input is text.
            if mm_embeds:
                inputs_embeds = self.model.get_input_embeddings(
                    input_ids, mm_embeds)
            else:
                inputs_embeds = self.model.get_input_embeddings(input_ids)
            return None, inputs_embeds
        else:
            # For text-only models, we use token ids as input.
            # While it is possible to use embeddings as input just like the
            # multimodal models, it is not desirable for performance since
            # then the embedding layer is not included in the CUDA graph.
            return input_ids, None
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea

    @torch.no_grad()
    def execute_model(
        self,
        scheduler_output: "SchedulerOutput",
<<<<<<< HEAD
    ) -> ModelRunnerOutput:
        # Update cached state
        self._update_states(scheduler_output)

        # If necessary, swap decodes/prompts to have all decodes on the start
        ensure_decodes_first(self.input_batch)

        # Prepare prompts/decodes info
        pd_info = self._get_prompts_and_decodes(scheduler_output)

        # Init
        num_prompts = len(pd_info.prompt_req_ids)
        num_decodes = len(pd_info.decode_req_ids)
        decode_data = None
        sampled_token_ids = [0] * self.input_batch.num_reqs

        # Run each prompt individually
        is_first = True
        for i in range(num_prompts):
            req_id = pd_info.prompt_req_ids[i]
            req_index = num_decodes + i
            assert req_index == self.input_batch.req_id_to_index[
                req_id]  # TODO: Remove
            req_state = self.requests[req_id]
            num_scheduled_tokens = pd_info.prompt_scheduled_tokens[i]
            prompt_len = num_scheduled_tokens
            seq_len = req_state.num_computed_tokens + num_scheduled_tokens

            # Prepare first prompt
            if is_first:
                prompt_data = self._prepare_prompt(req_index,
                                                   num_scheduled_tokens)
                is_first = False

            # Run forward pass
            with set_forward_context(prompt_data.attn_metadata,
                                     self.vllm_config):
                assert self.model is not None
                selected_token_ids = self.model(prompt_data.input_tokens,
                                                prompt_data.input_positions,
                                                prompt_data.attn_metadata,
                                                self.kv_caches)

            # In parallel to TPU execution, prepare the next iteration
            if i < num_prompts - 1:
                # There is next prompt => prepare it
                prompt_data = self._prepare_prompt(
                    req_index + 1, pd_info.prompt_scheduled_tokens[i + 1])
            elif i == num_prompts - 1 and num_decodes > 0:
                # There is next decode => prepare it
                decode_data = self._prepare_decode(pd_info.decode_req_ids)

            # Update cached state (if prompt is fully done)
            if seq_len >= len(req_state.prompt_token_ids):
                # Transfer sampled tokens from TPU to CPU
                selected_token_ids_cpu = selected_token_ids.cpu()

                # Get output token
                token_id = selected_token_ids_cpu[prompt_len - 1].item()
                sampled_token_ids[req_index] = token_id

                # Add output token to the request
                self.input_batch.token_ids_cpu[req_index, seq_len] = token_id
                self.input_batch.num_tokens[req_index] += 1
                req_state.output_token_ids.append(token_id)

        # Run decodes (a single batch)
        if num_decodes > 0:

            # Prepare decode (if was not yet prepared)
            if decode_data is None:
                decode_data = self._prepare_decode(pd_info.decode_req_ids)

            # Run forward pass
            with set_forward_context(decode_data.attn_metadata,
                                     self.vllm_config):
                assert self.model is not None
                selected_token_ids = self.model(decode_data.input_tokens,
                                                decode_data.input_positions,
                                                decode_data.attn_metadata,
                                                self.kv_caches)

            # Transfer sampled tokens from TPU to CPU
            decode_token_ids_cpu = selected_token_ids.cpu()
            # Convert to list
            decode_token_ids_list = decode_token_ids_cpu.tolist()

            # Update cached state for each decode request
            for i in range(num_decodes):
                req_id = pd_info.decode_req_ids[i]
                req_index = i
                assert req_index == self.input_batch.req_id_to_index[
                    req_id]  # TODO: Remove
                req_state = self.requests[req_id]
                seq_len = req_state.num_computed_tokens + 1

                token_id = decode_token_ids_list[i]
                sampled_token_ids[req_index] = token_id

                self.input_batch.token_ids_cpu[req_index, seq_len] = token_id
                self.input_batch.num_tokens[req_index] += 1
                req_state.output_token_ids.append(token_id)

        # Create output.
        all_req_ids = pd_info.decode_req_ids + pd_info.prompt_req_ids
        prompt_logprobs_dict: Dict[str, Optional[LogprobsTensors]] = {}
        for req_id in all_req_ids:
            prompt_logprobs_dict[req_id] = None

        model_runner_output = ModelRunnerOutput(
            req_ids=all_req_ids,
            req_id_to_index=self.input_batch.req_id_to_index,
            sampled_token_ids=[[token_id] for token_id in sampled_token_ids],
            spec_token_ids=None,
            logprobs=None,
            prompt_logprobs_dict=prompt_logprobs_dict,  # type: ignore[arg-type]
        )

=======
        intermediate_tensors: Optional[IntermediateTensors] = None,
    ) -> ModelRunnerOutput:
        # Update cached state
        self._update_states(scheduler_output)
        if not scheduler_output.total_num_scheduled_tokens:
            # Return empty ModelRunnerOutput if there's no work to do.
            return EMPTY_MODEL_RUNNER_OUTPUT

        if self.is_multimodal_model:
            # Run the multimodal encoder if any.
            self._execute_mm_encoder(scheduler_output)
            mm_embeds = self._gather_mm_embeddings(scheduler_output)
        else:
            mm_embeds = []
        xm.mark_step()
        # Prepare inputs
        attn_metadata, logits_indices, padded_num_reqs = self._prepare_inputs(
            scheduler_output)
        input_ids, inputs_embeds = self._get_model_inputs(
            self.input_ids, mm_embeds)
        xm.mark_step()
        num_reqs = self.input_batch.num_reqs
        # Run the decoder
        with set_forward_context(
                attn_metadata,
                self.vllm_config,
                num_tokens=scheduler_output.total_num_scheduled_tokens):
            hidden_states = self.model(
                input_ids=input_ids,
                positions=self.position_ids,
                inputs_embeds=inputs_embeds,
            )
        hidden_states = self.select_hidden_states(hidden_states,
                                                  logits_indices)
        logits = self.compute_logits(hidden_states)
        tpu_sampling_metadata = TPUSupportedSamplingMetadata.\
            from_input_batch(self.input_batch, padded_num_reqs, self.device)
        if scheduler_output.grammar_bitmask is not None:
            require_struct_decoding, grammar_bitmask_padded, arange = \
                self.prepare_structured_decoding_input(logits, scheduler_output)
            logits = self.structured_decode(require_struct_decoding,
                                            grammar_bitmask_padded, logits,
                                            arange)
        selected_token_ids = self.sample_from_logits(logits,
                                                     tpu_sampling_metadata)

        # NOTE (NickLucche) Use the original logits (before any penalties or
        # temperature scaling) for the top-k logprobs. We can't enforce it due
        # to recompilations outside torch.compiled code, so just make sure
        # `sample_from_logits` does not modify the logits in-place.
        logprobs = self.gather_logprobs(logits, selected_token_ids) \
            if tpu_sampling_metadata.logprobs else None

        # Remove padding on cpu and keep dynamic op outside of xla graph.
        selected_token_ids = selected_token_ids.cpu()[:num_reqs]
        logprobs_lists = logprobs.tolists() \
            if tpu_sampling_metadata.logprobs else None

        # Update the cache state concurrently. Code above will not block until
        # we use `selected_token_ids`. Add mark_step if post-processing changes
        request_seq_lens: list[tuple[int, CachedRequestState, int]] = []
        discard_sampled_tokens_req_indices = []
        for i, req_id in zip(range(num_reqs), self.input_batch.req_ids):
            assert req_id is not None
            req_state = self.requests[req_id]
            seq_len = (req_state.num_computed_tokens +
                       scheduler_output.num_scheduled_tokens[req_id])
            if seq_len >= req_state.num_tokens:
                request_seq_lens.append((i, req_state, seq_len))
            else:
                # Ignore the sampled token from the partial request.
                # Rewind the generator state as if the token was not sampled.
                generator = self.input_batch.generators.get(i)
                if generator is not None:
                    # This relies on cuda-specific torch-internal impl details
                    generator.set_offset(generator.get_offset() - 4)

                # Record the index of the request that should not be sampled,
                # so that we could clear the sampled tokens before returning.
                discard_sampled_tokens_req_indices.append(i)

        assert all(
            req_id is not None for req_id in
            self.input_batch.req_ids[:num_reqs]), "req_ids contains None"
        req_ids = cast(list[str], self.input_batch.req_ids[:num_reqs])

        prompt_logprobs_dict: dict[str, Optional[LogprobsTensors]] = {}
        for req_id in self.input_batch.req_ids[:num_reqs]:
            prompt_logprobs_dict[req_id] = None

        max_gen_len = selected_token_ids.shape[-1]
        if max_gen_len == 1:
            valid_sampled_token_ids = selected_token_ids.tolist()

            # Mask out the sampled tokens that should not be sampled.
            # TODO: Keep in sync with gpu_model_runner.py, in particular
            #       the "else" case here
            for i in discard_sampled_tokens_req_indices:
                valid_sampled_token_ids[i].clear()

            # Append sampled tokens
            for i, req_state, seq_len in request_seq_lens:
                token_id = valid_sampled_token_ids[i][0]
                self.input_batch.token_ids_cpu[i, seq_len] = token_id
                req_state.output_token_ids.append(token_id)
                self.input_batch.num_tokens[i] += 1

        else:
            valid_mask = selected_token_ids != INVALID_TOKEN_ID
            gen_lens = valid_mask.sum(dim=1).tolist()
            valid_sampled_token_ids = [
                seq.tolist()
                for seq in selected_token_ids[valid_mask].split(gen_lens)
            ]
            self.input_batch.num_tokens[:num_reqs] += gen_lens
            for i, req_state, seq_len in request_seq_lens:
                target_slice = slice(seq_len - gen_lens[i] + 1, seq_len + 1)
                self.input_batch.token_ids_cpu[
                    i, target_slice] = valid_sampled_token_ids[i]
                req_state.output_token_ids.extend(valid_sampled_token_ids[i])

        model_runner_output = ModelRunnerOutput(
            req_ids=req_ids,
            req_id_to_index=self.input_batch.req_id_to_index,
            sampled_token_ids=valid_sampled_token_ids,
            spec_token_ids=None,
            logprobs=logprobs_lists,
            prompt_logprobs_dict=prompt_logprobs_dict,
        )

        # Check there are no new graphs compiled - all the graphs should be
        # captured and compiled during warm up.
        self._verify_num_xla_graphs("execute_model")

>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
        return model_runner_output

    def load_model(self) -> None:
        self.device = self.device_config.device

        # NOTE(woosuk): While the executor assigns the TP ranks to the worker
        # process, the ranks can be different from the ranks internally assigned
        # by the xm runtime. Therefore, there is a mismatch in the rank
        # assignment between the gloo (cpu) runtime and the xm (tpu) runtime.
        # This is not a problem in linear layers because all-reduce is
        # rank-agnostic. However, it matters for all-gather as the ranks
        # determine the order of concatenating the output tensors.
        # As a workaround, we use the xm's rank assignment only when loading
        # the embedding weights.
        xm_tp_rank = xr.global_ordinal()
        with patch(
                "vllm.model_executor.layers.vocab_parallel_embedding."
                "get_tensor_model_parallel_rank",
                return_value=xm_tp_rank):
            model = get_model(vllm_config=self.vllm_config)
<<<<<<< HEAD
        model = model.eval()
        xm.mark_step()
        xm.wait_device_ops()
        model = ModelWrapperV1(model)
        self.model = torch.compile(model,
                                   backend="openxla",
                                   fullgraph=True,
                                   dynamic=False)

    def dummy_run(
        self,
        kv_caches,
        num_tokens: int,
        seq_len: Optional[int] = None,
        exec_mode: Optional[ExecutionMode] = None,
    ) -> None:
        assert seq_len is not None
        assert exec_mode is not None

        exec_mode = ExecutionMode(exec_mode)
        if exec_mode.is_prefill():
            seq_len = (seq_len + 15) // 16 * 16
            token_ids = torch.zeros((num_tokens, seq_len),
                                    dtype=torch.int32,
                                    device=self.device)
            position_ids = torch.zeros((num_tokens, seq_len),
                                       dtype=torch.int32,
                                       device=self.device)
            slot_mapping = torch.zeros((num_tokens, seq_len),
                                       dtype=torch.int64,
                                       device=self.device)
            if exec_mode == ExecutionMode.PREFILL:
                attn_metadata = PallasMetadata(
                    num_prefills=num_tokens,
                    num_prefill_tokens=num_tokens * seq_len,
                    num_decode_tokens=0,
                    slot_mapping=slot_mapping,
                    multi_modal_placeholder_index_maps=None,
                    enable_kv_scales_calculation=True,
                    block_tables=None,
                    context_lens=None,
                    effective_query_lens=None,
                )

            else:
                context_lens = torch.ones((num_tokens, ),
                                          dtype=torch.int32,
                                          device=self.device)

                block_tables = torch.zeros(
                    (num_tokens, self.max_num_blocks_per_req),
                    dtype=torch.int32,
                    device=self.device)

                effective_query_lens = torch.ones_like(context_lens)

                attn_metadata = PallasMetadata(
                    num_prefills=num_tokens,
                    num_prefill_tokens=num_tokens * seq_len,
                    num_decode_tokens=0,
                    slot_mapping=slot_mapping,
                    multi_modal_placeholder_index_maps=None,
                    enable_kv_scales_calculation=True,
                    block_tables=block_tables,
                    context_lens=context_lens,
                    effective_query_lens=effective_query_lens,
                )
        else:
            assert seq_len == 1
            token_ids = torch.zeros((num_tokens, seq_len),
                                    dtype=torch.int32,
                                    device=self.device)
            position_ids = torch.zeros((num_tokens, seq_len),
                                       dtype=torch.int32,
                                       device=self.device)
            slot_mapping = torch.zeros((num_tokens, seq_len),
                                       dtype=torch.int64,
                                       device=self.device)
            block_tables = torch.zeros(
                (num_tokens, self.max_num_blocks_per_req),
                dtype=torch.int32,
                device=self.device)
            context_lens = torch.ones((num_tokens, ),
                                      dtype=torch.int32,
                                      device=self.device)
            attn_metadata = PallasMetadata(
                num_prefills=0,
                num_prefill_tokens=0,
                num_decode_tokens=num_tokens * seq_len,
                slot_mapping=slot_mapping,
                multi_modal_placeholder_index_maps=None,
                enable_kv_scales_calculation=True,
                block_tables=block_tables,
                context_lens=context_lens,
            )

        # NOTE(woosuk): There are two stages of compilation: torch.compile and
        # XLA compilation. Using `mark_dynamic` can reduce the torch.compile
        # overhead by reusing the FX graph for different shapes.
        # However, the XLA graph will still require static shapes and needs to
        # be re-compiled for every different shapes. This overhead is inevitable
        # in the first run, but can be skipped afterwards as we cache the XLA
        # graphs in the disk (VLLM_XLA_CACHE_PATH).
        if exec_mode.is_prefill():
            # Prefll
            torch._dynamo.mark_dynamic(token_ids, 1)
            torch._dynamo.mark_dynamic(position_ids, 1)
            torch._dynamo.mark_dynamic(attn_metadata.slot_mapping, 1)
        else:
            # Decode
            torch._dynamo.mark_dynamic(token_ids, 0)
            torch._dynamo.mark_dynamic(position_ids, 0)
            torch._dynamo.mark_dynamic(attn_metadata.slot_mapping, 0)
            torch._dynamo.mark_dynamic(attn_metadata.context_lens, 0)
            torch._dynamo.mark_dynamic(attn_metadata.block_tables, 0)

        with set_forward_context(attn_metadata, self.vllm_config, 0):
            assert self.model is not None
            self.model(token_ids, position_ids, attn_metadata, kv_caches)

    def capture_model(self) -> None:
        """Compile the model."""

        # Prefill
        logger.info(
            "Compiling the model with different input shapes for prefill:")
        start = time.time()
        for batch_size in [1]:
            seq_len = 16
            while seq_len <= self.model_config.max_model_len:
                self.dummy_run(self.kv_caches,
                               batch_size,
                               seq_len,
                               exec_mode=ExecutionMode.PREFILL)
                xm.wait_device_ops()
                logger.info("  batch_size: %d, seq_len: %d", batch_size,
                            seq_len)
                num_tokens = batch_size * seq_len
                if num_tokens >= self.scheduler_config.max_num_batched_tokens:
                    break
                seq_len = seq_len * 2

        end = time.time()
        logger.info("    -- Compilation for prefill done in %.2f [secs].",
                    end - start)

        # Prefix prefill
        if self.scheduler_config.enable_chunked_prefill:
            logger.info("Compiling the model with different input shapes for "
                        "prefix prefill:")
            start = time.time()
            for batch_size in [1]:
                seq_len = 16
                while seq_len <= self.model_config.max_model_len:
                    self.dummy_run(self.kv_caches,
                                   batch_size,
                                   seq_len,
                                   exec_mode=ExecutionMode.PREFIX_PREFILL)
                    xm.wait_device_ops()
                    logger.info("  batch_size: %d, seq_len: %d", batch_size,
                                seq_len)
                    num_tokens = batch_size * seq_len
                    if (num_tokens
                            >= self.scheduler_config.max_num_batched_tokens):
                        break
                    seq_len = seq_len * 2
            end = time.time()
            logger.info(
                "    -- Compilation for prefix prefill done in %.2f [secs].",
                end - start)

        # Decode
        logger.info(
            "Compiling the model with different input shapes for decode:")
        start = time.time()
        seq_len = 1
        batch_size = 8  # Must be in sync with _get_padded_batch_size()
        while True:
            self.dummy_run(self.kv_caches,
                           batch_size,
                           seq_len,
                           exec_mode=ExecutionMode.DECODE)
            xm.wait_device_ops()
            logger.info("  batch_size: %d, seq_len: %d", batch_size, seq_len)

            if batch_size >= self.scheduler_config.max_num_seqs:
                break
            batch_size = batch_size + 16 if batch_size >= 16 else batch_size * 2

        end = time.time()
        logger.info("    -- Compilation for decode done in %.2f [secs].",
                    end - start)
=======
        if self.lora_config is not None:
            model = self.load_lora_model(model, self.model_config,
                                         self.scheduler_config,
                                         self.lora_config, self.device)

        # Sync all pending XLA execution during model initialization and weight
        # loading.
        xm.mark_step()
        xm.wait_device_ops()
        self.model = model
        self.sampler = TPUSampler()

    @torch.no_grad()
    def _dummy_run(self, num_tokens: int) -> None:
        if self.is_multimodal_model:
            input_ids = None
            inputs_embeds = torch.zeros((num_tokens, self.hidden_size),
                                        dtype=self.dtype,
                                        device=self.device)
        else:
            input_ids = torch.zeros((num_tokens),
                                    dtype=torch.int32,
                                    device=self.device)
            inputs_embeds = None
        actual_num_reqs = min(num_tokens, self.max_num_reqs)
        position_ids = torch.zeros(num_tokens,
                                   dtype=torch.int32,
                                   device=self.device)
        slot_mapping = torch.zeros(num_tokens,
                                   dtype=torch.int64,
                                   device=self.device)
        block_tables = torch.zeros(
            (self.max_num_reqs, self.block_table_cpu.shape[1]),
            dtype=torch.int32,
            device=self.device)
        query_lens = [1] * self.max_num_reqs
        query_start_loc = torch.cumsum(torch.tensor([0] + query_lens,
                                                    dtype=torch.int32),
                                       dim=0,
                                       dtype=torch.int32).to(self.device)
        context_lens = torch.ones((self.max_num_reqs, ),
                                  dtype=torch.int32,
                                  device=self.device)
        num_seqs = torch.tensor([actual_num_reqs],
                                dtype=torch.int32,
                                device=self.device)
        attn_metadata = PallasMetadata(
            slot_mapping=slot_mapping,
            block_tables=block_tables,
            context_lens=context_lens,
            query_start_loc=query_start_loc,
            num_seqs=num_seqs,
        )

        if self.is_multimodal_model:
            torch._dynamo.mark_dynamic(inputs_embeds, 0)
        else:
            torch._dynamo.mark_dynamic(input_ids, 0)
        torch._dynamo.mark_dynamic(position_ids, 0)
        torch._dynamo.mark_dynamic(attn_metadata.slot_mapping, 0)

        layer_names = get_layers_from_vllm_config(self.vllm_config,
                                                  Attention).keys()
        per_layer_attn_metadata = {
            layer_name: attn_metadata
            for layer_name in layer_names
        }

        with self.maybe_dummy_run_with_lora(
                self.lora_config,
                np.array([num_tokens], dtype=np.int32)), set_forward_context(
                    per_layer_attn_metadata, self.vllm_config, 0):
            out = self.model(input_ids=input_ids,
                             positions=position_ids,
                             inputs_embeds=inputs_embeds)
        self._hidden_states_dtype = out.dtype

    def _precompile_mm_encoder(self) -> None:
        # Pre-compile MM encoder for all supported data modalities.
        hf_config = self.vllm_config.model_config.hf_config
        for mode, max_items_by_mode in \
            self.max_num_mm_items_by_modality.items():
            logger.info(
                "Compiling Multimodal %s Encoder with different input"
                " shapes.", mode)
            start = time.perf_counter()
            # No padding for MM encoder just yet.
            for num_items in range(1, max_items_by_mode + 1):
                logger.info("  -- mode: %s items: %d", mode, num_items)
                batched_dummy_mm_inputs = self._get_mm_dummy_batch(
                    mode, num_items)
                # Run multimodal encoder.
                xm.mark_step()
                mm_embeds = self.model.\
                    get_multimodal_embeddings(**batched_dummy_mm_inputs)
                xm.mark_step()
                num_patches = mm_embeds[0].shape[0]
                items_size = num_patches * num_items

                # NOTE (NickLucche) pre-compile `get_input_embeddings` when mm
                # embeddings are present. We assume `--disable-mm-chunked`,
                # hence only whole items can be scheduled. This implies we just
                # need to compile when `num_items` fit the (padded) `input_ids`
                for num_tokens in self.num_tokens_paddings:
                    if num_tokens >= items_size:
                        # XLA Workaround: if torch.zeros(..device) is used, XLA
                        # compiles a scalar+expansion op, which won't match
                        # the graph generated at runtime. CPU->TPU must be used
                        placeholders_ids = torch.zeros(num_tokens,
                                                       dtype=torch.int32,
                                                       device="cpu")
                        # Align placeholders and actual num mm_embeddings.
                        placeholders_ids[:items_size] = \
                            hf_config.image_token_index

                        placeholders_ids = placeholders_ids.to(self.device)
                        # Assign outputs or the graph will be cut short.
                        a, b = self._get_model_inputs(placeholders_ids,
                                                      [mm_embeds])
                        assert a is None
                        xm.mark_step()

            # Pre-compile `get_input_embeddings` when mm_embeddings are not
            # present. Chunk is only made of text, no mm_placeholders.
            for num_tokens in self.num_tokens_paddings:
                placeholders_ids = torch.zeros(num_tokens,
                                               dtype=torch.int32,
                                               device="cpu")
                placeholders_ids = placeholders_ids.to(self.device)
                a, b = self._get_model_inputs(placeholders_ids, [])
                assert a is None
                xm.mark_step()

            xm.wait_device_ops()
            end = time.perf_counter()
            logger.info(
                "Multimodal %s Encoder compilation finished in in %.2f "
                "[secs].", mode, end - start)

    def _precompile_backbone(self) -> None:
        logger.info("Compiling the model with different input shapes.")
        start = time.perf_counter()
        for num_tokens in self.num_tokens_paddings:
            logger.info("  -- num_tokens: %d", num_tokens)
            self._dummy_run(num_tokens)
        xm.wait_device_ops()
        end = time.perf_counter()
        logger.info("Compilation finished in %.2f [secs].", end - start)
        self._update_num_xla_graphs("model backbone")

    def _precompile_select_hidden_states(self) -> None:
        # Compile hidden state selection function for bucketed
        # n_tokens x max_num_reqs. Graph is really small so this is fine.
        logger.info(
            "Compiling select_hidden_states with different input shapes.")
        start = time.perf_counter()
        hsize = self.model_config.get_hidden_size()
        for num_tokens in self.num_tokens_paddings:
            dummy_hidden = torch.zeros((num_tokens, hsize),
                                       device=self.device,
                                       dtype=self._hidden_states_dtype)
            torch._dynamo.mark_dynamic(dummy_hidden, 0)
            for num_reqs in self.num_reqs_paddings:
                indices = torch.zeros(num_reqs,
                                      dtype=torch.int32,
                                      device=self.device)
                torch._dynamo.mark_dynamic(indices, 0)
                self.select_hidden_states(dummy_hidden, indices)
                logger.info("  -- num_tokens: %d, num_seqs: %d", num_tokens,
                            num_reqs)
                # Requests can't be more than tokens. But do compile for the
                # next bigger value in case num_tokens uses bucketed padding.
                if num_reqs >= min(num_tokens, self.max_num_reqs):
                    break
        xm.wait_device_ops()
        end = time.perf_counter()
        logger.info("Compilation finished in %.2f [secs].", end - start)
        self._update_num_xla_graphs("select_hidden_states")

    def _precompile_compute_logits(self) -> None:
        logger.info("Compiling compute_logits with different input shapes.")
        start = time.perf_counter()
        hsize = self.model_config.get_hidden_size()
        for num_reqs in self.num_reqs_paddings:
            dummy_hidden = torch.zeros((num_reqs, hsize),
                                       device=self.device,
                                       dtype=self._hidden_states_dtype)
            torch._dynamo.mark_dynamic(dummy_hidden, 0)
            self.compute_logits(dummy_hidden)
            logger.info("  -- num_seqs: %d", num_reqs)
        xm.wait_device_ops()
        end = time.perf_counter()
        logger.info("Compilation finished in %.2f [secs].", end - start)
        self._update_num_xla_graphs("compute_logits")

    def _precompile_structured_decoding(self) -> None:
        logger.info(
            "Compiling structured_decoding with different input shapes.")
        start = time.perf_counter()
        for num_reqs in self.num_reqs_paddings:
            dummy_logits = torch.zeros((num_reqs, self.vocab_size),
                                       device=self.device,
                                       dtype=self._hidden_states_dtype)
            dummy_require_struct_decoding = \
                self.require_structured_out_cpu[:num_reqs].to(self.device)
            dummy_grammar_bitmask = \
                self.grammar_bitmask_cpu[:num_reqs].to(self.device)
            # The first dimension of the above 3 dummy tensors cannot be
            # mark_dynamic because some operations in structured_decode require
            # them to be static.
            arange = self.structured_decode_arange.to(self.device)
            self.structured_decode(dummy_require_struct_decoding,
                                   dummy_grammar_bitmask, dummy_logits, arange)
            logger.info("  -- num_seqs: %d", num_reqs)
        xm.wait_device_ops()
        end = time.perf_counter()
        logger.info("Compilation finished in %.2f [secs].", end - start)
        self._update_num_xla_graphs("structured_decoding")

    def _precompile_sample_from_logits(self) -> None:
        logger.info(
            "Compiling sample_from_logits with different input shapes.")
        start = time.perf_counter()
        for num_reqs in self.num_reqs_paddings:
            dummy_logits = torch.zeros((num_reqs, self.vocab_size),
                                       device=self.device,
                                       dtype=self._hidden_states_dtype)
            # The first dimension of dummy_logits cannot be mark_dynamic
            # because some operations in the sampler require it to be static.
            for all_greedy in [False, True]:
                generate_params_if_all_greedy = not all_greedy
                sampling_metadata = (
                    TPUSupportedSamplingMetadata.from_input_batch(
                        self.input_batch,
                        num_reqs,
                        self.device,
                        generate_params_if_all_greedy,
                    ))
                sampling_metadata.all_greedy = all_greedy
                self.sample_from_logits(dummy_logits, sampling_metadata)
            logger.info("  -- num_seqs: %d", num_reqs)
        xm.wait_device_ops()
        end = time.perf_counter()
        logger.info("Compilation finished in %.2f [secs].", end - start)
        self._update_num_xla_graphs("sample_from_logits")

    def _precompile_gather_logprobs(self) -> None:
        logger.info("Compiling gather_logprobs with different input shapes.")
        start = time.perf_counter()
        for num_reqs in self.num_reqs_paddings:
            dummy_logits = torch.zeros((num_reqs, self.vocab_size),
                                       device=self.device,
                                       dtype=self._hidden_states_dtype)
            dummy_tokens = torch.zeros((num_reqs, 1),
                                       dtype=torch.int64).to(self.device)
            self.gather_logprobs(dummy_logits, dummy_tokens)
            logger.info("  -- num_seqs: %d", num_reqs)
        xm.wait_device_ops()
        end = time.perf_counter()
        logger.info("Compilation finished in %.2f [secs].", end - start)
        self._update_num_xla_graphs("gather_logprobs")

    def capture_model(self) -> None:
        """
        Precompile all the subgraphs with possible input shapes.
        """
        self._precompile_mm_encoder()
        self._precompile_backbone()
        self._precompile_select_hidden_states()
        self._precompile_compute_logits()
        self._precompile_structured_decoding()
        self._precompile_sample_from_logits()
        self._precompile_gather_logprobs()

    def profile_run(
        self,
        num_tokens: int,
    ) -> None:
        # Profile with multimodal encoder & encoder cache.
        # TODO: handle encoder-decoder models once we support them.
        if (self.is_multimodal_model and self.max_num_encoder_input_tokens > 0
                and self.encoder_cache_size > 0):

            # NOTE: Currently model is profiled with a single non-text
            # modality with the max possible input tokens even when
            # it supports multiple.
            dummy_data_modality, max_num_mm_items = max(
                self.max_num_mm_items_by_modality.items(), key=lambda t: t[1])

            encoder_budget = min(self.max_num_encoder_input_tokens,
                                 self.encoder_cache_size)

            logger.info(
                "Encoder cache will be initialized with a budget of %d tokens,"
                " and profiled with %s %s items of the maximum feature size.",
                encoder_budget, max_num_mm_items, dummy_data_modality)

            # Create dummy batch of multimodal inputs.
            batched_dummy_mm_inputs = self._get_mm_dummy_batch(
                dummy_data_modality, max_num_mm_items)

            # Run multimodal encoder.
            # Isolate encoder graph from post-processing to minimize
            # impact of recompilation until it's fixed.
            start = time.perf_counter()
            xm.mark_step()
            dummy_encoder_outputs = self.model.get_multimodal_embeddings(
                **batched_dummy_mm_inputs)
            xm.mark_step()
            xm.wait_device_ops()
            end = time.perf_counter()
            logger.info(
                "Multimodal Encoder profiling finished in in %.2f [secs].",
                end - start)

            assert len(dummy_encoder_outputs) == max_num_mm_items, (
                "Expected dimension 0 of encoder outputs to match the number "
                f"of multimodal data items: {max_num_mm_items}, got "
                f"{len(dummy_encoder_outputs)=} instead. This is most likely "
                "due to the 'get_multimodal_embeddings' method of the model "
                "not implemented correctly.")

            # Cache the dummy encoder outputs.
            self.encoder_cache["tmp"] = dict(enumerate(dummy_encoder_outputs))

        # Trigger compilation for general shape.
        self._dummy_run(num_tokens)

        xm.mark_step()
        xm.wait_device_ops()
        self.encoder_cache.clear()
        gc.collect()
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea

    def initialize_kv_cache(self, kv_cache_config: KVCacheConfig) -> None:
        """
        Initialize KV cache based on `kv_cache_config`.
        Args:
<<<<<<< HEAD
            kv_cache_config: Configuration for the KV cache, including the KV 
            cache size of each layer
        """
        if len(kv_cache_config.groups) > 1:
=======
            kv_cache_config: Configuration for the KV cache, including the KV
            cache size of each layer
        """
        if len(kv_cache_config.kv_cache_groups) > 1:
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
            raise NotImplementedError(
                "Hybrid models with more than one KV cache type are not "
                "supported yet.")

<<<<<<< HEAD
        kv_caches: Dict[str, torch.Tensor] = {}

        for layer_name, layer_spec in kv_cache_config.kv_cache_spec.items():
            tensor_config = kv_cache_config.tensors[layer_name]
            assert tensor_config.size % layer_spec.page_size_bytes == 0
            num_blocks = tensor_config.size // layer_spec.page_size_bytes
            if isinstance(layer_spec, FullAttentionSpec):
                kv_cache_shape = PallasAttentionBackend.get_kv_cache_shape(
                    num_blocks, layer_spec.block_size, layer_spec.num_kv_heads,
                    layer_spec.head_size)
                dtype = layer_spec.dtype

                tpu_k_cache = torch.zeros(kv_cache_shape,
                                          dtype=dtype,
                                          device=self.device)
                tpu_v_cache = torch.zeros_like(tpu_k_cache)

                kv_caches[layer_name] = (tpu_k_cache, tpu_v_cache)
            else:
                raise NotImplementedError
=======
        self.input_batch = InputBatch(
            max_num_reqs=self.max_num_reqs,
            max_model_len=self.max_model_len,
            max_num_batched_tokens=self.max_num_tokens,
            device=self.device,
            pin_memory=self.pin_memory,
            vocab_size=self.model_config.get_vocab_size(),
            kv_cache_config=kv_cache_config,
        )
        assert self.block_table_cpu.dtype == self.input_batch.block_table[
            0].get_cpu_tensor().dtype

        kv_caches: dict[str, torch.Tensor] = {}

        for kv_cache_group in kv_cache_config.kv_cache_groups:
            kv_cache_spec = kv_cache_group.kv_cache_spec
            for layer_name in kv_cache_group.layer_names:
                tensor_config = kv_cache_config.tensors[layer_name]
                assert tensor_config.size % kv_cache_spec.page_size_bytes == 0
                num_blocks = tensor_config.size // kv_cache_spec.page_size_bytes
                if isinstance(kv_cache_spec, AttentionSpec):
                    kv_cache_shape = PallasAttentionBackend.get_kv_cache_shape(
                        num_blocks, kv_cache_spec.block_size,
                        kv_cache_spec.num_kv_heads, kv_cache_spec.head_size)
                    dtype = kv_cache_spec.dtype

                    tpu_kv_cache = torch.zeros(kv_cache_shape,
                                               dtype=dtype,
                                               device=self.device)

                    kv_caches[layer_name] = tpu_kv_cache
                else:
                    raise NotImplementedError
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea

        bind_kv_cache(
            kv_caches,
            self.vllm_config.compilation_config.static_forward_context,
            self.kv_caches)

<<<<<<< HEAD

class ModelWrapperV1(nn.Module):

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(
        self,
        token_ids: torch.Tensor,
        position_ids: torch.Tensor,
        attn_metadata: AttentionMetadata,
        kv_caches: List[Tuple[torch.Tensor, torch.Tensor]],
    ) -> torch.Tensor:
        """Executes the forward pass of the model and samples the next token.

        Args:
            token_ids: The input token IDs of shape [batch_size, seq_len].
            position_ids: The input position IDs of shape [batch_size, seq_len].
            attn_metadata: The Pallas attention metadata.
            input_lens: The actual input lengths of shape [batch_size].
            t: The sampling temperature of shape [batch_size].
            p: The top-p probability of shape [batch_size].
            num_samples: Number of samples to draw from each logits vector.
            kv_caches: The key and value caches. They can be None during the
                memory profiling at initialization.
        """
        # Skip this in memory profiling at initialization.
        if attn_metadata is not None and kv_caches[0][0].numel() > 0:
            # index_copy_(slot_mapping) only works when the inserted dimension
            # is 0. However, the KV cache in the Pallas backend has the shape
            # [num_kv_heads, num_blocks, block_size, head_size]. To make it
            # work, we need to flatten the first three dimensions and modify
            # the slot_mapping accordingly.
            num_kv_heads, num_blocks, block_size, _ = kv_caches[0][0].shape
            slot_mapping = attn_metadata.slot_mapping
            slot_mapping = slot_mapping.flatten()
            head_indicies = torch.arange(0,
                                         num_kv_heads,
                                         device=slot_mapping.device,
                                         dtype=slot_mapping.dtype)
            head_indicies *= block_size * num_blocks
            slot_mapping = slot_mapping.repeat_interleave(num_kv_heads).view(
                -1, num_kv_heads)
            slot_mapping = slot_mapping + head_indicies.view(1, -1)
            slot_mapping = slot_mapping.flatten()
            attn_metadata.slot_mapping = slot_mapping

        assert self.model is not None
        hidden_states = self.model(
            token_ids,
            position_ids,
            kv_caches,
            attn_metadata,
        )

        hidden_states = hidden_states.flatten(0, 1)
        logits = self.model.compute_logits(hidden_states, None)

        # Greedy sampling.
        argmax_token_ids = torch.argmax(logits, dim=-1, keepdim=True)
        argmax_token_ids = argmax_token_ids.squeeze(dim=-1)
        return argmax_token_ids


def swap_positions(b: InputBatch, id_1, id_2):
    assert id_1 != id_2
    req_id_1 = b.req_ids[id_1]
    req_id_2 = b.req_ids[id_2]
    assert req_id_1 is not None
    assert req_id_2 is not None
    assert id_1 == b.req_id_to_index[req_id_1]
    assert id_2 == b.req_id_to_index[req_id_2]

    b.req_ids[id_1], b.req_ids[id_2] = b.req_ids[id_2], b.req_ids[id_1]
    b.req_id_to_index[req_id_1], b.req_id_to_index[
        req_id_2] = b.req_id_to_index[req_id_2], b.req_id_to_index[req_id_1]

    ids = [id_1, id_2]
    rev_ids = [id_2, id_1]
    b.num_tokens[ids] = b.num_tokens[rev_ids]
    b.token_ids_cpu[ids] = b.token_ids_cpu[rev_ids]
    b.num_prompt_tokens[ids] = b.num_prompt_tokens[rev_ids]
    b.num_computed_tokens_cpu[ids] = b.num_computed_tokens_cpu[rev_ids]

    b.block_table.swap_row(id_1, id_2)

    b.temperature_cpu[ids] = b.temperature_cpu[rev_ids]
    b.top_p_cpu[ids] = b.top_p_cpu[rev_ids]
    b.top_k_cpu[ids] = b.top_k_cpu[rev_ids]
    b.frequency_penalties_cpu[ids] = b.frequency_penalties_cpu[rev_ids]
    b.presence_penalties_cpu[ids] = b.presence_penalties_cpu[rev_ids]
    b.repetition_penalties_cpu[ids] = b.repetition_penalties_cpu[rev_ids]

    b.min_tokens[id_1], b.min_tokens[id_2] = b.min_tokens[id_2], b.min_tokens[
        id_1]

    gen_1 = b.generators.pop(id_1, None)
    gen_2 = b.generators.pop(id_2, None)
    if gen_1 is not None:
        b.generators[id_2] = gen_1
    if gen_2 is not None:
        b.generators[id_1] = gen_2


def ensure_decodes_first(b: InputBatch):
    num_reqs = b.num_reqs
    while True:
        # Find the first prompt index
        first_prompt_index = None
        for i in range(num_reqs):
            if b.num_computed_tokens_cpu[i] < b.num_prompt_tokens[i]:
                first_prompt_index = i
                break
        if first_prompt_index is None:
            break

        # Find the last decode index
        last_decode_index = None
        for i in reversed(range(num_reqs)):
            if b.num_computed_tokens_cpu[i] >= b.num_prompt_tokens[i]:
                last_decode_index = i
                break
        if last_decode_index is None:
            break

        # Sanity
        assert first_prompt_index != last_decode_index

        # Check if done
        if first_prompt_index > last_decode_index:
            break

        # Swap
        swap_positions(b, first_prompt_index, last_decode_index)


def _get_padded_prompt_len(x: int) -> int:
    # NOTE(woosuk): The pallas FlashAttention kernel requires the sequence
    # length to be a multiple of 16. We pad the prompt length to the nearest
    # multiple of 16. This is also good for performance.
    if x <= 16:
        return 16
    return 1 << (x - 1).bit_length()


def _get_padded_batch_size(batch_size: int) -> int:
    # The GMM Pallas kernel requires num_tokens * topk to be a multiple of 16.
    # To meet this requirement in the simplest way, we set the minimal batch
    # size to 8.
    if batch_size <= 8:
        return 8
    else:
        return ((batch_size + 15) // 16) * 16
=======
    def reset_dynamo_cache(self):
        if self.is_multimodal_model:
            compiled_model = self.model.get_language_model().model
        else:
            compiled_model = self.model.model
        if isinstance(compiled_model, TorchCompileWrapperWithCustomDispatcher):
            logger.info("Clear dynamo cache and cached dynamo bytecode.")
            torch._dynamo.eval_frame.remove_from_cache(
                compiled_model.original_code_object)
            compiled_model.compiled_codes.clear()

    @torch.compile(backend="openxla", fullgraph=True, dynamic=False)
    def select_hidden_states(self, hidden_states, indices_do_sample):
        return hidden_states[indices_do_sample]

    @torch.compile(backend="openxla", fullgraph=True, dynamic=False)
    def compute_logits(self,
                       sample_hidden_states: torch.Tensor) -> torch.Tensor:
        return self.model.compute_logits(sample_hidden_states, None)

    @torch.compile(backend="openxla", fullgraph=True, dynamic=False)
    def sample_from_logits(
            self, logits: torch.Tensor,
            sampling_metadata: TPUSupportedSamplingMetadata) -> torch.Tensor:
        """
        Sample with xla-friendly function. This function is to be traced 
        separately from `forward` for lighter compilation overhead.
        """
        if sampling_metadata.all_greedy:
            out_tokens = torch.argmax(logits, dim=-1, keepdim=True)
        else:
            out_tokens = self.sampler(logits,
                                      sampling_metadata).sampled_token_ids
        return out_tokens

    @torch.compile(backend="openxla", fullgraph=True, dynamic=False)
    def gather_logprobs(self, logits: torch.Tensor,
                        sampled_tokens: torch.Tensor) -> LogprobsTensors:
        """
        Gather the top_logprobs with corresponding tokens. Use a fixed number
        of logprobs as an alternative to having multiple pre-compiled graphs.
        Select the number of logprobs actually demanded by each request on CPU.
        """
        logprobs = self.sampler.compute_logprobs(logits)
        return self.sampler.gather_logprobs(
            logprobs,
            self.model_config.max_logprobs,
            token_ids=sampled_tokens.squeeze(-1))

    @torch.compile(backend="openxla", fullgraph=True, dynamic=False)
    def structured_decode(self, require_struct_decoding: torch.Tensor,
                          grammar_bitmask: torch.Tensor, logits: torch.Tensor,
                          arange: torch.Tensor) -> torch.Tensor:
        return torch.where(
            require_struct_decoding,
            self.apply_grammar_bitmask(logits, grammar_bitmask, arange),
            logits)

    def apply_grammar_bitmask(self, logits: torch.Tensor,
                              grammar_bitmask: torch.Tensor,
                              arange: torch.Tensor):
        assert (logits.shape[0] == grammar_bitmask.shape[0])
        logits_cloned = logits.clone()
        for i in range(logits.shape[0]):
            unpacked_bitmask = (torch.bitwise_right_shift(
                grammar_bitmask[i][:, None], arange[None, :]) & 1) == 0
            unpacked_bitmask = unpacked_bitmask.reshape(-1)[:self.vocab_size]
            logits_cloned[i] = logits_cloned[i].masked_fill(
                unpacked_bitmask, -float("inf"))
        return logits_cloned

    def get_multimodal_embeddings(self, *args, **kwargs):
        return self.model.get_multimodal_embeddings(*args, **kwargs)

    def get_input_embeddings(self, *args, **kwargs):
        return self.model.get_input_embeddings(*args, **kwargs)

    def prepare_structured_decoding_input(
        self, logits: torch.Tensor, scheduler_output: "SchedulerOutput"
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        grammar_bitmask = scheduler_output.grammar_bitmask
        assert grammar_bitmask is not None
        num_reqs, _ = logits.shape

        # Reset pre-allocated tensors
        self.grammar_bitmask_cpu.zero_()
        self.require_structured_out_cpu.zero_()

        # We receive the structured output bitmask from the scheduler, but the
        # indices of the requests in the batch may not match the indices of
        # the bitmask since the scheduler doesn't know how the tpu runner is
        # ordering the requests in the batch. We need to match the order of
        # bitmask with the order of requests
        struct_out_indices: list[int] = []
        mask_indices: list[int] = []
        for req_id in self.input_batch.req_ids:
            mask_index = scheduler_output.structured_output_request_ids.get(
                req_id)
            if mask_index is None:
                continue
            batch_index = self.input_batch.req_id_to_index[req_id]
            struct_out_indices.append(batch_index)
            mask_indices.append(mask_index)
        self.grammar_bitmask_cpu[struct_out_indices] = torch.from_numpy(
            grammar_bitmask[mask_indices])
        # It's not guaranteed that all requests in this batch require
        # structured output, so create a bool tensor to represent
        # the requests that need structured output.
        struct_out_indices = torch.tensor(struct_out_indices, dtype=torch.long)
        self.require_structured_out_cpu[struct_out_indices] = True
        return self.require_structured_out_cpu[:num_reqs].to(logits.device), \
            self.grammar_bitmask_cpu[:num_reqs].to(logits.device), \
            self.structured_decode_arange.to(logits.device)

    def _get_mm_dummy_batch(self, modality: str,
                            batch_size: int) -> BatchedTensorInputs:
        # Dummy data for pre-compiling multimodal models.
        dummy_request_data = self.mm_registry.get_decoder_dummy_data(
            model_config=self.model_config,
            seq_len=self.max_num_tokens,
        )
        dummy_mm_data = dummy_request_data.multi_modal_data

        # Dummy data definition in V0 may contain multiple multimodal items
        # (e.g, multiple images) for a single request, therefore here we
        # always replicate first item by max_num_mm_items times since in V1
        # they are scheduled to be processed separately.
        assert isinstance(dummy_mm_data, MultiModalKwargs), (
            "Expected dummy multimodal data to be of type "
            f"MultiModalKwargs, got {type(dummy_mm_data)=} instead. "
            "This is most likely due to the model not having a merged "
            "processor.")

        # When models have a merged processor, their dummy data is
        # already batched `MultiModalKwargs`, therefore we take the first
        # `MultiModalKwargsItem` from the desired modality to profile on.
        dummy_mm_item = dummy_mm_data.get_item(modality=modality, item_index=0)
        dummy_mm_kwargs = MultiModalKwargs.from_items([dummy_mm_item])

        batched_dummy_mm_inputs = MultiModalKwargs.batch([dummy_mm_kwargs] *
                                                         batch_size)
        return MultiModalKwargs.as_kwargs(batched_dummy_mm_inputs,
                                          device=self.device)


def _get_req_paddings(min_req_size: int, max_req_size: int) -> list[int]:
    logger.info("Preparing request paddings:")
    # assert min_req_size is power of 2
    assert (min_req_size & (min_req_size - 1) == 0) and min_req_size > 0
    paddings: list = []
    num = max(MIN_NUM_SEQS, min_req_size)
    while num <= max_req_size and (len(paddings) == 0 or paddings[-1] != num):
        paddings.append(num)
        logger.info("    %d", num)
        num = _get_padded_num_reqs_with_upper_limit(num + 1, max_req_size)
    return paddings


def _get_padded_num_reqs_with_upper_limit(x: int, upper_limit: int) -> int:
    res = MIN_NUM_SEQS if x <= MIN_NUM_SEQS else 1 << (x - 1).bit_length()
    return min(res, upper_limit)


def _get_token_paddings(min_token_size: int, max_token_size: int,
                        padding_gap: int) -> list[int]:
    """Generate a list of padding size, starting from min_token_size, 
    ending with a number that can cover max_token_size
    
    If padding_gap == 0 then:
        increase 2X each time (exponential)
    else:
        first increase the size to twice, 
        then increase the padding size by padding_gap.
    """
    # assert min_token_size is power of 2
    assert (min_token_size & (min_token_size - 1) == 0) and min_token_size > 0
    paddings = []
    num = min_token_size

    if padding_gap == 0:
        logger.info("Using exponential token paddings:")
        while True:
            logger.info("    %d", num)
            paddings.append(num)
            if num >= max_token_size:
                break
            num *= 2
    else:
        logger.info("Using incremental token paddings:")
        while num <= padding_gap:
            logger.info("    %d", num)
            paddings.append(num)
            num *= 2
        num //= 2
        while num < max_token_size:
            num += padding_gap
            logger.info("    %d", num)
            paddings.append(num)

    return paddings


def _get_padded_token_len(paddings: list[int], x: int) -> int:
    """Return the first element in paddings list greater or equal to x.
    """
    index = bisect.bisect_left(paddings, x)
    assert index < len(paddings)
    return paddings[index]
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
