# SPDX-License-Identifier: Apache-2.0

<<<<<<< HEAD
import argparse
import dataclasses
import json
from dataclasses import dataclass
from typing import (TYPE_CHECKING, Any, Dict, List, Literal, Mapping, Optional,
                    Tuple, Type, Union, cast, get_args)

import torch

import vllm.envs as envs
from vllm import version
from vllm.config import (CacheConfig, CompilationConfig, ConfigFormat,
                         DecodingConfig, DeviceConfig, HfOverrides,
                         KVTransferConfig, LoadConfig, LoadFormat, LoRAConfig,
                         ModelConfig, ModelImpl, ObservabilityConfig,
                         ParallelConfig, PoolerConfig, PromptAdapterConfig,
                         SchedulerConfig, SpeculativeConfig, TaskOption,
                         TokenizerPoolConfig, VllmConfig)
from vllm.executor.executor_base import ExecutorBase
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization import QUANTIZATION_METHODS
from vllm.plugins import load_general_plugins
from vllm.transformers_utils.utils import check_gguf_file
from vllm.usage.usage_lib import UsageContext
from vllm.utils import FlexibleArgumentParser, StoreBoolean

if TYPE_CHECKING:
    from vllm.transformers_utils.tokenizer_group import BaseTokenizerGroup

logger = init_logger(__name__)

ALLOWED_DETAILED_TRACE_MODULES = ["model", "worker", "all"]

DEVICE_OPTIONS = [
    "auto",
    "cuda",
    "neuron",
    "cpu",
    "openvino",
    "tpu",
    "xpu",
    "hpu",
]


def nullable_str(val: str):
    if not val or val == "None":
        return None
    return val


def nullable_kvs(val: str) -> Optional[Mapping[str, int]]:
=======
# yapf: disable
import argparse
import dataclasses
import json
import re
import sys
import threading
import warnings
from dataclasses import MISSING, dataclass, fields, is_dataclass
from itertools import permutations
from typing import (Annotated, Any, Callable, Dict, List, Literal, Optional,
                    Type, TypeVar, Union, cast, get_args, get_origin)

import torch
from typing_extensions import TypeIs, deprecated

import vllm.envs as envs
from vllm.config import (BlockSize, CacheConfig, CacheDType, CompilationConfig,
                         ConfigFormat, ConfigType, DecodingConfig,
                         DetailedTraceModules, Device, DeviceConfig,
                         DistributedExecutorBackend, GuidedDecodingBackend,
                         GuidedDecodingBackendV1, HfOverrides, KVEventsConfig,
                         KVTransferConfig, LoadConfig, LoadFormat, LoRAConfig,
                         ModelConfig, ModelDType, ModelImpl, MultiModalConfig,
                         ObservabilityConfig, ParallelConfig, PoolerConfig,
                         PrefixCachingHashAlgo, PromptAdapterConfig,
                         SchedulerConfig, SchedulerPolicy, SpeculativeConfig,
                         TaskOption, TokenizerMode, TokenizerPoolConfig,
                         VllmConfig, get_attr_docs, get_field)
from vllm.executor.executor_base import ExecutorBase
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization import QuantizationMethods
from vllm.plugins import load_general_plugins
from vllm.reasoning import ReasoningParserManager
from vllm.test_utils import MODEL_WEIGHTS_S3_BUCKET, MODELS_ON_S3
from vllm.transformers_utils.utils import check_gguf_file
from vllm.usage.usage_lib import UsageContext
from vllm.utils import (STR_DUAL_CHUNK_FLASH_ATTN_VAL, FlexibleArgumentParser,
                        GiB_bytes, is_in_doc_build, is_in_ray_actor)

# yapf: enable

logger = init_logger(__name__)

# object is used to allow for special typing forms
T = TypeVar("T")
TypeHint = Union[type[Any], object]
TypeHintT = Union[type[T], object]


def parse_type(return_type: Callable[[str], T]) -> Callable[[str], T]:

    def _parse_type(val: str) -> T:
        try:
            if return_type is json.loads and not re.match("^{.*}$", val):
                return cast(T, nullable_kvs(val))
            return return_type(val)
        except ValueError as e:
            raise argparse.ArgumentTypeError(
                f"Value {val} cannot be converted to {return_type}.") from e

    return _parse_type


def optional_type(
        return_type: Callable[[str], T]) -> Callable[[str], Optional[T]]:

    def _optional_type(val: str) -> Optional[T]:
        if val == "" or val == "None":
            return None
        return parse_type(return_type)(val)

    return _optional_type


def union_dict_and_str(val: str) -> Optional[Union[str, dict[str, str]]]:
    if not re.match("^{.*}$", val):
        return str(val)
    return optional_type(json.loads)(val)


@deprecated(
    "Passing a JSON argument as a string containing comma separated key=value "
    "pairs is deprecated. This will be removed in v0.10.0. Please use a JSON "
    "string instead.")
def nullable_kvs(val: str) -> dict[str, int]:
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
    """Parses a string containing comma separate key [str] to value [int]
    pairs into a dictionary.

    Args:
        val: String value to be parsed.

    Returns:
        Dictionary with parsed values.
    """
<<<<<<< HEAD
    if len(val) == 0:
        return None

    out_dict: Dict[str, int] = {}
=======
    out_dict: dict[str, int] = {}
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
    for item in val.split(","):
        kv_parts = [part.lower().strip() for part in item.split("=")]
        if len(kv_parts) != 2:
            raise argparse.ArgumentTypeError(
                "Each item should be in the form KEY=VALUE")
        key, value = kv_parts

        try:
            parsed_value = int(value)
        except ValueError as exc:
            msg = f"Failed to parse value of item {key}={value}"
            raise argparse.ArgumentTypeError(msg) from exc

        if key in out_dict and out_dict[key] != parsed_value:
            raise argparse.ArgumentTypeError(
                f"Conflicting values specified for key: {key}")
        out_dict[key] = parsed_value

    return out_dict


<<<<<<< HEAD
@dataclass
class EngineArgs:
    """Arguments for vLLM engine."""
    model: str = 'facebook/opt-125m'
    served_model_name: Optional[Union[str, List[str]]] = None
    tokenizer: Optional[str] = None
    task: TaskOption = "auto"
    skip_tokenizer_init: bool = False
    tokenizer_mode: str = 'auto'
    trust_remote_code: bool = False
    allowed_local_media_path: str = ""
    download_dir: Optional[str] = None
    load_format: str = 'auto'
    config_format: ConfigFormat = ConfigFormat.AUTO
    dtype: str = 'auto'
    kv_cache_dtype: str = 'auto'
    seed: int = 0
    max_model_len: Optional[int] = None
    # Note: Specifying a custom executor backend by passing a class
    # is intended for expert use only. The API may change without
    # notice.
    distributed_executor_backend: Optional[Union[str,
                                                 Type[ExecutorBase]]] = None
    # number of P/D disaggregation (or other disaggregation) workers
    pipeline_parallel_size: int = 1
    tensor_parallel_size: int = 1
    max_parallel_loading_workers: Optional[int] = None
    block_size: Optional[int] = None
    enable_prefix_caching: Optional[bool] = None
    disable_sliding_window: bool = False
    use_v2_block_manager: bool = True
    swap_space: float = 4  # GiB
    cpu_offload_gb: float = 0  # GiB
    gpu_memory_utilization: float = 0.90
    max_num_batched_tokens: Optional[int] = None
    max_num_partial_prefills: Optional[int] = 1
    max_long_partial_prefills: Optional[int] = 1
    long_prefill_token_threshold: Optional[int] = 0
    max_num_seqs: Optional[int] = None
    max_logprobs: int = 20  # Default value for OpenAI Chat Completions API
    disable_log_stats: bool = False
    revision: Optional[str] = None
    code_revision: Optional[str] = None
    rope_scaling: Optional[Dict[str, Any]] = None
    rope_theta: Optional[float] = None
    hf_overrides: Optional[HfOverrides] = None
    tokenizer_revision: Optional[str] = None
    quantization: Optional[str] = None
    enforce_eager: Optional[bool] = None
    max_seq_len_to_capture: int = 8192
    disable_custom_all_reduce: bool = False
    tokenizer_pool_size: int = 0
    # Note: Specifying a tokenizer pool by passing a class
    # is intended for expert use only. The API may change without
    # notice.
    tokenizer_pool_type: Union[str, Type["BaseTokenizerGroup"]] = "ray"
    tokenizer_pool_extra_config: Optional[Dict[str, Any]] = None
    limit_mm_per_prompt: Optional[Mapping[str, int]] = None
    mm_processor_kwargs: Optional[Dict[str, Any]] = None
    disable_mm_preprocessor_cache: bool = False
    enable_lora: bool = False
    enable_lora_bias: bool = False
    max_loras: int = 1
    max_lora_rank: int = 16
    enable_prompt_adapter: bool = False
    max_prompt_adapters: int = 1
    max_prompt_adapter_token: int = 0
    fully_sharded_loras: bool = False
    lora_extra_vocab_size: int = 256
    long_lora_scaling_factors: Optional[Tuple[float]] = None
    lora_dtype: Optional[Union[str, torch.dtype]] = 'auto'
    max_cpu_loras: Optional[int] = None
    device: str = 'auto'
    num_scheduler_steps: int = 1
    multi_step_stream_outputs: bool = True
    ray_workers_use_nsight: bool = False
    num_gpu_blocks_override: Optional[int] = None
    num_lookahead_slots: int = 0
    model_loader_extra_config: Optional[dict] = None
    ignore_patterns: Optional[Union[str, List[str]]] = None
    preemption_mode: Optional[str] = None

    scheduler_delay_factor: float = 0.0
    enable_chunked_prefill: Optional[bool] = None

    guided_decoding_backend: str = 'xgrammar'
    logits_processor_pattern: Optional[str] = None
    # Speculative decoding configuration.
    speculative_model: Optional[str] = None
    speculative_model_quantization: Optional[str] = None
    speculative_draft_tensor_parallel_size: Optional[int] = None
    num_speculative_tokens: Optional[int] = None
    speculative_disable_mqa_scorer: Optional[bool] = False
    speculative_max_model_len: Optional[int] = None
    speculative_disable_by_batch_size: Optional[int] = None
    ngram_prompt_lookup_max: Optional[int] = None
    ngram_prompt_lookup_min: Optional[int] = None
    spec_decoding_acceptance_method: str = 'rejection_sampler'
    typical_acceptance_sampler_posterior_threshold: Optional[float] = None
    typical_acceptance_sampler_posterior_alpha: Optional[float] = None
    qlora_adapter_name_or_path: Optional[str] = None
    disable_logprobs_during_spec_decoding: Optional[bool] = None

    show_hidden_metrics_for_version: Optional[str] = None
    otlp_traces_endpoint: Optional[str] = None
    collect_detailed_traces: Optional[str] = None
    disable_async_output_proc: bool = False
    scheduling_policy: Literal["fcfs", "priority"] = "fcfs"
    scheduler_cls: Union[str, Type[object]] = "vllm.core.scheduler.Scheduler"

    override_neuron_config: Optional[Dict[str, Any]] = None
    override_pooler_config: Optional[PoolerConfig] = None
    compilation_config: Optional[CompilationConfig] = None
    worker_cls: str = "auto"

    kv_transfer_config: Optional[KVTransferConfig] = None

    generation_config: Optional[str] = None
    override_generation_config: Optional[Dict[str, Any]] = None
    enable_sleep_mode: bool = False
    model_impl: str = "auto"

    calculate_kv_scales: Optional[bool] = None

    additional_config: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if not self.tokenizer:
            self.tokenizer = self.model

        # Override the default value of enable_prefix_caching if it's not set
        # by user.
        if self.enable_prefix_caching is None:
            self.enable_prefix_caching = bool(envs.VLLM_USE_V1)

        # Override max_num_seqs if it's not set by user.
        if self.max_num_seqs is None:
            self.max_num_seqs = 256 if not envs.VLLM_USE_V1 else 1024

=======
def is_type(type_hint: TypeHint, type: TypeHintT) -> TypeIs[TypeHintT]:
    """Check if the type hint is a specific type."""
    return type_hint is type or get_origin(type_hint) is type


def contains_type(type_hints: set[TypeHint], type: TypeHintT) -> bool:
    """Check if the type hints contain a specific type."""
    return any(is_type(type_hint, type) for type_hint in type_hints)


def get_type(type_hints: set[TypeHint], type: TypeHintT) -> TypeHintT:
    """Get the specific type from the type hints."""
    return next((th for th in type_hints if is_type(th, type)), None)


def literal_to_kwargs(type_hints: set[TypeHint]) -> dict[str, Any]:
    """Convert Literal type hints to argparse kwargs."""
    type_hint = get_type(type_hints, Literal)
    choices = get_args(type_hint)
    choice_type = type(choices[0])
    if not all(isinstance(choice, choice_type) for choice in choices):
        raise ValueError(
            "All choices must be of the same type. "
            f"Got {choices} with types {[type(c) for c in choices]}")
    return {"type": choice_type, "choices": sorted(choices)}


def is_not_builtin(type_hint: TypeHint) -> bool:
    """Check if the class is not a built-in type."""
    return type_hint.__module__ != "builtins"


def get_kwargs(cls: ConfigType) -> dict[str, Any]:
    cls_docs = get_attr_docs(cls)
    kwargs = {}
    for field in fields(cls):
        # Get the set of possible types for the field
        type_hints: set[TypeHint] = set()
        if get_origin(field.type) in {Union, Annotated}:
            type_hints.update(get_args(field.type))
        else:
            type_hints.add(field.type)

        # If the field is a dataclass, we can use the model_validate_json
        generator = (th for th in type_hints if is_dataclass(th))
        dataclass_cls = next(generator, None)

        # Get the default value of the field
        if field.default is not MISSING:
            default = field.default
        elif field.default_factory is not MISSING:
            if is_dataclass(field.default_factory) and is_in_doc_build():
                default = {}
            else:
                default = field.default_factory()

        # Get the help text for the field
        name = field.name
        help = cls_docs[name].strip()
        # Escape % for argparse
        help = help.replace("%", "%%")

        # Initialise the kwargs dictionary for the field
        kwargs[name] = {"default": default, "help": help}

        # Set other kwargs based on the type hints
        json_tip = """\n\nShould either be a valid JSON string or JSON keys
        passed individually. For example, the following sets of arguments are
        equivalent:\n\n
        - `--json-arg '{"key1": "value1", "key2": {"key3": "value2"}}'`\n
        - `--json-arg.key1 value1 --json-arg.key2.key3 value2`\n\n"""
        if dataclass_cls is not None:
            dataclass_init = lambda x, f=dataclass_cls: f(**json.loads(x))
            # Special case for configs with a from_cli method
            if hasattr(dataclass_cls, "from_cli"):
                from_cli = dataclass_cls.from_cli
                dataclass_init = lambda x, f=from_cli: f(x)
            kwargs[name]["type"] = dataclass_init
            kwargs[name]["help"] += json_tip
        elif contains_type(type_hints, bool):
            # Creates --no-<name> and --<name> flags
            kwargs[name]["action"] = argparse.BooleanOptionalAction
        elif contains_type(type_hints, Literal):
            kwargs[name].update(literal_to_kwargs(type_hints))
        elif contains_type(type_hints, tuple):
            type_hint = get_type(type_hints, tuple)
            types = get_args(type_hint)
            tuple_type = types[0]
            assert all(t is tuple_type for t in types if t is not Ellipsis), (
                "All non-Ellipsis tuple elements must be of the same "
                f"type. Got {types}.")
            kwargs[name]["type"] = tuple_type
            kwargs[name]["nargs"] = "+" if Ellipsis in types else len(types)
        elif contains_type(type_hints, list):
            type_hint = get_type(type_hints, list)
            types = get_args(type_hint)
            assert len(types) == 1, (
                "List type must have exactly one type. Got "
                f"{type_hint} with types {types}")
            kwargs[name]["type"] = types[0]
            kwargs[name]["nargs"] = "+"
        elif contains_type(type_hints, int):
            kwargs[name]["type"] = int
            # Special case for large integers
            if name in {"max_model_len"}:
                kwargs[name]["type"] = human_readable_int
        elif contains_type(type_hints, float):
            kwargs[name]["type"] = float
        elif contains_type(type_hints,
                           dict) and (contains_type(type_hints, str) or any(
                               is_not_builtin(th) for th in type_hints)):
            kwargs[name]["type"] = union_dict_and_str
        elif contains_type(type_hints, dict):
            # Dict arguments will always be optional
            kwargs[name]["type"] = parse_type(json.loads)
            kwargs[name]["help"] += json_tip
        elif (contains_type(type_hints, str)
              or any(is_not_builtin(th) for th in type_hints)):
            kwargs[name]["type"] = str
        else:
            raise ValueError(
                f"Unsupported type {type_hints} for argument {name}.")

        # If the type hint was a sequence of literals, use the helper function
        # to update the type and choices
        if get_origin(kwargs[name].get("type")) is Literal:
            kwargs[name].update(literal_to_kwargs({kwargs[name]["type"]}))

        # If None is in type_hints, make the argument optional.
        # But not if it's a bool, argparse will handle this better.
        if type(None) in type_hints and not contains_type(type_hints, bool):
            kwargs[name]["type"] = optional_type(kwargs[name]["type"])
            if kwargs[name].get("choices"):
                kwargs[name]["choices"].append("None")
    return kwargs


@dataclass
class EngineArgs:
    """Arguments for vLLM engine."""
    model: str = ModelConfig.model
    served_model_name: Optional[Union[
        str, List[str]]] = ModelConfig.served_model_name
    tokenizer: Optional[str] = ModelConfig.tokenizer
    hf_config_path: Optional[str] = ModelConfig.hf_config_path
    task: TaskOption = ModelConfig.task
    skip_tokenizer_init: bool = ModelConfig.skip_tokenizer_init
    enable_prompt_embeds: bool = ModelConfig.enable_prompt_embeds
    tokenizer_mode: TokenizerMode = ModelConfig.tokenizer_mode
    trust_remote_code: bool = ModelConfig.trust_remote_code
    allowed_local_media_path: str = ModelConfig.allowed_local_media_path
    download_dir: Optional[str] = LoadConfig.download_dir
    load_format: str = LoadConfig.load_format
    config_format: str = ModelConfig.config_format
    dtype: ModelDType = ModelConfig.dtype
    kv_cache_dtype: CacheDType = CacheConfig.cache_dtype
    seed: Optional[int] = ModelConfig.seed
    max_model_len: Optional[int] = ModelConfig.max_model_len
    cuda_graph_sizes: list[int] = get_field(SchedulerConfig,
                                            "cuda_graph_sizes")
    # Note: Specifying a custom executor backend by passing a class
    # is intended for expert use only. The API may change without
    # notice.
    distributed_executor_backend: Optional[Union[
        DistributedExecutorBackend,
        Type[ExecutorBase]]] = ParallelConfig.distributed_executor_backend
    # number of P/D disaggregation (or other disaggregation) workers
    pipeline_parallel_size: int = ParallelConfig.pipeline_parallel_size
    tensor_parallel_size: int = ParallelConfig.tensor_parallel_size
    data_parallel_size: int = ParallelConfig.data_parallel_size
    data_parallel_size_local: Optional[int] = None
    data_parallel_address: Optional[str] = None
    data_parallel_rpc_port: Optional[int] = None
    enable_expert_parallel: bool = ParallelConfig.enable_expert_parallel
    max_parallel_loading_workers: Optional[
        int] = ParallelConfig.max_parallel_loading_workers
    block_size: Optional[BlockSize] = CacheConfig.block_size
    enable_prefix_caching: Optional[bool] = CacheConfig.enable_prefix_caching
    prefix_caching_hash_algo: PrefixCachingHashAlgo = \
        CacheConfig.prefix_caching_hash_algo
    disable_sliding_window: bool = ModelConfig.disable_sliding_window
    disable_cascade_attn: bool = ModelConfig.disable_cascade_attn
    use_v2_block_manager: bool = True
    swap_space: float = CacheConfig.swap_space
    cpu_offload_gb: float = CacheConfig.cpu_offload_gb
    gpu_memory_utilization: float = CacheConfig.gpu_memory_utilization
    max_num_batched_tokens: Optional[
        int] = SchedulerConfig.max_num_batched_tokens
    max_num_partial_prefills: int = SchedulerConfig.max_num_partial_prefills
    max_long_partial_prefills: int = SchedulerConfig.max_long_partial_prefills
    long_prefill_token_threshold: int = \
        SchedulerConfig.long_prefill_token_threshold
    max_num_seqs: Optional[int] = SchedulerConfig.max_num_seqs
    max_logprobs: int = ModelConfig.max_logprobs
    disable_log_stats: bool = False
    revision: Optional[str] = ModelConfig.revision
    code_revision: Optional[str] = ModelConfig.code_revision
    rope_scaling: dict[str, Any] = get_field(ModelConfig, "rope_scaling")
    rope_theta: Optional[float] = ModelConfig.rope_theta
    hf_token: Optional[Union[bool, str]] = ModelConfig.hf_token
    hf_overrides: Optional[HfOverrides] = \
        get_field(ModelConfig, "hf_overrides")
    tokenizer_revision: Optional[str] = ModelConfig.tokenizer_revision
    quantization: Optional[QuantizationMethods] = ModelConfig.quantization
    enforce_eager: bool = ModelConfig.enforce_eager
    max_seq_len_to_capture: int = ModelConfig.max_seq_len_to_capture
    disable_custom_all_reduce: bool = ParallelConfig.disable_custom_all_reduce
    # The following three fields are deprecated and will be removed in a future
    # release. Setting them will have no effect. Please remove them from your
    # configurations.
    tokenizer_pool_size: int = TokenizerPoolConfig.pool_size
    tokenizer_pool_type: str = TokenizerPoolConfig.pool_type
    tokenizer_pool_extra_config: dict = \
        get_field(TokenizerPoolConfig, "extra_config")
    limit_mm_per_prompt: dict[str, int] = \
        get_field(MultiModalConfig, "limit_per_prompt")
    mm_processor_kwargs: Optional[Dict[str, Any]] = \
        MultiModalConfig.mm_processor_kwargs
    disable_mm_preprocessor_cache: bool = \
        MultiModalConfig.disable_mm_preprocessor_cache
    # LoRA fields
    enable_lora: bool = False
    enable_lora_bias: bool = LoRAConfig.bias_enabled
    max_loras: int = LoRAConfig.max_loras
    max_lora_rank: int = LoRAConfig.max_lora_rank
    fully_sharded_loras: bool = LoRAConfig.fully_sharded_loras
    max_cpu_loras: Optional[int] = LoRAConfig.max_cpu_loras
    lora_dtype: Optional[Union[str, torch.dtype]] = LoRAConfig.lora_dtype
    lora_extra_vocab_size: int = LoRAConfig.lora_extra_vocab_size
    long_lora_scaling_factors: Optional[tuple[float, ...]] = \
        LoRAConfig.long_lora_scaling_factors
    # PromptAdapter fields
    enable_prompt_adapter: bool = False
    max_prompt_adapters: int = PromptAdapterConfig.max_prompt_adapters
    max_prompt_adapter_token: int = \
        PromptAdapterConfig.max_prompt_adapter_token

    device: Device = DeviceConfig.device
    num_scheduler_steps: int = SchedulerConfig.num_scheduler_steps
    multi_step_stream_outputs: bool = SchedulerConfig.multi_step_stream_outputs
    ray_workers_use_nsight: bool = ParallelConfig.ray_workers_use_nsight
    num_gpu_blocks_override: Optional[
        int] = CacheConfig.num_gpu_blocks_override
    num_lookahead_slots: int = SchedulerConfig.num_lookahead_slots
    model_loader_extra_config: dict = \
        get_field(LoadConfig, "model_loader_extra_config")
    ignore_patterns: Optional[Union[str,
                                    List[str]]] = LoadConfig.ignore_patterns
    preemption_mode: Optional[str] = SchedulerConfig.preemption_mode

    scheduler_delay_factor: float = SchedulerConfig.delay_factor
    enable_chunked_prefill: Optional[
        bool] = SchedulerConfig.enable_chunked_prefill
    disable_chunked_mm_input: bool = SchedulerConfig.disable_chunked_mm_input

    guided_decoding_backend: GuidedDecodingBackend = DecodingConfig.backend
    guided_decoding_disable_fallback: bool = DecodingConfig.disable_fallback
    guided_decoding_disable_any_whitespace: bool = \
        DecodingConfig.disable_any_whitespace
    guided_decoding_disable_additional_properties: bool = \
        DecodingConfig.disable_additional_properties
    logits_processor_pattern: Optional[
        str] = ModelConfig.logits_processor_pattern

    speculative_config: Optional[Dict[str, Any]] = None

    qlora_adapter_name_or_path: Optional[str] = None
    show_hidden_metrics_for_version: Optional[str] = \
        ObservabilityConfig.show_hidden_metrics_for_version
    otlp_traces_endpoint: Optional[str] = \
        ObservabilityConfig.otlp_traces_endpoint
    collect_detailed_traces: Optional[list[DetailedTraceModules]] = \
        ObservabilityConfig.collect_detailed_traces
    disable_async_output_proc: bool = not ModelConfig.use_async_output_proc
    scheduling_policy: SchedulerPolicy = SchedulerConfig.policy
    scheduler_cls: Union[str, Type[object]] = SchedulerConfig.scheduler_cls

    override_neuron_config: dict[str, Any] = \
        get_field(ModelConfig, "override_neuron_config")
    override_pooler_config: Optional[Union[dict, PoolerConfig]] = \
        ModelConfig.override_pooler_config
    compilation_config: Optional[CompilationConfig] = None
    worker_cls: str = ParallelConfig.worker_cls
    worker_extension_cls: str = ParallelConfig.worker_extension_cls

    kv_transfer_config: Optional[KVTransferConfig] = None
    kv_events_config: Optional[KVEventsConfig] = None

    generation_config: str = ModelConfig.generation_config
    enable_sleep_mode: bool = ModelConfig.enable_sleep_mode
    override_generation_config: dict[str, Any] = \
        get_field(ModelConfig, "override_generation_config")
    model_impl: str = ModelConfig.model_impl

    calculate_kv_scales: bool = CacheConfig.calculate_kv_scales

    additional_config: Optional[Dict[str, Any]] = None
    enable_reasoning: Optional[bool] = None  # DEPRECATED
    reasoning_parser: str = DecodingConfig.reasoning_backend

    use_tqdm_on_load: bool = LoadConfig.use_tqdm_on_load
    pt_load_map_location: str = LoadConfig.pt_load_map_location

    def __post_init__(self):
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
        # support `EngineArgs(compilation_config={...})`
        # without having to manually construct a
        # CompilationConfig object
        if isinstance(self.compilation_config, (int, dict)):
            self.compilation_config = CompilationConfig.from_cli(
                str(self.compilation_config))
<<<<<<< HEAD

=======
        if self.qlora_adapter_name_or_path is not None:
            warnings.warn(
                "The `qlora_adapter_name_or_path` is deprecated "
                "and will be removed in v0.10.0. ",
                DeprecationWarning,
                stacklevel=2,
            )
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
        # Setup plugins
        from vllm.plugins import load_general_plugins
        load_general_plugins()

    @staticmethod
    def add_cli_args(parser: FlexibleArgumentParser) -> FlexibleArgumentParser:
        """Shared CLI arguments for vLLM engine."""

        # Model arguments
<<<<<<< HEAD
        parser.add_argument(
            '--model',
            type=str,
            default=EngineArgs.model,
            help='Name or path of the huggingface model to use.')
        parser.add_argument(
            '--task',
            default=EngineArgs.task,
            choices=get_args(TaskOption),
            help='The task to use the model for. Each vLLM instance only '
            'supports one task, even if the same model can be used for '
            'multiple tasks. When the model only supports one task, ``"auto"`` '
            'can be used to select it; otherwise, you must specify explicitly '
            'which task to use.')
        parser.add_argument(
            '--tokenizer',
            type=nullable_str,
            default=EngineArgs.tokenizer,
            help='Name or path of the huggingface tokenizer to use. '
            'If unspecified, model name or path will be used.')
        parser.add_argument(
            '--skip-tokenizer-init',
            action='store_true',
            help='Skip initialization of tokenizer and detokenizer.')
        parser.add_argument(
            '--revision',
            type=nullable_str,
            default=None,
            help='The specific model version to use. It can be a branch '
            'name, a tag name, or a commit id. If unspecified, will use '
            'the default version.')
        parser.add_argument(
            '--code-revision',
            type=nullable_str,
            default=None,
            help='The specific revision to use for the model code on '
            'Hugging Face Hub. It can be a branch name, a tag name, or a '
            'commit id. If unspecified, will use the default version.')
        parser.add_argument(
            '--tokenizer-revision',
            type=nullable_str,
            default=None,
            help='Revision of the huggingface tokenizer to use. '
            'It can be a branch name, a tag name, or a commit id. '
            'If unspecified, will use the default version.')
        parser.add_argument(
            '--tokenizer-mode',
            type=str,
            default=EngineArgs.tokenizer_mode,
            choices=['auto', 'slow', 'mistral', 'custom'],
            help='The tokenizer mode.\n\n* "auto" will use the '
            'fast tokenizer if available.\n* "slow" will '
            'always use the slow tokenizer. \n* '
            '"mistral" will always use the `mistral_common` tokenizer. \n* '
            '"custom" will use --tokenizer to select the '
            'preregistered tokenizer.')
        parser.add_argument('--trust-remote-code',
                            action='store_true',
                            help='Trust remote code from huggingface.')
        parser.add_argument(
            '--allowed-local-media-path',
            type=str,
            help="Allowing API requests to read local images or videos "
            "from directories specified by the server file system. "
            "This is a security risk. "
            "Should only be enabled in trusted environments.")
        parser.add_argument('--download-dir',
                            type=nullable_str,
                            default=EngineArgs.download_dir,
                            help='Directory to download and load the weights, '
                            'default to the default cache dir of '
                            'huggingface.')
        parser.add_argument(
            '--load-format',
            type=str,
            default=EngineArgs.load_format,
            choices=[f.value for f in LoadFormat],
            help='The format of the model weights to load.\n\n'
            '* "auto" will try to load the weights in the safetensors format '
            'and fall back to the pytorch bin format if safetensors format '
            'is not available.\n'
            '* "pt" will load the weights in the pytorch bin format.\n'
            '* "safetensors" will load the weights in the safetensors format.\n'
            '* "npcache" will load the weights in pytorch format and store '
            'a numpy cache to speed up the loading.\n'
            '* "dummy" will initialize the weights with random values, '
            'which is mainly for profiling.\n'
            '* "tensorizer" will load the weights using tensorizer from '
            'CoreWeave. See the Tensorize vLLM Model script in the Examples '
            'section for more information.\n'
            '* "runai_streamer" will load the Safetensors weights using Run:ai'
            'Model Streamer \n'
            '* "bitsandbytes" will load the weights using bitsandbytes '
            'quantization.\n')
        parser.add_argument(
            '--config-format',
            default=EngineArgs.config_format,
            choices=[f.value for f in ConfigFormat],
            help='The format of the model config to load.\n\n'
            '* "auto" will try to load the config in hf format '
            'if available else it will try to load in mistral format ')
        parser.add_argument(
            '--dtype',
            type=str,
            default=EngineArgs.dtype,
            choices=[
                'auto', 'half', 'float16', 'bfloat16', 'float', 'float32'
            ],
            help='Data type for model weights and activations.\n\n'
            '* "auto" will use FP16 precision for FP32 and FP16 models, and '
            'BF16 precision for BF16 models.\n'
            '* "half" for FP16. Recommended for AWQ quantization.\n'
            '* "float16" is the same as "half".\n'
            '* "bfloat16" for a balance between precision and range.\n'
            '* "float" is shorthand for FP32 precision.\n'
            '* "float32" for FP32 precision.')
        parser.add_argument(
            '--kv-cache-dtype',
            type=str,
            choices=['auto', 'fp8', 'fp8_e5m2', 'fp8_e4m3'],
            default=EngineArgs.kv_cache_dtype,
            help='Data type for kv cache storage. If "auto", will use model '
            'data type. CUDA 11.8+ supports fp8 (=fp8_e4m3) and fp8_e5m2. '
            'ROCm (AMD GPU) supports fp8 (=fp8_e4m3)')
        parser.add_argument('--max-model-len',
                            type=int,
                            default=EngineArgs.max_model_len,
                            help='Model context length. If unspecified, will '
                            'be automatically derived from the model config.')
        parser.add_argument(
            '--guided-decoding-backend',
            type=str,
            default='xgrammar',
            help='Which engine will be used for guided decoding'
            ' (JSON schema / regex etc) by default. Currently support '
            'https://github.com/outlines-dev/outlines, '
            'https://github.com/mlc-ai/xgrammar, and '
            'https://github.com/noamgat/lm-format-enforcer.'
            ' Can be overridden per request via guided_decoding_backend'
            ' parameter.\n'
            'Backend-sepcific options can be supplied in a comma-separated '
            'list following a colon after the backend name. Valid backends and '
            'all available options are: [xgrammar:no-fallback, '
            'outlines:no-fallback, lm-format-enforcer:no-fallback]')
        parser.add_argument(
            '--logits-processor-pattern',
            type=nullable_str,
            default=None,
            help='Optional regex pattern specifying valid logits processor '
            'qualified names that can be passed with the `logits_processors` '
            'extra completion argument. Defaults to None, which allows no '
            'processors.')
        parser.add_argument(
            '--model-impl',
            type=str,
            default=EngineArgs.model_impl,
            choices=[f.value for f in ModelImpl],
            help='Which implementation of the model to use.\n\n'
            '* "auto" will try to use the vLLM implementation if it exists '
            'and fall back to the Transformers implementation if no vLLM '
            'implementation is available.\n'
            '* "vllm" will use the vLLM model implementation.\n'
            '* "transformers" will use the Transformers model '
            'implementation.\n')
        # Parallel arguments
        parser.add_argument(
            '--distributed-executor-backend',
            choices=['ray', 'mp', 'uni', 'external_launcher'],
            default=EngineArgs.distributed_executor_backend,
            help='Backend to use for distributed model '
            'workers, either "ray" or "mp" (multiprocessing). If the product '
            'of pipeline_parallel_size and tensor_parallel_size is less than '
            'or equal to the number of GPUs available, "mp" will be used to '
            'keep processing on a single host. Otherwise, this will default '
            'to "ray" if Ray is installed and fail otherwise. Note that tpu '
            'only supports Ray for distributed inference.')

        parser.add_argument('--pipeline-parallel-size',
                            '-pp',
                            type=int,
                            default=EngineArgs.pipeline_parallel_size,
                            help='Number of pipeline stages.')
        parser.add_argument('--tensor-parallel-size',
                            '-tp',
                            type=int,
                            default=EngineArgs.tensor_parallel_size,
                            help='Number of tensor parallel replicas.')
        parser.add_argument(
            '--max-parallel-loading-workers',
            type=int,
            default=EngineArgs.max_parallel_loading_workers,
            help='Load model sequentially in multiple batches, '
            'to avoid RAM OOM when using tensor '
            'parallel and large models.')
        parser.add_argument(
            '--ray-workers-use-nsight',
            action='store_true',
            help='If specified, use nsight to profile Ray workers.')
        # KV cache arguments
        parser.add_argument('--block-size',
                            type=int,
                            default=EngineArgs.block_size,
                            choices=[8, 16, 32, 64, 128],
                            help='Token block size for contiguous chunks of '
                            'tokens. This is ignored on neuron devices and '
                            'set to ``--max-model-len``. On CUDA devices, '
                            'only block sizes up to 32 are supported. '
                            'On HPU devices, block size defaults to 128.')

        parser.add_argument(
            "--enable-prefix-caching",
            action=argparse.BooleanOptionalAction,
            default=EngineArgs.enable_prefix_caching,
            help="Enables automatic prefix caching. "
            "Use ``--no-enable-prefix-caching`` to disable explicitly.",
        )
        parser.add_argument('--disable-sliding-window',
                            action='store_true',
                            help='Disables sliding window, '
                            'capping to sliding window size.')
        parser.add_argument('--use-v2-block-manager',
                            action='store_true',
                            default=True,
=======
        model_kwargs = get_kwargs(ModelConfig)
        model_group = parser.add_argument_group(
            title="ModelConfig",
            description=ModelConfig.__doc__,
        )
        if 'serve' not in sys.argv[1:] and '--help' not in sys.argv[1:]:
            model_group.add_argument("--model", **model_kwargs["model"])
        model_group.add_argument("--task", **model_kwargs["task"])
        model_group.add_argument("--tokenizer", **model_kwargs["tokenizer"])
        model_group.add_argument("--tokenizer-mode",
                                 **model_kwargs["tokenizer_mode"])
        model_group.add_argument("--trust-remote-code",
                                 **model_kwargs["trust_remote_code"])
        model_group.add_argument("--dtype", **model_kwargs["dtype"])
        model_group.add_argument("--seed", **model_kwargs["seed"])
        model_group.add_argument("--hf-config-path",
                                 **model_kwargs["hf_config_path"])
        model_group.add_argument("--allowed-local-media-path",
                                 **model_kwargs["allowed_local_media_path"])
        model_group.add_argument("--revision", **model_kwargs["revision"])
        model_group.add_argument("--code-revision",
                                 **model_kwargs["code_revision"])
        model_group.add_argument("--rope-scaling",
                                 **model_kwargs["rope_scaling"])
        model_group.add_argument("--rope-theta", **model_kwargs["rope_theta"])
        model_group.add_argument("--tokenizer-revision",
                                 **model_kwargs["tokenizer_revision"])
        model_group.add_argument("--max-model-len",
                                 **model_kwargs["max_model_len"])
        model_group.add_argument("--quantization", "-q",
                                 **model_kwargs["quantization"])
        model_group.add_argument("--enforce-eager",
                                 **model_kwargs["enforce_eager"])
        model_group.add_argument("--max-seq-len-to-capture",
                                 **model_kwargs["max_seq_len_to_capture"])
        model_group.add_argument("--max-logprobs",
                                 **model_kwargs["max_logprobs"])
        model_group.add_argument("--disable-sliding-window",
                                 **model_kwargs["disable_sliding_window"])
        model_group.add_argument("--disable-cascade-attn",
                                 **model_kwargs["disable_cascade_attn"])
        model_group.add_argument("--skip-tokenizer-init",
                                 **model_kwargs["skip_tokenizer_init"])
        model_group.add_argument("--enable-prompt-embeds",
                                 **model_kwargs["enable_prompt_embeds"])
        model_group.add_argument("--served-model-name",
                                 **model_kwargs["served_model_name"])
        # This one is a special case because it is the
        # opposite of ModelConfig.use_async_output_proc
        model_group.add_argument(
            "--disable-async-output-proc",
            action="store_true",
            default=EngineArgs.disable_async_output_proc,
            help="Disable async output processing. This may result in "
            "lower performance.")
        model_group.add_argument("--config-format",
                                 choices=[f.value for f in ConfigFormat],
                                 **model_kwargs["config_format"])
        # This one is a special case because it can bool
        # or str. TODO: Handle this in get_kwargs
        model_group.add_argument("--hf-token",
                                 type=str,
                                 nargs="?",
                                 const=True,
                                 default=model_kwargs["hf_token"]["default"],
                                 help=model_kwargs["hf_token"]["help"])
        model_group.add_argument("--hf-overrides",
                                 **model_kwargs["hf_overrides"])
        model_group.add_argument("--override-neuron-config",
                                 **model_kwargs["override_neuron_config"])
        model_group.add_argument("--override-pooler-config",
                                 **model_kwargs["override_pooler_config"])
        model_group.add_argument("--logits-processor-pattern",
                                 **model_kwargs["logits_processor_pattern"])
        model_group.add_argument("--generation-config",
                                 **model_kwargs["generation_config"])
        model_group.add_argument("--override-generation-config",
                                 **model_kwargs["override_generation_config"])
        model_group.add_argument("--enable-sleep-mode",
                                 **model_kwargs["enable_sleep_mode"])
        model_group.add_argument("--model-impl",
                                 choices=[f.value for f in ModelImpl],
                                 **model_kwargs["model_impl"])

        # Model loading arguments
        load_kwargs = get_kwargs(LoadConfig)
        load_group = parser.add_argument_group(
            title="LoadConfig",
            description=LoadConfig.__doc__,
        )
        load_group.add_argument("--load-format",
                                choices=[f.value for f in LoadFormat],
                                **load_kwargs["load_format"])
        load_group.add_argument("--download-dir",
                                **load_kwargs["download_dir"])
        load_group.add_argument("--model-loader-extra-config",
                                **load_kwargs["model_loader_extra_config"])
        load_group.add_argument("--ignore-patterns",
                                **load_kwargs["ignore_patterns"])
        load_group.add_argument("--use-tqdm-on-load",
                                **load_kwargs["use_tqdm_on_load"])
        load_group.add_argument(
            "--qlora-adapter-name-or-path",
            type=str,
            default=None,
            help="The `--qlora-adapter-name-or-path` has no effect, do not set"
            " it, and it  will be removed in v0.10.0.",
            deprecated=True,
        )
        load_group.add_argument('--pt-load-map-location',
                                **load_kwargs["pt_load_map_location"])

        # Guided decoding arguments
        guided_decoding_kwargs = get_kwargs(DecodingConfig)
        guided_decoding_group = parser.add_argument_group(
            title="DecodingConfig",
            description=DecodingConfig.__doc__,
        )
        guided_decoding_group.add_argument("--guided-decoding-backend",
                                           **guided_decoding_kwargs["backend"])
        guided_decoding_group.add_argument(
            "--guided-decoding-disable-fallback",
            **guided_decoding_kwargs["disable_fallback"])
        guided_decoding_group.add_argument(
            "--guided-decoding-disable-any-whitespace",
            **guided_decoding_kwargs["disable_any_whitespace"])
        guided_decoding_group.add_argument(
            "--guided-decoding-disable-additional-properties",
            **guided_decoding_kwargs["disable_additional_properties"])
        guided_decoding_group.add_argument(
            "--enable-reasoning",
            action=argparse.BooleanOptionalAction,
            deprecated=True,
            help="[DEPRECATED] The `--enable-reasoning` flag is deprecated as "
            "of v0.8.6. Use `--reasoning-parser` to specify the reasoning "
            "parser backend instead. This flag (`--enable-reasoning`) will be "
            "removed in v0.10.0. When `--reasoning-parser` is specified, "
            "reasoning mode is automatically enabled.")
        guided_decoding_group.add_argument(
            "--reasoning-parser",
            # This choices is a special case because it's not static
            choices=list(ReasoningParserManager.reasoning_parsers),
            **guided_decoding_kwargs["reasoning_backend"])

        # Parallel arguments
        parallel_kwargs = get_kwargs(ParallelConfig)
        parallel_group = parser.add_argument_group(
            title="ParallelConfig",
            description=ParallelConfig.__doc__,
        )
        parallel_group.add_argument(
            "--distributed-executor-backend",
            **parallel_kwargs["distributed_executor_backend"])
        parallel_group.add_argument(
            "--pipeline-parallel-size", "-pp",
            **parallel_kwargs["pipeline_parallel_size"])
        parallel_group.add_argument("--tensor-parallel-size", "-tp",
                                    **parallel_kwargs["tensor_parallel_size"])
        parallel_group.add_argument("--data-parallel-size", "-dp",
                                    **parallel_kwargs["data_parallel_size"])
        parallel_group.add_argument('--data-parallel-size-local',
                                    '-dpl',
                                    type=int,
                                    help='Number of data parallel replicas '
                                    'to run on this node.')
        parallel_group.add_argument('--data-parallel-address',
                                    '-dpa',
                                    type=str,
                                    help='Address of data parallel cluster '
                                    'head-node.')
        parallel_group.add_argument('--data-parallel-rpc-port',
                                    '-dpp',
                                    type=int,
                                    help='Port for data parallel RPC '
                                    'communication.')
        parallel_group.add_argument(
            "--enable-expert-parallel",
            **parallel_kwargs["enable_expert_parallel"])
        parallel_group.add_argument(
            "--max-parallel-loading-workers",
            **parallel_kwargs["max_parallel_loading_workers"])
        parallel_group.add_argument(
            "--ray-workers-use-nsight",
            **parallel_kwargs["ray_workers_use_nsight"])
        parallel_group.add_argument(
            "--disable-custom-all-reduce",
            **parallel_kwargs["disable_custom_all_reduce"])
        parallel_group.add_argument("--worker-cls",
                                    **parallel_kwargs["worker_cls"])
        parallel_group.add_argument("--worker-extension-cls",
                                    **parallel_kwargs["worker_extension_cls"])

        # KV cache arguments
        cache_kwargs = get_kwargs(CacheConfig)
        cache_group = parser.add_argument_group(
            title="CacheConfig",
            description=CacheConfig.__doc__,
        )
        cache_group.add_argument("--block-size", **cache_kwargs["block_size"])
        cache_group.add_argument("--gpu-memory-utilization",
                                 **cache_kwargs["gpu_memory_utilization"])
        cache_group.add_argument("--swap-space", **cache_kwargs["swap_space"])
        cache_group.add_argument("--kv-cache-dtype",
                                 **cache_kwargs["cache_dtype"])
        cache_group.add_argument("--num-gpu-blocks-override",
                                 **cache_kwargs["num_gpu_blocks_override"])
        cache_group.add_argument("--enable-prefix-caching",
                                 **cache_kwargs["enable_prefix_caching"])
        cache_group.add_argument("--prefix-caching-hash-algo",
                                 **cache_kwargs["prefix_caching_hash_algo"])
        cache_group.add_argument("--cpu-offload-gb",
                                 **cache_kwargs["cpu_offload_gb"])
        cache_group.add_argument("--calculate-kv-scales",
                                 **cache_kwargs["calculate_kv_scales"])

        # Tokenizer arguments
        tokenizer_kwargs = get_kwargs(TokenizerPoolConfig)
        tokenizer_group = parser.add_argument_group(
            title="TokenizerPoolConfig",
            description=TokenizerPoolConfig.__doc__,
        )
        tokenizer_group.add_argument("--tokenizer-pool-size",
                                     **tokenizer_kwargs["pool_size"])
        tokenizer_group.add_argument("--tokenizer-pool-type",
                                     **tokenizer_kwargs["pool_type"])
        tokenizer_group.add_argument("--tokenizer-pool-extra-config",
                                     **tokenizer_kwargs["extra_config"])

        # Multimodal related configs
        multimodal_kwargs = get_kwargs(MultiModalConfig)
        multimodal_group = parser.add_argument_group(
            title="MultiModalConfig",
            description=MultiModalConfig.__doc__,
        )
        multimodal_group.add_argument("--limit-mm-per-prompt",
                                      **multimodal_kwargs["limit_per_prompt"])
        multimodal_group.add_argument(
            "--mm-processor-kwargs",
            **multimodal_kwargs["mm_processor_kwargs"])
        multimodal_group.add_argument(
            "--disable-mm-preprocessor-cache",
            **multimodal_kwargs["disable_mm_preprocessor_cache"])

        # LoRA related configs
        lora_kwargs = get_kwargs(LoRAConfig)
        lora_group = parser.add_argument_group(
            title="LoRAConfig",
            description=LoRAConfig.__doc__,
        )
        lora_group.add_argument(
            "--enable-lora",
            action=argparse.BooleanOptionalAction,
            help="If True, enable handling of LoRA adapters.")
        lora_group.add_argument("--enable-lora-bias",
                                **lora_kwargs["bias_enabled"])
        lora_group.add_argument("--max-loras", **lora_kwargs["max_loras"])
        lora_group.add_argument("--max-lora-rank",
                                **lora_kwargs["max_lora_rank"])
        lora_group.add_argument("--lora-extra-vocab-size",
                                **lora_kwargs["lora_extra_vocab_size"])
        lora_group.add_argument(
            "--lora-dtype",
            **lora_kwargs["lora_dtype"],
        )
        lora_group.add_argument("--long-lora-scaling-factors",
                                **lora_kwargs["long_lora_scaling_factors"])
        lora_group.add_argument("--max-cpu-loras",
                                **lora_kwargs["max_cpu_loras"])
        lora_group.add_argument("--fully-sharded-loras",
                                **lora_kwargs["fully_sharded_loras"])

        # PromptAdapter related configs
        prompt_adapter_kwargs = get_kwargs(PromptAdapterConfig)
        prompt_adapter_group = parser.add_argument_group(
            title="PromptAdapterConfig",
            description=PromptAdapterConfig.__doc__,
        )
        prompt_adapter_group.add_argument(
            "--enable-prompt-adapter",
            action=argparse.BooleanOptionalAction,
            help="If True, enable handling of PromptAdapters.")
        prompt_adapter_group.add_argument(
            "--max-prompt-adapters",
            **prompt_adapter_kwargs["max_prompt_adapters"])
        prompt_adapter_group.add_argument(
            "--max-prompt-adapter-token",
            **prompt_adapter_kwargs["max_prompt_adapter_token"])

        # Device arguments
        device_kwargs = get_kwargs(DeviceConfig)
        device_group = parser.add_argument_group(
            title="DeviceConfig",
            description=DeviceConfig.__doc__,
        )
        device_group.add_argument("--device",
                                  **device_kwargs["device"],
                                  deprecated=True)

        # Speculative arguments
        speculative_group = parser.add_argument_group(
            title="SpeculativeConfig",
            description=SpeculativeConfig.__doc__,
        )
        speculative_group.add_argument(
            "--speculative-config",
            type=json.loads,
            default=None,
            help="The configurations for speculative decoding. Should be a "
            "JSON string.")

        # Observability arguments
        observability_kwargs = get_kwargs(ObservabilityConfig)
        observability_group = parser.add_argument_group(
            title="ObservabilityConfig",
            description=ObservabilityConfig.__doc__,
        )
        observability_group.add_argument(
            "--show-hidden-metrics-for-version",
            **observability_kwargs["show_hidden_metrics_for_version"])
        observability_group.add_argument(
            "--otlp-traces-endpoint",
            **observability_kwargs["otlp_traces_endpoint"])
        # TODO: generalise this special case
        choices = observability_kwargs["collect_detailed_traces"]["choices"]
        metavar = f"{{{','.join(choices)}}}"
        observability_kwargs["collect_detailed_traces"]["metavar"] = metavar
        observability_kwargs["collect_detailed_traces"]["choices"] += [
            ",".join(p)
            for p in permutations(get_args(DetailedTraceModules), r=2)
        ]
        observability_group.add_argument(
            "--collect-detailed-traces",
            **observability_kwargs["collect_detailed_traces"])

        # Scheduler arguments
        scheduler_kwargs = get_kwargs(SchedulerConfig)
        scheduler_group = parser.add_argument_group(
            title="SchedulerConfig",
            description=SchedulerConfig.__doc__,
        )
        scheduler_group.add_argument(
            "--max-num-batched-tokens",
            **scheduler_kwargs["max_num_batched_tokens"])
        scheduler_group.add_argument("--max-num-seqs",
                                     **scheduler_kwargs["max_num_seqs"])
        scheduler_group.add_argument(
            "--max-num-partial-prefills",
            **scheduler_kwargs["max_num_partial_prefills"])
        scheduler_group.add_argument(
            "--max-long-partial-prefills",
            **scheduler_kwargs["max_long_partial_prefills"])
        scheduler_group.add_argument('--cuda-graph-sizes',
                                     **scheduler_kwargs["cuda_graph_sizes"])
        scheduler_group.add_argument(
            "--long-prefill-token-threshold",
            **scheduler_kwargs["long_prefill_token_threshold"])
        scheduler_group.add_argument("--num-lookahead-slots",
                                     **scheduler_kwargs["num_lookahead_slots"])
        scheduler_group.add_argument("--scheduler-delay-factor",
                                     **scheduler_kwargs["delay_factor"])
        scheduler_group.add_argument("--preemption-mode",
                                     **scheduler_kwargs["preemption_mode"])
        scheduler_group.add_argument("--num-scheduler-steps",
                                     **scheduler_kwargs["num_scheduler_steps"])
        scheduler_group.add_argument(
            "--multi-step-stream-outputs",
            **scheduler_kwargs["multi_step_stream_outputs"])
        scheduler_group.add_argument("--scheduling-policy",
                                     **scheduler_kwargs["policy"])
        scheduler_group.add_argument(
            "--enable-chunked-prefill",
            **scheduler_kwargs["enable_chunked_prefill"])
        scheduler_group.add_argument(
            "--disable-chunked-mm-input",
            **scheduler_kwargs["disable_chunked_mm_input"])
        scheduler_group.add_argument("--scheduler-cls",
                                     **scheduler_kwargs["scheduler_cls"])

        # vLLM arguments
        vllm_kwargs = get_kwargs(VllmConfig)
        vllm_group = parser.add_argument_group(
            title="VllmConfig",
            description=VllmConfig.__doc__,
        )
        vllm_group.add_argument("--kv-transfer-config",
                                **vllm_kwargs["kv_transfer_config"])
        vllm_group.add_argument('--kv-events-config',
                                **vllm_kwargs["kv_events_config"])
        vllm_group.add_argument("--compilation-config", "-O",
                                **vllm_kwargs["compilation_config"])
        vllm_group.add_argument("--additional-config",
                                **vllm_kwargs["additional_config"])

        # Other arguments
        parser.add_argument('--use-v2-block-manager',
                            action='store_true',
                            default=True,
                            deprecated=True,
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
                            help='[DEPRECATED] block manager v1 has been '
                            'removed and SelfAttnBlockSpaceManager (i.e. '
                            'block manager v2) is now the default. '
                            'Setting this flag to True or False'
                            ' has no effect on vLLM behavior.')
<<<<<<< HEAD
        parser.add_argument(
            '--num-lookahead-slots',
            type=int,
            default=EngineArgs.num_lookahead_slots,
            help='Experimental scheduling config necessary for '
            'speculative decoding. This will be replaced by '
            'speculative config in the future; it is present '
            'to enable correctness tests until then.')

        parser.add_argument('--seed',
                            type=int,
                            default=EngineArgs.seed,
                            help='Random seed for operations.')
        parser.add_argument('--swap-space',
                            type=float,
                            default=EngineArgs.swap_space,
                            help='CPU swap space size (GiB) per GPU.')
        parser.add_argument(
            '--cpu-offload-gb',
            type=float,
            default=0,
            help='The space in GiB to offload to CPU, per GPU. '
            'Default is 0, which means no offloading. Intuitively, '
            'this argument can be seen as a virtual way to increase '
            'the GPU memory size. For example, if you have one 24 GB '
            'GPU and set this to 10, virtually you can think of it as '
            'a 34 GB GPU. Then you can load a 13B model with BF16 weight, '
            'which requires at least 26GB GPU memory. Note that this '
            'requires fast CPU-GPU interconnect, as part of the model is '
            'loaded from CPU memory to GPU memory on the fly in each '
            'model forward pass.')
        parser.add_argument(
            '--gpu-memory-utilization',
            type=float,
            default=EngineArgs.gpu_memory_utilization,
            help='The fraction of GPU memory to be used for the model '
            'executor, which can range from 0 to 1. For example, a value of '
            '0.5 would imply 50%% GPU memory utilization. If unspecified, '
            'will use the default value of 0.9. This is a per-instance '
            'limit, and only applies to the current vLLM instance.'
            'It does not matter if you have another vLLM instance running '
            'on the same GPU. For example, if you have two vLLM instances '
            'running on the same GPU, you can set the GPU memory utilization '
            'to 0.5 for each instance.')
        parser.add_argument(
            '--num-gpu-blocks-override',
            type=int,
            default=None,
            help='If specified, ignore GPU profiling result and use this number'
            ' of GPU blocks. Used for testing preemption.')
        parser.add_argument('--max-num-batched-tokens',
                            type=int,
                            default=EngineArgs.max_num_batched_tokens,
                            help='Maximum number of batched tokens per '
                            'iteration.')
        parser.add_argument(
            "--max-num-partial-prefills",
            type=int,
            default=EngineArgs.max_num_partial_prefills,
            help="For chunked prefill, the max number of concurrent \
            partial prefills."
            "Defaults to 1",
        )
        parser.add_argument(
            "--max-long-partial-prefills",
            type=int,
            default=EngineArgs.max_long_partial_prefills,
            help="For chunked prefill, the maximum number of prompts longer "
            "than --long-prefill-token-threshold that will be prefilled "
            "concurrently. Setting this less than --max-num-partial-prefills "
            "will allow shorter prompts to jump the queue in front of longer "
            "prompts in some cases, improving latency. Defaults to 1.")
        parser.add_argument(
            "--long-prefill-token-threshold",
            type=float,
            default=EngineArgs.long_prefill_token_threshold,
            help="For chunked prefill, a request is considered long if the "
            "prompt is longer than this number of tokens. Defaults to 4%% of "
            "the model's context length.",
        )
        parser.add_argument('--max-num-seqs',
                            type=int,
                            default=EngineArgs.max_num_seqs,
                            help='Maximum number of sequences per iteration.')
        parser.add_argument(
            '--max-logprobs',
            type=int,
            default=EngineArgs.max_logprobs,
            help=('Max number of log probs to return logprobs is specified in'
                  ' SamplingParams.'))
        parser.add_argument('--disable-log-stats',
                            action='store_true',
                            help='Disable logging statistics.')
        # Quantization settings.
        parser.add_argument('--quantization',
                            '-q',
                            type=nullable_str,
                            choices=[*QUANTIZATION_METHODS, None],
                            default=EngineArgs.quantization,
                            help='Method used to quantize the weights. If '
                            'None, we first check the `quantization_config` '
                            'attribute in the model config file. If that is '
                            'None, we assume the model weights are not '
                            'quantized and use `dtype` to determine the data '
                            'type of the weights.')
        parser.add_argument(
            '--rope-scaling',
            default=None,
            type=json.loads,
            help='RoPE scaling configuration in JSON format. '
            'For example, ``{"rope_type":"dynamic","factor":2.0}``')
        parser.add_argument('--rope-theta',
                            default=None,
                            type=float,
                            help='RoPE theta. Use with `rope_scaling`. In '
                            'some cases, changing the RoPE theta improves the '
                            'performance of the scaled model.')
        parser.add_argument('--hf-overrides',
                            type=json.loads,
                            default=EngineArgs.hf_overrides,
                            help='Extra arguments for the HuggingFace config. '
                            'This should be a JSON string that will be '
                            'parsed into a dictionary.')
        parser.add_argument('--enforce-eager',
                            action='store_true',
                            help='Always use eager-mode PyTorch. If False, '
                            'will use eager mode and CUDA graph in hybrid '
                            'for maximal performance and flexibility.')
        parser.add_argument('--max-seq-len-to-capture',
                            type=int,
                            default=EngineArgs.max_seq_len_to_capture,
                            help='Maximum sequence length covered by CUDA '
                            'graphs. When a sequence has context length '
                            'larger than this, we fall back to eager mode. '
                            'Additionally for encoder-decoder models, if the '
                            'sequence length of the encoder input is larger '
                            'than this, we fall back to the eager mode.')
        parser.add_argument('--disable-custom-all-reduce',
                            action='store_true',
                            default=EngineArgs.disable_custom_all_reduce,
                            help='See ParallelConfig.')
        parser.add_argument('--tokenizer-pool-size',
                            type=int,
                            default=EngineArgs.tokenizer_pool_size,
                            help='Size of tokenizer pool to use for '
                            'asynchronous tokenization. If 0, will '
                            'use synchronous tokenization.')
        parser.add_argument('--tokenizer-pool-type',
                            type=str,
                            default=EngineArgs.tokenizer_pool_type,
                            help='Type of tokenizer pool to use for '
                            'asynchronous tokenization. Ignored '
                            'if tokenizer_pool_size is 0.')
        parser.add_argument('--tokenizer-pool-extra-config',
                            type=nullable_str,
                            default=EngineArgs.tokenizer_pool_extra_config,
                            help='Extra config for tokenizer pool. '
                            'This should be a JSON string that will be '
                            'parsed into a dictionary. Ignored if '
                            'tokenizer_pool_size is 0.')

        # Multimodal related configs
        parser.add_argument(
            '--limit-mm-per-prompt',
            type=nullable_kvs,
            default=EngineArgs.limit_mm_per_prompt,
            # The default value is given in
            # MultiModalRegistry.init_mm_limits_per_prompt
            help=('For each multimodal plugin, limit how many '
                  'input instances to allow for each prompt. '
                  'Expects a comma-separated list of items, '
                  'e.g.: `image=16,video=2` allows a maximum of 16 '
                  'images and 2 videos per prompt. Defaults to 1 for '
                  'each modality.'))
        parser.add_argument(
            '--mm-processor-kwargs',
            default=None,
            type=json.loads,
            help=('Overrides for the multimodal input mapping/processing, '
                  'e.g., image processor. For example: ``{"num_crops": 4}``.'))
        parser.add_argument(
            '--disable-mm-preprocessor-cache',
            action='store_true',
            help='If true, then disables caching of the multi-modal '
            'preprocessor/mapper. (not recommended)')

        # LoRA related configs
        parser.add_argument('--enable-lora',
                            action='store_true',
                            help='If True, enable handling of LoRA adapters.')
        parser.add_argument('--enable-lora-bias',
                            action='store_true',
                            help='If True, enable bias for LoRA adapters.')
        parser.add_argument('--max-loras',
                            type=int,
                            default=EngineArgs.max_loras,
                            help='Max number of LoRAs in a single batch.')
        parser.add_argument('--max-lora-rank',
                            type=int,
                            default=EngineArgs.max_lora_rank,
                            help='Max LoRA rank.')
        parser.add_argument(
            '--lora-extra-vocab-size',
            type=int,
            default=EngineArgs.lora_extra_vocab_size,
            help=('Maximum size of extra vocabulary that can be '
                  'present in a LoRA adapter (added to the base '
                  'model vocabulary).'))
        parser.add_argument(
            '--lora-dtype',
            type=str,
            default=EngineArgs.lora_dtype,
            choices=['auto', 'float16', 'bfloat16'],
            help=('Data type for LoRA. If auto, will default to '
                  'base model dtype.'))
        parser.add_argument(
            '--long-lora-scaling-factors',
            type=nullable_str,
            default=EngineArgs.long_lora_scaling_factors,
            help=('Specify multiple scaling factors (which can '
                  'be different from base model scaling factor '
                  '- see eg. Long LoRA) to allow for multiple '
                  'LoRA adapters trained with those scaling '
                  'factors to be used at the same time. If not '
                  'specified, only adapters trained with the '
                  'base model scaling factor are allowed.'))
        parser.add_argument(
            '--max-cpu-loras',
            type=int,
            default=EngineArgs.max_cpu_loras,
            help=('Maximum number of LoRAs to store in CPU memory. '
                  'Must be >= than max_loras. '
                  'Defaults to max_loras.'))
        parser.add_argument(
            '--fully-sharded-loras',
            action='store_true',
            help=('By default, only half of the LoRA computation is '
                  'sharded with tensor parallelism. '
                  'Enabling this will use the fully sharded layers. '
                  'At high sequence length, max rank or '
                  'tensor parallel size, this is likely faster.'))
        parser.add_argument('--enable-prompt-adapter',
                            action='store_true',
                            help='If True, enable handling of PromptAdapters.')
        parser.add_argument('--max-prompt-adapters',
                            type=int,
                            default=EngineArgs.max_prompt_adapters,
                            help='Max number of PromptAdapters in a batch.')
        parser.add_argument('--max-prompt-adapter-token',
                            type=int,
                            default=EngineArgs.max_prompt_adapter_token,
                            help='Max number of PromptAdapters tokens')
        parser.add_argument("--device",
                            type=str,
                            default=EngineArgs.device,
                            choices=DEVICE_OPTIONS,
                            help='Device type for vLLM execution.')
        parser.add_argument('--num-scheduler-steps',
                            type=int,
                            default=1,
                            help=('Maximum number of forward steps per '
                                  'scheduler call.'))

        parser.add_argument(
            '--multi-step-stream-outputs',
            action=StoreBoolean,
            default=EngineArgs.multi_step_stream_outputs,
            nargs="?",
            const="True",
            help='If False, then multi-step will stream outputs at the end '
            'of all steps')
        parser.add_argument(
            '--scheduler-delay-factor',
            type=float,
            default=EngineArgs.scheduler_delay_factor,
            help='Apply a delay (of delay factor multiplied by previous '
            'prompt latency) before scheduling next prompt.')
        parser.add_argument(
            '--enable-chunked-prefill',
            action=StoreBoolean,
            default=EngineArgs.enable_chunked_prefill,
            nargs="?",
            const="True",
            help='If set, the prefill requests can be chunked based on the '
            'max_num_batched_tokens.')

        parser.add_argument(
            '--speculative-model',
            type=nullable_str,
            default=EngineArgs.speculative_model,
            help=
            'The name of the draft model to be used in speculative decoding.')
        # Quantization settings for speculative model.
        parser.add_argument(
            '--speculative-model-quantization',
            type=nullable_str,
            choices=[*QUANTIZATION_METHODS, None],
            default=EngineArgs.speculative_model_quantization,
            help='Method used to quantize the weights of speculative model. '
            'If None, we first check the `quantization_config` '
            'attribute in the model config file. If that is '
            'None, we assume the model weights are not '
            'quantized and use `dtype` to determine the data '
            'type of the weights.')
        parser.add_argument(
            '--num-speculative-tokens',
            type=int,
            default=EngineArgs.num_speculative_tokens,
            help='The number of speculative tokens to sample from '
            'the draft model in speculative decoding.')
        parser.add_argument(
            '--speculative-disable-mqa-scorer',
            action='store_true',
            help=
            'If set to True, the MQA scorer will be disabled in speculative '
            ' and fall back to batch expansion')
        parser.add_argument(
            '--speculative-draft-tensor-parallel-size',
            '-spec-draft-tp',
            type=int,
            default=EngineArgs.speculative_draft_tensor_parallel_size,
            help='Number of tensor parallel replicas for '
            'the draft model in speculative decoding.')

        parser.add_argument(
            '--speculative-max-model-len',
            type=int,
            default=EngineArgs.speculative_max_model_len,
            help='The maximum sequence length supported by the '
            'draft model. Sequences over this length will skip '
            'speculation.')

        parser.add_argument(
            '--speculative-disable-by-batch-size',
            type=int,
            default=EngineArgs.speculative_disable_by_batch_size,
            help='Disable speculative decoding for new incoming requests '
            'if the number of enqueue requests is larger than this value.')

        parser.add_argument(
            '--ngram-prompt-lookup-max',
            type=int,
            default=EngineArgs.ngram_prompt_lookup_max,
            help='Max size of window for ngram prompt lookup in speculative '
            'decoding.')

        parser.add_argument(
            '--ngram-prompt-lookup-min',
            type=int,
            default=EngineArgs.ngram_prompt_lookup_min,
            help='Min size of window for ngram prompt lookup in speculative '
            'decoding.')

        parser.add_argument(
            '--spec-decoding-acceptance-method',
            type=str,
            default=EngineArgs.spec_decoding_acceptance_method,
            choices=['rejection_sampler', 'typical_acceptance_sampler'],
            help='Specify the acceptance method to use during draft token '
            'verification in speculative decoding. Two types of acceptance '
            'routines are supported: '
            '1) RejectionSampler which does not allow changing the '
            'acceptance rate of draft tokens, '
            '2) TypicalAcceptanceSampler which is configurable, allowing for '
            'a higher acceptance rate at the cost of lower quality, '
            'and vice versa.')

        parser.add_argument(
            '--typical-acceptance-sampler-posterior-threshold',
            type=float,
            default=EngineArgs.typical_acceptance_sampler_posterior_threshold,
            help='Set the lower bound threshold for the posterior '
            'probability of a token to be accepted. This threshold is '
            'used by the TypicalAcceptanceSampler to make sampling decisions '
            'during speculative decoding. Defaults to 0.09')

        parser.add_argument(
            '--typical-acceptance-sampler-posterior-alpha',
            type=float,
            default=EngineArgs.typical_acceptance_sampler_posterior_alpha,
            help='A scaling factor for the entropy-based threshold for token '
            'acceptance in the TypicalAcceptanceSampler. Typically defaults '
            'to sqrt of --typical-acceptance-sampler-posterior-threshold '
            'i.e. 0.3')

        parser.add_argument(
            '--disable-logprobs-during-spec-decoding',
            action=StoreBoolean,
            default=EngineArgs.disable_logprobs_during_spec_decoding,
            nargs="?",
            const="True",
            help='If set to True, token log probabilities are not returned '
            'during speculative decoding. If set to False, log probabilities '
            'are returned according to the settings in SamplingParams. If '
            'not specified, it defaults to True. Disabling log probabilities '
            'during speculative decoding reduces latency by skipping logprob '
            'calculation in proposal sampling, target sampling, and after '
            'accepted tokens are determined.')

        parser.add_argument('--model-loader-extra-config',
                            type=nullable_str,
                            default=EngineArgs.model_loader_extra_config,
                            help='Extra config for model loader. '
                            'This will be passed to the model loader '
                            'corresponding to the chosen load_format. '
                            'This should be a JSON string that will be '
                            'parsed into a dictionary.')
        parser.add_argument(
            '--ignore-patterns',
            action="append",
            type=str,
            default=[],
            help="The pattern(s) to ignore when loading the model."
            "Default to `original/**/*` to avoid repeated loading of llama's "
            "checkpoints.")
        parser.add_argument(
            '--preemption-mode',
            type=str,
            default=None,
            help='If \'recompute\', the engine performs preemption by '
            'recomputing; If \'swap\', the engine performs preemption by '
            'block swapping.')

        parser.add_argument(
            "--served-model-name",
            nargs="+",
            type=str,
            default=None,
            help="The model name(s) used in the API. If multiple "
            "names are provided, the server will respond to any "
            "of the provided names. The model name in the model "
            "field of a response will be the first name in this "
            "list. If not specified, the model name will be the "
            "same as the ``--model`` argument. Noted that this name(s) "
            "will also be used in `model_name` tag content of "
            "prometheus metrics, if multiple names provided, metrics "
            "tag will take the first one.")
        parser.add_argument('--qlora-adapter-name-or-path',
                            type=str,
                            default=None,
                            help='Name or path of the QLoRA adapter.')

        parser.add_argument('--show-hidden-metrics-for-version',
                            type=str,
                            default=None,
                            help='Enable deprecated Prometheus metrics that '
                            'have been hidden since the specified version. '
                            'For example, if a previously deprecated metric '
                            'has been hidden since the v0.7.0 release, you '
                            'use --show-hidden-metrics-for-version=0.7 as a '
                            'temporary escape hatch while you migrate to new '
                            'metrics. The metric is likely to be removed '
                            'completely in an upcoming release.')

        parser.add_argument(
            '--otlp-traces-endpoint',
            type=str,
            default=None,
            help='Target URL to which OpenTelemetry traces will be sent.')
        parser.add_argument(
            '--collect-detailed-traces',
            type=str,
            default=None,
            help="Valid choices are " +
            ",".join(ALLOWED_DETAILED_TRACE_MODULES) +
            ". It makes sense to set this only if ``--otlp-traces-endpoint`` is"
            " set. If set, it will collect detailed traces for the specified "
            "modules. This involves use of possibly costly and or blocking "
            "operations and hence might have a performance impact.")

        parser.add_argument(
            '--disable-async-output-proc',
            action='store_true',
            default=EngineArgs.disable_async_output_proc,
            help="Disable async output processing. This may result in "
            "lower performance.")

        parser.add_argument(
            '--scheduling-policy',
            choices=['fcfs', 'priority'],
            default="fcfs",
            help='The scheduling policy to use. "fcfs" (first come first served'
            ', i.e. requests are handled in order of arrival; default) '
            'or "priority" (requests are handled based on given '
            'priority (lower value means earlier handling) and time of '
            'arrival deciding any ties).')

        parser.add_argument(
            '--scheduler-cls',
            default=EngineArgs.scheduler_cls,
            help='The scheduler class to use. "vllm.core.scheduler.Scheduler" '
            'is the default scheduler. Can be a class directly or the path to '
            'a class of form "mod.custom_class".')

        parser.add_argument(
            '--override-neuron-config',
            type=json.loads,
            default=None,
            help="Override or set neuron device configuration. "
            "e.g. ``{\"cast_logits_dtype\": \"bloat16\"}``.")
        parser.add_argument(
            '--override-pooler-config',
            type=PoolerConfig.from_json,
            default=None,
            help="Override or set the pooling method for pooling models. "
            "e.g. ``{\"pooling_type\": \"mean\", \"normalize\": false}``.")

        parser.add_argument('--compilation-config',
                            '-O',
                            type=CompilationConfig.from_cli,
                            default=None,
                            help='torch.compile configuration for the model.'
                            'When it is a number (0, 1, 2, 3), it will be '
                            'interpreted as the optimization level.\n'
                            'NOTE: level 0 is the default level without '
                            'any optimization. level 1 and 2 are for internal '
                            'testing only. level 3 is the recommended level '
                            'for production.\n'
                            'To specify the full compilation config, '
                            'use a JSON string.\n'
                            'Following the convention of traditional '
                            'compilers, using -O without space is also '
                            'supported. -O3 is equivalent to -O 3.')

        parser.add_argument('--kv-transfer-config',
                            type=KVTransferConfig.from_cli,
                            default=None,
                            help='The configurations for distributed KV cache '
                            'transfer. Should be a JSON string.')

        parser.add_argument(
            '--worker-cls',
            type=str,
            default="auto",
            help='The worker class to use for distributed execution.')
        parser.add_argument(
            "--generation-config",
            type=nullable_str,
            default=None,
            help="The folder path to the generation config. "
            "Defaults to None, no generation config is loaded, vLLM defaults "
            "will be used. If set to 'auto', the generation config will be "
            "loaded from model path. If set to a folder path, the generation "
            "config will be loaded from the specified folder path. If "
            "`max_new_tokens` is specified in generation config, then "
            "it sets a server-wide limit on the number of output tokens "
            "for all requests.")

        parser.add_argument(
            "--override-generation-config",
            type=json.loads,
            default=None,
            help="Overrides or sets generation config in JSON format. "
            "e.g. ``{\"temperature\": 0.5}``. If used with "
            "--generation-config=auto, the override parameters will be merged "
            "with the default config from the model. If generation-config is "
            "None, only the override parameters are used.")

        parser.add_argument("--enable-sleep-mode",
                            action="store_true",
                            default=False,
                            help="Enable sleep mode for the engine. "
                            "(only cuda platform is supported)")

        parser.add_argument(
            '--calculate-kv-scales',
            action='store_true',
            help='This enables dynamic calculation of '
            'k_scale and v_scale when kv-cache-dtype is fp8. '
            'If calculate-kv-scales is false, the scales will '
            'be loaded from the model checkpoint if available. '
            'Otherwise, the scales will default to 1.0.')

        parser.add_argument(
            "--additional-config",
            type=json.loads,
            default=None,
            help="Additional config for specified platform in JSON format. "
            "Different platforms may support different configs. Make sure the "
            "configs are valid for the platform you are using. The input format"
            " is like '{\"config_key\":\"config_value\"}'")
=======
        parser.add_argument('--disable-log-stats',
                            action='store_true',
                            help='Disable logging statistics.')

>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
        return parser

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace):
        # Get the list of attributes of this dataclass.
        attrs = [attr.name for attr in dataclasses.fields(cls)]
        # Set the attributes from the parsed arguments.
        engine_args = cls(**{attr: getattr(args, attr) for attr in attrs})
        return engine_args

    def create_model_config(self) -> ModelConfig:
<<<<<<< HEAD
        return ModelConfig(
            model=self.model,
            task=self.task,
            # We know this is not None because we set it in __post_init__
            tokenizer=cast(str, self.tokenizer),
=======
        # gguf file needs a specific model loader and doesn't use hf_repo
        if check_gguf_file(self.model):
            self.quantization = self.load_format = "gguf"

        # NOTE: This is to allow model loading from S3 in CI
        if (not isinstance(self, AsyncEngineArgs) and envs.VLLM_CI_USE_S3
                and self.model in MODELS_ON_S3
                and self.load_format == LoadFormat.AUTO):  # noqa: E501
            self.model = f"{MODEL_WEIGHTS_S3_BUCKET}/{self.model}"
            self.load_format = LoadFormat.RUNAI_STREAMER

        return ModelConfig(
            model=self.model,
            hf_config_path=self.hf_config_path,
            task=self.task,
            tokenizer=self.tokenizer,
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
            tokenizer_mode=self.tokenizer_mode,
            trust_remote_code=self.trust_remote_code,
            allowed_local_media_path=self.allowed_local_media_path,
            dtype=self.dtype,
            seed=self.seed,
            revision=self.revision,
            code_revision=self.code_revision,
            rope_scaling=self.rope_scaling,
            rope_theta=self.rope_theta,
<<<<<<< HEAD
=======
            hf_token=self.hf_token,
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
            hf_overrides=self.hf_overrides,
            tokenizer_revision=self.tokenizer_revision,
            max_model_len=self.max_model_len,
            quantization=self.quantization,
            enforce_eager=self.enforce_eager,
            max_seq_len_to_capture=self.max_seq_len_to_capture,
            max_logprobs=self.max_logprobs,
            disable_sliding_window=self.disable_sliding_window,
<<<<<<< HEAD
            skip_tokenizer_init=self.skip_tokenizer_init,
=======
            disable_cascade_attn=self.disable_cascade_attn,
            skip_tokenizer_init=self.skip_tokenizer_init,
            enable_prompt_embeds=self.enable_prompt_embeds,
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
            served_model_name=self.served_model_name,
            limit_mm_per_prompt=self.limit_mm_per_prompt,
            use_async_output_proc=not self.disable_async_output_proc,
            config_format=self.config_format,
            mm_processor_kwargs=self.mm_processor_kwargs,
            disable_mm_preprocessor_cache=self.disable_mm_preprocessor_cache,
            override_neuron_config=self.override_neuron_config,
            override_pooler_config=self.override_pooler_config,
            logits_processor_pattern=self.logits_processor_pattern,
            generation_config=self.generation_config,
            override_generation_config=self.override_generation_config,
            enable_sleep_mode=self.enable_sleep_mode,
            model_impl=self.model_impl,
        )

    def create_load_config(self) -> LoadConfig:
<<<<<<< HEAD
=======

        if self.quantization == "bitsandbytes":
            self.load_format = "bitsandbytes"

>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
        return LoadConfig(
            load_format=self.load_format,
            download_dir=self.download_dir,
            model_loader_extra_config=self.model_loader_extra_config,
            ignore_patterns=self.ignore_patterns,
<<<<<<< HEAD
        )

    def create_engine_config(self,
                             usage_context: Optional[UsageContext] = None
                             ) -> VllmConfig:
        from vllm.platforms import current_platform
        current_platform.pre_register_and_update()

        if envs.VLLM_USE_V1:
            self._override_v1_engine_args(usage_context)

        # gguf file needs a specific model loader and doesn't use hf_repo
        if check_gguf_file(self.model):
            self.quantization = self.load_format = "gguf"

        # bitsandbytes quantization needs a specific model loader
        # so we make sure the quant method and the load format are consistent
        if (self.quantization == "bitsandbytes" or
           self.qlora_adapter_name_or_path is not None) and \
           self.load_format != "bitsandbytes":
            raise ValueError(
                "BitsAndBytes quantization and QLoRA adapter only support "
                f"'bitsandbytes' load format, but got {self.load_format}")

        if (self.load_format == "bitsandbytes" or
            self.qlora_adapter_name_or_path is not None) and \
            self.quantization != "bitsandbytes":
            raise ValueError(
                "BitsAndBytes load format and QLoRA adapter only support "
                f"'bitsandbytes' quantization, but got {self.quantization}")

        assert self.cpu_offload_gb >= 0, (
            "CPU offload space must be non-negative"
            f", but got {self.cpu_offload_gb}")

        device_config = DeviceConfig(device=self.device)
        model_config = self.create_model_config()

        if (model_config.is_multimodal_model and not envs.VLLM_USE_V1
                and self.enable_prefix_caching):
            logger.warning("--enable-prefix-caching is currently not "
                           "supported for multimodal models in v0 and "
                           "has been disabled.")
            self.enable_prefix_caching = False
=======
            use_tqdm_on_load=self.use_tqdm_on_load,
            pt_load_map_location=self.pt_load_map_location,
        )

    def create_speculative_config(
        self,
        target_model_config: ModelConfig,
        target_parallel_config: ParallelConfig,
        enable_chunked_prefill: bool,
        disable_log_stats: bool,
    ) -> Optional["SpeculativeConfig"]:
        """Initializes and returns a SpeculativeConfig object based on
        `speculative_config`.

        This function utilizes `speculative_config` to create a
        SpeculativeConfig object. The `speculative_config` can either be
        provided as a JSON string input via CLI arguments or directly as a
        dictionary from the engine.
        """
        if self.speculative_config is None:
            return None

        # Note(Shangming): These parameters are not obtained from the cli arg
        # '--speculative-config' and must be passed in when creating the engine
        # config.
        self.speculative_config.update({
            "target_model_config": target_model_config,
            "target_parallel_config": target_parallel_config,
            "enable_chunked_prefill": enable_chunked_prefill,
            "disable_log_stats": disable_log_stats,
        })
        speculative_config = SpeculativeConfig.from_dict(
            self.speculative_config)

        return speculative_config

    def create_engine_config(
        self,
        usage_context: Optional[UsageContext] = None,
    ) -> VllmConfig:
        """
        Create the VllmConfig.

        NOTE: for autoselection of V0 vs V1 engine, we need to
        create the ModelConfig first, since ModelConfig's attrs
        (e.g. the model arch) are needed to make the decision.

        This function set VLLM_USE_V1=X if VLLM_USE_V1 is
        unspecified by the user.

        If VLLM_USE_V1 is specified by the user but the VllmConfig
        is incompatible, we raise an error.
        """
        from vllm.platforms import current_platform
        current_platform.pre_register_and_update()

        device_config = DeviceConfig(device=current_platform.device_type)
        model_config = self.create_model_config()

        # * If VLLM_USE_V1 is unset, we enable V1 for "supported features"
        #   and fall back to V0 for experimental or unsupported features.
        # * If VLLM_USE_V1=1, we enable V1 for supported + experimental
        #   features and raise error for unsupported features.
        # * If VLLM_USE_V1=0, we disable V1.
        use_v1 = False
        try_v1 = envs.VLLM_USE_V1 or not envs.is_set("VLLM_USE_V1")
        if try_v1 and self._is_v1_supported_oracle(model_config):
            use_v1 = True

        # If user explicitly set VLLM_USE_V1, sanity check we respect it.
        if envs.is_set("VLLM_USE_V1"):
            assert use_v1 == envs.VLLM_USE_V1
        # Otherwise, set the VLLM_USE_V1 variable globally.
        else:
            envs.set_vllm_use_v1(use_v1)

        # Set default arguments for V0 or V1 Engine.
        if use_v1:
            self._set_default_args_v1(usage_context)
        else:
            self._set_default_args_v0(model_config)

        assert self.enable_chunked_prefill is not None

        if envs.VLLM_ATTENTION_BACKEND in [STR_DUAL_CHUNK_FLASH_ATTN_VAL]:
            assert self.enforce_eager, (
                "Cuda graph is not supported with DualChunkFlashAttention. "
                "To run the model in eager mode, set 'enforce_eager=True' "
                "or use '--enforce-eager' in the CLI.")
            assert current_platform.is_cuda(), (
                "DualChunkFlashAttention is only supported on CUDA platform.")
            assert not use_v1, (
                "DualChunkFlashAttention is not supported on V1 engine. "
                "To run the model in V0 engine, try set 'VLLM_USE_V1=0'")
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea

        cache_config = CacheConfig(
            block_size=self.block_size,
            gpu_memory_utilization=self.gpu_memory_utilization,
            swap_space=self.swap_space,
            cache_dtype=self.kv_cache_dtype,
            is_attention_free=model_config.is_attention_free,
            num_gpu_blocks_override=self.num_gpu_blocks_override,
            sliding_window=model_config.get_sliding_window(),
            enable_prefix_caching=self.enable_prefix_caching,
<<<<<<< HEAD
            cpu_offload_gb=self.cpu_offload_gb,
            calculate_kv_scales=self.calculate_kv_scales,
        )
        parallel_config = ParallelConfig(
            pipeline_parallel_size=self.pipeline_parallel_size,
            tensor_parallel_size=self.tensor_parallel_size,
            max_parallel_loading_workers=self.max_parallel_loading_workers,
            disable_custom_all_reduce=self.disable_custom_all_reduce,
            tokenizer_pool_config=TokenizerPoolConfig.create_config(
                self.tokenizer_pool_size,
                self.tokenizer_pool_type,
                self.tokenizer_pool_extra_config,
            ),
            ray_workers_use_nsight=self.ray_workers_use_nsight,
            distributed_executor_backend=self.distributed_executor_backend,
            worker_cls=self.worker_cls,
        )

        max_model_len = model_config.max_model_len
        use_long_context = max_model_len > 32768
        if self.enable_chunked_prefill is None:
            # If not explicitly set, enable chunked prefill by default for
            # long context (> 32K) models. This is to avoid OOM errors in the
            # initial memory profiling phase.

            # For multimodal models and models with MLA, chunked prefill is
            # disabled by default in V0, but enabled by design in V1
            if model_config.is_multimodal_model or model_config.use_mla:
                self.enable_chunked_prefill = bool(envs.VLLM_USE_V1)

            elif use_long_context:
                is_gpu = device_config.device_type == "cuda"
                use_sliding_window = (model_config.get_sliding_window()
                                      is not None)
                use_spec_decode = self.speculative_model is not None
                from vllm.platforms import current_platform
                if (is_gpu and not use_sliding_window and not use_spec_decode
                        and not self.enable_lora
                        and not self.enable_prompt_adapter
                        and model_config.runner_type != "pooling"
                        and not current_platform.is_rocm()):
                    self.enable_chunked_prefill = True
                    logger.warning(
                        "Chunked prefill is enabled by default for models with "
                        "max_model_len > 32K. Currently, chunked prefill might "
                        "not work with some features or models. If you "
                        "encounter any issues, please disable chunked prefill "
                        "by setting --enable-chunked-prefill=False.")
            if self.enable_chunked_prefill is None:
                self.enable_chunked_prefill = False

        if not self.enable_chunked_prefill and use_long_context:
            logger.warning(
                "The model has a long context length (%s). This may cause OOM "
                "errors during the initial memory profiling phase, or result "
                "in low performance due to small KV cache space. Consider "
                "setting --max-model-len to a smaller value.", max_model_len)
        elif (self.enable_chunked_prefill
              and model_config.runner_type == "pooling"):
            msg = "Chunked prefill is not supported for pooling models"
            raise ValueError(msg)

        speculative_config = SpeculativeConfig.maybe_create_spec_config(
            target_model_config=model_config,
            target_parallel_config=parallel_config,
            target_dtype=self.dtype,
            speculative_model=self.speculative_model,
            speculative_model_quantization = \
                self.speculative_model_quantization,
            speculative_draft_tensor_parallel_size = \
                self.speculative_draft_tensor_parallel_size,
            num_speculative_tokens=self.num_speculative_tokens,
            speculative_disable_mqa_scorer=self.speculative_disable_mqa_scorer,
            speculative_disable_by_batch_size=self.
            speculative_disable_by_batch_size,
            speculative_max_model_len=self.speculative_max_model_len,
            enable_chunked_prefill=self.enable_chunked_prefill,
            disable_log_stats=self.disable_log_stats,
            ngram_prompt_lookup_max=self.ngram_prompt_lookup_max,
            ngram_prompt_lookup_min=self.ngram_prompt_lookup_min,
            draft_token_acceptance_method=\
                self.spec_decoding_acceptance_method,
            typical_acceptance_sampler_posterior_threshold=self.
            typical_acceptance_sampler_posterior_threshold,
            typical_acceptance_sampler_posterior_alpha=self.
            typical_acceptance_sampler_posterior_alpha,
            disable_logprobs=self.disable_logprobs_during_spec_decoding,
=======
            prefix_caching_hash_algo=self.prefix_caching_hash_algo,
            cpu_offload_gb=self.cpu_offload_gb,
            calculate_kv_scales=self.calculate_kv_scales,
        )

        # Get the current placement group if Ray is initialized and
        # we are in a Ray actor. If so, then the placement group will be
        # passed to spawned processes.
        placement_group = None
        if is_in_ray_actor():
            import ray

            # This call initializes Ray automatically if it is not initialized,
            # but we should not do this here.
            placement_group = ray.util.get_current_placement_group()

        # Local DP size defaults to global DP size if not set.
        data_parallel_size_local = self.data_parallel_size if (
            self.data_parallel_size_local
            is None) else self.data_parallel_size_local

        # DP address, used in multi-node case for torch distributed group
        # and ZMQ sockets.
        data_parallel_address = self.data_parallel_address if (
            self.data_parallel_address
            is not None) else ParallelConfig.data_parallel_master_ip

        # This port is only used when there are remote data parallel engines,
        # otherwise the local IPC transport is used.
        data_parallel_rpc_port = self.data_parallel_rpc_port if (
            self.data_parallel_rpc_port
            is not None) else ParallelConfig.data_parallel_rpc_port

        parallel_config = ParallelConfig(
            pipeline_parallel_size=self.pipeline_parallel_size,
            tensor_parallel_size=self.tensor_parallel_size,
            data_parallel_size=self.data_parallel_size,
            data_parallel_size_local=data_parallel_size_local,
            data_parallel_master_ip=data_parallel_address,
            data_parallel_rpc_port=data_parallel_rpc_port,
            enable_expert_parallel=self.enable_expert_parallel,
            max_parallel_loading_workers=self.max_parallel_loading_workers,
            disable_custom_all_reduce=self.disable_custom_all_reduce,
            ray_workers_use_nsight=self.ray_workers_use_nsight,
            placement_group=placement_group,
            distributed_executor_backend=self.distributed_executor_backend,
            worker_cls=self.worker_cls,
            worker_extension_cls=self.worker_extension_cls,
        )

        speculative_config = self.create_speculative_config(
            target_model_config=model_config,
            target_parallel_config=parallel_config,
            enable_chunked_prefill=self.enable_chunked_prefill,
            disable_log_stats=self.disable_log_stats,
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
        )

        # Reminder: Please update docs/source/features/compatibility_matrix.md
        # If the feature combo become valid
        if self.num_scheduler_steps > 1:
            if speculative_config is not None:
                raise ValueError("Speculative decoding is not supported with "
                                 "multi-step (--num-scheduler-steps > 1)")
            if self.enable_chunked_prefill and self.pipeline_parallel_size > 1:
                raise ValueError("Multi-Step Chunked-Prefill is not supported "
                                 "for pipeline-parallel-size > 1")
            from vllm.platforms import current_platform
            if current_platform.is_cpu():
                logger.warning("Multi-Step (--num-scheduler-steps > 1) is "
                               "currently not supported for CPUs and has been "
                               "disabled.")
                self.num_scheduler_steps = 1

        # make sure num_lookahead_slots is set the higher value depending on
        # if we are using speculative decoding or multi-step
        num_lookahead_slots = max(self.num_lookahead_slots,
                                  self.num_scheduler_steps - 1)
        num_lookahead_slots = num_lookahead_slots \
            if speculative_config is None \
            else speculative_config.num_lookahead_slots

<<<<<<< HEAD
        if not self.use_v2_block_manager:
            logger.warning(
                "[DEPRECATED] Block manager v1 has been removed, "
                "and setting --use-v2-block-manager to True or False has "
                "no effect on vLLM behavior. Please remove "
                "--use-v2-block-manager in your engine argument. "
                "If your use case is not supported by "
                "SelfAttnBlockSpaceManager (i.e. block manager v2),"
                " please file an issue with detailed information.")

=======
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
        scheduler_config = SchedulerConfig(
            runner_type=model_config.runner_type,
            max_num_batched_tokens=self.max_num_batched_tokens,
            max_num_seqs=self.max_num_seqs,
            max_model_len=model_config.max_model_len,
<<<<<<< HEAD
            num_lookahead_slots=num_lookahead_slots,
            delay_factor=self.scheduler_delay_factor,
            enable_chunked_prefill=self.enable_chunked_prefill,
=======
            cuda_graph_sizes=self.cuda_graph_sizes,
            num_lookahead_slots=num_lookahead_slots,
            delay_factor=self.scheduler_delay_factor,
            enable_chunked_prefill=self.enable_chunked_prefill,
            disable_chunked_mm_input=self.disable_chunked_mm_input,
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
            is_multimodal_model=model_config.is_multimodal_model,
            preemption_mode=self.preemption_mode,
            num_scheduler_steps=self.num_scheduler_steps,
            multi_step_stream_outputs=self.multi_step_stream_outputs,
            send_delta_data=(envs.VLLM_USE_RAY_SPMD_WORKER
                             and parallel_config.use_ray),
            policy=self.scheduling_policy,
            scheduler_cls=self.scheduler_cls,
            max_num_partial_prefills=self.max_num_partial_prefills,
            max_long_partial_prefills=self.max_long_partial_prefills,
            long_prefill_token_threshold=self.long_prefill_token_threshold,
        )

        lora_config = LoRAConfig(
            bias_enabled=self.enable_lora_bias,
            max_lora_rank=self.max_lora_rank,
            max_loras=self.max_loras,
            fully_sharded_loras=self.fully_sharded_loras,
            lora_extra_vocab_size=self.lora_extra_vocab_size,
            long_lora_scaling_factors=self.long_lora_scaling_factors,
            lora_dtype=self.lora_dtype,
            max_cpu_loras=self.max_cpu_loras if self.max_cpu_loras
            and self.max_cpu_loras > 0 else None) if self.enable_lora else None

<<<<<<< HEAD
        if self.qlora_adapter_name_or_path is not None and \
            self.qlora_adapter_name_or_path != "":
            if self.model_loader_extra_config is None:
                self.model_loader_extra_config = {}
            self.model_loader_extra_config[
                "qlora_adapter_name_or_path"] = self.qlora_adapter_name_or_path
=======
        # bitsandbytes pre-quantized model need a specific model loader
        if model_config.quantization == "bitsandbytes":
            self.quantization = self.load_format = "bitsandbytes"
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea

        load_config = self.create_load_config()

        prompt_adapter_config = PromptAdapterConfig(
            max_prompt_adapters=self.max_prompt_adapters,
            max_prompt_adapter_token=self.max_prompt_adapter_token) \
                                        if self.enable_prompt_adapter else None

        decoding_config = DecodingConfig(
<<<<<<< HEAD
            guided_decoding_backend=self.guided_decoding_backend)

        show_hidden_metrics = False
        if self.show_hidden_metrics_for_version is not None:
            show_hidden_metrics = version._prev_minor_version_was(
                self.show_hidden_metrics_for_version)

        detailed_trace_modules = []
        if self.collect_detailed_traces is not None:
            detailed_trace_modules = self.collect_detailed_traces.split(",")
        for m in detailed_trace_modules:
            if m not in ALLOWED_DETAILED_TRACE_MODULES:
                raise ValueError(
                    f"Invalid module {m} in collect_detailed_traces. "
                    f"Valid modules are {ALLOWED_DETAILED_TRACE_MODULES}")
        observability_config = ObservabilityConfig(
            show_hidden_metrics=show_hidden_metrics,
            otlp_traces_endpoint=self.otlp_traces_endpoint,
            collect_model_forward_time="model" in detailed_trace_modules
            or "all" in detailed_trace_modules,
            collect_model_execute_time="worker" in detailed_trace_modules
            or "all" in detailed_trace_modules,
=======
            backend=self.guided_decoding_backend,
            disable_fallback=self.guided_decoding_disable_fallback,
            disable_any_whitespace=self.guided_decoding_disable_any_whitespace,
            disable_additional_properties=\
                self.guided_decoding_disable_additional_properties,
            reasoning_backend=self.reasoning_parser
        )

        observability_config = ObservabilityConfig(
            show_hidden_metrics_for_version=self.
            show_hidden_metrics_for_version,
            otlp_traces_endpoint=self.otlp_traces_endpoint,
            collect_detailed_traces=self.collect_detailed_traces,
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
        )

        config = VllmConfig(
            model_config=model_config,
            cache_config=cache_config,
            parallel_config=parallel_config,
            scheduler_config=scheduler_config,
            device_config=device_config,
            lora_config=lora_config,
            speculative_config=speculative_config,
            load_config=load_config,
            decoding_config=decoding_config,
            observability_config=observability_config,
            prompt_adapter_config=prompt_adapter_config,
            compilation_config=self.compilation_config,
            kv_transfer_config=self.kv_transfer_config,
<<<<<<< HEAD
            additional_config=self.additional_config,
        )

        if envs.VLLM_USE_V1:
            self._override_v1_engine_config(config)
        return config

    def _override_v1_engine_args(self, usage_context: UsageContext) -> None:
        """
        Override the EngineArgs's args based on the usage context for V1.
        """
        assert envs.VLLM_USE_V1, "V1 is not enabled"

        # V1 always uses chunked prefills.
        self.enable_chunked_prefill = True
        # When no user override, set the default values based on the usage
        # context.
        # Use different default values for different hardware.
        from vllm.platforms import current_platform
        device_name = current_platform.get_device_name().lower()
        if "h100" in device_name or "h200" in device_name:
            # For H100 and H200, we use larger default values.
=======
            kv_events_config=self.kv_events_config,
            additional_config=self.additional_config,
        )

        return config

    def _is_v1_supported_oracle(self, model_config: ModelConfig) -> bool:
        """Oracle for whether to use V0 or V1 Engine by default."""

        #############################################################
        # Unsupported Feature Flags on V1.

        if (self.load_format == LoadFormat.TENSORIZER.value
                or self.load_format == LoadFormat.SHARDED_STATE.value):
            _raise_or_fallback(
                feature_name=f"--load_format {self.load_format}",
                recommend_to_remove=False)
            return False

        if (self.logits_processor_pattern
                != EngineArgs.logits_processor_pattern):
            _raise_or_fallback(feature_name="--logits-processor-pattern",
                               recommend_to_remove=False)
            return False

        if self.preemption_mode != SchedulerConfig.preemption_mode:
            _raise_or_fallback(feature_name="--preemption-mode",
                               recommend_to_remove=True)
            return False

        if (self.disable_async_output_proc
                != EngineArgs.disable_async_output_proc):
            _raise_or_fallback(feature_name="--disable-async-output-proc",
                               recommend_to_remove=True)
            return False

        if self.scheduling_policy != SchedulerConfig.policy:
            _raise_or_fallback(feature_name="--scheduling-policy",
                               recommend_to_remove=False)
            return False

        if self.num_scheduler_steps != SchedulerConfig.num_scheduler_steps:
            _raise_or_fallback(feature_name="--num-scheduler-steps",
                               recommend_to_remove=True)
            return False

        if self.scheduler_delay_factor != SchedulerConfig.delay_factor:
            _raise_or_fallback(feature_name="--scheduler-delay-factor",
                               recommend_to_remove=True)
            return False

        if self.guided_decoding_backend not in get_args(
                GuidedDecodingBackendV1):
            _raise_or_fallback(
                feature_name=
                f"--guided-decoding-backend={self.guided_decoding_backend}",
                recommend_to_remove=False)
            return False

        # Need at least Ampere for now (FA support required).
        # Skip this check if we are running on a non-GPU platform,
        # or if the device capability is not available
        # (e.g. in a Ray actor without GPUs).
        from vllm.platforms import current_platform
        if (current_platform.is_cuda()
                and current_platform.get_device_capability()
                and current_platform.get_device_capability().major < 8):
            _raise_or_fallback(feature_name="Compute Capability < 8.0",
                               recommend_to_remove=False)
            return False

        # No Fp8 KV cache so far.
        if self.kv_cache_dtype != "auto":
            fp8_attention = self.kv_cache_dtype.startswith("fp8")
            will_use_fa = (
                current_platform.is_cuda()
                and not envs.is_set("VLLM_ATTENTION_BACKEND")
            ) or envs.VLLM_ATTENTION_BACKEND == "FLASH_ATTN_VLLM_V1"
            supported = False
            if current_platform.is_rocm():
                supported = True
            elif fp8_attention and will_use_fa:
                from vllm.attention.utils.fa_utils import (
                    flash_attn_supports_fp8)
                supported = flash_attn_supports_fp8()
            if not supported:
                _raise_or_fallback(feature_name="--kv-cache-dtype",
                                   recommend_to_remove=False)
                return False

        # No Prompt Adapter so far.
        if self.enable_prompt_adapter:
            _raise_or_fallback(feature_name="--enable-prompt-adapter",
                               recommend_to_remove=False)
            return False

        # No text embedding inputs so far.
        if self.enable_prompt_embeds:
            _raise_or_fallback(feature_name="--enable-prompt-embeds",
                               recommend_to_remove=False)
            return False

        # Only Fp16 and Bf16 dtypes since we only support FA.
        V1_SUPPORTED_DTYPES = [torch.bfloat16, torch.float16]
        if model_config.dtype not in V1_SUPPORTED_DTYPES:
            _raise_or_fallback(feature_name=f"--dtype {model_config.dtype}",
                               recommend_to_remove=False)
            return False

        # Some quantization is not compatible with torch.compile.
        V1_UNSUPPORTED_QUANT = ["gguf"]
        if model_config.quantization in V1_UNSUPPORTED_QUANT:
            _raise_or_fallback(
                feature_name=f"--quantization {model_config.quantization}",
                recommend_to_remove=False)
            return False

        # No Embedding Models so far.
        if model_config.task not in ["generate"]:
            _raise_or_fallback(feature_name=f"--task {model_config.task}",
                               recommend_to_remove=False)
            return False

        # No Mamba or Encoder-Decoder so far.
        if not model_config.is_v1_compatible:
            _raise_or_fallback(feature_name=model_config.architectures,
                               recommend_to_remove=False)
            return False

        # No Concurrent Partial Prefills so far.
        if (self.max_num_partial_prefills
                != SchedulerConfig.max_num_partial_prefills
                or self.max_long_partial_prefills
                != SchedulerConfig.max_long_partial_prefills):
            _raise_or_fallback(feature_name="Concurrent Partial Prefill",
                               recommend_to_remove=False)
            return False

        # No OTLP observability so far.
        if (self.otlp_traces_endpoint or self.collect_detailed_traces):
            _raise_or_fallback(feature_name="--otlp-traces-endpoint",
                               recommend_to_remove=False)
            return False

        # V1 supports N-gram, Medusa, and Eagle speculative decoding.
        is_ngram_enabled = False
        is_eagle_enabled = False
        is_medusa_enabled = False
        if self.speculative_config is not None:
            # This is supported but experimental (handled below).
            speculative_method = self.speculative_config.get("method")
            if speculative_method:
                if speculative_method in ("ngram", "[ngram]"):
                    is_ngram_enabled = True
                elif speculative_method == "medusa":
                    is_medusa_enabled = True
                elif speculative_method in ("eagle", "eagle3"):
                    is_eagle_enabled = True
            else:
                speculative_model = self.speculative_config.get("model")
                if speculative_model in ("ngram", "[ngram]"):
                    is_ngram_enabled = True
            if not (is_ngram_enabled or is_eagle_enabled or is_medusa_enabled):
                # Other speculative decoding methods are not supported yet.
                _raise_or_fallback(feature_name="Speculative Decoding",
                                   recommend_to_remove=False)
                return False

        # No XFormers so far.
        V1_BACKENDS = [
            "FLASH_ATTN_VLLM_V1",
            "FLASH_ATTN",
            "PALLAS",
            "PALLAS_VLLM_V1",
            "TRITON_ATTN_VLLM_V1",
            "TRITON_MLA",
            "FLASHMLA",
            "FLASHINFER",
            "FLASHINFER_VLLM_V1",
            "ROCM_AITER_MLA",
        ]
        if (envs.is_set("VLLM_ATTENTION_BACKEND")
                and envs.VLLM_ATTENTION_BACKEND not in V1_BACKENDS):
            name = f"VLLM_ATTENTION_BACKEND={envs.VLLM_ATTENTION_BACKEND}"
            _raise_or_fallback(feature_name=name, recommend_to_remove=True)
            return False

        # Platforms must decide if they can support v1 for this model
        if not current_platform.supports_v1(model_config=model_config):
            _raise_or_fallback(
                feature_name=f"device type={current_platform.device_type}",
                recommend_to_remove=False)
            return False
        #############################################################
        # Experimental Features - allow users to opt in.

        # Signal Handlers requires running in main thread.
        if (threading.current_thread() != threading.main_thread()
                and _warn_or_fallback("Engine in background thread")):
            return False

        if (self.pipeline_parallel_size > 1
                and self.distributed_executor_backend
                not in ("ray", "mp", "external_launcher")):
            name = "Pipeline Parallelism without Ray distributed executor " \
                    "or multiprocessing executor or external launcher"
            _raise_or_fallback(feature_name=name, recommend_to_remove=False)
            return False

        # Non-[CUDA, TPU] may be supported on V1, but off by default for now.
        v0_hardware = not any(
            (current_platform.is_cuda(), current_platform.is_tpu()))
        if v0_hardware and _warn_or_fallback(  # noqa: SIM103
                current_platform.device_name):
            return False
        #############################################################

        return True

    def _set_default_args_v0(self, model_config: ModelConfig) -> None:
        """Set Default Arguments for V0 Engine."""

        max_model_len = model_config.max_model_len
        use_long_context = max_model_len > 32768
        if self.enable_chunked_prefill is None:
            # Chunked prefill not supported for Multimodal or MLA in V0.
            if model_config.is_multimodal_model or model_config.use_mla:
                self.enable_chunked_prefill = False

            # Enable chunked prefill by default for long context (> 32K)
            # models to avoid OOM errors in initial memory profiling phase.
            elif use_long_context:
                from vllm.platforms import current_platform
                is_gpu = current_platform.is_cuda()
                use_sliding_window = (model_config.get_sliding_window()
                                      is not None)
                use_spec_decode = self.speculative_config is not None

                if (is_gpu and not use_sliding_window and not use_spec_decode
                        and not self.enable_lora
                        and not self.enable_prompt_adapter
                        and model_config.runner_type != "pooling"):
                    self.enable_chunked_prefill = True
                    logger.warning(
                        "Chunked prefill is enabled by default for models "
                        "with max_model_len > 32K. Chunked prefill might "
                        "not work with some features or models. If you "
                        "encounter any issues, please disable by launching "
                        "with --enable-chunked-prefill=False.")

            if self.enable_chunked_prefill is None:
                self.enable_chunked_prefill = False

        if not self.enable_chunked_prefill and use_long_context:
            logger.warning(
                "The model has a long context length (%s). This may cause"
                "OOM during the initial memory profiling phase, or result "
                "in low performance due to small KV cache size. Consider "
                "setting --max-model-len to a smaller value.", max_model_len)
        elif (self.enable_chunked_prefill
              and model_config.runner_type == "pooling"):
            msg = "Chunked prefill is not supported for pooling models"
            raise ValueError(msg)

        # if using prefix caching, we must set a hash algo
        if self.enable_prefix_caching:
            # Disable prefix caching for multimodal models for VLLM_V0.
            if model_config.is_multimodal_model:
                logger.warning(
                    "--enable-prefix-caching is not supported for multimodal "
                    "models in V0 and has been disabled.")
                self.enable_prefix_caching = False

            # VLLM_V0 only supports builtin hash algo for prefix caching.
            if self.prefix_caching_hash_algo == "sha256":
                raise ValueError(
                    "sha256 is not supported for prefix caching in V0 engine. "
                    "Please use 'builtin'.")

        # Set max_num_seqs to 256 for VLLM_V0.
        if self.max_num_seqs is None:
            self.max_num_seqs = 256

    def _set_default_args_v1(self, usage_context: UsageContext) -> None:
        """Set Default Arguments for V1 Engine."""

        # V1 always uses chunked prefills.
        self.enable_chunked_prefill = True

        # V1 enables prefix caching by default.
        if self.enable_prefix_caching is None:
            self.enable_prefix_caching = True

        # V1 should use the new scheduler by default.
        # Swap it only if this arg is set to the original V0 default
        if self.scheduler_cls == EngineArgs.scheduler_cls:
            self.scheduler_cls = "vllm.v1.core.sched.scheduler.Scheduler"

        # When no user override, set the default values based on the usage
        # context.
        # Use different default values for different hardware.

        # Try to query the device name on the current platform. If it fails,
        # it may be because the platform that imports vLLM is not the same
        # as the platform that vLLM is running on (e.g. the case of scaling
        # vLLM with Ray) and has no GPUs. In this case we use the default
        # values for non-H100/H200 GPUs.
        from vllm.platforms import current_platform
        try:
            device_memory = current_platform.get_device_total_memory()
            device_name = current_platform.get_device_name().lower()
        except Exception:
            # This is only used to set default_max_num_batched_tokens
            device_memory = 0

        # NOTE(Kuntai): Setting large `max_num_batched_tokens` for A100 reduces
        # throughput, see PR #17885 for more details.
        # So here we do an extra device name check to prevent such regression.
        if device_memory >= 70 * GiB_bytes and "a100" not in device_name:
            # For GPUs like H100 and MI300x, use larger default values.
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
            default_max_num_batched_tokens = {
                UsageContext.LLM_CLASS: 16384,
                UsageContext.OPENAI_API_SERVER: 8192,
            }
<<<<<<< HEAD
=======
            default_max_num_seqs = 1024
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
        else:
            # TODO(woosuk): Tune the default values for other hardware.
            default_max_num_batched_tokens = {
                UsageContext.LLM_CLASS: 8192,
                UsageContext.OPENAI_API_SERVER: 2048,
            }
<<<<<<< HEAD

        if (self.max_num_batched_tokens is None
                and usage_context in default_max_num_batched_tokens):
            self.max_num_batched_tokens = default_max_num_batched_tokens[
                usage_context]
            logger.warning(
                "Setting max_num_batched_tokens to %d for %s usage context.",
                self.max_num_batched_tokens, usage_context.value)

    def _override_v1_engine_config(self, engine_config: VllmConfig) -> None:
        """
        Override the EngineConfig's configs based on the usage context for V1.
        """
        assert envs.VLLM_USE_V1, "V1 is not enabled"
=======
            default_max_num_seqs = 256

        # tpu specific default values.
        if current_platform.is_tpu():
            default_max_num_batched_tokens_tpu = {
                UsageContext.LLM_CLASS: {
                    'V6E': 2048,
                    'V5E': 1024,
                    'V5P': 512,
                },
                UsageContext.OPENAI_API_SERVER: {
                    'V6E': 1024,
                    'V5E': 512,
                    'V5P': 256,
                }
            }

        use_context_value = usage_context.value if usage_context else None
        if (self.max_num_batched_tokens is None
                and usage_context in default_max_num_batched_tokens):
            if current_platform.is_tpu():
                chip_name = current_platform.get_device_name()
                if chip_name in default_max_num_batched_tokens_tpu[
                        usage_context]:
                    self.max_num_batched_tokens = \
                        default_max_num_batched_tokens_tpu[
                            usage_context][chip_name]
                else:
                    self.max_num_batched_tokens = \
                        default_max_num_batched_tokens[usage_context]
            else:
                self.max_num_batched_tokens = default_max_num_batched_tokens[
                    usage_context]
            logger.debug(
                "Setting max_num_batched_tokens to %d for %s usage context.",
                self.max_num_batched_tokens, use_context_value)

        if self.max_num_seqs is None:
            self.max_num_seqs = default_max_num_seqs

            logger.debug("Setting max_num_seqs to %d for %s usage context.",
                         self.max_num_seqs, use_context_value)
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea


@dataclass
class AsyncEngineArgs(EngineArgs):
    """Arguments for asynchronous vLLM engine."""
    disable_log_requests: bool = False

    @staticmethod
    def add_cli_args(parser: FlexibleArgumentParser,
                     async_args_only: bool = False) -> FlexibleArgumentParser:
<<<<<<< HEAD
=======
        # Initialize plugin to update the parser, for example, The plugin may
        # adding a new kind of quantization method to --quantization argument or
        # a new device to --device argument.
        load_general_plugins()
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
        if not async_args_only:
            parser = EngineArgs.add_cli_args(parser)
        parser.add_argument('--disable-log-requests',
                            action='store_true',
                            help='Disable logging requests.')
<<<<<<< HEAD
        # Initialize plugin to update the parser, for example, The plugin may
        # adding a new kind of quantization method to --quantization argument or
        # a new device to --device argument.
        load_general_plugins()
=======
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
        from vllm.platforms import current_platform
        current_platform.pre_register_and_update(parser)
        return parser


<<<<<<< HEAD
=======
def _raise_or_fallback(feature_name: str, recommend_to_remove: bool):
    if envs.is_set("VLLM_USE_V1") and envs.VLLM_USE_V1:
        raise NotImplementedError(
            f"VLLM_USE_V1=1 is not supported with {feature_name}.")
    msg = f"{feature_name} is not supported by the V1 Engine. "
    msg += "Falling back to V0. "
    if recommend_to_remove:
        msg += f"We recommend to remove {feature_name} from your config "
        msg += "in favor of the V1 Engine."
    logger.warning(msg)


def _warn_or_fallback(feature_name: str) -> bool:
    if envs.is_set("VLLM_USE_V1") and envs.VLLM_USE_V1:
        logger.warning(
            "Detected VLLM_USE_V1=1 with %s. Usage should "
            "be considered experimental. Please report any "
            "issues on Github.", feature_name)
        should_exit = False
    else:
        logger.info(
            "%s is experimental on VLLM_USE_V1=1. "
            "Falling back to V0 Engine.", feature_name)
        should_exit = True
    return should_exit


def human_readable_int(value):
    """Parse human-readable integers like '1k', '2M', etc.
    Including decimal values with decimal multipliers.

    Examples:
    - '1k' -> 1,000
    - '1K' -> 1,024
    - '25.6k' -> 25,600
    """
    value = value.strip()
    match = re.fullmatch(r'(\d+(?:\.\d+)?)([kKmMgGtT])', value)
    if match:
        decimal_multiplier = {
            'k': 10**3,
            'm': 10**6,
            'g': 10**9,
        }
        binary_multiplier = {
            'K': 2**10,
            'M': 2**20,
            'G': 2**30,
        }

        number, suffix = match.groups()
        if suffix in decimal_multiplier:
            mult = decimal_multiplier[suffix]
            return int(float(number) * mult)
        elif suffix in binary_multiplier:
            mult = binary_multiplier[suffix]
            # Do not allow decimals with binary multipliers
            try:
                return int(number) * mult
            except ValueError as e:
                raise argparse.ArgumentTypeError("Decimals are not allowed " \
                f"with binary suffixes like {suffix}. Did you mean to use " \
                f"{number}{suffix.lower()} instead?") from e

    # Regular plain number.
    return int(value)


>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
# These functions are used by sphinx to build the documentation
def _engine_args_parser():
    return EngineArgs.add_cli_args(FlexibleArgumentParser())


def _async_engine_args_parser():
    return AsyncEngineArgs.add_cli_args(FlexibleArgumentParser(),
                                        async_args_only=True)
