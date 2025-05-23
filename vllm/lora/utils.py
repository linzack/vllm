# SPDX-License-Identifier: Apache-2.0

import os
import re
<<<<<<< HEAD
from typing import List, Optional, Set, Tuple, Type, Union
=======
from typing import Optional, Union
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea

import huggingface_hub
from huggingface_hub.utils import (EntryNotFoundError, HfHubHTTPError,
                                   HFValidationError, RepositoryNotFoundError)
from torch import nn
from transformers import PretrainedConfig

from vllm.config import LoRAConfig
from vllm.logger import init_logger
from vllm.lora.fully_sharded_layers import (
    ColumnParallelLinearWithShardedLoRA,
    MergedColumnParallelLinearWithShardedLoRA,
<<<<<<< HEAD
    MergedQKVParallelLinearWithShardedLora, QKVParallelLinearWithShardedLora,
=======
    MergedQKVParallelLinearWithShardedLoRA, QKVParallelLinearWithShardedLoRA,
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
    RowParallelLinearWithShardedLoRA)
# being imported for _all_lora_classes below
# yapf conflicts with isort for this block
# yapf: disable
from vllm.lora.layers import (BaseLayerWithLoRA, ColumnParallelLinearWithLoRA,
<<<<<<< HEAD
                              LinearScalingRotaryEmbeddingWithLora,
                              LogitsProcessorWithLoRA,
                              MergedColumnParallelLinearWithLoRA,
                              MergedQKVParallelLinearWithLora,
                              QKVParallelLinearWithLora,
=======
                              LinearScalingRotaryEmbeddingWithLoRA,
                              LogitsProcessorWithLoRA,
                              MergedColumnParallelLinearWithLoRA,
                              MergedQKVParallelLinearWithLoRA,
                              QKVParallelLinearWithLoRA,
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
                              ReplicatedLinearWithLoRA,
                              RowParallelLinearWithLoRA,
                              VocabParallelEmbeddingWithLoRA)
from vllm.model_executor.layers.linear import LinearBase
# yapf: enable
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.models.utils import WeightsMapper

logger = init_logger(__name__)

<<<<<<< HEAD
_all_lora_classes: Set[Type[BaseLayerWithLoRA]] = {
    VocabParallelEmbeddingWithLoRA,
    ColumnParallelLinearWithLoRA,
    MergedColumnParallelLinearWithLoRA,
    QKVParallelLinearWithLora,
    MergedQKVParallelLinearWithLora,
=======
_all_lora_classes: set[type[BaseLayerWithLoRA]] = {
    VocabParallelEmbeddingWithLoRA,
    ColumnParallelLinearWithLoRA,
    MergedColumnParallelLinearWithLoRA,
    QKVParallelLinearWithLoRA,
    MergedQKVParallelLinearWithLoRA,
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
    RowParallelLinearWithLoRA,
    ReplicatedLinearWithLoRA,
    LogitsProcessorWithLoRA,
    ColumnParallelLinearWithShardedLoRA,
<<<<<<< HEAD
    QKVParallelLinearWithShardedLora,
    MergedColumnParallelLinearWithShardedLoRA,
    MergedQKVParallelLinearWithShardedLora,
    RowParallelLinearWithShardedLoRA,
    LinearScalingRotaryEmbeddingWithLora,
=======
    QKVParallelLinearWithShardedLoRA,
    MergedColumnParallelLinearWithShardedLoRA,
    MergedQKVParallelLinearWithShardedLoRA,
    RowParallelLinearWithShardedLoRA,
    LinearScalingRotaryEmbeddingWithLoRA,
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
}


def from_layer(layer: nn.Module,
               max_loras: int,
               lora_config: LoRAConfig,
<<<<<<< HEAD
               packed_modules_list: List,
=======
               packed_modules_list: list,
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
               model_config: Optional[PretrainedConfig] = None) -> nn.Module:
    for lora_cls in _all_lora_classes:
        # specifying kwargs so they can be easily accessed in decorator
        if lora_cls.can_replace_layer(source_layer=layer,
                                      lora_config=lora_config,
                                      packed_modules_list=packed_modules_list,
                                      model_config=model_config):
<<<<<<< HEAD
            ret = lora_cls(layer)
            ret.create_lora_weights(max_loras, lora_config, model_config)
            return ret

    # The Case for HFCompatibleLinear
    if (hasattr(layer, "get_lora_class")
            and layer.__class__.__name__ == "HFCompatibleLinear"):
        lora_cls = layer.get_lora_class(lora_config.fully_sharded_loras)
        ret = lora_cls(layer)
        ret.create_lora_weights(max_loras, lora_config, model_config)
        return ret
=======
            instance_layer = lora_cls(layer)
            instance_layer.create_lora_weights(max_loras, lora_config,
                                               model_config)
            return instance_layer
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
    return layer


def from_layer_logits_processor(
    layer: LogitsProcessor,
    lm_head: ParallelLMHead,
    max_loras: int,
    lora_config: LoRAConfig,
    model_config: Optional[PretrainedConfig] = None,
) -> LogitsProcessorWithLoRA:
    ret = LogitsProcessorWithLoRA(layer, lm_head.embedding_dim,
                                  lm_head.weight.dtype, lm_head.weight.device,
                                  lm_head.get_sharded_to_full_mapping())
    ret.create_lora_weights(max_loras, lora_config, model_config)
    return ret


def replace_submodule(model: nn.Module, module_name: str,
                      new_module: nn.Module) -> nn.Module:
    """Replace a submodule in a model with a new module."""
    parent = model.get_submodule(".".join(module_name.split(".")[:-1]))
    target_name = module_name.split(".")[-1]
    setattr(parent, target_name, new_module)
    return new_module


def parse_fine_tuned_lora_name(
        name: str,
        weights_mapper: Optional[WeightsMapper] = None
<<<<<<< HEAD
) -> Tuple[str, bool, bool]:
=======
) -> tuple[str, bool, bool]:
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
    """Parse the name of lora weights.

    args:
        name: the name of the fine-tuned LoRA, e.g.
            base_model.model.dense1.weight
        weights_mapper: maps the name of weight, e.g.
            `model.` -> `language_model.model.`,
    return:
<<<<<<< HEAD
        Tuple(module_name, is_lora_a):
=======
        tuple(module_name, is_lora_a):
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
            module_name: the name of the module, e.g. model.dense1,
            is_lora_a whether the tensor is lora_a or lora_b.
            is_bias whether the tensor is lora bias.
    """

<<<<<<< HEAD
    # LoRA weight qualified name always starts with `base_model.model.`,
    # so we remove the prefix `base_model.model.` to make the following
    # mapping correctly.
    if "base_model.model." in name:
=======
    # LoRA weight qualified name usually starts with `base_model.model.`,
    # so we remove the prefix `base_model.model.` to make the following
    # mapping correctly.
    if name.startswith("base_model.model."):
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
        name = name.replace("base_model.model.", "")
        name = weights_mapper._map_name(name) if weights_mapper else name
        # recover the prefix `base_model.model.`
        name = "base_model.model." + name
<<<<<<< HEAD
=======
    else:
        name = weights_mapper._map_name(name) if weights_mapper else name

    # In some situations, we may not start with `base_model.model.`.
    # If we don't (e.g., ibm-granite/granite-speech-3.3-8b),
    # we should keep the prefix intact.
    start_index = 2 if name.startswith("base_model.model.") else 0
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea

    parts = name.split(".")
    if parts[-1] == "weight" and (parts[-2] == "lora_A"
                                  or parts[-2] == "lora_B"):
<<<<<<< HEAD
        new_name = ".".join(parts[2:-2])
        return new_name, parts[-2] == "lora_A", False

    if parts[-1] == "lora_embedding_A" or parts[-1] == "lora_embedding_B":
        new_name = ".".join(parts[2:-1])
        return new_name, parts[-1] == "lora_embedding_A", False

    if parts[-1] == "bias":
        new_name = ".".join(parts[2:-2])
=======
        new_name = ".".join(parts[start_index:-2])
        return new_name, parts[-2] == "lora_A", False

    if parts[-1] == "lora_embedding_A" or parts[-1] == "lora_embedding_B":
        new_name = ".".join(parts[start_index:-1])
        return new_name, parts[-1] == "lora_embedding_A", False

    if parts[-1] == "bias":
        new_name = ".".join(parts[start_index:-2])
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
        return new_name, False, True

    raise ValueError(f"{name} is unsupported LoRA weight")


<<<<<<< HEAD
def is_regex_target_modules(load_modules: Union[str, List[str]],
                            expected_lora_modules: List[str]) -> bool:
=======
def is_regex_target_modules(load_modules: Union[str, list[str]],
                            expected_lora_modules: list[str]) -> bool:
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
    """
    PEFT supports passing `target_modules` in the form of regular expressions, 
    such as `model.*(q_proj|k_proj|v_proj)$`. This function is mainly used to 
    determine whether the suffix in the regular expression is present in the 
    `expected_lora_modules`.
    """

    def is_valid_regex(pattern):
        try:
            re.compile(pattern)
            return True
        except re.error:
            return False

    def is_subset(sub_list, full_list):
        return set(sub_list).issubset(set(full_list))

    # Similar to PEFT's processing logic, regex-related operations are only
    #  executed when the load_modules is a `str`.
    if not isinstance(load_modules, str):
        return False

    if is_valid_regex(load_modules):
        match = re.search(r"\((.*?)\)\$?$", load_modules)
        if match:
            suffix = match.group(1).split("|")
            return is_subset(suffix, expected_lora_modules)
    return False


<<<<<<< HEAD
def get_supported_lora_modules(model: nn.Module) -> List[str]:
    """
    In vLLM, all linear layers support LoRA.
    """
    supported_lora_modules: Set[str] = set()
=======
def get_supported_lora_modules(model: nn.Module) -> list[str]:
    """
    In vLLM, all linear layers support LoRA.
    """
    supported_lora_modules: set[str] = set()
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
    # step1: traverse the model to get all the linear subfixes.
    for name, module in model.named_modules():
        if isinstance(module, (LinearBase, )):
            supported_lora_modules.add(name.split(".")[-1])
    # step 2: get the embedding modules if the model's mbedding_modules
    # is not empty.
    if model.embedding_modules:
        for name in model.embedding_modules:
            supported_lora_modules.add(name)
    return list(supported_lora_modules)


def get_adapter_absolute_path(lora_path: str) -> str:
    """
    Resolves the given lora_path to an absolute local path.

    If the lora_path is identified as a Hugging Face model identifier,
    it will download the model and return the local snapshot path.
    Otherwise, it treats the lora_path as a local file path and
    converts it to an absolute path.

    Parameters:
    lora_path (str): The path to the lora model, which can be an absolute path,
                     a relative path, or a Hugging Face model identifier.

    Returns:
    str: The resolved absolute local path to the lora model.
    """

    # Check if the path is an absolute path. Return it no matter exists or not.
    if os.path.isabs(lora_path):
        return lora_path

    # If the path starts with ~, expand the user home directory.
    if lora_path.startswith('~'):
        return os.path.expanduser(lora_path)

    # Check if the expanded relative path exists locally.
    if os.path.exists(lora_path):
        return os.path.abspath(lora_path)

    # If the path does not exist locally, assume it's a Hugging Face repo.
    try:
        local_snapshot_path = huggingface_hub.snapshot_download(
            repo_id=lora_path)
    except (HfHubHTTPError, RepositoryNotFoundError, EntryNotFoundError,
            HFValidationError):
        # Handle errors that may occur during the download
        # Return original path instead instead of throwing error here
        logger.exception("Error downloading the HuggingFace model")
        return lora_path

    return local_snapshot_path
