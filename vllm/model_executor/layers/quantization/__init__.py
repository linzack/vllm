# SPDX-License-Identifier: Apache-2.0

<<<<<<< HEAD
from typing import Dict, List, Type
=======
from typing import Literal, get_args
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea

from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig)

<<<<<<< HEAD
QUANTIZATION_METHODS: List[str] = [
=======
QuantizationMethods = Literal[
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
    "aqlm",
    "awq",
    "deepspeedfp",
    "tpu_int8",
    "fp8",
    "ptpc_fp8",
    "fbgemm_fp8",
    "modelopt",
<<<<<<< HEAD
    # The order of gptq methods is important for config.py iteration over
    # override_quantization_method(..)
    "marlin",
    "gguf",
    "gptq_marlin_24",
    "gptq_marlin",
=======
    "modelopt_fp4",
    "marlin",
    "bitblas",
    "gguf",
    "gptq_marlin_24",
    "gptq_marlin",
    "gptq_bitblas",
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
    "awq_marlin",
    "gptq",
    "compressed-tensors",
    "bitsandbytes",
    "qqq",
    "hqq",
    "experts_int8",
    "neuron_quant",
    "ipex",
    "quark",
<<<<<<< HEAD
    "moe_wna16"
]
=======
    "moe_wna16",
    "torchao",
    "auto-round",
]
QUANTIZATION_METHODS: list[str] = list(get_args(QuantizationMethods))
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea

# The customized quantization methods which will be added to this dict.
_CUSTOMIZED_METHOD_TO_QUANT_CONFIG = {}


def register_quantization_config(quantization: str):
    """Register a customized vllm quantization config.

    When a quantization method is not supported by vllm, you can register a customized
    quantization config to support it.

    Args:
        quantization (str): The quantization method name.

    Examples:
        >>> from vllm.model_executor.layers.quantization import register_quantization_config
        >>> from vllm.model_executor.layers.quantization import get_quantization_config
        >>> from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
        >>>
        >>> @register_quantization_config("my_quant")
        ... class MyQuantConfig(QuantizationConfig):
        ...     pass
        >>>
        >>> get_quantization_config("my_quant")
        <class 'MyQuantConfig'>
    """  # noqa: E501

    def _wrapper(quant_config_cls):
        if quantization in QUANTIZATION_METHODS:
            raise ValueError(
                f"The quantization method `{quantization}` is already exists.")
        if not issubclass(quant_config_cls, QuantizationConfig):
            raise ValueError("The quantization config must be a subclass of "
                             "`QuantizationConfig`.")
        _CUSTOMIZED_METHOD_TO_QUANT_CONFIG[quantization] = quant_config_cls
        QUANTIZATION_METHODS.append(quantization)
        return quant_config_cls

    return _wrapper


<<<<<<< HEAD
def get_quantization_config(quantization: str) -> Type[QuantizationConfig]:
=======
def get_quantization_config(quantization: str) -> type[QuantizationConfig]:
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
    if quantization not in QUANTIZATION_METHODS:
        raise ValueError(f"Invalid quantization method: {quantization}")

    # lazy import to avoid triggering `torch.compile` too early
    from vllm.model_executor.layers.quantization.quark.quark import QuarkConfig

    from .aqlm import AQLMConfig
<<<<<<< HEAD
    from .awq import AWQConfig
    from .awq_marlin import AWQMarlinConfig
=======
    from .auto_round import AutoRoundConfig
    from .awq import AWQConfig
    from .awq_marlin import AWQMarlinConfig
    from .bitblas import BitBLASConfig
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
    from .bitsandbytes import BitsAndBytesConfig
    from .compressed_tensors.compressed_tensors import (  # noqa: E501
        CompressedTensorsConfig)
    from .deepspeedfp import DeepSpeedFPConfig
    from .experts_int8 import ExpertsInt8Config
    from .fbgemm_fp8 import FBGEMMFp8Config
    from .fp8 import Fp8Config
    from .gguf import GGUFConfig
    from .gptq import GPTQConfig
<<<<<<< HEAD
=======
    from .gptq_bitblas import GPTQBitBLASConfig
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
    from .gptq_marlin import GPTQMarlinConfig
    from .gptq_marlin_24 import GPTQMarlin24Config
    from .hqq_marlin import HQQMarlinConfig
    from .ipex_quant import IPEXConfig
    from .marlin import MarlinConfig
<<<<<<< HEAD
    from .modelopt import ModelOptFp8Config
=======
    from .modelopt import ModelOptFp8Config, ModelOptNvFp4Config
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
    from .moe_wna16 import MoeWNA16Config
    from .neuron_quant import NeuronQuantConfig
    from .ptpc_fp8 import PTPCFp8Config
    from .qqq import QQQConfig
<<<<<<< HEAD
    from .tpu_int8 import Int8TpuConfig

    method_to_config: Dict[str, Type[QuantizationConfig]] = {
=======
    from .torchao import TorchAOConfig
    from .tpu_int8 import Int8TpuConfig

    method_to_config: dict[str, type[QuantizationConfig]] = {
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
        "aqlm": AQLMConfig,
        "awq": AWQConfig,
        "deepspeedfp": DeepSpeedFPConfig,
        "tpu_int8": Int8TpuConfig,
        "fp8": Fp8Config,
        "fbgemm_fp8": FBGEMMFp8Config,
        "modelopt": ModelOptFp8Config,
<<<<<<< HEAD
        # The order of gptq methods is important for config.py iteration over
        # override_quantization_method(..)
        "marlin": MarlinConfig,
        "gguf": GGUFConfig,
        "gptq_marlin_24": GPTQMarlin24Config,
        "gptq_marlin": GPTQMarlinConfig,
=======
        "modelopt_fp4": ModelOptNvFp4Config,
        "marlin": MarlinConfig,
        "bitblas": BitBLASConfig,
        "gguf": GGUFConfig,
        "gptq_marlin_24": GPTQMarlin24Config,
        "gptq_marlin": GPTQMarlinConfig,
        "gptq_bitblas": GPTQBitBLASConfig,
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
        "awq_marlin": AWQMarlinConfig,
        "gptq": GPTQConfig,
        "compressed-tensors": CompressedTensorsConfig,
        "bitsandbytes": BitsAndBytesConfig,
        "ptpc_fp8": PTPCFp8Config,
        "qqq": QQQConfig,
        "hqq": HQQMarlinConfig,
        "experts_int8": ExpertsInt8Config,
        "neuron_quant": NeuronQuantConfig,
        "ipex": IPEXConfig,
        "quark": QuarkConfig,
        "moe_wna16": MoeWNA16Config,
<<<<<<< HEAD
=======
        "torchao": TorchAOConfig,
        "auto-round": AutoRoundConfig,
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
    }
    # Update the `method_to_config` with customized quantization methods.
    method_to_config.update(_CUSTOMIZED_METHOD_TO_QUANT_CONFIG)

    return method_to_config[quantization]


__all__ = [
    "QuantizationConfig",
<<<<<<< HEAD
    "get_quantization_config",
    "QUANTIZATION_METHODS",
]
=======
    "QuantizationMethods",
    "get_quantization_config",
    "QUANTIZATION_METHODS",
]
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
