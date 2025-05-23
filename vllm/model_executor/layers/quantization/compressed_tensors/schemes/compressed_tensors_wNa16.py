# SPDX-License-Identifier: Apache-2.0

<<<<<<< HEAD
from typing import Callable, List, Optional, Set
=======
from typing import Callable, Optional
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea

import torch
from compressed_tensors.quantization import ActivationOrdering

from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.compressed_tensors.schemes import (
    CompressedTensorsScheme)
from vllm.model_executor.layers.quantization.kernels.mixed_precision import (
    MPLinearLayerConfig, choose_mp_linear_kernel)
from vllm.model_executor.layers.quantization.utils.marlin_utils import (
    marlin_repeat_scales_on_all_ranks)
<<<<<<< HEAD
from vllm.model_executor.parameter import (BasevLLMParameter,
                                           ChannelQuantScaleParameter,
                                           GroupQuantScaleParameter,
                                           PackedvLLMParameter,
                                           RowvLLMParameter)
=======
# yapf conflicts with isort for this block
# yapf: disable
from vllm.model_executor.parameter import (BasevLLMParameter,
                                           ChannelQuantScaleParameter,
                                           GroupQuantScaleParameter,
                                           PackedColumnParameter,
                                           PackedvLLMParameter,
                                           RowvLLMParameter)
# yapf: enable
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
from vllm.scalar_type import scalar_types

logger = init_logger(__name__)

__all__ = ["CompressedTensorsWNA16"]
WNA16_SUPPORTED_TYPES_MAP = {
    4: scalar_types.uint4b8,
    8: scalar_types.uint8b128
}
<<<<<<< HEAD
=======
WNA16_ZP_SUPPORTED_TYPES_MAP = {4: scalar_types.uint4, 8: scalar_types.uint8}
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
WNA16_SUPPORTED_BITS = list(WNA16_SUPPORTED_TYPES_MAP.keys())


class CompressedTensorsWNA16(CompressedTensorsScheme):
<<<<<<< HEAD
    _kernel_backends_being_used: Set[str] = set()
=======
    _kernel_backends_being_used: set[str] = set()
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea

    def __init__(self,
                 strategy: str,
                 num_bits: int,
                 group_size: Optional[int] = None,
<<<<<<< HEAD
=======
                 symmetric: Optional[bool] = True,
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
                 actorder: Optional[ActivationOrdering] = None):

        self.pack_factor = 32 // num_bits
        self.strategy = strategy
<<<<<<< HEAD
=======
        self.symmetric = symmetric
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
        self.group_size = -1 if group_size is None else group_size
        self.has_g_idx = actorder == ActivationOrdering.GROUP

        if self.group_size == -1 and self.strategy != "channel":
            raise ValueError("Marlin kernels require group quantization or "
                             "channelwise quantization, but found no group "
                             "size and strategy is not channelwise.")

        if num_bits not in WNA16_SUPPORTED_TYPES_MAP:
            raise ValueError(
                f"Unsupported num_bits = {num_bits}. "
                f"Supported num_bits = {WNA16_SUPPORTED_TYPES_MAP.keys()}")

<<<<<<< HEAD
        self.quant_type = WNA16_SUPPORTED_TYPES_MAP[num_bits]
=======
        self.quant_type = (WNA16_ZP_SUPPORTED_TYPES_MAP[num_bits]
                           if not self.symmetric else
                           WNA16_SUPPORTED_TYPES_MAP[num_bits])
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea

    @classmethod
    def get_min_capability(cls) -> int:
        # ampere and up
        return 80

    def create_weights(self, layer: torch.nn.Module, output_size: int,
<<<<<<< HEAD
                       input_size: int, output_partition_sizes: List[int],
=======
                       input_size: int, output_partition_sizes: list[int],
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
                       input_size_per_partition: int,
                       params_dtype: torch.dtype, weight_loader: Callable,
                       **kwargs):

        output_size_per_partition = sum(output_partition_sizes)

        mp_linear_kernel_config = MPLinearLayerConfig(
            full_weight_shape=(input_size, output_size),
            partition_weight_shape=\
                (input_size_per_partition, output_size_per_partition),
            weight_type=self.quant_type,
            act_type=params_dtype,
            group_size=self.group_size,
<<<<<<< HEAD
            zero_points=False,
=======
            zero_points=not self.symmetric,
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
            has_g_idx=self.has_g_idx
        )

        kernel_type = choose_mp_linear_kernel(mp_linear_kernel_config)

        if kernel_type.__name__ not in self._kernel_backends_being_used:
            logger.info("Using %s for CompressedTensorsWNA16",
                        kernel_type.__name__)
            self._kernel_backends_being_used.add(kernel_type.__name__)

        # If group_size is -1, we are in channelwise case.
        group_size = self.group_size if self.group_size != -1 else input_size
        row_parallel = (input_size != input_size_per_partition)
        partition_scales = not marlin_repeat_scales_on_all_ranks(
            self.has_g_idx, self.group_size, row_parallel)

        scales_and_zp_size = input_size // group_size

        if partition_scales:
            assert input_size_per_partition % group_size == 0
            scales_and_zp_size = input_size_per_partition // group_size

        weight = PackedvLLMParameter(input_dim=1,
                                     output_dim=0,
                                     weight_loader=weight_loader,
                                     packed_factor=self.pack_factor,
                                     packed_dim=1,
                                     data=torch.empty(
                                         output_size_per_partition,
                                         input_size_per_partition //
                                         self.pack_factor,
                                         dtype=torch.int32,
                                     ))

        weight_scale_args = {
            "weight_loader":
            weight_loader,
            "data":
            torch.empty(
                output_size_per_partition,
                scales_and_zp_size,
                dtype=params_dtype,
            )
        }
<<<<<<< HEAD
        if not partition_scales:
            weight_scale = ChannelQuantScaleParameter(output_dim=0,
                                                      **weight_scale_args)
=======

        zeros_args = {
            "weight_loader":
            weight_loader,
            "data":
            torch.zeros(
                output_size_per_partition // self.pack_factor,
                scales_and_zp_size,
                dtype=torch.int32,
            )
        }

        if not partition_scales:
            weight_scale = ChannelQuantScaleParameter(output_dim=0,
                                                      **weight_scale_args)

            if not self.symmetric:
                qzeros = PackedColumnParameter(output_dim=0,
                                               packed_dim=0,
                                               packed_factor=self.pack_factor,
                                               **zeros_args)
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
        else:
            weight_scale = GroupQuantScaleParameter(output_dim=0,
                                                    input_dim=1,
                                                    **weight_scale_args)
<<<<<<< HEAD
=======
            if not self.symmetric:
                qzeros = PackedvLLMParameter(input_dim=1,
                                             output_dim=0,
                                             packed_dim=0,
                                             packed_factor=self.pack_factor,
                                             **zeros_args)
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea

        # A 2D array defining the original shape of the weights
        # before packing
        weight_shape = BasevLLMParameter(data=torch.empty(2,
                                                          dtype=torch.int64),
                                         weight_loader=weight_loader)

        layer.register_parameter("weight_packed", weight)
        layer.register_parameter("weight_scale", weight_scale)
        layer.register_parameter("weight_shape", weight_shape)

<<<<<<< HEAD
=======
        if not self.symmetric:
            layer.register_parameter("weight_zero_point", qzeros)

>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
        # group index (for activation reordering)
        if self.has_g_idx:
            weight_g_idx = RowvLLMParameter(data=torch.empty(
                input_size_per_partition,
                dtype=torch.int32,
            ),
                                            input_dim=0,
                                            weight_loader=weight_loader)
            layer.register_parameter("weight_g_idx", weight_g_idx)

        self.kernel = kernel_type(mp_linear_kernel_config,
                                  w_q_param_name="weight_packed",
                                  w_s_param_name="weight_scale",
<<<<<<< HEAD
                                  w_zp_param_name=None,
=======
                                  w_zp_param_name="weight_zero_point",
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
                                  w_gidx_param_name="weight_g_idx")

    # Checkpoints are serialized in compressed-tensors format, which is
    # different from the format the kernel may want. Handle repacking here.
    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        self.kernel.process_weights_after_loading(layer)

    def apply_weights(self, layer: torch.nn.Module, x: torch.Tensor,
                      bias: Optional[torch.Tensor]) -> torch.Tensor:
        return self.kernel.apply_weights(layer, x, bias)
