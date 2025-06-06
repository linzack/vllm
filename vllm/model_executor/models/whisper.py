# SPDX-License-Identifier: Apache-2.0

import math
<<<<<<< HEAD
from typing import (Iterable, List, Mapping, Optional, Set, Tuple, TypedDict,
                    Union)

import numpy as np
import torch
from torch import nn
from transformers.models.whisper.modeling_whisper import sinusoids

from vllm.attention import Attention, AttentionMetadata, AttentionType
from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.inputs import INPUT_REGISTRY, DummyData, InputContext
=======
from collections.abc import Iterable, Mapping, Sequence
from typing import Optional, TypedDict, Union

import torch
from torch import nn
from transformers import (BatchFeature, WhisperConfig, WhisperFeatureExtractor,
                          WhisperProcessor)
from transformers.models.whisper.modeling_whisper import sinusoids

from vllm.attention import Attention, AttentionType
from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import get_tensor_model_parallel_world_size
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
from vllm.logger import init_logger
from vllm.model_executor.layers.activation import get_act_fn
from vllm.model_executor.layers.linear import (ColumnParallelLinear,
                                               QKVParallelLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig)
<<<<<<< HEAD
from vllm.model_executor.layers.sampler import Sampler, SamplerOutput
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.multimodal import (MULTIMODAL_REGISTRY, MultiModalKwargs,
                             NestedTensors)
from vllm.multimodal.audio import resample_audio
from vllm.sequence import SequenceData
from vllm.transformers_utils.processor import cached_processor_from_config

from .interfaces import SupportsMultiModal, SupportsTranscription
from .utils import AutoWeightsLoader, WeightsMapper, make_layers
=======
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.multimodal import MULTIMODAL_REGISTRY, NestedTensors
from vllm.multimodal.inputs import (MultiModalDataDict, MultiModalFieldConfig,
                                    MultiModalKwargs)
from vllm.multimodal.parse import MultiModalDataItems, MultiModalDataParser
from vllm.multimodal.processing import (BaseProcessingInfo,
                                        EncDecMultiModalProcessor,
                                        PromptReplacement, PromptUpdate)
from vllm.multimodal.profiling import BaseDummyInputsBuilder

from .interfaces import (MultiModalEmbeddings, SupportsMultiModal,
                         SupportsTranscription, SupportsV0Only)
from .utils import (AutoWeightsLoader, WeightsMapper, cast_overflow_tensors,
                    make_layers)
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea

logger = init_logger(__name__)


class WhisperAudioInputs(TypedDict):
    input_features: NestedTensors
    """Shape: `(batch_size, 128, M)`"""


class WhisperPositionalEmbedding(nn.Embedding):

<<<<<<< HEAD
    def __init__(self,
                 num_positions: int,
                 embedding_dim: int,
                 padding_idx: Optional[int] = None):
=======
    def __init__(self, num_positions: int, embedding_dim: int):
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
        super().__init__(num_positions, embedding_dim)

    def forward(self, position_ids):
        return self.weight[position_ids]


class WhisperAttention(nn.Module):

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        bias: bool = True,
        attn_type: AttentionType = AttentionType.DECODER,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.embed_dim = embed_dim
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        if self.total_num_heads >= tp_size:
            # Number of heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_heads % tp_size == 0
        else:
            # Number of heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert tp_size % self.total_num_heads == 0
        self.num_kv_heads = max(1, self.total_num_heads // tp_size)
        self.head_dim = self.embed_dim // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.attn_type = attn_type

        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: "
                f"{self.embed_dim} and `num_heads`: {num_heads}).")
        self.scaling = self.head_dim**-0.5

        self._init_qkv(embed_dim, bias, quant_config, prefix=prefix)
        self.out_proj = RowParallelLinear(
            input_size=embed_dim,
            output_size=embed_dim,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.out_proj",
        )
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
            attn_type=self.attn_type,
        )

    def _init_qkv(
        self,
        embed_dim: int,
        bias: bool = True,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        self.qkv_proj = QKVParallelLinear(
            hidden_size=embed_dim,
            head_size=self.head_dim,
            total_num_heads=self.total_num_heads,
            total_num_kv_heads=self.total_num_heads,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
<<<<<<< HEAD
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
=======
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
    ):
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

<<<<<<< HEAD
        attn_output = self.attn(q, k, v, kv_cache, attn_metadata)
=======
        attn_output = self.attn(q, k, v)
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea

        output, _ = self.out_proj(attn_output)

        return output


class WhisperCrossAttention(WhisperAttention):

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        bias: bool = True,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__(
            embed_dim=embed_dim,
            num_heads=num_heads,
            bias=bias,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=prefix,
            attn_type=AttentionType.ENCODER_DECODER,
        )

    def _init_qkv(
        self,
        embed_dim: int,
        bias: bool = True,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        self.q_proj = ColumnParallelLinear(
            input_size=embed_dim,
            output_size=embed_dim,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.q_proj",
        )
        self.kv_proj = QKVParallelLinear(
            hidden_size=embed_dim,
            head_size=self.head_dim,
            total_num_heads=0,
            total_num_kv_heads=self.total_num_heads,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.kv_proj",
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor],
<<<<<<< HEAD
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
=======
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
    ):
        q, _ = self.q_proj(hidden_states)

        # Encoder hidden states are only computed once during prefill phase.
        # Afterwards, the keys and values should be available in the kv-cache.
        if encoder_hidden_states is not None:
            kv, _ = self.kv_proj(encoder_hidden_states)
            k, v = kv.split([self.kv_size, self.kv_size], dim=-1)
        else:
            k = v = None

<<<<<<< HEAD
        attn_output = self.attn(
            q,
            k,
            v,
            kv_cache,
            attn_metadata,
        )
=======
        attn_output = self.attn(q, k, v)
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea

        output, _ = self.out_proj(attn_output)

        return output


class WhisperMLP(nn.Module):

    def __init__(
        self,
        embed_dim: int,
        ffn_dim: int,
        act_fn: str,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()

        self.activation_fn = get_act_fn(act_fn)
        self.fc1 = ColumnParallelLinear(
            input_size=embed_dim,
            output_size=ffn_dim,
            quant_config=quant_config,
            prefix=f"{prefix}.fc1",
        )
        self.fc2 = RowParallelLinear(
            input_size=ffn_dim,
            output_size=embed_dim,
            quant_config=quant_config,
            prefix=f"{prefix}.fc2",
        )

    def forward(self, hidden_states: torch.Tensor):
        hidden_states, _ = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states, _ = self.fc2(hidden_states)
        return hidden_states


class WhisperEncoderLayer(nn.Module):

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config

        self.embed_dim = config.d_model
        self.self_attn = WhisperAttention(
            embed_dim=self.embed_dim,
            num_heads=config.encoder_attention_heads,
            attn_type=AttentionType.ENCODER,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn",
        )
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.mlp = WhisperMLP(
            embed_dim=config.d_model,
            ffn_dim=config.encoder_ffn_dim,
            act_fn=config.activation_function,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp",
        )
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
<<<<<<< HEAD
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ):
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            kv_cache=kv_cache,
            attn_metadata=attn_metadata,
        )
=======
    ):
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states = self.self_attn(hidden_states=hidden_states)
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

<<<<<<< HEAD
        if hidden_states.isinf().any() or hidden_states.isnan().any():
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states,
                                        min=-clamp_value,
                                        max=clamp_value)
=======
        hidden_states = cast_overflow_tensors(hidden_states)
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea

        return hidden_states


class WhisperDecoderLayer(nn.Module):

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config

        self.self_attn = WhisperAttention(
            embed_dim=config.d_model,
            num_heads=config.decoder_attention_heads,
            attn_type=AttentionType.DECODER,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn",
        )
        self.self_attn_layer_norm = nn.LayerNorm(config.d_model)
        self.encoder_attn = WhisperCrossAttention(
            embed_dim=config.d_model,
            num_heads=config.decoder_attention_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.encoder_attn",
        )
        self.encoder_attn_layer_norm = nn.LayerNorm(config.d_model)
        self.mlp = WhisperMLP(
            embed_dim=config.d_model,
            ffn_dim=config.decoder_ffn_dim,
            act_fn=config.activation_function,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp",
        )
        self.final_layer_norm = nn.LayerNorm(config.d_model)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor],
<<<<<<< HEAD
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ):
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states = self.self_attn(hidden_states=hidden_states,
                                       kv_cache=kv_cache,
                                       attn_metadata=attn_metadata)
=======
    ):
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states = self.self_attn(hidden_states=hidden_states)
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.encoder_attn_layer_norm(hidden_states)
        hidden_states = self.encoder_attn(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
<<<<<<< HEAD
            kv_cache=kv_cache,
            attn_metadata=attn_metadata,
=======
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class WhisperEncoder(nn.Module):

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        embed_dim = config.d_model
        self.num_mel_bins = config.num_mel_bins
<<<<<<< HEAD
        self.padding_idx = config.pad_token_id
=======
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
        self.max_source_positions = config.max_source_positions
        self.embed_scale = (math.sqrt(embed_dim)
                            if config.scale_embedding else 1.0)

        self.conv1 = nn.Conv1d(self.num_mel_bins,
                               embed_dim,
                               kernel_size=3,
                               padding=1)
        self.conv2 = nn.Conv1d(embed_dim,
                               embed_dim,
                               kernel_size=3,
                               stride=2,
                               padding=1)
        self.embed_positions = nn.Embedding(self.max_source_positions,
                                            embed_dim)
        self.start_layer, self.end_layer, self.layers = make_layers(
            config.encoder_layers,
            lambda prefix: WhisperEncoderLayer(vllm_config=vllm_config,
                                               prefix=f"{prefix}.layers"),
            prefix=f"{prefix}.layers",
        )
        self.layer_norm = nn.LayerNorm(config.d_model)

        with torch.no_grad():
            self.embed_positions.weight.copy_(
                sinusoids(*self.embed_positions.weight.shape))

<<<<<<< HEAD
    def forward(
        self,
        input_features: Union[torch.Tensor, List[torch.Tensor]],
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
    ):
=======
    def forward(self, input_features: Union[torch.Tensor, list[torch.Tensor]]):
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
        hidden_states = []
        for features in input_features:
            embeds = nn.functional.gelu(self.conv1(features))
            embeds = nn.functional.gelu(self.conv2(embeds))
            embeds = embeds.permute(1, 0)
            embeds = embeds + self.embed_positions.weight[:embeds.size(0), :]
            hidden_states.append(embeds)
        hidden_states = torch.cat(hidden_states)

<<<<<<< HEAD
        for idx, encoder_layer in enumerate(self.layers):
            hidden_states = encoder_layer(
                hidden_states,
                kv_cache=kv_caches[idx],
                attn_metadata=attn_metadata,
            )
=======
        for encoder_layer in self.layers:
            hidden_states = encoder_layer(hidden_states)
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea

        hidden_states = self.layer_norm(hidden_states)
        return hidden_states


class WhisperDecoder(nn.Module):

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.layerdrop = config.decoder_layerdrop
        self.padding_idx = config.pad_token_id
        self.max_target_positions = config.max_target_positions
        self.max_source_positions = config.max_source_positions
        self.embed_scale = (math.sqrt(config.d_model)
                            if config.scale_embedding else 1.0)

        self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model,
                                         self.padding_idx)
        self.embed_positions = WhisperPositionalEmbedding(
            self.max_target_positions, config.d_model)
        self.start_layer, self.end_layer, self.layers = make_layers(
            config.decoder_layers,
            lambda prefix: WhisperDecoderLayer(vllm_config=vllm_config,
                                               prefix=f"{prefix}.layers"),
            prefix=f"{prefix}.layers",
        )
        self.layer_norm = nn.LayerNorm(config.d_model)

    def forward(
        self,
        input_ids,
        positions: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor],
<<<<<<< HEAD
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
=======
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
    ):
        inputs_embeds = self.get_input_embeddings(input_ids)
        positions = self.embed_positions(positions)
        hidden_states = inputs_embeds + positions

<<<<<<< HEAD
        for idx, decoder_layer in enumerate(self.layers):
            hidden_states = decoder_layer(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                kv_cache=kv_caches[idx],
                attn_metadata=attn_metadata,
=======
        for decoder_layer in self.layers:
            hidden_states = decoder_layer(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
            )

        hidden_states = self.layer_norm(hidden_states)
        return hidden_states

    def get_input_embeddings(
        self,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        return self.embed_tokens(input_ids)


class WhisperModel(nn.Module):

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.encoder = WhisperEncoder(vllm_config=vllm_config,
                                      prefix=f"{prefix}.encoder")
        self.decoder = WhisperDecoder(vllm_config=vllm_config,
                                      prefix=f"{prefix}.decoder")

    def forward(
        self,
<<<<<<< HEAD
        input_features: Optional[Union[torch.Tensor, List[torch.Tensor]]],
        input_ids: Optional[torch.Tensor],
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        encoder_outputs = self.get_encoder_outputs(
            input_features,
            kv_caches=kv_caches,
            attn_metadata=attn_metadata,
        )
=======
        input_features: Optional[Union[torch.Tensor, list[torch.Tensor]]],
        input_ids: Optional[torch.Tensor],
        positions: torch.Tensor,
    ) -> torch.Tensor:
        encoder_outputs = self.get_encoder_outputs(input_features)
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
        decoder_outputs = self.decoder(
            input_ids=input_ids,
            positions=positions,
            encoder_hidden_states=encoder_outputs,
<<<<<<< HEAD
            kv_caches=kv_caches,
            attn_metadata=attn_metadata,
=======
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
        )
        return decoder_outputs

    def get_encoder_outputs(
        self,
<<<<<<< HEAD
        input_features: Optional[Union[torch.Tensor, List[torch.Tensor]]],
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
    ) -> Optional[torch.Tensor]:
        if input_features is None:
            return None
        return self.encoder(
            input_features,
            kv_caches=kv_caches,
            attn_metadata=attn_metadata,
        )

    def load_weights(self, weights: Iterable[Tuple[str,
                                                   torch.Tensor]]) -> Set[str]:
=======
        input_features: Optional[Union[torch.Tensor, list[torch.Tensor]]],
    ) -> Optional[torch.Tensor]:
        if input_features is None:
            return None
        return self.encoder(input_features)

    def load_weights(self, weights: Iterable[tuple[str,
                                                   torch.Tensor]]) -> set[str]:
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            (".self_attn.qkv_proj", ".self_attn.q_proj", "q"),
            (".self_attn.qkv_proj", ".self_attn.k_proj", "k"),
            (".self_attn.qkv_proj", ".self_attn.v_proj", "v"),
            (".encoder_attn.kv_proj", ".encoder_attn.k_proj", "k"),
            (".encoder_attn.kv_proj", ".encoder_attn.v_proj", "v"),
        ]
        params_dict = dict(self.named_parameters())
<<<<<<< HEAD
        loaded_params: Set[str] = set()
=======
        loaded_params: set[str] = set()
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
        for name, loaded_weight in weights:
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue

                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue

                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params


<<<<<<< HEAD
def get_max_whisper_audio_tokens(ctx: InputContext) -> int:
    return ctx.model_config.hf_config.max_source_positions


def dummy_encoder_data_for_whisper(ctx: InputContext, seq_len: int,
                                   mm_counts: Mapping[str, int]):
    assert mm_counts["audio"] == 1
    num_tokens = get_max_whisper_audio_tokens(ctx)
    processor = cached_processor_from_config(ctx.model_config)
    chunk_length = processor.feature_extractor.chunk_length
    sampling_rate = processor.feature_extractor.sampling_rate
    num_samples = chunk_length * sampling_rate
    return DummyData(
        SequenceData.from_prompt_token_counts((0, num_tokens)),
        {"audio": [(np.zeros(num_samples), sampling_rate)]},
    )


def input_processor_for_whisper(ctx: InputContext, inputs):
    multi_modal_data = inputs["encoder"]["multi_modal_data"]
    if isinstance(multi_modal_data["audio"], list):
        assert len(multi_modal_data["audio"]) == 1
        multi_modal_data["audio"] = multi_modal_data["audio"][0]
    # Resample and process audio
    audio, orig_sr = multi_modal_data["audio"]
    processor = cached_processor_from_config(ctx.model_config)
    target_sr = processor.feature_extractor.sampling_rate
    audio = resample_audio(audio, orig_sr=orig_sr, target_sr=target_sr)
    multi_modal_data["audio"] = (audio, target_sr)
    # Pre-allocate placeholder tokens in encoder sequence
    num_tokens = get_max_whisper_audio_tokens(ctx)
    inputs["encoder"]["prompt_token_ids"] = [0] * num_tokens
    return inputs


def input_mapper_for_whisper(
    ctx: InputContext,
    multi_modal_data: Union[np.ndarray, List[np.ndarray]],
) -> MultiModalKwargs:
    if not isinstance(multi_modal_data, list):
        multi_modal_data = [multi_modal_data]

    assert len(multi_modal_data) == 1

    if len(multi_modal_data) == 0:
        return MultiModalKwargs()

    processor = cached_processor_from_config(ctx.model_config)
    sampling_rate = processor.feature_extractor.sampling_rate

    audios = [audio for audio, _ in multi_modal_data]

    kwargs = processor(audios,
                       sampling_rate=sampling_rate,
                       return_tensors="pt")
    kwargs["input_features"] = kwargs["input_features"].squeeze(0).to(
        ctx.model_config.dtype)

    return MultiModalKwargs(kwargs)


@INPUT_REGISTRY.register_dummy_encoder_data(dummy_encoder_data_for_whisper)
@INPUT_REGISTRY.register_input_processor(input_processor_for_whisper)
@MULTIMODAL_REGISTRY.register_input_mapper("audio", input_mapper_for_whisper)
@MULTIMODAL_REGISTRY.register_max_multimodal_tokens(
    "audio", get_max_whisper_audio_tokens)
class WhisperForConditionalGeneration(nn.Module, SupportsTranscription,
                                      SupportsMultiModal):
=======
class WhisperProcessingInfo(BaseProcessingInfo):

    def get_hf_config(self) -> WhisperConfig:
        return self.ctx.get_hf_config(WhisperConfig)

    def get_hf_processor(self,
                         sampling_rate: Optional[int] = None
                         ) -> WhisperProcessor:
        return self.ctx.get_hf_processor(WhisperProcessor)

    def get_supported_mm_limits(self) -> Mapping[str, Optional[int]]:
        return {"audio": 1}

    def get_feature_extractor(self) -> WhisperFeatureExtractor:
        hf_processor = self.get_hf_processor()
        feature_extractor = hf_processor.feature_extractor  # type: ignore
        assert isinstance(feature_extractor, WhisperFeatureExtractor)
        return feature_extractor

    def get_num_audio_tokens(self) -> int:
        return self.get_hf_config().max_source_positions


class WhisperDummyInputsBuilder(BaseDummyInputsBuilder[WhisperProcessingInfo]):

    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        num_audios = mm_counts.get("audio", 0)

        return "<|startoftranscript|>" * num_audios

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> MultiModalDataDict:
        feature_extractor = self.info.get_feature_extractor()

        sampling_rate = feature_extractor.sampling_rate
        audio_len = feature_extractor.chunk_length * sampling_rate
        num_audios = mm_counts.get("audio", 0)

        return {
            "audio":
            self._get_dummy_audios(length=audio_len, num_audios=num_audios)
        }


class WhisperMultiModalProcessor(
        EncDecMultiModalProcessor[WhisperProcessingInfo]):

    def _get_data_parser(self) -> MultiModalDataParser:
        feature_extractor = self.info.get_feature_extractor()
        return MultiModalDataParser(target_sr=feature_extractor.sampling_rate)

    @property
    def pad_dummy_encoder_prompt(self) -> bool:
        return True

    def create_encoder_prompt(
        self,
        prompt: Union[str, list[int]],
        mm_data: MultiModalDataDict,
    ) -> Union[str, list[int]]:
        # Strictly speaking, whisper encoder only accept audio features.
        # We create a dummy encoder prompt here which will be padded to
        # num_audio_tokens. So that we can create dummy data from this
        # for encoder profiling.
        return [0]

    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        if mm_data:
            feature_extractor = self.info.get_feature_extractor(**mm_kwargs)
            mm_data = dict(audio=mm_data.pop("audios"))
            mm_kwargs = dict(
                **mm_kwargs,
                sampling_rate=feature_extractor.sampling_rate,
            )
        processed_outputs = super()._call_hf_processor(
            prompt=prompt,
            mm_data=mm_data,
            mm_kwargs=mm_kwargs,
        )
        if "labels" in processed_outputs:
            processed_outputs["input_ids"] = processed_outputs.pop("labels")
        return processed_outputs

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        return dict(input_features=MultiModalFieldConfig.batched("audio"))

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargs,
    ) -> Sequence[PromptUpdate]:
        num_tokens = self.info.get_num_audio_tokens()
        return [
            PromptReplacement(
                modality="audio",
                target=[0],
                replacement=[0] * num_tokens,
            )
        ]


@MULTIMODAL_REGISTRY.register_processor(WhisperMultiModalProcessor,
                                        info=WhisperProcessingInfo,
                                        dummy_inputs=WhisperDummyInputsBuilder)
class WhisperForConditionalGeneration(nn.Module, SupportsTranscription,
                                      SupportsMultiModal, SupportsV0Only):
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
    packed_modules_mapping = {
        "self_attn.qkv_proj": [
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
        ],
        "encoder_attn.kv_proj": ["encoder_attn.k_proj", "encoder_attn.v_proj"],
    }

    hf_to_vllm_mapper = WeightsMapper(orig_to_new_substr={
        ".fc1.": ".mlp.fc1.",
        ".fc2.": ".mlp.fc2."
    })

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.config = config
        self.dtype = vllm_config.model_config.dtype

        self.model = WhisperModel(vllm_config=vllm_config, prefix=prefix)
        self.unpadded_vocab_size = config.vocab_size
        self.proj_out = ParallelLMHead(config.vocab_size,
                                       config.d_model,
                                       quant_config=quant_config)
        self.proj_out = self.proj_out.tie_weights(
            self.model.decoder.embed_tokens)
        logit_scale = getattr(config, "logit_scale", 1.0)
        self.logits_processor = LogitsProcessor(self.unpadded_vocab_size,
                                                config.vocab_size, logit_scale)
<<<<<<< HEAD
        self.sampler = Sampler()
=======
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
<<<<<<< HEAD
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
=======
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
        **kwargs,
    ) -> torch.Tensor:
        audio_input = self._parse_and_validate_audio_input(**kwargs)
        decoder_outputs = self.model(
            input_features=audio_input["input_features"],
            input_ids=input_ids,
            positions=positions,
<<<<<<< HEAD
            kv_caches=kv_caches,
            attn_metadata=attn_metadata,
        )
        return decoder_outputs

    def get_multimodal_embeddings(
        self,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        **kwargs,
    ) -> Optional[NestedTensors]:
        # TODO: This method does not obey the interface for SupportsMultiModal.
        # Refactor this once encoder/decoder support is implemented in V1.
        audio_input = self._parse_and_validate_audio_input(**kwargs)
        return self.model.get_encoder_outputs(
            audio_input["input_features"],
            kv_caches=kv_caches,
            attn_metadata=attn_metadata,
        )
=======
        )
        return decoder_outputs

    def get_language_model(self) -> torch.nn.Module:
        return self.model.decoder

    def get_multimodal_embeddings(
            self, **kwargs: object) -> Optional[MultiModalEmbeddings]:
        # TODO: This method does not obey the interface for SupportsMultiModal.
        # Refactor this once encoder/decoder support is implemented in V1.
        audio_input = self._parse_and_validate_audio_input(**kwargs)
        return self.model.get_encoder_outputs(audio_input["input_features"])
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea

    def get_input_embeddings(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: Optional[NestedTensors] = None,
<<<<<<< HEAD
        attn_metadata: Optional[AttentionMetadata] = None,
=======
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
    ) -> torch.Tensor:
        # TODO: This method just returns the decoder sequence embeddings since
        # Whisper does not have encoder text tokens. Refactor this once
        # encoder/decoder support is implemented in V1.
        return self.model.decoder.get_input_embeddings(input_ids)

    def _parse_and_validate_audio_input(
            self, **kwargs: object) -> WhisperAudioInputs:
        input_features = kwargs.pop("input_features", None)

        if input_features is not None:
            if not isinstance(input_features, (torch.Tensor, list)):
                raise ValueError("Incorrect type of audio features. "
                                 f"Got type: {type(input_features)}")
<<<<<<< HEAD
            input_features = [feat.to(self.dtype) for feat in input_features]
=======
            input_features = torch.cat(
                [feat.to(self.dtype) for feat in input_features])
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea

        return WhisperAudioInputs(input_features=input_features)

    def compute_logits(self, hidden_states: torch.Tensor,
                       sampling_metadata: SamplingMetadata) -> torch.Tensor:
        logits = self.logits_processor(self.proj_out, hidden_states,
                                       sampling_metadata)
        return logits

<<<<<<< HEAD
    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        next_tokens = self.sampler(logits, sampling_metadata)
        return next_tokens

    def load_weights(self, weights: Iterable[Tuple[str,
                                                   torch.Tensor]]) -> Set[str]:
=======
    def load_weights(self, weights: Iterable[tuple[str,
                                                   torch.Tensor]]) -> set[str]:
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
        loader = AutoWeightsLoader(self, skip_prefixes=["proj_out."])

        # add fake zeros bias for k_proj to state_dict
        weights = _create_fake_bias_for_k_proj(weights)
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)


def _create_fake_bias_for_k_proj(
<<<<<<< HEAD
    weights: Iterable[Tuple[str, torch.Tensor]]
) -> Iterable[Tuple[str, torch.Tensor]]:
    """
    Create full zeros bias for k_proj weight in self-attention layers.
    So that the bias for k_proj in qkv_proj can be initialized with zeros.
    """
    for name, weight in weights:
        if name.endswith(".self_attn.k_proj.weight"):
=======
    weights: Iterable[tuple[str, torch.Tensor]]
) -> Iterable[tuple[str, torch.Tensor]]:
    """
    Create full zeros bias for k_proj weight in self-attn and x-attn layers.
    So that the bias for k_proj in qkv_proj can be initialized with zeros.
    """
    for name, weight in weights:
        if name.endswith(".k_proj.weight"):
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
            bias = torch.zeros(weight.size(0))
            bias_name = name.replace("weight", "bias")
            yield from [(name, weight), (bias_name, bias)]
        yield name, weight
