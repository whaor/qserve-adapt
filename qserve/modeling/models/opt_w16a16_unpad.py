from typing import Dict, List, Optional

import qserve_backend.fused_attention as fused_attention

# import gc
import torch
from flash_attn.flash_attn_interface import flash_attn_varlen_func
from qserve_backend import fused_kernels
import torch
from torch import nn
from transformers import OPTConfig

import qserve.utils.constants
# opt model directly uses nn.ReLU and nn.LayerNorm under fp16
# do not need to load from qserve.modeling

from qserve.modeling.layers.sampler import Sampler
from qserve.sampling_params import SamplingParams
from qserve.utils.input_metadata import InputMetadata
from qserve.utils.quant_config import QServeQuantConfig
from qserve.utils.weight_utils import (
    convert_pyslice_to_tensor,
    hf_model_weights_iterator,
    load_padded_tensor_parallel_vocab,
    load_tensor_parallel_weights,
)

max_seq_len = qserve.utils.constants.max_seq_len


class OPTLearnedPositionalEmbedding(nn.Embedding):
    """
    This module learns positional embeddings up to a fixed maximum size.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int):
        # OPT is set up so that if padding_idx is specified then offset the embedding ids by 2
        # and adjust num_embeddings appropriately. Other models don't have this hack
        self.offset = 2
        super().__init__(num_embeddings + self.offset, embedding_dim)

    def forward(self, attention_mask: torch.LongTensor, past_key_values_length: int = 0):
        """`input_ids_shape` is expected to be [bsz x seqlen]."""
        attention_mask = attention_mask.long()

        # create positions depending on attention_mask
        positions = (torch.cumsum(attention_mask, dim=1).type_as(attention_mask) * attention_mask).long() - 1

        # cut positions if `past_key_values_length` is > 0
        positions = positions[:, past_key_values_length:]

        return super().forward(positions + self.offset)


class OPTMLP(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()
        hidden_size = args.hidden_size
        ffn_dim = args.ffn_dim
        self.dropout = args.dropout
        self.use_int8 = True

        self.up_proj = nn.Linear(
            hidden_size, ffn_dim, bias=True, # OPTConfig's enable_bias is default to True
        )
        self.down_proj = nn.Linear(
            ffn_dim, hidden_size, bias=True
        )
        
        self.act_fn = nn.ReLU()
    
    def forward(self, x):
        # FP16 in, FP16 out
        x = self.up_proj(x)
        x = self.act_fn(x)
        x = self.down_proj(x)
        # original huggingface implementation has dropout here
        x = nn.functional.dropout(x, p=self.dropout, training=self.training)
        return x
    
class OPTAttention(nn.Module):
    def __init__(
            self,
            args,
            layer_idx: int,
            kv_cache_config: Optional[Dict] = None,
    ) -> None:
        super().__init__()
        self.hidden_size = args.hidden_size
        self.num_heads = args.num_attention_heads
        self.max_position_embeddings = args.max_position_embeddings
        self.layer_idx = layer_idx
        self.head_dim = self.hidden_size // self.num_heads
        
        self.scaling = self.head_dim ** (-0.5)
        self.use_int8 = True

        if kv_cache_config is None:
            self.kv_cache_config = {"INT4_ENABLED": False, "ZEROS_ENABLED": False}
            print("[Warning] kv cache config is not provided, using default config KV8")
        else:
            self.kv_cache_config = kv_cache_config
        

        self.qkv_proj = nn.Linear(
            self.hidden_size,
            3 * self.hidden_size,
            bias=args.enable_bias
        )
        self.o_proj = nn.Linear(
            self.hidden_size,
            self.hidden_size,
            bias=args.enable_bias
        )

        self.kv_max_seq_len = min(max_seq_len, self.max_position_embeddings)
    
    def forward(
            self,
            hidden_states,
            input_metadata: InputMetadata,
    ):
        qkv = self.qkv_proj(hidden_states)

        if input_metadata.is_prompt:
            # without rotary embedding
            fused_attention.apply_update_kv_cache(
                qkv,
                input_metadata.context_lens,
                input_metadata.padding_offsets,
                input_metadata.block_tables[self.layer_idx],
                self.num_heads, # q heads
                self.num_heads, # kv heads
                input_metadata.max_seq_len,
                64, # tokens_per_block
                
                self.hidden_size
                * (1 if self.use_int8 else 2)
                // (2 if self.kv_cache_config["INT4_ENABLED"] else 1),
                self.head_dim, # size_per_token = hidden_size * sizeof(dtype)

                self.kv_cache_config["INT4_ENABLED"], # int4_kv
                self.kv_cache_config["ZEROS_ENABLED"], # kv_cache_with_zeros
            )

            q, k, v = qkv.split(
                [self.hidden_size, self.hidden_size, self.hidden_size], dim=-1
            )
            q = q.reshape(q.size(0), self.num_heads, self.head_dim)
            k = k.reshape(k.size(0), self.num_heads, self.head_dim)
            v = v.reshape(v.size(0), self.num_heads, self.head_dim)

            attn_output = flash_attn_varlen_func(
                q,
                k,
                v,
                cu_seqlens_q=input_metadata.cu_seqlens,
                cu_seqlens_k=input_metadata.cu_seqlens,
                max_seqlen_q=input_metadata.max_seq_len,
                max_seqlen_k=input_metadata.max_seq_len,
                dropout_p=0.0, # when doing evaluation, should be 0.0
                causal=True,
            )
            attn_output = attn_output.reshape(q.size(0), -1)
        else:
            q, k, v = qkv.split(
                [self.hidden_size, self.hidden_size, self.hidden_size], dim=-1
            )
            q = q.reshape(q.size(0), self.num_heads, self.head_dim)
            k = k.reshape(k.size(0), self.num_heads, self.head_dim)
            v = v.reshape(v.size(0), self.num_heads, self.head_dim)

            kv_pointers = input_metadata.block_tables[self.layer_idx]
            lengths_per_sample = input_metadata.context_lens
            alibi_slopes = None
            memory_max_len = self.kv_max_seq_len
            tokens_per_block = 64
            
            attn_output = fused_attention.normal_single_query_attention(
                q,
                k,
                v,
                kv_pointers,
                input_metadata.context_lens, # lengths_per_sample
                None, # alibi_slopes
                self.kv_max_seq_len,
                64, # tokens_per_block
                
                self.hidden_size
                * (1 if self.use_int8 else 2)
                // (2 if self.kv_cache_config["INT4_ENABLED"] else 1),
                self.head_dim, # size_per_token = hidden_size * sizeof(dtype)

                input_metadata.max_seq_len, # timestep
                self.kv_cache_config["INT4_ENABLED"], # int4_kv
                self.kv_cache_config["ZEROS_ENABLED"], # kv_cache_with_zeros
            )

            attn_output = attn_output.reshape(q.size(0), -1)

        output = self.o_proj(attn_output)
        return output


class OPTDecoderLayer(nn.Module):
    def __init__(
            self,
            config: OPTConfig,
            layer_idx: int,
            kv_cache_config: Optional[Dict] = None,
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.use_int8 = True
        self.self_attn = OPTAttention(
            config,
            layer_idx=layer_idx,
            kv_cache_config=kv_cache_config,
        )
        self.mlp = OPTMLP(config)

        self.input_layernorm = nn.LayerNorm(
            self.hidden_size, elementwise_affine=config.layer_norm_elementwise_affine
        )

        self.post_attention_layernorm = nn.LayerNorm(
            self.hidden_size, elementwise_affine=config.layer_norm_elementwise_affine
        )

    def forward(
            self,
            hidden_states: torch.Tensor,
            input_metadata: InputMetadata,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        # Self Attention
        hidden_states = self.self_attn(hidden_states, input_metadata)
        hidden_states = residual + hidden_states
        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states
    
class OPTModel(nn.Module):
    def __init__(
            self,
            config: OPTConfig,
            quant_kv_cache: bool = True,
            kv_cache_config: Optional[Dict] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.kv_length = 0
        self.dropout = config.dropout
        self.layerdrop = config.layerdrop
        self.padding_idx = config.pad_token_id
        self.max_target_positions = config.max_position_embeddings
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.word_embed_proj_dim, self.padding_idx
        )
        self.embed_positions = OPTLearnedPositionalEmbedding(config.max_position_embeddings, config.hidden_size)

        if config.word_embed_proj_dim != config.hidden_size:
            self.project_out = nn.Linear(config.hidden_size, config.word_embed_proj_dim, bias=False)
        else:
            self.project_out = None

        if config.word_embed_proj_dim != config.hidden_size:
            self.project_in = nn.Linear(config.word_embed_proj_dim, config.hidden_size, bias=False)
        else:
            self.project_in = None

        if config.do_layer_norm_before and not config._remove_final_layer_norm:
            self.final_layer_norm = nn.LayerNorm(
                config.hidden_size, elementwise_affine=config.layer_norm_elementwise_affine
            )
        else:
            self.final_layer_norm = None

        self.layers = nn.ModuleList(
            [
                (
                    OPTDecoderLayer(config, i, kv_cache_config)
                    if quant_kv_cache
                    else None
                )
                for i in range(config.num_hidden_layers)
            ]
        )
    
    def forward(
            self,
            input_ids: torch.Tensor,
            input_metadata: InputMetadata,
            inputs_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        with torch.no_grad():
            
            # retrieve input_ids and inputs_embeds
            if input_ids is not None and inputs_embeds is not None:
                raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
            elif input_ids is not None:
                input_shape = input_ids.size()
                input_ids = input_ids.view(-1, input_shape[-1])
            elif inputs_embeds is not None:
                input_shape = inputs_embeds.size()[:-1]
            else:
                raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

            if inputs_embeds is None:
                inputs_embeds = self.embed_tokens(input_ids)

            batch_size, seq_length = input_shape
            self.kv_length += seq_length

            # construct attention mask
            attention_mask = torch.ones((batch_size, self.kv_length), device=inputs_embeds.device)
            
            pos_embeds = self.embed_positions(attention_mask, self.kv_length - seq_length)

            if self.project_in is not None:
                inputs_embeds = self.project_in(inputs_embeds)
            
            hidden_states = inputs_embeds + pos_embeds
            
            for i in range(len(self.layers)):
                layer = self.layers[i]
                hidden_states = layer(hidden_states, input_metadata)
            
            if self.final_layer_norm is not None:
                hidden_states = self.final_layer_norm(hidden_states)

            if self.project_out is not None:
                hidden_states = self.project_out(hidden_states)

            return hidden_states

class OPTForCausalLM(nn.Module):
    def __init__(
            self,
            config: OPTConfig,
            sampling_params: SamplingParams,
            quant_config: Optional[QServeQuantConfig] = QServeQuantConfig(weight_bits=4),
            kv_cache_config: Optional[Dict] = None,
            quant_path: Optional[str] = None,
    ) -> None:
        quant_kv_cache = True
        super().__init__()
        self.config = config
        self.quant_config = quant_config
        self.model = OPTModel(
            config, quant_kv_cache, kv_cache_config=kv_cache_config
        )
        self.lm_head = nn.Linear(config.word_embed_proj_dim, config.vocab_size, bias=False)
        self._column_parallel_layers = []
        self._row_parallel_layers = ["o_proj", "down_proj"]
        self.sampler = Sampler(sampling_params)

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        
        if quant_path is not None:
            self.load_weights(quant_path)

    def forward(
            self,
            input_ids: torch.Tensor,
            input_metadata: InputMetadata,
            inputs_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        hidden_states = self.model(input_ids, input_metadata, inputs_embeds)
        if input_metadata.is_prompt:
            output = self.lm_head(
                hidden_states[input_metadata.cu_seqlens[1:] - 1, :]
            )
        else:
            output = self.lm_head(hidden_states)
        return output
    
    def sample(
        self,
        input_ids: torch.Tensor,
        logits: torch.Tensor,
        input_metadata: InputMetadata,
    ):
        return self.sampler(input_ids, logits, input_metadata)
    
    def load_weights(
        self,
        model_name_or_path: str,
        cache_dir: Optional[str] = None,
        load_format: str = "auto",
        revision: Optional[str] = None,
    ):
        if self.quant_config is None:
            col_weight_suffixes = ["weight"]
            row_weight_suffixes = ["weight"]
        else:
            col_weight_suffixes = self.quant_config.get_col_parallel_tensor_names()
            row_weight_suffixes = self.quant_config.get_row_parallel_tensor_names()

        column_parallel_weights: List[str] = []
        for layer in self._column_parallel_layers:
            for suffix in col_weight_suffixes:
                column_parallel_weights.append(f"{layer}.{suffix}")
        row_parallel_weights: List[str] = []
        for layer in self._row_parallel_layers:
            for suffix in row_weight_suffixes:
                row_parallel_weights.append(f"{layer}.{suffix}")

        # TODO fix the tp parallelism
        # tp_size = get_tensor_model_parallel_world_size()
        # tp_rank = get_tensor_model_parallel_rank()
        tp_size = 1
        tp_rank = 0

        q_proj_shard_size = self.config.hidden_size // tp_size
        num_kv_heads_replicas = max(1, tp_size // self.config.num_key_value_heads)
        num_kv_heads_per_gpu = max(1, self.config.num_key_value_heads // tp_size)
        kv_proj_shard_size = (
            self.config.hidden_size
            // self.config.num_attention_heads
            * num_kv_heads_per_gpu
        )
        attention_weight_specs = [
            # (weight_name, shard_size, offset)
            ("q_proj", q_proj_shard_size, 0),
            ("k_proj", kv_proj_shard_size, q_proj_shard_size),
            ("v_proj", kv_proj_shard_size, q_proj_shard_size + kv_proj_shard_size),
        ]
        state_dict = self.state_dict()

        for name, loaded_weight in hf_model_weights_iterator(
            model_name_or_path, cache_dir, load_format, revision
        ):
            if "rotary_emb.inv_freq" in name:
                continue
            # bias is useless for llama
            if "bias" in name:
                pass
                # continue
            # if "norm" in name:
            #     continue

            packed_dim = None
            is_transposed = False
            if self.quant_config is not None:
                packed_dim = self.quant_config.get_packed_dim(name)
                is_transposed = self.quant_config.is_transposed(name)
            if is_transposed:
                loaded_weight = convert_pyslice_to_tensor(loaded_weight)
                loaded_weight = loaded_weight.T

            is_attention_weight = False
            for weight_name, shard_size, offset in attention_weight_specs:
                if weight_name not in name:
                    continue
                # print(weight_name)
                param = state_dict[name.replace(weight_name, "qkv_proj")]
                if is_transposed:
                    param = param.T

                if packed_dim is not None:
                    shard_dim = 0 if not is_transposed else 1
                    if packed_dim == shard_dim:
                        shard_size //= self.quant_config.pack_factor
                        offset //= self.quant_config.pack_factor

                if weight_name in ["k_proj", "v_proj"]:
                    shard_id = tp_rank // num_kv_heads_replicas
                else:
                    shard_id = tp_rank
                loaded_weight = loaded_weight[
                    shard_size * shard_id : shard_size * (shard_id + 1)
                ]
                if "s2_scales" in name or "s2_zeros" in name:
                    param_slice = param.data[:, offset : offset + shard_size]
                else:
                    param_slice = param.data[offset : offset + shard_size]
                assert param_slice.shape == loaded_weight.shape

                param_slice.copy_(loaded_weight)
                is_attention_weight = True
                break
            if is_attention_weight:
                continue

            is_gate_up_weight = False
            for stride_id, weight_name in enumerate(["gate_proj", "up_proj"]):
                if weight_name not in name:
                    continue
                param = state_dict[name.replace(weight_name, "gate_up_proj")]
                if is_transposed:
                    param = param.T

                if "s2_scales" in name or "s2_zeros" in name:
                    shard_size = param.shape[1] // 2
                    param_slice = param.data[
                        :, stride_id * shard_size : (stride_id + 1) * shard_size
                    ]
                else:
                    shard_size = param.shape[0] // 2
                    param_slice = param.data[
                        shard_size * stride_id : shard_size * (stride_id + 1)
                    ]
                loaded_weight = loaded_weight[
                    shard_size * tp_rank : shard_size * (tp_rank + 1)
                ]
                assert param_slice.shape == loaded_weight.shape
                param_slice.copy_(loaded_weight)
                is_gate_up_weight = True
                break
            if is_gate_up_weight:
                continue

            param = state_dict[name]
            if is_transposed:
                param = param.T

            if "embed_tokens" in name or "lm_head" in name:
                load_padded_tensor_parallel_vocab(param, loaded_weight, tp_rank)
                continue

            load_tensor_parallel_weights(
                param,
                loaded_weight,
                name,
                column_parallel_weights,
                row_parallel_weights,
                tp_rank,
            )

# 参考huggingface的class OPTDecoder
# huggingface OPTModel是基于OPTDecoder的封装，额外封装了一些对于我们的实验没用的功能
