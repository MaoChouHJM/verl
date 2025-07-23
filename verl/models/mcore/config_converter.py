# Copyright 2025 Bytedance Ltd. and/or its affiliates
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# convert huggingface config to mcore transformer config


from unittest import removeResult
import torch
import torch.nn.functional as F
from megatron.core import parallel_state as mpu
from megatron.core.transformer import MLATransformerConfig, TransformerConfig
from transformers import PretrainedConfig


def _get_base_transformer_config(
    hf_config: PretrainedConfig, dtype: torch.dtype, **override_transformer_config_kwargs
) -> dict:
    """
    Create a base TransformerConfig with common parameters across different model architectures.
    TODO: (ycl) use dataclass or converter config?

    Args:
        hf_config: HuggingFace model configuration
        dtype: Data type for the model
        override_transformer_config_kwargs: Additional parameters to override defaults

    Returns:
        TransformerConfig with common parameters
    """

    # Common parallel state parameters
    overlap_p2p_comm = (
        mpu.get_virtual_pipeline_model_parallel_world_size() is not None
        and mpu.get_virtual_pipeline_model_parallel_world_size() > 1
    )
    batch_p2p_comm = False
    

    # Base configuration with common parameters
    base_config = {
        # Model architecture parameters
        "num_layers": hf_config.num_hidden_layers,
        "hidden_size": hf_config.hidden_size,
        "num_attention_heads": hf_config.num_attention_heads,
        "num_query_groups": hf_config.num_key_value_heads,
        "ffn_hidden_size": hf_config.intermediate_size,
        "attention_dropout": hf_config.attention_dropout,
        "hidden_dropout": getattr(hf_config, "hidden_dropout", 0.0),
        "kv_channels": getattr(hf_config, "head_dim", None),
        "layernorm_epsilon": hf_config.rms_norm_eps,
        "add_bias_linear": True,
        # Activation and normalization
        "activation_func": F.silu,
        "normalization": "RMSNorm",
        "gated_linear_unit": True,
        # Data types
        "pipeline_dtype": dtype,
        "params_dtype": dtype,
        "bf16": dtype is torch.bfloat16,
        # Parallel configuration
        "tensor_model_parallel_size": mpu.get_tensor_model_parallel_world_size(),
        "pipeline_model_parallel_size": mpu.get_pipeline_model_parallel_world_size(),
        "expert_model_parallel_size": mpu.get_expert_model_parallel_world_size(),
        "expert_tensor_parallel_size": mpu.get_expert_tensor_parallel_world_size(),
        "virtual_pipeline_model_parallel_size": mpu.get_virtual_pipeline_model_parallel_world_size(),
        "context_parallel_size": mpu.get_context_parallel_world_size(),
        "overlap_p2p_comm": overlap_p2p_comm,
        "batch_p2p_comm": batch_p2p_comm,
        "sequence_parallel": mpu.get_tensor_model_parallel_world_size() > 1,
        # Common settings
        "variable_seq_lengths": True,
        "masked_softmax_fusion": True,
        "moe_token_dispatcher_type": "alltoall",
    }

    # Update with any provided overrides
    # override_transformer_config_kwargs as kwargs shall never be none
    base_config.update(override_transformer_config_kwargs)

    return base_config


def _get_mla_transformer_config(
    hf_config: PretrainedConfig, mla_rope_config: dict, dtype: torch.dtype, **override_transformer_config_kwargs
) -> dict:
    """
    Create a MLATransformerConfig with common parameters across different model architectures.
    This is specifically for MLA models like DeepseekV3.

    Args:
        hf_config: HuggingFace model configuration
        mla_rope_config: MLA specific RoPE configuration
        dtype: Data type for the model
        override_transformer_config_kwargs: Additional parameters to override defaults

    Returns:
        MLATransformerConfig with common parameters
    """
    base_config = _get_base_transformer_config(hf_config=hf_config, dtype=dtype, **override_transformer_config_kwargs)
    mla_config = {
        # MLA specific parameters
        "q_lora_rank": hf_config.q_lora_rank,
        "kv_lora_rank": hf_config.kv_lora_rank,
        "qk_head_dim": hf_config.qk_nope_head_dim,
        "qk_pos_emb_head_dim": hf_config.qk_rope_head_dim,
        "v_head_dim": hf_config.v_head_dim,
        "rotary_base": hf_config.rope_theta,
        "rotary_scaling_factor": mla_rope_config["factor"],
        "rope_type": mla_rope_config["type"],
        "max_position_embeddings": mla_rope_config["original_max_position_embeddings"],
        "beta_fast": mla_rope_config["beta_fast"],
        "beta_slow": mla_rope_config["beta_slow"],
        "mscale": mla_rope_config["mscale"],
        "mscale_all_dim": mla_rope_config["mscale_all_dim"],
    }

    base_config.update(mla_config)
    return base_config


def hf_to_mcore_config_dense(
    hf_config: PretrainedConfig, dtype: torch.dtype, **override_transformer_config_kwargs
) -> TransformerConfig:
    # for LlamaForCausalLM or Qwen2ForCausalLM
    qkv_bias = True if "Qwen2ForCausalLM" in hf_config.architectures else getattr(hf_config, "attention_bias", False)
    qk_layernorm = True if "Qwen3ForCausalLM" in hf_config.architectures else False

    args: dict = _get_base_transformer_config(
        hf_config=hf_config,
        dtype=dtype,
        use_cpu_initialization=False,
        add_bias_linear=False,
        add_qkv_bias=qkv_bias,
        qk_layernorm=qk_layernorm,
    )
    # override_transformer_config_kwargs as kwargs shall never be none
    args.update(override_transformer_config_kwargs)
    print(f"Overridden TF init config: {args}")
    return TransformerConfig(**args)


def hf_to_mcore_config_qwen2moe(
    hf_config: PretrainedConfig, dtype: torch.dtype, **override_transformer_config_kwargs
) -> TransformerConfig:
    args: dict = _get_base_transformer_config(
        hf_config=hf_config,
        dtype=dtype,
        use_cpu_initialization=False,
        add_bias_linear=False,
        layernorm_epsilon=hf_config.rms_norm_eps,
        # MoE specific
        moe_ffn_hidden_size=hf_config.moe_intermediate_size,
        moe_router_bias_update_rate=0.001,
        moe_router_topk=hf_config.num_experts_per_tok,
        num_moe_experts=hf_config.num_experts,
        moe_shared_expert_intermediate_size=hf_config.shared_expert_intermediate_size,
        moe_aux_loss_coeff=hf_config.router_aux_loss_coef,
        # moe_aux_loss_coeff=0.0,
        moe_router_load_balancing_type="none",  # turn off aux_loss as it hurts perf in RL
        moe_shared_expert_overlap=True,
        moe_grouped_gemm=True,
        moe_router_score_function="softmax",
        # Other optimizations
        persist_layer_norm=True,
        bias_activation_fusion=True,
        bias_dropout_fusion=True,
        # Qwen specific
        moe_router_pre_softmax=True,
        add_qkv_bias=True,
    )
    # override_transformer_config_kwargs as kwargs shall never be none
    args.update(override_transformer_config_kwargs)
    print(f"Overridden TF init config: {args}")
    return TransformerConfig(**args)


def hf_to_mcore_config_mixtral(
    hf_config: PretrainedConfig, dtype: torch.dtype, **override_transformer_config_kwargs
) -> TransformerConfig:
    args: dict = _get_base_transformer_config(
        hf_config=hf_config,
        dtype=dtype,
        use_cpu_initialization=False,
        add_bias_linear=False,
        layernorm_epsilon=hf_config.rms_norm_eps,
        # MoE specific
        num_moe_experts=hf_config.num_local_experts,
        moe_aux_loss_coeff=hf_config.router_aux_loss_coef,
        moe_router_topk=hf_config.num_experts_per_tok,
        moe_router_pre_softmax=True,
        moe_router_load_balancing_type="none",  # turn off aux_loss as it hurts perf in RL
        moe_router_score_function="softmax",
        moe_shared_expert_intermediate_size=None,  # mixtral has no shared expert
        moe_shared_expert_overlap=False,  # mixtral has no shared expert
        moe_ffn_hidden_size=hf_config.intermediate_size,
        moe_router_bias_update_rate=0.001,
        moe_permute_fusion=True, # need TE 2.1+
        moe_grouped_gemm=True,
        # Other optimizations
        persist_layer_norm=True,
        apply_rope_fusion=True,
        bias_activation_fusion=True,
        bias_dropout_fusion=True,
    )
    # override_transformer_config_kwargs as kwargs shall never be none
    args.update(override_transformer_config_kwargs)
    print(f"Overridden TF init config: {args}")
    return TransformerConfig(**args)


def hf_to_mcore_config_qwen3moe(
    hf_config: PretrainedConfig, dtype: torch.dtype, **override_transformer_config_kwargs
) -> TransformerConfig:
    args: dict = _get_base_transformer_config(
        hf_config=hf_config,
        dtype=dtype,
        use_cpu_initialization=False,
        add_bias_linear=False,
        layernorm_epsilon=hf_config.rms_norm_eps,
        # MoE specific
        moe_ffn_hidden_size=hf_config.moe_intermediate_size,
        moe_router_bias_update_rate=0.001,
        moe_router_topk=hf_config.num_experts_per_tok,
        num_moe_experts=hf_config.num_experts,
        moe_aux_loss_coeff=hf_config.router_aux_loss_coef,
        # moe_aux_loss_coeff=0.0,
        moe_router_load_balancing_type="none",  # turn off aux_loss as it hurts perf in RL
        moe_grouped_gemm=True,
        moe_router_score_function="softmax",
        # Other optimizations
        persist_layer_norm=True,
        bias_activation_fusion=True,
        bias_dropout_fusion=True,
        # Qwen specific
        moe_router_pre_softmax=False,
        qk_layernorm=True,
    )
    # override_transformer_config_kwargs as kwargs shall never be none
    args.update(override_transformer_config_kwargs)
    print(f"Overridden TF init config: {args}")
    return TransformerConfig(**args)


def hf_to_mcore_config_dpskv3(
    hf_config: PretrainedConfig, dtype: torch.dtype, **override_transformer_config_kwargs
) -> MLATransformerConfig:
    # DeepseekV3ForCausalLM
    from megatron.core.transformer.enums import AttnBackend

    #from .patch_v012 import apply_patch

    #apply_patch()

    mla_rope_config = {
        "beta_fast": 32,
        "beta_slow": 1,
        "factor": 1,
        "mscale": 1.0,
        "mscale_all_dim": 1.0,
        "original_max_position_embeddings": 4096,
        "type": "rope",
    }
    if "rope_scaling" in hf_config and hf_config.rope_scaling is not None:
        mla_rope_config.update(hf_config.rope_scaling)
    #moe_layer_freq = [1] * hf_config.num_hidden_layers
    #for i in range(min(hf_config.first_k_dense_replace, hf_config.num_hidden_layers)):
    #    moe_layer_freq[i] = 0

    # disable MTP and quantization for now
    if "num_nextn_predict_layers" in hf_config:
        assert hf_config.num_nextn_predict_layers == 0, "MTP is not supported for now, please modify the config.json to set num_nextn_predict_layers to 0"
    #assert "quantization_config" not in hf_config or not hf_config.quantization_config, "quantization is not supported for now, please modify the config.json to remove quantization_config"

    args: dict = _get_mla_transformer_config(
        hf_config=hf_config,
        mla_rope_config=mla_rope_config,
        dtype=dtype,
        # Additional parameters
        use_cpu_initialization=False,
        add_bias_linear=False,
        attention_backend=AttnBackend.fused,
        qk_layernorm=True,
        # Standard MoE parameters
        moe_ffn_hidden_size=hf_config.moe_intermediate_size,
        moe_token_dispatcher_type="flex",
        moe_router_bias_update_rate=0.001,
        moe_router_enable_expert_bias=True,
        moe_router_topk=hf_config.num_experts_per_tok,
        num_moe_experts=hf_config.n_routed_experts,
        moe_shared_expert_intermediate_size=hf_config.moe_intermediate_size * hf_config.n_shared_experts,
        moe_aux_loss_coeff=getattr(hf_config, "aux_loss_alpha", 0.0001),
        moe_router_load_balancing_type="seq_aux_loss",
        # moe_permute_fusion=True, # need TE 2.1+
        moe_grouped_gemm=True,
        moe_router_score_function="sigmoid",
        moe_router_pre_softmax=True,
        moe_router_topk_scaling_factor=hf_config.routed_scaling_factor,
        #moe_layer_freq=moe_layer_freq,
        # mcore 0.12 moe
        moe_router_dtype="fp32",
        # Other optimizations
        # deallocate_pipeline_outputs=True,
        # gradient_accumulation_fusion=True,
        persist_layer_norm=True,
        bias_activation_fusion=True,
        bias_dropout_fusion=True,
        apply_rope_fusion=True, # False when 0.12.1
        async_tensor_model_parallel_allreduce=True,
        attention_softmax_in_fp32=False,
        batch_p2p_comm=False,
        cp_comm_type="p2p",
        cross_entropy_fusion_impl="te",
        cross_entropy_loss_fusion=True,
        disable_bf16_reduced_precision_matmul=False,
        distribute_saved_activations=False,
        gradient_accumulation_fusion=True,
        moe_permute_fusion=True,
        moe_router_force_load_balancing=True,
        overlap_p2p_comm=True,
        **override_transformer_config_kwargs,
    )

    def core_transformer_config_from_args(args):
        # Config class.
        config_class = MLATransformerConfig
        # Translate args to core transformer configuration
        kw_args = {}
        import dataclasses
        for f in dataclasses.fields(config_class):
            if f.name in args:
                kw_args[f.name] = args[f.name]

        if 'no_persist_layer_norm' in args:
            kw_args['persist_layer_norm'] = not args['no_persist_layer_norm']

        if 'apply_layernorm_1p' in args:
            kw_args['layernorm_zero_centered_gamma'] = args['apply_layernorm_1p']

        if 'norm_epsilon' in args:
            kw_args['layernorm_epsilon'] = args['norm_epsilon']

        kw_args['deallocate_pipeline_outputs'] = True

        if 'params_dtype' in args:
            kw_args['pipeline_dtype'] = args['params_dtype']

        if 'overlap_p2p_comm' in args:
            kw_args['batch_p2p_comm'] = not args['overlap_p2p_comm']

        if 'num_experts' in args:
            kw_args['num_moe_experts'] = args['num_experts']

        if 'rotary_interleaved' in args:
            kw_args['rotary_interleaved'] = args['rotary_interleaved']

        if 'decoder_first_pipeline_num_layers' in args:
            kw_args['num_layers_in_first_pipeline_stage']= args['decoder_first_pipeline_num_layers']

        if 'decoder_last_pipeline_num_layers' in args:
            kw_args['num_layers_in_last_pipeline_stage']= args['decoder_last_pipeline_num_layers']

        if 'fp8_param_gather' in args:
            kw_args['fp8_param'] = args['fp8_param_gather']

        if 'swiglu' in args and args['swiglu']:
            kw_args['activation_func'] = F.silu
            kw_args['gated_linear_unit'] = True
            if 'bias_swiglu_fusion' in args:
                kw_args['bias_activation_fusion'] = args['bias_swiglu_fusion']
        else:
            if 'bias_gelu_fusion' in args:
                kw_args['bias_activation_fusion'] = args['bias_gelu_fusion']

        if 'squared_relu' in args and args['squared_relu']:
            if 'swiglu' in args:
                assert not args['swiglu']
            kw_args['activation_func'] = squared_relu

        if 'init_method_xavier_uniform' in args and args['init_method_xavier_uniform']:
            kw_args['init_method'] = torch.nn.init.xavier_uniform_
            kw_args['scaled_init_method'] = torch.nn.init.xavier_uniform_

        if 'group_query_attention' in args and args['group_query_attention']:
            if 'num_query_groups' in args:
                kw_args['num_query_groups'] = args['num_query_groups']
        else:
            kw_args['num_query_groups'] = None

        if 'config_logger_dir' in args:
            kw_args['config_logger_dir'] = args['config_logger_dir']

        if 'cp_comm_type' in args and len(args['cp_comm_type']) == 1:
            kw_args['cp_comm_type'] = args['cp_comm_type'][0]

        if 'is_hybrid_model' in args and args['is_hybrid_model']:
            kw_args['is_hybrid_model'] = args['is_hybrid_model']

        # Return config.
        return kw_args

    transformer_config = MLATransformerConfig(**core_transformer_config_from_args(args))
    import dataclasses, json
    config_dict = dataclasses.asdict(transformer_config)
    json_str = json.dumps(config_dict, indent=4, default=lambda o: str(o), sort_keys=True)
    from verl.utils.logger.aggregate_logger import print_rank_0
    print_rank_0(f"Overridden MLA TF init config: {json_str}")
    # MTP
    if "num_nextn_predict_layers" in hf_config:
        transformer_config.mtp_num_layers = hf_config.num_nextn_predict_layers
        if transformer_config.mtp_num_layers == 0:
            transformer_config.mtp_num_layers = None
        transformer_config.mtp_loss_scaling_factor = 0.1

    return transformer_config


def hf_to_mcore_config_qwen2_5_vl(
    hf_config: PretrainedConfig, dtype: torch.dtype, **override_transformer_config_kwargs
) -> TransformerConfig:
    # Qwen2_5_VLForConditionalGeneration

    args = _get_base_transformer_config(
        hf_config=hf_config,
        dtype=dtype,
        add_bias_linear=False,
        # qwen specific
        add_qkv_bias=True,
        mrope_section=hf_config.rope_scaling["mrope_section"],
    )
    # override_transformer_config_kwargs as kwargs shall never be none
    args.update(override_transformer_config_kwargs)
    print(f"Overridden TF init config: {args}")
    return TransformerConfig(**args)


def hf_to_mcore_config_llama4(
    hf_config: PretrainedConfig, dtype: torch.dtype, **override_transformer_config_kwargs
) -> TransformerConfig:
    # Llama4ForConditionalGeneration
    raise NotImplementedError("Llama4ForConditionalGeneration is not supported yet")


def hf_to_mcore_config_keye_qwen3_slowfast(
    hf_config: PretrainedConfig, dtype: torch.dtype, **override_transformer_config_kwargs
) -> TransformerConfig:
    return hf_to_mcore_config_dpskv3(hf_config,dtype, **override_transformer_config_kwargs)
