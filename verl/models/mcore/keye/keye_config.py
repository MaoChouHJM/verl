import torch
from typing import List
from dataclasses import dataclass, fields
from megatron.core.transformer import TransformerConfig
from megatron.core.transformer.pipeline_parallel_layer_layout import PipelineParallelLayerLayout
from megatron.training.activations import quick_gelu
from megatron.core import parallel_state


@dataclass
class VisionTransformerConfig(TransformerConfig):
    image_size: int = 384
    patch_size: int = 14
    in_channels: int = 3
    spatial_merge_size: int = 2
    tokens_per_second: int = 2
    temporal_patch_size: int = 2
    post_layer_norm: bool = True
    interpolate_pos_encoding: bool = True
    image_token_id: int = 151655
    video_token_id: int = 151656
    fast_video_token_id: int = 151678
    
    def __init__(self, base: TransformerConfig):
        config = {f.name : getattr(base, f.name) for f in fields(TransformerConfig)}
        super().__init__(**config)

def get_vision_model_config(args, base: TransformerConfig) -> VisionTransformerConfig:
    config = VisionTransformerConfig(base)
    config.hidden_size = 1152
    config.ffn_hidden_size = 4304
    config.num_layers = 27
    config.num_attention_heads = 16
    config.add_bias_linear = True
    config.add_qkv_bias = True
    config.hidden_dropout = 0.0
    config.attention_dropout = 0.0
    
    # config.gated_linear_unit = False # no gated
    # config.activation_func = quick_gelu # hidden_act
    config.kv_channels = config.hidden_size // config.num_attention_heads
    config.num_query_groups = config.num_attention_heads # no GQA
    config.layernorm_zero_centered_gamma = False # False
    config.apply_query_key_layer_scaling = False # factor=math.sqrt(head_dim)
    config.bias_activation_fusion = False # no swiglu, set false
    config.bias_dropout_fusion = False # no dropout, set false
    config.attention_softmax_in_fp32 = True # use True
    config.normalization = 'LayerNorm' # use RMSNorm

    # parallel setting
    config.pipeline_model_parallel_size = 1
    config.num_layers_in_first_pipeline_stage = None
    config.pipeline_model_parallel_layout = PipelineParallelLayerLayout.from_str(
        "E" + "t" * config.num_layers,
        config.pipeline_model_parallel_size
    )
    config.tensor_model_parallel_size = 1
    config.expert_model_parallel_size = 1
    config.context_parallel_size = 1
    config.sequence_parallel = False
    config.tp_comm_overlap = False
    config.gated_linear_unit =False
    
    return config


def get_vision_projection_config(config, embed_dim, spatial_merge_size):
    # merger: 
    # context_dim = hidden_size * merge_size**2
    # out_hidden_size = hidden_size
    # context_dim -> context_dim -> out_hidden_size
    # MLP: 
    # input_size -> ffn_hidden_size -> hidden_size
    # spec: LN -> Linear(bias=True) -> GELU -> Linear(bias=True)
    config.gated_linear_unit = False
    config.bias_activation_fusion = False
    config.add_bias_linear = True
    config.ffn_hidden_size = embed_dim * (spatial_merge_size ** 2)
    config.activation_func = torch.nn.functional.gelu
    config.tp_comm_overlap = False
    config.sequence_parallel = False
    return config
