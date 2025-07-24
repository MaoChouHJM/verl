from collections import OrderedDict, namedtuple
from typing import Dict, Literal, Optional, Union, List, Tuple
import logging
import numpy as np
import functools
import dataclasses

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from einops import rearrange

from megatron.core import tensor_parallel
from megatron.core.inference.contexts import BaseInferenceContext
from megatron.core.inference_params import InferenceParams
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer import MegatronModule, TransformerConfig, ModuleSpec
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.tensor_parallel import (
    gather_from_sequence_parallel_region,
    scatter_to_sequence_parallel_region,
)
from megatron.core.models.gpt import GPTModel
from megatron.core.utils import divide
from megatron.training import get_args
import megatron.core.parallel_state as mpu

import transformer_engine.pytorch as te

from .keye_config import VisionTransformerConfig
from .ulysses_parallel import (
    get_local_sequence,
    get_local_sequence_boundary,
    UlyssesAttention,
    AllGather
)


from flash_attn.layers.rotary import apply_rotary_emb
class SiglipVisionEmbeddings(nn.Module):
    def __init__(self, config: VisionTransformerConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.patch_embedding = nn.Conv2d(
            in_channels=config.in_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            padding="valid",
        )

        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)
        self.packing_position_embedding = nn.Embedding(32768, self.embed_dim)

    def interpolate_pos_encoding(self, embeddings: torch.Tensor, height: int, width: int, is_after_patchify: bool = False) -> torch.Tensor:
        """
        This method allows to interpolate the pre-trained position encodings, to be able to use the model on higher resolution
        images. This method is also adapted to support torch.jit tracing and no class embeddings.

        Adapted from:
        - https://github.com/facebookresearch/dino/blob/de9ee3df6cf39fac952ab558447af1fa1365362a/vision_transformer.py#L174-L194, and
        - https://github.com/facebookresearch/dinov2/blob/e1277af2ba9496fbadf7aec6eba56e8d882d1e35/dinov2/models/vision_transformer.py#L179-L211
        """

        num_patches = embeddings.shape[1]
        num_positions = self.position_embedding.weight.shape[0]

        # always interpolate when tracing to ensure the exported model works for dynamic input shapes
        if not torch.jit.is_tracing() and num_patches == num_positions and height == width:
            return self.position_embedding(self.position_ids)

        patch_pos_embed = self.position_embedding.weight.unsqueeze(0)

        dim = embeddings.shape[-1]

        if is_after_patchify:
            new_height = height
            new_width = width
        else:
            new_height = height // self.patch_size
            new_width = width // self.patch_size

        sqrt_num_positions = int(num_positions**0.5)
        patch_pos_embed = patch_pos_embed.reshape(1, sqrt_num_positions, sqrt_num_positions, dim)
        patch_pos_embed = patch_pos_embed.permute(0, 3, 1, 2)

        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed,
            size=(new_height, new_width),
            mode="bilinear",
            align_corners=False,
        )

        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return patch_pos_embed

    def flatten_list(self, image_grid_thw: torch.Tensor) -> List[List[int]]:
        tmp_image_grid_thw = list()
        for image_grid in image_grid_thw.tolist():
            if isinstance(image_grid, list):
                tmp_image_grid_thw.extend(image_grid)
            else:
                tmp_image_grid_thw.append(image_grid)
        return tmp_image_grid_thw

    def forward(
        self,
        pixel_values: torch.FloatTensor,
        position_ids: torch.Tensor,
        image_grid_thw: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        assert pixel_values.dim() == 5 and self.config.interpolate_pos_encoding
        assert position_ids is not None

        batch_size, squence_len, channel, height, width = pixel_values.shape
        target_dtype = self.patch_embedding.weight.dtype
        pixel_values = rearrange(pixel_values, "b l c h w -> (b l) c h w")
        patch_embeds = self.patch_embedding(pixel_values.to(
            dtype=target_dtype))  # shape = [*, width, grid, grid]
        embeddings = patch_embeds.flatten(-2).squeeze(-1)
        embeddings = rearrange(embeddings, "(b l) d -> b l d", b=batch_size, l=squence_len)

        if image_grid_thw is not None:
            # flatten_image_grid_thw = self.flatten_list(image_grid_thw)
            # print(f"{flatten_image_grid_thw=}")
            assert batch_size == 1
            start = 0
            # assert sum([np.prod(x) for x in flatten_image_grid_thw]) == embeddings.shape[1], (flatten_image_grid_thw, embeddings.shape)
            embeddings = embeddings.squeeze(0)
            tmp_embeddings = list()
            for image_grid in image_grid_thw:
                t, h, w = image_grid
                for _ in range(t):
                    end = start + h * w
                    image_embeddings = embeddings[start: end, :]
                    image_embeddings = image_embeddings + \
                        self.interpolate_pos_encoding(image_embeddings, h, w, True).squeeze(0)
                    tmp_embeddings.append(image_embeddings)
                    start = end
            embeddings = torch.concat(tmp_embeddings, dim=0).unsqueeze(0)
        else:
            embeddings = embeddings + self.packing_position_embedding(position_ids)
        return embeddings

class SigLIPRotaryEmbedding(nn.Module):
    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        self.dim = dim
        self.theta = theta
        #self.rope_init()
        self.inv_freq = 1.0 / (self.theta ** (torch.arange(0, self.dim, 2, dtype=torch.float32,device=torch.cuda.current_device()) / self.dim))

    def rope_init(self):
        inv_freq = 1.0 / (self.theta ** (torch.arange(0, self.dim, 2, dtype=torch.float) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seqlen: int) -> torch.Tensor:
        seq = torch.arange(seqlen, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(seq, self.inv_freq)
        return freqs

def apply_rotary_pos_emb_flashatt(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    cos = cos.chunk(2, dim=-1)[0].contiguous()
    sin = sin.chunk(2, dim=-1)[0].contiguous()
    q_embed = apply_rotary_emb(q.float(), cos.float(), sin.float()).type_as(q)
    k_embed = apply_rotary_emb(k.float(), cos.float(), sin.float()).type_as(k)
    return q_embed, k_embed
    
class Projector(nn.Module):
    def __init__(self,
                 config: VisionTransformerConfig,
                 out_features: int):
        super().__init__()

        self.config = config
        self.out_features = out_features
        self.hidden_size = (
            config.hidden_size
            * config.spatial_merge_size
            * config.spatial_merge_size
        )
        self.merge_kernel_size = (2, 2)
        self.pre_norm = nn.LayerNorm(self.hidden_size, eps=1e-5)
        self.linear_1 = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_2 = nn.Linear(self.hidden_size, out_features, bias=True)

    def forward(self, hidden_states: torch.Tensor, packed_seq_params, thw):
        cu_seqlens = packed_seq_params.cu_seqlens_q
        m1, m2 = self.merge_kernel_size
        processed_features = list()
        for i in range(cu_seqlens.shape[0] - 1):
            start = cu_seqlens[i]
            end = cu_seqlens[i + 1]
            image_feature = hidden_states[:, start: end, :].squeeze(0)
            image_grid = thw[i]
            t, h, w = image_grid
            from einops import rearrange
            image_feature = rearrange(image_feature, "(t h p1 w p2) d -> (t h w) (p1 p2 d)", t=t, h=h // m1, p1=m1, w=w // m2, p2=m2)
            image_feature = self.pre_norm(image_feature)
            image_feature = self.linear_1(image_feature)
            image_feature = F.gelu(image_feature)
            image_feature = self.linear_2(image_feature)
            processed_features.append(image_feature)
        hidden_states = torch.cat(processed_features, dim = 0)

        return hidden_states



class SiglipMLP(nn.Module):
    def __init__(self, config: VisionTransformerConfig):
        super().__init__()
        self.config = config
        self.linear_fc1 = nn.Linear(
            in_features=config.hidden_size,
            out_features=config.ffn_hidden_size,
            bias = True
        )
        self.linear_fc2 = nn.Linear(
            in_features=config.ffn_hidden_size,
            out_features=config.hidden_size,
            bias = True
        )
        self.activation = F.gelu
        
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.linear_fc1(hidden_states)
        hidden_states = self.activation(hidden_states, approximate="tanh")
        hidden_states = self.linear_fc2(hidden_states)
        return hidden_states


class SiglipAttention(nn.Module):
    def __init__(self,
                 config: VisionTransformerConfig):
        super().__init__()
        self.config = config
        self.query_projection_size = config.kv_channels * config.num_attention_heads
        self.kv_projection_size = config.kv_channels * config.num_query_groups
        self.hidden_size_per_attention_head = divide(
            self.query_projection_size, config.num_attention_heads
        )
        self.key_hidden_size = self.hidden_size_per_attention_head
        self.val_hidden_size = self.hidden_size_per_attention_head

        self.checkpoint_core_attention = (
            self.config.recompute_granularity == 'selective'
        )
        self.attn_mask_type = AttnMaskType.padding
        self.kept_packed_seq_params = set(
            field.name for field in dataclasses.fields(PackedSeqParams)
        )

        self.scale = self.hidden_size_per_attention_head**-0.5
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size,bias = True)
        self.dist_attn = UlyssesAttention()

    def _checkpoint_attention_forward(
        self,
        query,
        key,
        value,
        attention_mask,
        rotary_pos_emb=None,
        attn_mask_type=None,
        packed_seq_params=None,
    ):
        def custom_forward(*inputs):
            query = inputs[0]
            key = inputs[1]
            value = inputs[2]
            attention_mask = inputs[3]
            attn_mask_type = inputs[5]
            attn_mask_type = AttnMaskType(attn_mask_type.item())
            packed_seq_kwargs = (
                {key: getattr(packed_seq_params, key) for key in self.kept_packed_seq_params}
                if packed_seq_params is not None
                else {}
            )
            output_ = self.core_attention(
                query,
                key,
                value,
                attention_mask,
                attn_mask_type=attn_mask_type.name,
                **packed_seq_kwargs,
            )
            return output_
        if attn_mask_type is None:
            attn_mask_type = self.attn_mask_type
        attn_mask_type = torch.tensor([attn_mask_type.value], dtype=torch.int)
        hidden_states = tensor_parallel.checkpoint(
            custom_forward, False, query, key, value, attention_mask, rotary_pos_emb, attn_mask_type
        )
        return hidden_states

    def forward(self,
                hidden_states: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                packed_seq_params: Optional[PackedSeqParams] = None,
                rotary_pos_emb: Optional[torch.Tensor] = None):

        batch_size,seq_length,_ = hidden_states.shape
        queries = self.q_proj(hidden_states)
        keys = self.k_proj(hidden_states)
        values = self.v_proj(hidden_states)


        if rotary_pos_emb is None:
            queries = queries.view(batch_size, seq_length, self.config.num_attention_heads, self.hidden_size_per_attention_head).transpose(1, 2)
            keys = keys.view(batch_size, seq_length, self.config.num_attention_heads, self.hidden_size_per_attention_head).transpose(1, 2)
            values = values.view(batch_size, seq_length, self.config.num_attention_heads, self.hidden_size_per_attention_head).transpose(1, 2)
        else:
            #assert cu_seqlens is not None, "Rope support flash attn only."
            cos, sin = rotary_pos_emb
            queries = queries.view(batch_size, seq_length, self.config.num_attention_heads, self.hidden_size_per_attention_head)
            keys = keys.view(batch_size, seq_length, self.config.num_attention_heads, self.hidden_size_per_attention_head)
            queries, keys = apply_rotary_pos_emb_flashatt(queries, keys, cos, sin)
            queries = queries.transpose(1, 2)
            keys = keys.transpose(1, 2)
            values = values.view(batch_size, seq_length, self.config.num_attention_heads, self.hidden_size_per_attention_head).transpose(1, 2)

        queries = queries.transpose(1, 2).squeeze(0)
        keys = keys.transpose(1, 2).squeeze(0)
        values = values.transpose(1, 2).squeeze(0)

        cu_seqlens_q = packed_seq_params.cu_seqlens_q
        cu_seqlens_kv = packed_seq_params.cu_seqlens_kv

        from flash_attn import flash_attn_func, flash_attn_varlen_func
        # max_seqlen_q = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()
        # max_seqlen_k = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()
        # assert cu_seqlens[-1].item() == queries.shape[0] == keys.shape[0] == values.shape[0], (cu_seqlens, queries.shape, keys.shape, values.shape)
        if mpu.get_context_parallel_world_size() > 1:
            attn_output = self.dist_attn(
                queries.unsqueeze(0),
                keys.unsqueeze(0),
                values.unsqueeze(0),
                packed_seq_params.cu_seqlens_q,
                packed_seq_params.cu_seqlens_kv,
                packed_seq_params.max_seqlen_q,
                packed_seq_params.max_seqlen_kv,
                causal=False,
            )
            attn_output = attn_output.squeeze(0)
        else:
            attn_output = flash_attn_varlen_func(
                queries,
                keys,
                values,
                packed_seq_params.cu_seqlens_q,
                packed_seq_params.cu_seqlens_kv,
                packed_seq_params.max_seqlen_q,
                packed_seq_params.max_seqlen_kv,
                causal=False,
                softmax_scale=self.scale
            )
        
        attn_output = attn_output.flatten(-2).unsqueeze(0)

        hidden_states = self.out_proj(attn_output)
        
        return hidden_states


class SiglipEncoderLayer(nn.Module):
    def __init__(self, config: VisionTransformerConfig):
        super().__init__()
        
        self.attention = SiglipAttention(config)
        self.mlp = SiglipMLP(config)
        self.layer_norm1 = nn.LayerNorm(config.hidden_size, eps=config.layernorm_epsilon)
        self.layer_norm2 = nn.LayerNorm(config.hidden_size, eps=config.layernorm_epsilon)

    def forward(self,
                hidden_states: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                packed_seq_params: Optional[PackedSeqParams] = None,
                rotary_pos_emb: Optional[torch.Tensor] = None):
        
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)
        hidden_states = self.attention(hidden_states, attention_mask, packed_seq_params,rotary_pos_emb)
        hidden_states = residual + hidden_states
        
        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        
        return hidden_states


class SiglipVisionModel(nn.Module):
    def __init__(self,
                 config: VisionTransformerConfig,
                 layer_spec: ModuleSpec,
                 vp_stage: Optional[int] = None):
        super().__init__()

        self.config = config
        self.embeddings = SiglipVisionEmbeddings(config)
        self.layers = nn.ModuleList([SiglipEncoderLayer(config) for _ in range(config.num_layers)])
        self.final_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layernorm_epsilon)

        head_dim = divide(
            config.kv_channels * config.num_attention_heads, config.num_attention_heads
        )
        self.rotary_pos_emb = SigLIPRotaryEmbedding(head_dim // 2)
        self.checkpoint = functools.partial(checkpoint, use_reentrant=True)

    @staticmethod
    def flatten_list(image_grid_thw):
        tmp_image_grid_thw = list()
        for image_grid in image_grid_thw:
            if isinstance(image_grid, list):
                tmp_image_grid_thw.extend(image_grid)
            else:
                tmp_image_grid_thw.append(image_grid)
        return tmp_image_grid_thw

    def forward(self,
                pixel_values: torch.Tensor,
                position_ids: torch.Tensor,
                vision_grid_thw: torch.Tensor,
                packed_seq_params: PackedSeqParams):
        hidden_states = self.embeddings(
            pixel_values,
            position_ids,
            vision_grid_thw
        )
        
        assert hidden_states.size(1) % mpu.get_context_parallel_world_size() == 0, \
            f"UlysessParallel requires sequence length {hidden_states.size(1)} dividable by sq_size {mpu.get_context_parallel_world_size()}"

        flatten_image_grid_thw = self.flatten_list(vision_grid_thw)
        width_position_ids,height_position_ids = None,None
        if width_position_ids is None or height_position_ids is None:
            split_hids = list()
            split_wids = list()
            for t, h, w in flatten_image_grid_thw:
                image_pids = torch.arange(t * h * w, device=hidden_states.device) % (h * w)
                sample_hids = image_pids // w
                sample_wids = image_pids % w
                split_hids.append(sample_hids)
                split_wids.append(sample_wids)
            width_position_ids = torch.concat(split_wids, dim=0)
            height_position_ids = torch.concat(split_hids, dim=0)
        window_indices, cu_seqlens_within_windows = None, None
        pids = torch.stack([height_position_ids, width_position_ids], dim=-1)
        max_grid_size = pids.max() + 1
        rope_emb_max_grid = self.rotary_pos_emb(max_grid_size)
        rope_emb = rope_emb_max_grid[pids].flatten(1)
        rope_emb = rope_emb.repeat(1, 2)
        rope_emb = (rope_emb.cos(), rope_emb.sin())
        
        if mpu.get_context_parallel_world_size() > 1:
            start, end = get_local_sequence_boundary(hidden_states.size(1))
            hidden_states = get_local_sequence(hidden_states, 1)
            sin, cos = rope_emb
            rope_emb = (sin[start: end, :], cos[start:end, :])

        for layer in self.layers:
            hidden_states = self.checkpoint(
                layer.__call__,
                hidden_states,
                None,
                packed_seq_params,
                rope_emb,
            )
        hidden_states = self.final_layernorm(hidden_states)
        
        if mpu.get_context_parallel_world_size() > 1:
            hidden_states = AllGather.apply(
                hidden_states, mpu.get_context_parallel_group(), 1
            )
        return hidden_states


class KeyeModelSlowFast(MegatronModule):
    def __init__(
        self,
        transformer_config: TransformerConfig,
        transformer_layer_spec: ModuleSpec,
        vision_config: VisionTransformerConfig,
        fast_vision_config: VisionTransformerConfig,
        vision_layer_spec: ModuleSpec,
        fast_vision_layer_spec: ModuleSpec,
        pre_process: bool = True,
        post_process: bool = True,
        mtp_block_spec: Optional[ModuleSpec] = None,
        vp_stage: Optional[int] = None,
    ) -> None:
        super().__init__(config=transformer_config)
        args = get_args()
        self.config = transformer_config
        self.vision_config = vision_config
        self.pre_process = pre_process
        self.post_process = post_process
        self.vp_stage = vp_stage
        if self.pre_process:
            self.visual = SiglipVisionModel(vision_config, vision_layer_spec, vp_stage)
            self.mlp_AR = Projector(vision_config, transformer_config.hidden_size)

            # self.fast_visual = SiglipVisionModel(fast_vision_config, fast_vision_layer_spec, vp_stage)
            # self.fast_mlp_AR = Projector(fast_vision_config, transformer_config.hidden_size)

        self.language_model = GPTModel(
            config=transformer_config,
            transformer_layer_spec=transformer_layer_spec,
            vocab_size=args.padded_vocab_size,
            max_sequence_length=327680,
            pre_process=pre_process,
            post_process=post_process,
            fp16_lm_cross_entropy=False,
            parallel_output=True,
            share_embeddings_and_output_weights=False,
            position_embedding_type="mrope",
            rotary_percent=1.0,
            rotary_base=10000,
            rope_scaling=False,
            mtp_block_spec=mtp_block_spec,
            vp_stage=vp_stage,
        )
        self.share_embeddings_and_output_weights = (
            self.language_model.share_embeddings_and_output_weights
        )

    def shared_embedding_or_output_weight(self):
        """This is a convenience method to surface the language model's word embeddings, which is
        necessary for `finalize_model_grads._allreduce_word_embedding_grads`."""
        return self.language_model.shared_embedding_or_output_weight()

    def get_transformer_callables_by_layer(self, layer_number: int):
        return self.language_model.get_transformer_callables_by_layer(layer_number)

    def build_schedule_plan(
        self,
        input_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        inference_params: Optional[InferenceParams] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
        extra_block_kwargs: dict = None,
        runtime_gather_output: Optional[bool] = None,
        loss_mask: Optional[torch.Tensor] = None,
    ):
        return self.language_model.build_schedule_plan(
                    input_ids,
                    position_ids,
                    attention_mask,
                    decoder_input,
                    labels,
                    inference_params,
                    packed_seq_params,
                    extra_block_kwargs,
                    runtime_gather_output,
                    loss_mask,
               )

    def set_input_tensor(self, input_tensor) -> None:
        # This is usually handled in schedules.py but some inference code still
        # gives us non-lists or None
        if not isinstance(input_tensor, list):
            input_tensor = [input_tensor]

        self.language_model.set_input_tensor(input_tensor[0])

    def freeze(
        self, freeze_language_model: bool, freeze_vision_model: bool, freeze_vision_projection: bool
    ):
        """Freeze model modules.

        Make specific modules non-trainable by setting requires_grad to False for the module's parameters.

        Args:
            freeze_language_model (bool): Freeze the language model module.
            freeze_vision_model (bool): Freeze the vision model module.
            freeze_vision_projection (bool): Freeze the vision projection module.
        """
        modules = []
        if freeze_language_model and self.language_model is not None:
            modules.append(self.language_model)
        if freeze_vision_model and self.visual is not None:
            modules.append(self.visual)
        if freeze_vision_projection and self.mlp_AR is not None:
            modules.append(self.mlp_AR)

        for module in modules:
            for param in module.parameters():
                param.requires_grad = False

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        inference_context: BaseInferenceContext = None,
        packed_seq_params: PackedSeqParams = None,
        extra_block_kwargs: dict = None,
        runtime_gather_output: Optional[bool] = None,
        *,
        vision_data: Optional[torch.Tensor] = None,
        vision_position_ids: Optional[torch.Tensor] = None,
        vision_grid_thw: Optional[torch.Tensor] = None,
        vision_packed_seq_params: Optional[PackedSeqParams] = None,
        fast_vision_data: Optional[torch.Tensor] = None,
        fast_vision_position_ids: Optional[torch.Tensor] = None,
        fast_vision_grid_thw: Optional[torch.Tensor] = None,
        fast_vision_packed_seq_params: Optional[PackedSeqParams] = None,
        inference_params: Optional[InferenceParams] = None,
        loss_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.pre_process:
            assert vision_data is not None
            vision_data = vision_data.unsqueeze(0)
            vision_outputs = self.visual(
                pixel_values=vision_data,
                position_ids=vision_position_ids,
                vision_grid_thw=vision_grid_thw,
                packed_seq_params=vision_packed_seq_params,
            )
            vision_embeds = self.mlp_AR(vision_outputs, packed_seq_params=vision_packed_seq_params, thw = vision_grid_thw)
            if fast_vision_data is not None:
                fast_vision_data = fast_vision_data.unsqueeze(0)
                fast_vision_outputs = self.visual(
                    pixel_values=fast_vision_data,
                    position_ids=fast_vision_position_ids,
                    vision_grid_thw=fast_vision_grid_thw,
                    packed_seq_params=fast_vision_packed_seq_params,
                )
                fast_vision_embeds = self.mlp_AR(fast_vision_outputs, packed_seq_params=fast_vision_packed_seq_params, thw=fast_vision_grid_thw)

            mask = (input_ids == self.vision_config.image_token_id) | (input_ids == self.vision_config.video_token_id)
            mask = mask.view([-1, 1])

            fast_mask = input_ids == self.vision_config.fast_video_token_id
            fast_mask = fast_mask.view([-1, 1])
            # num_vision_tokens = mask.sum().item()
            # print(f"embed_shape={vision_embeds.shape}")
            # num_vision_features = vision_embeds.size(0)
            # if num_vision_tokens != num_vision_features:
            #     raise ValueError(f"Vision features and vision tokens not match: {num_vision_features=}, {num_vision_tokens=}")

            input_embeds = self.language_model.embedding(input_ids, position_ids=position_ids)  # [s, b, h]

            if self.config.sequence_parallel:
                input_embeds = gather_from_sequence_parallel_region(input_embeds)

            # print(f"input_embed={input_embeds.shape}, mask={mask.shape}, vision={vision_embeds.shape}")
            vision_mask = mask.unsqueeze(-1).expand_as(input_embeds).to(input_embeds.device)
            input_embeds = input_embeds.masked_scatter(
                vision_mask, vision_embeds.to(input_embeds.device, input_embeds.dtype))

            if fast_vision_data is not None:
                fast_vision_mask = fast_mask.unsqueeze(-1).expand_as(input_embeds).to(input_embeds.device)
                input_embeds = input_embeds.masked_scatter(
                        fast_vision_mask, fast_vision_embeds.to(input_embeds.device, input_embeds.dtype))
            
            if self.config.sequence_parallel:
                input_embeds = scatter_to_sequence_parallel_region(input_embeds)

            decoder_input = input_embeds


        output = self.language_model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            decoder_input=decoder_input,
            labels=labels,
            inference_context=inference_context,
            packed_seq_params=packed_seq_params,
            extra_block_kwargs=extra_block_kwargs,
            runtime_gather_output=runtime_gather_output,
            inference_params=inference_params,
            loss_mask=loss_mask,
        )
        return output


def _load_state_dict_hook_ignore_param_names(
    param_names: List[str], module: torch.nn.Module, incompatible_keys: namedtuple
):
    """Hook to ignore missing keys during checkpoint loading.

    By default, this should not be used to avoid accidentally missing weights in checkpoint loading.

    Example use case: Use this for the vision projection if you want to load a checkpoint that contains vision and language model weights
    but not the vision projection weights.

    Args:
        param_names (list of str): Parameter names allowed to be missing when calling load_state_dict.
        module (torch.nn.Module): The torch module this hook applies to. Unused here but required by the torch API.
        incompatible_keys (namedtuple): Namedtuple with fields missing_keys and unexpected_keys, which collect the missing and unexpected
            keys when calling load_state_dict on this torch module, respectively.
    """
    for param_name in param_names:
        if param_name in incompatible_keys.missing_keys:
            logging.getLogger(__name__).warning(
                f"{param_name} being removed from incompatible_keys.missing_keys in QWen2VLModel"
            )
            incompatible_keys.missing_keys.remove(param_name)
