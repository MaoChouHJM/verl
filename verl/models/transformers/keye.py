from typing import Optional, Tuple

import torch

from .flash_attention_utils import flash_attention_forward


from recovlm.models.keye.modeling_keye import (
        KeyeAttention,
        apply_multimodal_rotary_pos_emb,
        repeat_kv,
)


def keye_attn_forward(
    self : "KeyeAttention",
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
    **kwargs,
):
    bsz, q_len, _ = hidden_states.size()
    q= self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim)
    query_states = self.q_norm(q)
    key_states = self.k_norm(self.k_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim))
    value_states = self.v_proj(hidden_states)

    query_states = query_states.transpose(1, 2)
    key_states = key_states.transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    # Because the input can be padded, the absolute sequence length depends on the max position id.
    if position_embeddings is None:
        cos, sin = self.rotary_emb(value_states, position_ids)
    else:
        cos, sin = position_embeddings

    query_states, key_states = apply_multimodal_rotary_pos_emb(
        query_states, key_states, cos, sin, self.rope_scaling["mrope_section"]
    )

    # repeat k/v heads if n_kv_heads < n_heads
    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)
    dropout_rate = 0.0 if not self.training else self.attention_dropout

    sliding_window = None
    if (
        self.config.use_sliding_window
        and getattr(self.config, "sliding_window", None) is not None
        and self.layer_idx >= self.config.max_window_layers
    ):
        sliding_window = self.config.sliding_window
 
    attn_output, _ = flash_attention_forward(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        dropout=dropout_rate,
        sliding_window=sliding_window,
        position_ids=position_ids,  # important: pass position ids
    )  # (batch_size, seq_length, num_head / sp_size, head_size)

    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, None, None

