# Copyright 2025 Bytedance Ltd. and/or its affiliates
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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

import torch
from typing import Any, Dict, List, Optional, Tuple, Union
from megatron.core import parallel_state as mpu
from megatron.core.packed_seq_params import PackedSeqParams

PackingResult = Dict[str, Union[torch.Tensor, List[torch.Tensor]]]

def preprocess_packed_seqs(
    input_ids: torch.Tensor, attention_mask: torch.Tensor, pre_process: bool = True
) -> tuple[torch.Tensor, PackedSeqParams]:
    """
    Preprocess packed sequences
    CP splits sequence into CP*2 chunks, and each GPU gets 2 chunks (GPU0 gets first and last chunks, GPU1
    gets second and second last chunks, and so on), this is for load balancing with causal masking.
    See https://github.com/NVIDIA/TransformerEngine/issues/1368
    """
    batch_size = input_ids.shape[0]

    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    tp_size = mpu.get_tensor_model_parallel_world_size()
    cp_size = mpu.get_context_parallel_world_size()
    cp_rank = mpu.get_context_parallel_rank()
    align_size = tp_size * cp_size * 2 if cp_size > 1 else tp_size

    pad_size = (align_size - seqlens_in_batch % align_size) % align_size
    seqlens_in_batch_padded = seqlens_in_batch + pad_size
    cu_seqlens = torch.zeros(batch_size + 1, dtype=torch.int32, device=input_ids.device)
    cu_seqlens[1:] = torch.cumsum(seqlens_in_batch, dim=0)
    cu_seqlens_padded = torch.zeros(batch_size + 1, dtype=torch.int32, device=input_ids.device)
    cu_seqlens_padded[1:] = torch.cumsum(seqlens_in_batch_padded, dim=0)
    max_seqlen_in_batch = seqlens_in_batch_padded.max().item()

    shape = list(input_ids.shape[1:])
    shape[0] = seqlens_in_batch_padded.sum().item() // cp_size
    if pre_process:
        input_ids_rmpad = torch.zeros(shape, dtype=input_ids.dtype, device=input_ids.device)
        for i in range(batch_size):
            if cp_size <= 1:
                seqlen = seqlens_in_batch[i]
                input_ids_rmpad[cu_seqlens_padded[i] : cu_seqlens_padded[i] + seqlen] = input_ids[i, attention_mask[i]]
                continue
            seqlen = seqlens_in_batch_padded[i] // cp_size
            half_seqlen = seqlen // 2
            start_idx = cu_seqlens_padded[i] // cp_size
            # split to 2 chunks
            d = input_ids[i, attention_mask[i]]
            input_ids_rmpad[start_idx : start_idx + half_seqlen] = d[
                half_seqlen * cp_rank : half_seqlen * (cp_rank + 1)
            ]

            remain_start = seqlens_in_batch_padded[i] - half_seqlen * (cp_rank + 1)
            remain_end = seqlens_in_batch_padded[i] - half_seqlen * cp_rank
            remain_end = min(remain_end, d.shape[0])
            remain_len = remain_end - remain_start
            if remain_len > 0:
                input_ids_rmpad[start_idx + half_seqlen : start_idx + half_seqlen + remain_len] = d[
                    remain_start:remain_end
                ]

    packed_seq_params = PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=cu_seqlens_padded,
        max_seqlen_q=max_seqlen_in_batch,
        cu_seqlens_kv=cu_seqlens_padded,
        max_seqlen_kv=max_seqlen_in_batch,
        cu_seqlens_q_padded=cu_seqlens_padded,
        cu_seqlens_kv_padded=cu_seqlens_padded,
    )
    if pre_process:
        return input_ids_rmpad.unsqueeze(0), packed_seq_params
    else:
        return input_ids, packed_seq_params


def postprocess_packed_seqs(
    output: torch.Tensor,
    packed_seq_params: PackedSeqParams,
    attention_mask: torch.Tensor,
    batch_size: int,
    seq_len: int,
    post_process: bool = True,
) -> torch.Tensor:
    """
    Postprocess packed sequences
    """
    if not post_process:
        return output
    shape = [batch_size, seq_len] + list(output.shape[2:])  # 1,packed, dim -> batch_size, seq_len, dim
    output_new = torch.zeros(shape, dtype=output.dtype, device=output.device)

    cp_size = mpu.get_context_parallel_world_size()
    # all gather output across context parallel group
    if cp_size > 1:
        # output shape: [1, packed_len, hidden_dim]
        # need to gather across cp group and concatenate in sequence dimension
        output_list = [torch.empty_like(output) for _ in range(cp_size)]
        torch.distributed.all_gather(output_list, output.detach(), group=mpu.get_context_parallel_group())
        output_list[mpu.get_context_parallel_rank()] = output
    else:
        output_list = [output]
    for i in range(batch_size):
        if cp_size <= 1:
            s = attention_mask[i].sum().item()
            output_new[i, attention_mask[i]] = output[0][
                packed_seq_params.cu_seqlens_q_padded[i] : packed_seq_params.cu_seqlens_q_padded[i] + s
            ]
            continue
        s_len_padded_chunk = (
            packed_seq_params.cu_seqlens_q_padded[i + 1] - packed_seq_params.cu_seqlens_q_padded[i]
        ) // cp_size
        half_seqlen = s_len_padded_chunk // 2
        s_len = attention_mask[i].sum().item()
        s_len_padded = s_len_padded_chunk * cp_size
        tmp = torch.empty(s_len_padded, *output.shape[2:], device=output.device)
        for j in range(cp_size):
            o = output_list[j][0]
            # split to 2 chunks
            packed_start_idx = packed_seq_params.cu_seqlens_q_padded[i] // cp_size
            o0, o1 = (
                o[packed_start_idx : packed_start_idx + half_seqlen],
                o[packed_start_idx + half_seqlen : packed_start_idx + s_len_padded_chunk],
            )
            tmp[j * half_seqlen : (j + 1) * half_seqlen] = o0
            tmp[s_len_padded - (j + 1) * half_seqlen : s_len_padded - j * half_seqlen] = o1
        output_new[i, attention_mask[i]] = tmp[:s_len]

    return output_new


def remove_left_padding(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    position_ids: torch.Tensor,
    sequence_parallel: bool = False,
    pre_process: bool = True,
):
    """
    Remove left padding from input_ids, attention_mask and position_ids
    return new_input_ids, new_attention_mask, new_position_ids
    """
    assert attention_mask.ndim == 2
    assert position_ids.ndim == 2
    cp_size = mpu.get_context_parallel_world_size()
    assert cp_size == 1, "Context parallel size without seq_pack is not supported"
    batch_size = input_ids.shape[0]
    shape = list(input_ids.shape)  # batch_size, seq_len,...
    seq_lens = attention_mask.sum(dim=1)
    seq_len = seq_lens.max().item()
    if sequence_parallel:
        sp_world_size = mpu.get_tensor_model_parallel_world_size()
        pad_size = (sp_world_size - seq_len % sp_world_size) % sp_world_size
        seq_len = seq_len + pad_size
    shape[1] = seq_len
    if pre_process:
        new_input_ids = torch.zeros(dtype=input_ids.dtype, device=input_ids.device, size=shape)
    new_attention_mask = torch.zeros(
        dtype=attention_mask.dtype, device=attention_mask.device, size=(batch_size, seq_len)
    )
    new_position_ids = torch.zeros(dtype=position_ids.dtype, device=position_ids.device, size=(batch_size, seq_len))
    for i in range(batch_size):
        if pre_process:
            new_input_ids[i, : seq_lens[i]] = input_ids[i, attention_mask[i]]
        new_attention_mask[i, : seq_lens[i]] = attention_mask[i, attention_mask[i]]
        new_position_ids[i, : seq_lens[i]] = position_ids[i, attention_mask[i]]
    if pre_process:
        return new_input_ids, new_attention_mask, new_position_ids
    else:
        return input_ids, new_attention_mask, new_position_ids


def recover_left_padding(
    result,
    attention_mask: torch.Tensor,
    original_attention_mask: torch.Tensor,
    origin_seqlen: int,
    post_process: bool = True,
):
    """
    Recover left padding from result
    return result
    """
    if not post_process:
        return result
    shape = list(result.shape)
    batch_size = shape[0]
    shape[1] = origin_seqlen
    new_result = torch.zeros(dtype=result.dtype, device=result.device, size=shape)
    for i in range(batch_size):
        new_result[i, original_attention_mask[i]] = result[i, attention_mask[i]]
    return new_result

class SlowFastVisionPadder:
    """
    给slow fast的padding，最多使用4+6个token
    """
    MAX_PAD_LENGTH = 4+6 #
    def __init__(self, processor):

        # 这个padder这里使用了非slowfast版本的get_rope_index,这不重要,因为pad是不会用来计算损失的
        from examples.keye.processors.utils_slowfast import get_rope_index
        self.get_rope_index = get_rope_index
        self.processor = processor
        self.patch_size = processor.image_processor.patch_size
        self.merge_size = processor.image_processor.merge_size
        assert self.merge_size == 2, f"SlowFastVisionPadder does not support self.merge_size({self.merge_size}) != 2"
        self.image_pad = processor.tokenizer.encode("<|image_pad|>")[0]
        self.video_pad = processor.tokenizer.encode("<|video_pad|>")[0]
        fast_video_pad = processor.tokenizer.encode("<|fast_video_pad|>")
        assert len(fast_video_pad) == 1, "Decode fast_video_pad failed: {}".format(fast_video_pad)
        self.fast_video_pad = fast_video_pad[0]
        self.vision_start = processor.tokenizer.encode("<|vision_start|>")[0]
        self.vision_end = processor.tokenizer.encode("<|vision_end|>")[0]
        self.frame = processor.tokenizer.encode("<|frame|>")[0]

    def __call__(self, packed_pixel_values, packed_pixel_values_videos, packed_fast_pixel_values_videos):
          paddings = []
          n_pixel_values = sum([x.shape[0] for x in packed_pixel_values], 0)
          n_pixel_values_videos = sum([x.shape[0] for x in packed_pixel_values_videos], 0)
          n_fast_pixel_values_videos = sum([x.shape[0] for x in packed_fast_pixel_values_videos], 0)

          if n_pixel_values % 8 == 4: paddings.append(self.gen_img_pad(n_merged_slow_tokens=1))
          elif n_pixel_values == 0: paddings.append(self.gen_img_pad(n_merged_slow_tokens=2))

          paddings.append(
            self.gen_video_pad(
              n_merged_slow_tokens=1 if n_pixel_values_videos % 8 == 4 else 2, 
              n_merged_fast_tokens=1 if n_fast_pixel_values_videos % 8 == 4 else 2, 
              )
          )

          return paddings

    def gen_img_pad(self, n_merged_slow_tokens=1):
        input_ids = [self.vision_start] + [self.image_pad] * n_merged_slow_tokens + [self.vision_end]
        inputs = {
            "input_ids": torch.tensor([input_ids], dtype=torch.int64),
            "attention_mask": torch.tensor([[1] * (n_merged_slow_tokens + 2)], dtype=torch.int64),
            "pixel_values": torch.rand(n_merged_slow_tokens * 4, 3, self.patch_size, self.patch_size).float(),
            "image_grid_thw": torch.tensor([[1, 2, n_merged_slow_tokens * 2]], dtype=torch.int64),
            "loss_mask": torch.zeros(len(input_ids), dtype=torch.int64),
        }
        inputs["position_ids"] = self.get_rope_index(
          inputs["input_ids"],
          image_grid_thw=inputs.get("image_grid_thw"),
          video_grid_thw=inputs.get("video_grid_thw"),
          spatial_merge_size=self.merge_size,
          image_token_id=self.image_pad,
          video_token_id=self.video_pad,
          vision_start_token_id=self.vision_start
        )

        return inputs

    def gen_video_pad(self, n_merged_slow_tokens=1, n_merged_fast_tokens=2):
        """
        demo: 
        'input_ids':
        Tensor: shape=(1, 42), dtype=torch.int64, device=cpu, data=tensor([151652,     27,     91,   6763])...tensor([  6213,     91,     29, 151653])
        'attention_mask':
        Tensor: shape=(1, 42), dtype=torch.int64, device=cpu, data=tensor([1, 1, 1, 1])...tensor([1, 1, 1, 1])
        'pixel_values_videos':
        Tensor: shape=(16, 3, 14, 14), dtype=torch.float32, device=cpu, data=tensor([-1., -1., -1., -1.])...tensor([-1., -1., -1., -1.])
        'video_grid_thw':
        Tensor: shape=(1, 3), dtype=torch.int64, device=cpu, data=tensor([1, 4, 4])...tensor([1, 4, 4])
        'fast_pixel_values_videos':
        Tensor: shape=(32, 3, 14, 14), dtype=torch.float32, device=cpu, data=tensor([-1., -1., -1., -1.])...tensor([-1., -1., -1., -1.])
        'fast_video_grid_thw':
        Tensor: shape=(1, 3), dtype=torch.int64, device=cpu, data=tensor([1, 8, 4])...tensor([1, 8, 4])
        """
        # 标准是这个
        # # <|frame|>ts<|placeholder|><|placeholder|> ... n_slow ... <|placeholder|><|fast_start|><|fast_placeholder|><|fast_placeholder|> ... n_fast ... <|fast_placeholder|><|fast_end|>
        # 但是我们不需要那么多token
        #   只需要<|placeholder|><|placeholder|> ... n_slow ... <|placeholder|><|fast_placeholder|><|fast_placeholder|> ... n_fast ... <|fast_placeholder|>
        # total_slow_tokens = pass # 
        # video_inputs = (
        input_ids = [self.vision_start] + [self.video_pad] * n_merged_slow_tokens + [self.fast_video_pad] * n_merged_fast_tokens + [self.vision_end]
        inputs = {
            "input_ids": torch.tensor([input_ids], dtype=torch.int64),
            "attention_mask": torch.tensor([[1] * len(input_ids)], dtype=torch.int64),
            "video_grid_thw": torch.tensor([[1, 2, n_merged_slow_tokens * 2]], dtype=torch.int64),
            "fast_video_grid_thw": torch.tensor([[1, 2, n_merged_fast_tokens * 2]], dtype=torch.int64),
            "fast_pixel_values_videos": torch.rand(n_merged_fast_tokens * 4, 3, self.patch_size, self.patch_size).float(),
            "pixel_values_videos": torch.rand(n_merged_slow_tokens * 4, 3, self.patch_size, self.patch_size).float(),
            "loss_mask": torch.zeros(len(input_ids), dtype=torch.int64),
        }
        inputs["position_ids"] = self.get_rope_index(
          inputs["input_ids"],
          image_grid_thw=inputs.get("image_grid_thw"),
          video_grid_thw=inputs.get("video_grid_thw"),
          spatial_merge_size=self.merge_size,
          image_token_id=self.image_pad,
          video_token_id=self.video_pad,
          vision_start_token_id=self.vision_start
        )

        return inputs

def rmpad_and_cp_padding(input_ids: torch.Tensor, attention_mask: torch.Tensor, pre_process: bool = True, dbg_cp_size: int = 1) -> tuple[torch.Tensor, PackedSeqParams]: 
    """
    Preprocess packed sequences
    CP splits sequence into CP*2 chunks, and each GPU gets 2 chunks (GPU0 gets first and last chunks, GPU1
    gets second and second last chunks, and so on), this is for load balancing with causal masking.
    See https://github.com/NVIDIA/TransformerEngine/issues/1368
    """
    batch_size = input_ids.shape[0]

    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    # NOTE(huangjiaming): here we not support tp + sp
    #tp_size = mpu.get_tensor_model_parallel_world_size()
    tp_size = 1
    cp_size = mpu.get_context_parallel_world_size() if dbg_cp_size is None else dbg_cp_size
    align_size = tp_size * cp_size * 2 if cp_size > 1 else tp_size

    pad_size = (align_size - seqlens_in_batch % align_size) % align_size
    seqlens_in_batch_padded = seqlens_in_batch + pad_size
    cu_seqlens = torch.zeros(batch_size + 1, dtype=torch.int32, device=input_ids.device)
    cu_seqlens[1:] = torch.cumsum(seqlens_in_batch, dim=0)
    cu_seqlens_padded = torch.zeros(batch_size + 1, dtype=torch.int32, device=input_ids.device)
    cu_seqlens_padded[1:] = torch.cumsum(seqlens_in_batch_padded, dim=0)
    max_seqlen_in_batch = seqlens_in_batch_padded.max().item()

    shape = list(input_ids.shape[1:])
    if len(shape) == 1:
        shape[0] = seqlens_in_batch_padded.sum().item()
    elif len(shape) == 2:  # mrope
        shape[-1] = seqlens_in_batch_padded.sum().item()
    else:
        raise ValueError(f'shape can only be 2-D ids or 3-D mrope position_ids.')

    if pre_process:
        input_ids_rmpad = torch.zeros(shape, dtype=input_ids.dtype, device=input_ids.device)
        for i in range(batch_size):
            try:
                seqlen = seqlens_in_batch[i]
            except Exception as e:
                raise RuntimeError(f'{seqlens_in_batch=}\n{attention_mask.shape=}\n{input_ids.shape=}')
            input_ids_rmpad[..., cu_seqlens_padded[i] : cu_seqlens_padded[i] + seqlen] = input_ids[i, ...,  attention_mask[i].bool()]

    packed_seq_params = PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=cu_seqlens_padded,
        max_seqlen_q=max_seqlen_in_batch,
        cu_seqlens_kv=cu_seqlens_padded,
        max_seqlen_kv=max_seqlen_in_batch,
        cu_seqlens_q_padded=cu_seqlens_padded,
        cu_seqlens_kv_padded=cu_seqlens_padded,
    )
    if pre_process:
        return input_ids_rmpad.unsqueeze(0), packed_seq_params
    else:
        return input_ids, packed_seq_params

def get_batch_on_this_cp_rank(batch: Dict[str, Any],
                              dbg_cp_size: Optional[int] = None,
                              dbg_cp_rank: Optional[int] = None,
                              ):
    cp_size = mpu.get_context_parallel_world_size() if dbg_cp_size is None else dbg_cp_size
    if cp_size > 1:
        cp_rank = mpu.get_context_parallel_rank() if dbg_cp_rank is None else dbg_cp_rank
        cu_seqlens = batch["cu_seqlens"]

        total_tokens = cu_seqlens[-1].item()
        seq_starts = cu_seqlens[:-1]
        seq_lens = cu_seqlens[1:] - seq_starts
        num_sequences = seq_lens.size(0)
        K = seq_lens // (2 * cp_size)
        seq_ids_per_token = torch.repeat_interleave(
            torch.arange(num_sequences, device=cu_seqlens.device), repeats=seq_lens)
        intra_seq_indices = torch.arange(total_tokens, device=cu_seqlens.device) - \
            seq_starts[seq_ids_per_token]
        K_per_token = K[seq_ids_per_token]
        K_per_token[K_per_token == 0] = 1
        chunk_id_per_token = intra_seq_indices // K_per_token
        mask = (chunk_id_per_token == cp_rank) | (chunk_id_per_token == 2 * cp_size - cp_rank - 1)
        final_indices = torch.where(mask)[0]

        cp_split_keys = ["input_ids", "position_ids", "vision_mask_index", "fast_vision_mask_index"]
        for k in cp_split_keys:
            if k in batch:
                batch[k] = batch[k].index_select(-1, final_indices)

    return batch

def keye_slowfast_preprocess_packed_seq(
    input_ids: torch.Tensor,  # [bs, prompt_length + response_length]
    position_ids: Optional[torch.Tensor],
    attention_mask: torch.Tensor,  # [bs, prompt_length + response_length]
    multi_modal_inputs: Dict[str, torch.Tensor],
    slowfast_padder: SlowFastVisionPadder,
    pre_process: bool = True,
    image_token_id: int = 151655,
    video_token_id: int = 151656,
    fast_video_token_id: int = 151678,
    dbg_cp_size: Optional[int] = None,
    ) -> PackingResult:
    packed_input_ids_rmpad, _ = rmpad_and_cp_padding(input_ids, attention_mask, pre_process, dbg_cp_size=dbg_cp_size)

    packed_position_ids_rmpad, packed_seq_params = rmpad_and_cp_padding(position_ids, attention_mask, pre_process, dbg_cp_size=dbg_cp_size)

    cp_size = mpu.get_context_parallel_world_size() if dbg_cp_size is None else dbg_cp_size

    if pre_process:
        cu_seqlens = packed_seq_params.cu_seqlens_q_padded.clone()
        packed_input_ids = [packed_input_ids_rmpad]
        packed_position_ids = [packed_position_ids_rmpad]
        packed_vision_data = []
        packed_vision_grid_thw = []
        fast_packed_vision_data = []
        fast_packed_vision_grid_thw = []

        # get vision info from multi_modal_inputs, each has been already concatenated
        if "pixel_values" in multi_modal_inputs and len(multi_modal_inputs["pixel_values"]):
            packed_vision_data.append(multi_modal_inputs["pixel_values"])
            packed_vision_grid_thw.append(multi_modal_inputs["image_grid_thw"])
        if "pixel_values_videos" in multi_modal_inputs:
            packed_vision_data.append(multi_modal_inputs["pixel_values_videos"])
            packed_vision_grid_thw.append(multi_modal_inputs["video_grid_thw"])
        if 'fast_pixel_values_videos' in multi_modal_inputs:
            fast_packed_vision_data.append(multi_modal_inputs["fast_pixel_values_videos"])
            fast_packed_vision_grid_thw.append(multi_modal_inputs["fast_video_grid_thw"])

        # padding to trigger VIT
        vit_paddings = slowfast_padder(packed_vision_data, [], fast_packed_vision_data)
        for pad in vit_paddings:
            all_one_attention_mask = torch.ones_like(pad['input_ids'])
            pad['input_ids'], _ = rmpad_and_cp_padding(pad['input_ids'], all_one_attention_mask,
                                    pre_process, dbg_cp_size=dbg_cp_size)
            pad['position_ids'], _ = rmpad_and_cp_padding(pad['position_ids'].transpose(0, 1), all_one_attention_mask,
                                    pre_process, dbg_cp_size=dbg_cp_size)

            packed_input_ids.append(pad['input_ids'])
            packed_position_ids.append(pad['position_ids'])
            try:
                cu_seqlens = torch.cat([cu_seqlens,
                    (cu_seqlens[-1] + len(pad['input_ids'][0])).unsqueeze(0)])
            except Exception as e:
                raise RuntimeError(f'{cu_seqlens=}\n{cu_seqlens.shape=}\n{e=}')
            if cp_size > 1 and cu_seqlens[-1] % (2 * cp_size) != 0:
                raise RuntimeError(f"cu_seqlens={cu_seqlens}")
            if "pixel_values" in pad and len(pad["pixel_values"]):
                packed_vision_data.append(pad["pixel_values"])
                packed_vision_grid_thw.append(pad["image_grid_thw"])
            if "pixel_values_videos" in pad:
                packed_vision_data.append(pad["pixel_values_videos"])
                packed_vision_grid_thw.append(pad["video_grid_thw"])
            if 'fast_pixel_values_videos' in pad:
                fast_packed_vision_data.append(pad["fast_pixel_values_videos"])
                fast_packed_vision_grid_thw.append(pad["fast_video_grid_thw"])



        packed_input_ids = torch.cat(packed_input_ids, dim=-1)
        packed_position_ids = torch.cat(packed_position_ids, dim=-1)
        packed_vision_data = None if len(packed_vision_data) == 0 else \
            torch.cat(packed_vision_data, dim=0)
        packed_vision_grid_thw = None if len(packed_vision_grid_thw) == 0 else \
            torch.cat(packed_vision_grid_thw, dim=0)
        fast_packed_vision_data = None if len(fast_packed_vision_data) == 0 else \
            torch.cat(fast_packed_vision_data, dim=0)
        fast_packed_vision_grid_thw = None if len(fast_packed_vision_grid_thw) == 0 else \
            torch.cat(fast_packed_vision_grid_thw, dim=0)

        batch = {
        "input_ids": packed_input_ids,
        "position_ids": packed_position_ids,
        "vision_data": packed_vision_data,
        "vision_grid_thw": packed_vision_grid_thw,
        "cu_seqlens": cu_seqlens,
        }
        def _get_seq_params(vision_grid_thw: torch.Tensor) -> torch.Tensor:
            cu_seqlens = [0]
            max_seqlen = -1
            position_ids = []
            for idx, thw in enumerate(vision_grid_thw):
                numel = thw[0] * thw[1] * thw[2]
                cu_seqlens.append(cu_seqlens[-1] + numel)
                max_seqlen = max(max_seqlen, numel)
                position_ids.append(torch.arange(numel) % (numel / thw.size(0)))
            return (
                torch.tensor(cu_seqlens, dtype=torch.int32),
                max_seqlen,
                torch.concat(position_ids, dim=0),
            )

        vision_seq_params = _get_seq_params(packed_vision_grid_thw)
        batch["vision_cu_seqlens"] = vision_seq_params[0]
        batch["vision_max_seqlen"] = vision_seq_params[1]
        batch["vision_position_ids"] = vision_seq_params[2]

        def get_vision_mask_index(input_ids, mask):
            result = torch.full(input_ids.shape, -1, dtype=torch.int32)
            num_mask = mask.sum().item()
            result[mask] = torch.arange(num_mask, dtype=torch.int32)
            return result

        batch["vision_mask_index"] = get_vision_mask_index(
            packed_input_ids,
            (packed_input_ids == image_token_id) | (packed_input_ids == video_token_id)
        )
        assert (batch["vision_mask_index"] != -1).sum().item() * 4 == batch["vision_data"].shape[0]

        if fast_packed_vision_data is not None:
            batch["fast_vision_data"] = fast_packed_vision_data
            batch["fast_vision_grid_thw"] = fast_packed_vision_grid_thw
            fast_vision_seq_params = _get_seq_params(fast_packed_vision_grid_thw)
            batch["fast_vision_cu_seqlens"] = fast_vision_seq_params[0]
            batch["fast_vision_max_seqlen"] = fast_vision_seq_params[1]
            batch["fast_vision_position_ids"] = fast_vision_seq_params[2]
            batch["fast_vision_mask_index"] = get_vision_mask_index(
                packed_input_ids,
                packed_input_ids == fast_video_token_id,
            )
            assert (batch["fast_vision_mask_index"] != -1).sum().item() * 4 == batch["fast_vision_data"].shape[0]
        batch["max_seqlen"] = (batch["cu_seqlens"][1:] - batch["cu_seqlens"][:-1]).max()
        batch["position_ids"] = batch["position_ids"].transpose(0, 1)
    else:
        raise ValueError(f"Not supported preprocess is False")


    return batch, packed_seq_params
