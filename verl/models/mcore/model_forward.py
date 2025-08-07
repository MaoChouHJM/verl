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

from codecs import ascii_encode

from megatron.core import parallel_state as mpu

from verl.utils.megatron_utils import unwrap_model

from .util import (SlowFastVisionPadder, get_batch_on_this_cp_rank,
                   keye_slowfast_preprocess_packed_seq,
                   postprocess_packed_seqs, preprocess_packed_seqs,
                   recover_left_padding, remove_left_padding)


def gptmodel_forward(
    model,
    input_ids,
    attention_mask,
    position_ids,
    sequence_parallel,
    value_model=False,
    pack_seqs=True,
    logits_processor=None,
    logits_processor_args: dict = None,
    **kwargs,
):
    """Default forward pass for GPT models with optional sequence packing."""
    pre_process = unwrap_model(model).pre_process
    post_process = unwrap_model(model).post_process
    if pack_seqs:
        batch_size, seq_len = attention_mask.shape[:2]
        input_ids_rmpad, packed_seq_params = preprocess_packed_seqs(input_ids, attention_mask, pre_process=pre_process)
        input_ids_rmpad = input_ids_rmpad.contiguous()
        output_orig = model(
            input_ids=input_ids_rmpad,
            attention_mask=None,
            position_ids=position_ids,
            packed_seq_params=packed_seq_params,
        )
        if post_process and logits_processor is not None:
            args = {
                k: preprocess_packed_seqs(v, attention_mask, pre_process=True)[0]
                for k, v in logits_processor_args.items()
            }
            output_dict = logits_processor(output_orig, **args)
            output = {
                k: postprocess_packed_seqs(
                    v, packed_seq_params, attention_mask, batch_size, seq_len, post_process=post_process
                )
                for k, v in output_dict.items()
            }
        else:
            output = postprocess_packed_seqs(
                output_orig, packed_seq_params, attention_mask, batch_size, seq_len, post_process=post_process
            )
    else:
        assert logits_processor is None, "logits_processor is not supported for non-packed sequence"
        batch_size, sequence_length = attention_mask.shape
        new_input_ids, new_attention_mask, new_position_ids = remove_left_padding(
            input_ids, attention_mask, position_ids, sequence_parallel, pre_process=pre_process
        )
        output = model(input_ids=new_input_ids, attention_mask=new_attention_mask, position_ids=new_position_ids)
        output = recover_left_padding(
            output, new_attention_mask, attention_mask, sequence_length, post_process=post_process
        )
    if value_model and post_process:
        output = output[..., 0]
    return output


def gptmodel_forward_qwen2_5_vl(
    model,
    input_ids,
    attention_mask,
    position_ids,
    sequence_parallel,
    value_model=False,
    pack_seqs=True,
    multi_modal_inputs=None,
    logits_processor=None,
    logits_processor_args: dict = None,
    **kwargs,
):
    from megatron.core import parallel_state as mpu

    assert mpu.get_context_parallel_world_size() == 1, "qwen2_5_vl's context parallel is not accurate yet"
    pre_process = unwrap_model(model).pre_process
    post_process = unwrap_model(model).post_process
    pixel_values = (
        multi_modal_inputs["pixel_values"].to(input_ids.device) if "pixel_values" in multi_modal_inputs else None
    )
    image_grid_thw = (
        multi_modal_inputs["image_grid_thw"].to(input_ids.device) if "image_grid_thw" in multi_modal_inputs else None
    )
    if pack_seqs:
        batch_size, seq_len = attention_mask.shape[:2]
        input_ids_rmpad, packed_seq_params = preprocess_packed_seqs(input_ids, attention_mask, pre_process=True)
        input_ids_rmpad = input_ids_rmpad.contiguous()
        output_orig = model(
            input_ids=input_ids_rmpad,
            attention_mask=None,
            position_ids=position_ids,
            packed_seq_params=packed_seq_params,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
        )

        if post_process and logits_processor is not None:
            args = {
                k: preprocess_packed_seqs(v, attention_mask, pre_process=True)[0]
                for k, v in logits_processor_args.items()
            }
            output_dict = logits_processor(output_orig, **args)
            output = {
                k: postprocess_packed_seqs(
                    v, packed_seq_params, attention_mask, batch_size, seq_len, post_process=post_process
                )
                for k, v in output_dict.items()
            }
        else:
            output = postprocess_packed_seqs(
                output_orig, packed_seq_params, attention_mask, batch_size, seq_len, post_process=post_process
            )
    else:
        batch_size, sequence_length = attention_mask.shape
        new_input_ids, new_attention_mask, new_position_ids = remove_left_padding(
            input_ids, attention_mask, position_ids, sequence_parallel, pre_process=pre_process
        )
        output = model(
            input_ids=new_input_ids,
            position_ids=new_position_ids,
            attention_mask=new_attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
        )
        output = recover_left_padding(
            output, new_attention_mask, attention_mask, sequence_length, post_process=post_process
        )
    if value_model and post_process:
        output = output[..., 0]
    return output

def gptmodel_forward_keye_qwen3(
    model,
    input_ids,
    attention_mask,
    position_ids,
    value_model=False,
    pack_seqs=True,
    multi_modal_inputs=None,
    logits_processor=None,
    logits_processor_args: dict = None,
    **kwargs,
):
    post_process = unwrap_model(model).post_process
    vp_stage = unwrap_model(model).vp_stage
    if not hasattr(gptmodel_forward_keye_qwen3, 'slowfast_padder'):
        base_model_dir = unwrap_model(model).model_path
        from transformers import AutoProcessor
        processor = AutoProcessor.from_pretrained(base_model_dir, trust_remote_code=True)
        gptmodel_forward_keye_qwen3.slowfast_padder = SlowFastVisionPadder(processor)
    
    if pack_seqs:
        batch_size, seq_len = attention_mask.shape[:2]
        batch, packed_seq_params_ori = keye_slowfast_preprocess_packed_seq(
            input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            multi_modal_inputs=multi_modal_inputs,
            slowfast_padder=gptmodel_forward_keye_qwen3.slowfast_padder,
            pre_process=True,
            image_token_id=unwrap_model(model).hf_config.image_token_id,
            video_token_id=unwrap_model(model).hf_config.video_token_id,
            )
        batch = get_batch_on_this_cp_rank(batch)
        from megatron.core.packed_seq_params import PackedSeqParams
        packed_seq_params = PackedSeqParams(
                qkv_format="thd",
                cu_seqlens_q=batch["cu_seqlens"].cpu(),
                cu_seqlens_kv=batch["cu_seqlens"].cpu(),
                max_seqlen_q=batch["max_seqlen"].cpu(),
                max_seqlen_kv=batch["max_seqlen"].cpu(),
            )
        vision_packed_seq_params = None
        if mpu.is_pipeline_first_stage(ignore_virtual=False, vp_stage=vp_stage):
            if vp_stage is None or vp_stage == 0:
                if 'vision_cu_seqlens' in batch:
                    vision_packed_seq_params = PackedSeqParams(
                        qkv_format="thd",
                        cu_seqlens_q=batch["vision_cu_seqlens"].cpu(),
                        cu_seqlens_kv=batch["vision_cu_seqlens"].cpu(),
                        max_seqlen_q=batch["vision_max_seqlen"].cpu(),
                        max_seqlen_kv=batch["vision_max_seqlen"].cpu(),
                    )

        output_tensor = model(
            input_ids=batch.get("input_ids", None),
            position_ids=batch.get("position_ids", None),
            attention_mask=batch.get("attention_mask", None),
            labels=batch.get("labels", None),
            packed_seq_params=packed_seq_params,
            vision_data=batch.get("vision_data", None),
            vision_position_ids=batch.get("vision_position_ids", None),
            vision_grid_thw=batch.get("vision_grid_thw", None),
            vision_packed_seq_params=vision_packed_seq_params,
            loss_mask=batch.get("loss_mask", None),
        )

        if post_process and logits_processor is not None:
            args = {
                k: get_batch_on_this_cp_rank(keye_slowfast_preprocess_packed_seq(
                    v,
                    position_ids=position_ids,
                    attention_mask=attention_mask,
                    multi_modal_inputs=multi_modal_inputs,
                    pre_process=True,
                    image_token_id=unwrap_model(model).hf_config.image_token_id,
                    video_token_id=unwrap_model(model).hf_config.video_token_id,
                    )[0]["input_ids"])
                for k, v in logits_processor_args.items()
            }
            output_dict = logits_processor(output_tensor, **args)
            output = {
                k: postprocess_packed_seqs(
                    v, packed_seq_params_ori, attention_mask, batch_size, seq_len, post_process=post_process
                )
                for k, v in output_dict.items()
            }
        else:
            output = postprocess_packed_seqs(
                output_tensor, packed_seq_params_ori, attention_mask, batch_size, seq_len, post_process=post_process
            )
    else:
        raise ValueError(f"slowfast model does not support non-packed sequence")
    if value_model and post_process:
        output = output[..., 0]
    return output

def gptmodel_forward_keye_qwen3_slowfast(
    model,
    input_ids,
    attention_mask,
    position_ids,
    value_model=False,
    pack_seqs=True,
    multi_modal_inputs=None,
    logits_processor=None,
    logits_processor_args: dict = None,
    **kwargs,
):
    post_process = unwrap_model(model).post_process
    vp_stage = unwrap_model(model).vp_stage
    if not hasattr(gptmodel_forward_keye_qwen3_slowfast, 'slowfast_padder'):
        base_model_dir = unwrap_model(model).model_path
        from transformers import AutoProcessor
        processor = AutoProcessor.from_pretrained(base_model_dir, trust_remote_code=True)
        gptmodel_forward_keye_qwen3_slowfast.slowfast_padder = SlowFastVisionPadder(processor)
    
    if pack_seqs:
        batch_size, seq_len = attention_mask.shape[:2]
        batch, packed_seq_params_ori = keye_slowfast_preprocess_packed_seq(
            input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            multi_modal_inputs=multi_modal_inputs,
            slowfast_padder=gptmodel_forward_keye_qwen3_slowfast.slowfast_padder,
            pre_process=True,
            image_token_id=unwrap_model(model).hf_config.image_token_id,
            video_token_id=unwrap_model(model).hf_config.video_token_id,
            fast_video_token_id=unwrap_model(model).hf_config.fast_video_token_id,
            )
        batch = get_batch_on_this_cp_rank(batch)
        from megatron.core.packed_seq_params import PackedSeqParams
        packed_seq_params = PackedSeqParams(
                qkv_format="thd",
                cu_seqlens_q=batch["cu_seqlens"],
                cu_seqlens_kv=batch["cu_seqlens"],
                max_seqlen_q=batch["max_seqlen"],
                max_seqlen_kv=batch["max_seqlen"],
            )

        vision_packed_seq_params = None
        fast_vision_packed_seq_params = None

        if mpu.is_pipeline_first_stage(ignore_virtual=False, vp_stage=vp_stage):
            if vp_stage is None or vp_stage == 0:
                if "vision_cu_seqlens" in vision_packed_seq_params:
                    vision_packed_seq_params = PackedSeqParams(
                        qkv_format="thd",
                        cu_seqlens_q=batch["vision_cu_seqlens"],
                        cu_seqlens_kv=batch["vision_cu_seqlens"],
                        max_seqlen_q=batch["vision_max_seqlen"],
                        max_seqlen_kv=batch["vision_max_seqlen"],
                    )
                if "fast_vision_cu_seqlens" in vision_packed_seq_params:
                    fast_vision_packed_seq_params = PackedSeqParams(
                        qkv_format="thd",
                        cu_seqlens_q=batch.get("fast_vision_cu_seqlens", None),
                        cu_seqlens_kv=batch.get("fast_vision_cu_seqlens", None),
                        max_seqlen_q=batch.get("fast_vision_max_seqlen", None),
                        max_seqlen_kv=batch.get("fast_vision_max_seqlen", None),
                    )

        output_tensor = model(
            input_ids=batch.get("input_ids", None),
            position_ids=batch.get("position_ids", None),
            attention_mask=batch.get("attention_mask", None),
            labels=batch.get("labels", None),
            packed_seq_params=packed_seq_params,
            vision_data=batch.get("vision_data", None),
            vision_position_ids=batch.get("vision_position_ids", None),
            vision_grid_thw=batch.get("vision_grid_thw", None),
            vision_packed_seq_params=vision_packed_seq_params,
            vision_mask_index=batch.get("vision_mask_index", None),
            fast_vision_data=batch.get("fast_vision_data", None),
            fast_vision_position_ids=batch.get("fast_vision_position_ids", None),
            fast_vision_grid_thw=batch.get("fast_vision_grid_thw", None),
            fast_vision_packed_seq_params=fast_vision_packed_seq_params,
            fast_vision_mask_index=batch.get("fast_vision_mask_index", None),
            loss_mask=batch.get("loss_mask", None),
        )

        if post_process and logits_processor is not None:
            args = {
                k: get_batch_on_this_cp_rank(keye_slowfast_preprocess_packed_seq(
                    v,
                    position_ids=position_ids,
                    attention_mask=attention_mask,
                    multi_modal_inputs=multi_modal_inputs,
                    slowfast_padder=gptmodel_forward_keye_qwen3_slowfast.slowfast_padder,
                    pre_process=True,
                    image_token_id=unwrap_model(model).hf_config.image_token_id,
                    video_token_id=unwrap_model(model).hf_config.video_token_id,
                    fast_video_token_id=unwrap_model(model).hf_config.fast_video_token_id,
                    )[0]["input_ids"])
                for k, v in logits_processor_args.items()
            }
            output_dict = logits_processor(output_tensor, **args)
            output = {
                k: postprocess_packed_seqs(
                    v, packed_seq_params_ori, attention_mask, batch_size, seq_len, post_process=post_process
                )
                for k, v in output_dict.items()
            }
        else:
            output = postprocess_packed_seqs(
                output_tensor, packed_seq_params_ori, attention_mask, batch_size, seq_len, post_process=post_process
            )
    else:
        raise ValueError(f"slowfast model does not support non-packed sequence")
    if value_model and post_process:
        output = output[..., 0]
    return output
