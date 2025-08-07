import torch
from typing import Any, Dict, List, Optional, Tuple, Union
from megatron.core import parallel_state as mpu
from megatron.core.packed_seq_params import PackedSeqParams
from transformers import PreTrainedTokenizer, ProcessorMixin, AutoProcessor, AutoConfig
from copy import deepcopy


PackingResult = Dict[str, Union[torch.Tensor, List[torch.Tensor]]]

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

    return batch

def convert_sample_to_verl_input(samples):
    from collections import defaultdict
    batch_size = len(samples)
    max_len = max([i['input_ids'].shape[-1] for i in samples])
    attention_mask = torch.zeros((batch_size, max_len), dtype=torch.int32)
    for idx, s in enumerate(samples):
        seq_len = s['input_ids'].shape[-1]
        attention_mask[idx, : seq_len] = 1 
        padding_len = max_len - seq_len
        s['input_ids']  = torch.cat([s['input_ids'], torch.zeros((1, padding_len), dtype=s['input_ids'].dtype)], dim=-1)
        input_ids = s['input_ids']
        s['position_ids']  = torch.cat([s['position_ids'], torch.zeros((3, 1, padding_len), dtype=s['position_ids'].dtype)], dim=-1)
       
    VISION_KEYS = ['pixel_values', 'image_grid_thw', 'pixel_values_videos',
                   'video_grid_thw', 'fast_pixel_values_videos', 'fast_video_grid_thw']
    
    multi_modal_input = defaultdict(list)
    for key in VISION_KEYS:
        for s in samples:
            if key in s:
                multi_modal_input[key].append(s[key])
    
    for k in multi_modal_input.keys():
        multi_modal_input[k] = torch.cat(multi_modal_input[k], dim=0)

    cat = torch.cat([s['input_ids'] for s in samples], dim=0)

    return torch.cat([s['input_ids'] for s in samples], dim=0), torch.cat([s['position_ids'] for s in samples], dim=1).transpose(0, 1), attention_mask, multi_modal_input


def test_cp():
    hf_processor = AutoProcessor.from_pretrained("/mmu_mllm_hdd_2/zhouyang12/models/Keye-8B-demo_hf_vit_rope_slowfast_0714",
                                                 trust_remote_code=True)
    slowfast_padder = SlowFastVisionPadder(hf_processor)

    from transformers.feature_extraction_utils import BatchFeature
    torch.serialization.add_safe_globals([BatchFeature])

    pack_input = torch.load("/nlp_group/huangjiaming/logits-distill/pack_input.pth")
    input_ids, position_ids, attention_mask, multi_modal_input = convert_sample_to_verl_input(pack_input)
    batch = keye_slowfast_preprocess_packed_seq(input_ids, position_ids, attention_mask, multi_modal_input,
                                                slowfast_padder, pre_process=True, dbg_cp_size=2)

    # compare total
    pack_output = torch.load("/nlp_group/huangjiaming/logits-distill/pack_output.pth")
    for k in pack_output.keys():
        if k in ["loss_mask", "sample_idx", "epoch_idx", "labels"]:
            continue 
        try:
            is_equal = torch.all(batch[k] == pack_output[k])
        except Exception as e:
            raise ValueError(f'{k=}\n{e}\n{batch[k].shape=}\n{pack_output[k].shape=}')
        print(f'{k=}\n{is_equal=}')
        #if not is_equal:
        #    print(f"{k=} {is_equal=} {batch[k].shape=} {pack_output[k].shape=}")

    before_cp_total_len = batch["input_ids"].shape
    print(f"{before_cp_total_len=}")
    # compare cp
    batch_cp_0 = get_batch_on_this_cp_rank(deepcopy(batch), dbg_cp_size=2, dbg_cp_rank=0)
    batch_cp_1 = get_batch_on_this_cp_rank(deepcopy(batch), dbg_cp_size=2, dbg_cp_rank=1)
    pack_output_0 = torch.load("/nlp_group/huangjiaming/logits-distill/pack_output_0.pth")
    pack_output_1 = torch.load("/nlp_group/huangjiaming/logits-distill/pack_output_1.pth")
    #is_equal = torch.all(batch["position_ids"] == pack_output["position_ids"])
    for k in pack_output.keys():
        if k in ["loss_mask", "sample_idx", "epoch_idx", "labels"]:
            continue 
        is_equal_0 = torch.all(batch_cp_0[k] == pack_output_0[k])
        is_equal_1 = torch.all(batch_cp_1[k] == pack_output_1[k])
        print(f'{k=}\n{is_equal_0=} {is_equal_1=}')

def convert_swift_input_to_verl_input(samples):
    MAX_PROMPT_LENGTH = 1024 * 3
    MAX_RESPONSE_LENGTH = 1024 * 5
    hf_config = AutoConfig.from_pretrained("/hetu_group/jky/misc/tools/swift_20250508/playground/keye_8b_20250613/rl/20250703.1.r1reward/global_step80_hf", trust_remote_code=True)
    from collections import defaultdict
    import numpy as np
    from verl import DataProto

    dp_rank = mpu.get_data_parallel_rank()
    print(f'{dp_rank=}')
    path = f"/hetu_group/jky/misc/tools/swift_20250508_0528/playground/keye_8b_20250613/rl/20250805.1.rft__qwen2_5_vl_72b_v2__trainvit__fromtkdhjrft__math__fromrl80/tools/outputs_2nd/rank{dp_rank}_globalstep0_debug_inputs.pt"
    samples = torch.load(path, map_location='cpu')
    batch_size = 1
    valid_resp_length = samples['inputs']['logits_to_keep']
    input_ids = samples['inputs']['input_ids']
    old_log_probs = samples['inputs']['old_per_token_logps']
    advantages = samples['inputs']['advantages']
    from verl.utils.dataset.keye_utils.keye_vl_utils import get_rope_index
    position_ids, _ = get_rope_index(
        input_ids,
        samples['inputs'].get('image_grid_thw', None),
        samples['inputs'].get('video_grid_thw', None),
        spatial_merge_size=hf_config.vision_config.spatial_merge_size,
        image_token_id=hf_config.image_token_id,
        video_token_id=hf_config.video_token_id,
        vision_start_token_id=hf_config.vision_start_token_id,
        tokens_per_second=hf_config.vision_config.tokens_per_second,
    )
    attention_mask = torch.zeros((batch_size, MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH), dtype=torch.int32)
    new_input_ids = torch.zeros((batch_size, MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH), dtype=torch.int32)
    new_response = new_input_ids[:, -MAX_RESPONSE_LENGTH:]
    new_position_ids = torch.zeros((3, batch_size, MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH), dtype=torch.int32)
    new_old_log_probs = torch.zeros((batch_size, MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH), dtype=torch.int32)

    new_input_ids[0, MAX_PROMPT_LENGTH + valid_resp_length - input_ids.shape[-1] : \
                      MAX_PROMPT_LENGTH + valid_resp_length] = input_ids[0]

    new_position_ids[:, 0, MAX_PROMPT_LENGTH + valid_resp_length - position_ids.shape[-1] : \
                      MAX_PROMPT_LENGTH + valid_resp_length] = position_ids[:, 0, :]
    attention_mask[0, MAX_PROMPT_LENGTH + valid_resp_length - input_ids.shape[-1] : \
                      MAX_PROMPT_LENGTH + valid_resp_length] = 1
    new_old_log_probs[0, MAX_PROMPT_LENGTH : MAX_PROMPT_LENGTH + old_log_probs.shape[-1]] = \
                      old_log_probs[0]
       
    VISION_KEYS = ['pixel_values', 'image_grid_thw', 'pixel_values_videos',
                   'video_grid_thw', 'fast_pixel_values_videos', 'fast_video_grid_thw']
    
    multi_modal_input = {}
    for key in VISION_KEYS:
        if key in samples['inputs']:
            multi_modal_input[key] = samples['inputs'][key]
    
    tensors = defaultdict(list)
    non_tensors = defaultdict(list)
    tensors['responses'] = new_response
    tensors['input_ids'] = new_input_ids
    tensors['attention_mask'] = attention_mask
    tensors['position_ids'] = new_position_ids.transpose(0, 1)
    tensors['old_log_probs'] = new_old_log_probs
    tensors['advantages'] = advantages
    non_tensors['multi_modal_inputs'] = np.array([multi_modal_input], dtype=object)
  
    batch: DataProto = DataProto.from_single_dict({**tensors, **non_tensors})
    # original verl data proto select
    select_keys = ["responses", "input_ids", "attention_mask", "position_ids", "old_log_probs", "advantages"]
    #if self.config.use_kl_loss:
    #    select_keys.append("ref_log_prob")
    #self.has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()

    data = batch.select(select_keys, ["multi_modal_inputs"])
    return data.make_iterator(
        mini_batch_size=self.config.ppo_mini_batch_size,
        epochs=self.config.ppo_epochs,
        seed=self.config.data_loader_seed,
        dataloader_kwargs={"shuffle": self.config.shuffle},
    )

#
#
#    MAX_PROMPT_LENGTH = 1024 * 3
#    MAX_RESPONSE_LENGTH = 1024 * 5
#    from collections import defaultdict
#    batch_size = 1
#    valid_resp_length = samples['inputs']['logits_to_keep']
#    input_ids = samples['inputs']['input_ids']
#    hf_config = AutoConfig.from_pretrained("/hetu_group/jky/misc/tools/swift_20250508/playground/keye_8b_20250613/rl/20250703.1.r1reward/global_step80_hf", trust_remote_code=True)
#    from verl.utils.dataset.keye_utils.keye_vl_utils import get_rope_index
#    position_ids, _ = get_rope_index(
#        input_ids,
#        samples['inputs'].get('image_grid_thw', None),
#        samples['inputs'].get('video_grid_thw', None),
#        spatial_merge_size=hf_config.vision_config.spatial_merge_size,
#        image_token_id=hf_config.image_token_id,
#        video_token_id=hf_config.video_token_id,
#        vision_start_token_id=hf_config.vision_start_token_id,
#        tokens_per_second=hf_config.vision_config.tokens_per_second,
#    )
#    attention_mask = torch.zeros((batch_size, MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH), dtype=torch.int32)
#    new_input_ids = torch.zeros((batch_size, MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH), dtype=torch.int32)
#    new_response = new_input_ids[:, -MAX_RESPONSE_LENGTH:]
#    new_position_ids = torch.zeros((3, batch_size, MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH), dtype=torch.int32)
#
#    new_input_ids[0, MAX_PROMPT_LENGTH + valid_resp_length - input_ids.shape[-1] : \
#                      MAX_PROMPT_LENGTH + valid_resp_length] = input_ids[0]
#    new_position_ids[:, 0, MAX_PROMPT_LENGTH + valid_resp_length - position_ids.shape[-1] : \
#                      MAX_PROMPT_LENGTH + valid_resp_length] = position_ids[:, 0, :]
#    attention_mask[0, MAX_PROMPT_LENGTH + valid_resp_length - input_ids.shape[-1] : \
#                      MAX_PROMPT_LENGTH + valid_resp_length] = 1
#       
#    VISION_KEYS = ['pixel_values', 'image_grid_thw', 'pixel_values_videos',
#                   'video_grid_thw', 'fast_pixel_values_videos', 'fast_video_grid_thw']
#    
#    multi_modal_input = defaultdict(list)
#    for key in VISION_KEYS:
#        if key in samples['inputs']:
#            multi_modal_input[key].append(samples['inputs'][key])
#    
#    for k in multi_modal_input.keys():
#        multi_modal_input[k] = torch.cat(multi_modal_input[k], dim=0)
#
#    return new_input_ids, new_position_ids.transpose(0, 1), attention_mask, multi_modal_input

if __name__ == "__main__":
    swift_input = torch.load("/hetu_group/jky/misc/tools/swift_20250508_0528/playground/keye_8b_20250613/rl/20250805.1.rft__qwen2_5_vl_72b_v2__trainvit__fromtkdhjrft__math__fromrl80/tools/outputs_2nd/rank0_globalstep0_debug_inputs.pt")
    verl_input = convert_swift_input_to_verl_input(swift_input)
