import torch
from verl.protocol import DataProto
from transformers import AutoTokenizer


def find_diff_indices_and_values(tensor1: torch.Tensor, tensor2: torch.Tensor):
    """
    找出两个形状相同的张量中，位置相同但值不同的索引和值。

    Args:
        tensor1: 第一个张量。
        tensor2: 第二个张量。

    Returns:
        一个字典，包含以下键：
        - 'diff_indices': 一个张量，包含所有值不同的元素的索引。
        - 'values_tensor1': 一个张量，包含在tensor1中对应这些索引的值。
        - 'values_tensor2': 一个张量，包含在tensor2中对应这些索引的值。
    """
    if tensor1.shape != tensor2.shape:
        raise ValueError("两个张量的形状必须相同。")

    # 找出值不相等的元素
    diff_mask = (tensor1 != tensor2)

    # 获取不相等的元素的索引
    # torch.nonzero() 返回一个二维张量，每一行是一个索引
    diff_indices = torch.nonzero(diff_mask)

    if diff_indices.numel() == 0:
        print("两个张量完全相同，没有不同的值。")
        return {
            'diff_indices': torch.empty(0, tensor1.dim(), dtype=torch.long),
            'values_tensor1': torch.empty(0, dtype=tensor1.dtype),
            'values_tensor2': torch.empty(0, dtype=tensor2.dtype)
        }

    # 使用索引从原始张量中获取对应的值
    # 注意：对于多维张量，直接使用 diff_indices 进行索引会返回一个展平的张量。
    # 如果需要保持维度结构，可能需要更复杂的索引，但对于获取值，这种方式是有效的。
    values_tensor1 = tensor1[diff_mask]
    values_tensor2 = tensor2[diff_mask]

    return {
        'diff_indices': diff_indices,
        'values_tensor1': values_tensor1,
        'values_tensor2': values_tensor2
    }

hf_tokenizer = AutoTokenizer.from_pretrained("/mmu_mllm_hdd_2/wenbin/SFT/Keye-8B/RL/ColdStart/20250716.6.cold_start_v0.from_cotmix_lr1e-6__freezevit__lcy100percent__adddhjrft_tk_jkbadcaserft_jk5wrft__real__newrewardmodel_alsocotmix/output/v1-20250716-195523/checkpoint-127",  trust_remote_code=True)


verl_res_list = []
for i in range(4):
    dt = DataProto.load_from_disk(f"/nlp_group/huangjiaming/logits-distill/dataset_res_{i}.data_proto")
    verl_res_list.append(dt)
  

swift_res_path = "/hetu_group/jky/misc/tools/swift_20250508/playground/keye_8b_20250613/rl/20250704.1.r1reward_tpl_v4__fixanswerblank/tools/output/dataset_debug.pt"
swift_res_pt =  torch.load(swift_res_path, map_location="cpu")

def diff_input_ids(verl_pt, swift_data):
    print(f'{swift_data.keys()=}')
    attention_mask = verl_pt.batch["attention_mask"][0]
    verl_prompt_ids_padded = verl_pt.batch["input_ids"][0]
    verl_prompt_ids = verl_prompt_ids_padded[attention_mask.bool()]
    verl_position_ids_padded = verl_pt.batch["position_ids"][0]
    verl_position_ids = verl_position_ids_padded[:, attention_mask.bool()]
    print(f"{verl_position_ids.shape=}")

    swift_response_length = swift_data["logits_to_keep"]
    swift_res_input_ids_len = swift_data["input_ids"].shape[1]
    swift_prompt_ids = swift_data["input_ids"][0, : swift_res_input_ids_len - swift_response_length]
    swift_res_position_ids_len = swift_data["position_ids"].shape[2]
    swift_position_ids = swift_data["position_ids"].squeeze(1)[:, : swift_res_position_ids_len - swift_response_length]
    print(f"{swift_position_ids.shape=}")
    try:
        is_input_ids_match = torch.all(verl_prompt_ids == swift_prompt_ids)
        is_position_ids_match = torch.all(verl_position_ids == swift_position_ids)
    except Exception as e:
        print(f'{verl_pt=}\n\n{swift_data=}')
        verl_prompt_str = "".join(hf_tokenizer.batch_decode(verl_prompt_ids))
        swift_prompt_str = "".join(hf_tokenizer.batch_decode(swift_prompt_ids))
        print(f'{verl_prompt_str=}\n\n{swift_prompt_str=}')
        raise RuntimeError(f'{e=}')
    if is_input_ids_match and is_position_ids_match:
        print(f'[SUCCESS] input_ids, position ids  matched.')
    else:
        print(f'[FAILED]  not matched {is_input_ids_match=} {is_position_ids_match=}')
        if not is_input_ids_match:
            verl_prompt_str = "".join(hf_tokenizer.batch_decode(verl_prompt_ids))
            swift_prompt_str = "".join(hf_tokenizer.batch_decode(swift_prompt_ids))
            print(f'[DEBUG INFO] {verl_prompt_str=} {swift_prompt_str=}')
        if not is_position_ids_match:
            print(f'[DEBUG INFO] {verl_position_ids=}\n\n{swift_position_ids=}')
            result_1d = find_diff_indices_and_values(verl_position_ids, swift_position_ids)
            
            if result_1d['diff_indices'].numel() > 0:
                print("不同的索引 (1D):", result_1d['diff_indices'].squeeze()) # squeeze for 1D
                print("tensor1 对应的值:", result_1d['values_tensor1'])
                print("tensor2 对应的值:", result_1d['values_tensor2'])
                print("\n详细列表:")
                for i in range(result_1d['diff_indices'].shape[0]):
                    idx = result_1d['diff_indices'][i].tolist() # Convert to list for printing
                    val1 = result_1d['values_tensor1'][i].item()
                    val2 = result_1d['values_tensor2'][i].item()
                    print(f"索引: {idx}, tensor1值: {val1}, tensor2值: {val2}")
            else:
                print("两个张量完全相同。")
   
    # compare images
    if "pixel_values" in swift_data and "image_grid_thw" in swift_data:
        swift_pixel_values = swift_data['pixel_values']
        print(f"{swift_pixel_values.shape=}")
        swift_image_grid_thw = swift_data['image_grid_thw']
        print(f"{swift_image_grid_thw.shape=}")
        verl_pixel_values = verl_pt.non_tensor_batch['multi_modal_inputs'][0]['pixel_values']
        verl_image_grid_thw = verl_pt.non_tensor_batch['multi_modal_inputs'][0]['image_grid_thw']
        print(f"{verl_pixel_values.shape=}")
        print(f"{verl_image_grid_thw.shape=}")
        is_pixel_value_match = torch.all(verl_pixel_values == swift_pixel_values)
        is_image_grid_thw_match = torch.all(verl_image_grid_thw == swift_image_grid_thw)
        print(f'{is_pixel_value_match=} {is_image_grid_thw_match=}')
        if is_pixel_value_match and is_image_grid_thw_match:
            print(f'[SUCCESS] pixel_value, image_grid_thw  matched.')
        else:
            print(f'[FAILED]  not matched {is_pixel_value_match=} {is_image_grid_thw_match=}')
    # compare videos
    if "pixel_values_videos" in swift_data and "video_grid_thw" in swift_data:
        swift_pixel_values_videos = swift_data['pixel_values_videos']
        swift_video_grid_thw = swift_data['video_grid_thw']
        verl_pixel_values_videos = verl_pt.non_tensor_batch['multi_modal_inputs'][0]['pixel_values_videos']
        verl_video_grid_thw = verl_pt.non_tensor_batch['multi_modal_inputs'][0]['video_grid_thw']
        print(f"{swift_pixel_values_videos.shape=}")
        print(f"{verl_pixel_values_videos.shape=}")
        is_pixel_value_match = torch.all(verl_pixel_values_videos == swift_pixel_values_videos)
        is_video_grid_thw_match = torch.all(verl_video_grid_thw == swift_video_grid_thw)
        print(f'{is_pixel_value_match=} {is_video_grid_thw_match=}')
        if is_pixel_value_match and is_video_grid_thw_match:
            print(f'[SUCCESS] pixel_value_videos, video_grid_thw  matched.')
        else:
            print(f'[FAILED]  not matched {is_pixel_value_match=} {is_video_grid_thw_match=}')
            #result_1d = find_diff_indices_and_values(swift_pixel_values_videos, verl_pixel_values_videos)
            #
            #if result_1d['diff_indices'].numel() > 0:
            #    print("不同的索引 (1D):", result_1d['diff_indices'].squeeze()) # squeeze for 1D
            #    print("tensor1 对应的值:", result_1d['values_tensor1'])
            #    print("tensor2 对应的值:", result_1d['values_tensor2'])
            #    print("\n详细列表:")
            #    for i in range(result_1d['diff_indices'].shape[0]):
            #        idx = result_1d['diff_indices'][i].tolist() # Convert to list for printing
            #        val1 = result_1d['values_tensor1'][i].item()
            #        val2 = result_1d['values_tensor2'][i].item()
            #        print(f"索引: {idx}, tensor1值: {val1}, tensor2值: {val2}")
            #else:
            #    print("两个张量完全相同。")


print(f'**********Compare Text data***********')
diff_input_ids(verl_res_list[0], swift_res_pt[2])
print(f'**********Compare One-Photo data***********')
diff_input_ids(verl_res_list[1], swift_res_pt[0])
print(f'**********Compare Multi-Photo data***********')
diff_input_ids(verl_res_list[2], swift_res_pt[4])
print(f'**********Compare Video data***********')
diff_input_ids(verl_res_list[3], swift_res_pt[-1])
