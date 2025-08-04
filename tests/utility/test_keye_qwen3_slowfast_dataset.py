import json
import sys
import token
from multiprocessing import process

from omegaconf import DictConfig, OmegaConf
from torchdata.stateful_dataloader import StatefulDataLoader

from verl.protocol import DataProto
from verl.utils import hf_processor, hf_tokenizer
from verl.utils.dataset.keye_qwen3_slowfast_dataset import (
    KeyeQwen3SlowFastDataset, collate_fn)

config = OmegaConf.create({
    "base_model_dir" : "/mmu_mllm_hdd_2/zhouyang12/models/Keye-8B-demo_hf_vit_rope_slowfast_0714",
    "custom_chat_template": "{% set image_count = namespace(value=0) %}{% set video_count = namespace(value=0) %}{% for message in messages %}<|im_start|>{{ message['role'] }}\n{% if message['content'] is string %}{{ message['content'] }}<|im_end|>\n{% else %}{% for content in message['content'] %}{% if content['type'] == 'image' or 'image' in content or 'image_url' in content %}{% set image_count.value = image_count.value + 1 %}{% if add_vision_id %}Picture {{ image_count.value }}: {% endif %}<|vision_start|><|image_pad|><|vision_end|>{% elif content['type'] == 'video' or 'video' in content %}{% set video_count.value = video_count.value + 1 %}{% if add_vision_id %}Video {{ video_count.value }}: {% endif %}<|vision_start|><|video_pad|><|vision_end|>{% elif 'text' in content %}{{ content['text'] }}{% endif %}{% endfor %}<|im_end|>\n{% endif %}{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
    })

tokenizer = hf_tokenizer(config.base_model_dir, trust_remote_code=True)
processor = hf_processor(config.base_model_dir, trust_remote_code=True, use_fast=True)



if config.get("custom_chat_template", None) is not None:
    tokenizer.chat_template = config.custom_chat_template
    processor.chat_template = config.custom_chat_template



image_text_path = "/nlp_group/huangjiaming/data/0623_longcot_video_multi_images_16frame_processed.parquet"
video_path = "/nlp_group/huangjiaming/data/0623_longcot_video_all_processed.parquet"
cot_path =  "/nlp_group/huangjiaming/kai-verl/OpenR1_Math_220k_rule_long_cot_new_think_token.parquet"


test_path = "/nlp_group/huangjiaming/logits-distill/merged_file.parquet"
dataset = KeyeQwen3SlowFastDataset(
        test_path,
        tokenizer,
        config,
        processor)

data_loader = StatefulDataLoader(
        dataset=dataset,
        batch_size=4,
        num_workers=2,
        drop_last=True,
        collate_fn=collate_fn,
        sampler=None,
    )

item = 0
for batch_dict in data_loader:
    batch: DataProto = DataProto.from_single_dict(batch_dict)
    print(batch)
    #batch.save_to_disk(f"/nlp_group/huangjiaming/logits-distill/dataset_res_{item}.data_proto")
    #item += 1

#    batch = batch_dict
#    model_input = batch.non_tensor_batch["multi_modal_inputs"]
#    empty_model_inputs = [x for x in model_input if x == {}]
#    assert len(empty_model_inputs) == 0, f"{len(empty_model_inputs)=} {len(batch)=}"

    #print(f"Received batch: ")
    #break
    #break
