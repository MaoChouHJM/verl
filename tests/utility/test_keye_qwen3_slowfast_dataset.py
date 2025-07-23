from multiprocessing import process
import sys
import json
import token
from torchdata.stateful_dataloader import StatefulDataLoader
from verl.utils.dataset.keye_qwen3_slowfast_dataset import collate_fn
from verl.utils.dataset.keye_qwen3_slowfast_dataset import KeyeQwen3SlowFastDataset
from omegaconf import DictConfig, OmegaConf
from verl.protocol import DataProto
from verl.utils import hf_processor, hf_tokenizer


config = OmegaConf.create({
    "base_model_dir" : "/mmu_mllm_hdd_2/zhouyang12/output1/Keye/0.9.3/Stage2/8b/slowfast-0721-0717-v2/step27000/global_step27000/converted/"
    })

tokenizer = hf_tokenizer(config.base_model_dir, trust_remote_code=True)
processor = hf_processor(config.base_model_dir, trust_remote_code=True, use_fast=True)

dataset = KeyeQwen3SlowFastDataset(
        "/nlp_group/huangjiaming/kai-verl/OpenR1_Math_220k_rule_long_cot_new_think_token.parquet",
        tokenizer,
        config,
        processor)

data_loader = StatefulDataLoader(
        dataset=dataset,
        batch_size=1,
        num_workers=0,
        drop_last=True,
        collate_fn=collate_fn,
        sampler=None,
    )

item = 0
for batch_dict in data_loader:
#    print(f'Received batch: {batch}')
    batch: DataProto = DataProto.from_single_dict(batch_dict)
    item += 1
    print(item)
    break

#    batch = batch_dict
#    model_input = batch.non_tensor_batch["multi_modal_inputs"]
#    empty_model_inputs = [x for x in model_input if x == {}]
#    assert len(empty_model_inputs) == 0, f"{len(empty_model_inputs)=} {len(batch)=}"

    #print(f"Received batch: ")
    #break
