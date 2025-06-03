import sys
import json
from torchdata.stateful_dataloader import StatefulDataLoader
from verl.utils.dataset.rl_dataset import collate_fn
from verl.utils.dataset.qwen3_rl_dataset import Qwen3RLHFDataset
from omegaconf import DictConfig, OmegaConf
from verl.protocol import DataProto


config = OmegaConf.create({
    "base_model_dir" : "/mmu_mllm_hdd_2/wenbin/SFT/Keye-8B/20250528.CoT_Mix_tianke_v3.from_mpo_v1_from_19083/output/v1-20250528-213601/checkpoint-5154"
    })

dataset = Qwen3RLHFDataset(
        ["/nlp_group/huangjiaming/kai-verl/dataset_MMPR_K12_nn_addTokenLen__mmpr1.1_minlen30_sample5w__fixsystem__instuctnothink__new_think_token__fixnothink.parquet"],
        None,
        config,
        None)

data_loader = StatefulDataLoader(
        dataset=dataset,
        batch_size=1,
        num_workers=8,
        drop_last=True,
        collate_fn=collate_fn,
        sampler=None,
    )

item = 0
for batch_dict in data_loader:
#    print(f'Received batch: {batch}')
    batch: DataProto = DataProto.from_single_dict(batch_dict)
    item += 1

#    batch = batch_dict
#    model_input = batch.non_tensor_batch["multi_modal_inputs"]
#    empty_model_inputs = [x for x in model_input if x == {}]
#    assert len(empty_model_inputs) == 0, f"{len(empty_model_inputs)=} {len(batch)=}"

    #print(f"Received batch: ")
    #break
