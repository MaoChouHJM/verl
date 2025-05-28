import sys
import json
from torchdata.stateful_dataloader import StatefulDataLoader
from verl.utils.dataset.rl_dataset import collate_fn
from verl.utils.dataset.qwen3_rl_dataset import Qwen3RLHFDataset
from omegaconf import DictConfig, OmegaConf
from verl.protocol import DataProto


config = OmegaConf.create({
    "hf_dataset_config" : "/llm_reco/lingzhixin/recovlm_qw0510/recovlm/examples/vlm/keye/debug_keye_8B256.json"
    })

dataset = Qwen3RLHFDataset(
        "/nlp_group/huangjiaming/kai-verl/single.parquet",
        None,
        config,
        None)

data_loader = StatefulDataLoader(
        dataset=dataset,
        batch_size=1,
        num_workers=0,
        drop_last=True,
        collate_fn=collate_fn,
        sampler=None,
    )

for batch_dict in data_loader:
#    print(f'Received batch: {batch}')
    batch: DataProto = DataProto.from_single_dict(batch_dict)
    print(batch.batch["position_ids"].shape)
    break

#    batch = batch_dict
#    model_input = batch.non_tensor_batch["multi_modal_inputs"]
#    empty_model_inputs = [x for x in model_input if x == {}]
#    assert len(empty_model_inputs) == 0, f"{len(empty_model_inputs)=} {len(batch)=}"

    #print(f"Received batch: ")
    #break
