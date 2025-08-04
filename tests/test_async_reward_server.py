import sys
import os
import ray
from functools import partial

from verl.protocol import DataProto, DataProtoItem
from verl.single_controller.ray.base import RayWorkerGroup
from verl.workers.reward_workers import RewardWorker
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto, DataProtoFuture
from keye_reward import KeyeComputeReward, ModelBaseAccuracy
from verl.workers.async_reward_manager import AsyncRewardWorkerManager
from verl.workers.async_reward_workers import AsyncRewardWorker
from omegaconf import OmegaConf
import tracemalloc


batch = DataProto.load_from_disk("/nlp_group/yuanjiawei05/logits-distill/dataproto/test_batch_step_211.pkl")
tracemalloc.start()


def reward_fn():
    pass

def val_reward_fn():
    pass


def to_dataproto(item: DataProtoItem) -> DataProto:
    return DataProto(
        batch=item.batch,
        non_tensor_batch=item.non_tensor_batch,
        meta_info=item.meta_info
    )

config_dict = {
    "reward_model": {
        "launch_reward_fn_async": True,
        "enable_reward_workers": True
    }
}

config = OmegaConf.create(config_dict)

resource_pool = RayResourcePool([2, 2, 2, 2], use_gpu=True, max_colocate_count=1)
class_with_args = RayClassWithInitArgs(cls=RewardWorker, config=config, reward_fn=reward_fn, val_reward_fn=val_reward_fn)
rm_wg = RayWorkerGroup(resource_pool, class_with_args)

batch_padded, pad_size = pad_dataproto_to_divisor(batch, 8)

rm_wg.init_worker(batch_padded)
AsyncRewardWorkerManager(config, rm_wg, reward_fn, val_reward_fn)




