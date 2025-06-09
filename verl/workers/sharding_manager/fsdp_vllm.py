# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

import inspect
import logging
import os
import time
from contextlib import contextmanager
import torch
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp.api import (FullStateDictConfig,
                                        ShardedStateDictConfig, StateDictType)
from torch.distributed.fsdp.fully_sharded_data_parallel import \
    FullyShardedDataParallel as FSDP

from verl import DataProto
from verl.protocol import all_gather_data_proto
from verl.third_party.vllm import LLM
from verl.third_party.vllm import parallel_state as vllm_ps
from verl.third_party.vllm import vllm_version
from verl.utils.debug import GPUMemoryLogger, log_gpu_memory_usage
from verl.utils.vllm_utils import patch_vllm_moe_model_weight_loader

from .base import BaseShardingManager

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class FSDPVLLMShardingManager(BaseShardingManager):
    def __init__(
        self,
        module: FSDP,
        inference_engine: LLM,
        model_config,
        full_params: bool = False,
        device_mesh: DeviceMesh = None,
    ):
        self.module = module
        self.inference_engine = inference_engine
        self.model_config = model_config
        self.device_mesh = device_mesh

        # Full params
        self.full_params = full_params
        if full_params:
            FSDP.set_state_dict_type(
                self.module, state_dict_type=StateDictType.FULL_STATE_DICT, state_dict_config=FullStateDictConfig()
            )
        else:
            FSDP.set_state_dict_type(
                self.module,
                state_dict_type=StateDictType.SHARDED_STATE_DICT,
                state_dict_config=ShardedStateDictConfig(),
            )

        self.tp_size = vllm_ps.get_tensor_model_parallel_world_size()
        self.tp_rank = vllm_ps.get_tensor_model_parallel_rank()

        # Note that torch_random_states may be different on each dp rank
        self.torch_random_states = torch.cuda.get_rng_state()
        # get a random rng states
        if self.device_mesh is not None:
            gen_dp_rank = self.device_mesh["dp"].get_local_rank()
            torch.cuda.manual_seed(gen_dp_rank + 1000)  # make sure all tp ranks have the same random states
            self.gen_random_states = torch.cuda.get_rng_state()
            torch.cuda.set_rng_state(self.torch_random_states)
        else:
            self.gen_random_states = None
        self.timing_data = {}

    @GPUMemoryLogger(role="fsdp vllm sharding_manager", logger=logger)
    def __enter__(self):
        # NOTE: Basically, we only need `torch.cuda.empty_cache()` before vllm wake_up and
        # after vllm sleep, since vllm has its own caching memory allocator CuMemAllocator.
        # Out of vllm scope, we should avoid empty cache to let pytorch using caching memory
        # to speed up memory allocations.
        #
        # pytorch: https://pytorch.org/docs/stable/notes/cuda.html#memory-management
        # vllm: https://github.com/vllm-project/vllm/blob/v0.7.3/vllm/device_allocator/cumem.py#L103
        with self.timing_record("rollout_sharding/enter"):
            with self.timing_record("rollout_sharding/enter/empty_cache_before_state_dict"):
                torch.cuda.empty_cache()
            log_gpu_memory_usage("Before state_dict() in sharding manager memory", logger=logger)
            with self.timing_record("rollout_sharding/enter/get_state_dict"):
                params = self.module.state_dict()
            log_gpu_memory_usage("After state_dict() in sharding manager memory", logger=logger)
            # Copy, not share memory
            load_format = "hf" if self.full_params else "dtensor"

            if vllm_version in (
                "0.5.4",
                "0.6.3",
            ):
                with self.timing_record("rollout_sharding/enter/sync_model_weights_legacy"):
                    self.inference_engine.sync_model_weights(params, load_format=load_format)
                log_gpu_memory_usage("After sync model weights in sharding manager", logger=logger)
                with self.timing_record("rollout_sharding/enter/del_params_legacy"):
                    del params
            else:
                #with self.timing_record("rollout_sharding/enter/wake_up_weights"):
                if "tags" in inspect.signature(self.inference_engine.wake_up).parameters:
                    with self.timing_record("rollout_sharding/enter/wake_up_weights"):
                        self.inference_engine.wake_up(tags=["weights"])
                else:
                    with self.timing_record("rollout_sharding/enter/wake_up"):
                        self.inference_engine.wake_up()

                # update model params
                with self.timing_record("rollout_sharding/enter/update_params"):
                    self.update_params(params)
                log_gpu_memory_usage("After sync model weights in sharding manager", logger=logger)
                with self.timing_record("rollout_sharding/enter/del_params"):
                    del params
                with self.timing_record("rollout_sharding/enter/empty_cache_after_delparams"):
                    torch.cuda.empty_cache()
                with self.timing_record("rollout_sharding/enter/wake_up_kv_cache"):
                    if "tags" in inspect.signature(self.inference_engine.wake_up).parameters:
                        self.inference_engine.wake_up(tags=["kv_cache"])

            log_gpu_memory_usage("After del state_dict and empty_cache in sharding manager", logger=logger)

            # TODO: offload FSDP model weights
            # self.module.cpu()
            # torch.cuda.empty_cache()
            # if torch.distributed.get_rank() == 0:
            # print(f'after model to cpu in sharding manager memory allocated: {torch.cuda.memory_allocated() / 1e9}GB, reserved: {torch.cuda.memory_reserved() / 1e9}GB')

            # important: need to manually set the random states of each tp to be identical.
            if self.device_mesh is not None:
                with self.timing_record("rollout_sharding/enter/setup_rng_state"):
                    self.torch_random_states = torch.cuda.get_rng_state()
                    torch.cuda.set_rng_state(self.gen_random_states)

    @GPUMemoryLogger(role="fsdp vllm sharding_manager", logger=logger)
    def __exit__(self, exc_type, exc_value, traceback):
        # TODO(ZSL): check this
        with self.timing_record("rollout_sharding/exit"):
            if vllm_version in (
                "0.5.4",
                "0.6.3",
            ):
                with self.timing_record("rollout_sharding/exit/offload_model_weights"):
                    self.inference_engine.offload_model_weights()
            else:
                with self.timing_record("rollout_sharding/exit/sleep"):
                    self.inference_engine.sleep(level=1)

            # self.module.to('cuda')
            # if torch.distributed.get_rank() == 0:
            #     print(f'after actor module to cuda in sharding manager memory allocated: {torch.cuda.memory_allocated() / 1e9}GB, reserved: {torch.cuda.memory_reserved() / 1e9}GB')
            with self.timing_record("rollout_sharding_manager/exit/set_train_mode"):
                self.module.train()
            
            # add empty cache after each compute
            with self.timing_record("rollout_sharding_manager/exit/empty_cache_after_compute"):
                torch.cuda.empty_cache()

            # restore random states
            if self.device_mesh is not None:
                with self.timing_record("rollout_sharding_manager/exit/restore_rng_state"):
                    self.gen_random_states = torch.cuda.get_rng_state()
                    torch.cuda.set_rng_state(self.torch_random_states)

    @GPUMemoryLogger(role="fsdp vllm sharding_manager", logger=logger)
    def preprocess_data(self, data: DataProto) -> DataProto:
        """All gather across tp group to make each rank has identical input."""
        if self.tp_size == 1:
            return data

        # TODO: Current impl doesn't consider FSDP with torch micro-dp
        if vllm_version in (
            "0.5.4",
            "0.6.3",
        ):
            group = vllm_ps.get_tensor_model_parallel_group()
        else:
            group = vllm_ps.get_tensor_model_parallel_group().device_group

        all_gather_data_proto(data=data, process_group=group)
        return data

    @GPUMemoryLogger(role="fsdp vllm sharding_manager", logger=logger)
    def postprocess_data(self, data: DataProto) -> DataProto:
        """Get chunk data of this tp rank since we do all gather in preprocess."""
        if self.tp_size == 1:
            return data

        return data.chunk(chunks=self.tp_size)[self.tp_rank]

    def update_params(self, updated_params):
        model = self.inference_engine.llm_engine.model_executor.driver_worker.worker.model_runner.model
        with self.timing_record("rollout_sharding/enter/update_params/patch_moe_loader"):
            patch_vllm_moe_model_weight_loader(model)
        world_size = torch.distributed.get_world_size()
        with self.timing_record("rollout_sharding/enter/update_params/load_weights"):
            loaded_params = model.load_weights(
                ((name, param.full_tensor() if world_size != 1 and hasattr(param, "full_tensor") else param) for name, param in updated_params.items())
            )
        logger.info(f"vLLM load weights, loaded_params: {len(loaded_params)}")

    @contextmanager
    def timing_record(self, method_name : str, **kwargs):
        start_time = time.perf_counter()
        try:
            yield
        finally:
            end_time = time.perf_counter()
            duration = end_time - start_time
            assert method_name not in self.timing_data, method_name
            self.timing_data[method_name] = duration
            #wandb.log({method_name : duration})
            for k,v in kwargs:
                name = method_name + '/' + k
                assert name not in self.timing_data
                self.timing_data[name] = v
                #wandb.log({name : v})

