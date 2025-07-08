# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
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
"""
This file contains a Megatron style Hybrid Engine that shares the weights of the actor with the inference engine.
"""

import asyncio
import logging
import os
import time

import torch
from sglang.srt.entrypoints.engine import Engine
from torch import nn
from torch.distributed.device_mesh import DeviceMesh
from contextlib import contextmanager

from verl.protocol import DataProto, all_gather_data_proto
from verl.utils.debug import GPUMemoryLogger, log_gpu_memory_usage
from verl.utils.debug.performance import _timer
from verl.utils.megatron_utils import per_tensor_generator, load_megatron_model_to_gpu, offload_megatron_model_to_cpu

from .base import BaseShardingManager

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_PPO_LOGGING_LEVEL", "WARN"))


"""
Megatron Hybrid Engine:
- During training, only the current pp stage holds the parameters
- Before inference, broadcast the parameters of the current pp rank to all other pp ranks (all pp ranks holds all the parameters)
- Bind the parameters to the inference engine
- Do inference in tp. pp is treated as additional dp
- After inference, all the parameters that doesn't belong to this pp rank is freed.
"""


class MegatronSGLangShardingManager(BaseShardingManager):
    def __init__(
        self,
        actor_module: nn.ModuleList,
        inference_engine: Engine,
        model_config,
        transformer_config,
        layer_name_mapping,
        weight_converter,
        offload_param,
        device_mesh: DeviceMesh | None = None,
    ):
        self.actor_module = actor_module
        self.inference_engine = inference_engine
        self.model_config = model_config
        self.transformer_config = transformer_config
        self.layer_name_mapping = layer_name_mapping
        self.weight_converter = weight_converter
        self.offload_param = offload_param
        self.device_mesh = device_mesh

        if self.device_mesh is not None:
            self.infer_tp_size = self.device_mesh["tp"].mesh.size()[0]
        else:
            self.infer_tp_size = self.inference_engine._tp_size

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

    @GPUMemoryLogger(role="MegatronSGLangShardingManager enter", logger=logger)
    def __enter__(self):
        self.timing = {}
        with self.timing_record("rollout_sharding/enter"), _timer("reshard", self.timing):
            with self.timing_record("rollout_sharding/per_tensor_generator"):
                per_tensor_param = per_tensor_generator(
                    self.actor_module,
                    self.model_config,
                    self.weight_converter,
                    self.transformer_config,
                    self.layer_name_mapping,
                )
            loop = asyncio.get_event_loop()
            with self.timing_record("rollout_sharding/update_weights"):
                loop.run_until_complete(self.update_weights(per_tensor_param))
            # important: need to manually set the random states of each tp to be identical.
            if self.device_mesh is not None:
                self.torch_random_states = torch.cuda.get_rng_state()
                torch.cuda.set_rng_state(self.gen_random_states)

    @GPUMemoryLogger(role="MegatronSGLangShardingManager exit", logger=logger)
    def __exit__(self, exc_type, exc_value, traceback):
        with self.timing_record("rollout_sharding/exit"):
            log_gpu_memory_usage("Before SGLang offload in sharding manager", logger=logger)
            loop = asyncio.get_event_loop()
            with self.timing_record("rollout_sharding/release_memory"):
                loop.run_until_complete(self.release_memory())
            log_gpu_memory_usage("After SGLang offload in sharding manager", logger=logger)

            for model in self.actor_module:
                model.train()
            # add empty cache after each compute
            with self.timing_record("rollout_sharding/empty_cache"):
                torch.cuda.empty_cache()

            # restore random states
            if self.device_mesh is not None:
                self.gen_random_states = torch.cuda.get_rng_state()
                torch.cuda.set_rng_state(self.torch_random_states)

    async def update_weights(self, params):
        timing = {}
        with _timer("resume_memory_occupation", timing):
            if self.device_mesh["tp"].get_local_rank() == 0:
                await self.inference_engine.resume_memory_occupation()

        # Most naive implementation, can optimize a lot if it is bottleneck from sglang Engine weight update
        # named_tensors = [(k, v) for k, v in params.items()]
        named_tensors = params
        load_format = None
        with _timer("weight_update_total", timing):
            start_time = time.time()
            for tensor_index, (name, tensor) in enumerate(named_tensors):
                cur_time = time.time()
                timing[f'generate_weight_{name}'] = cur_time - start_time
                timing[f'weight_shape_{name}'] = tensor.shape
                timing[f'weight_dtype_{name}'] = tensor.dtype
                with _timer(f"update_tensor_{name}", timing):
                    if self.device_mesh["tp"].get_local_rank() == 0:
                        success, sglang_time = await self.inference_engine.update_weights_from_tensor(
                            named_tensors=[
                                (
                                    name,
                                    tensor.detach(),
                                )
                            ],
                            load_format=load_format,
                            flush_cache=False,
                        )
                        timing[f'sglang_cost_time_{name}'] = sglang_time
                with _timer(f"flush_cache_{name}", timing):
                    if self.device_mesh["tp"].get_local_rank() == 0:
                        await self.inference_engine.flush_cache()
                start_time = time.time()

        if not hasattr(self, '_first_call_update_weights') or not self._first_call_update_weights:
            self._first_call_update_weights = True

            if self.device_mesh.get_rank() == 0:
                import pickle
                from datetime import datetime
                now = datetime.now()
                formatted_date_time = now.strftime("%Y-%m-%d_%H_%M_%S")
                dump_file = os.getcwd() + f"/cost_{formatted_date_time}.pkl"
                with open(dump_file, 'wb') as f:
                    pickle.dump(timing, f) 
                    print(f'update_weights_from_tensor perf file: {dump_file} has dumped')

            
    async def release_memory(self):
        if self.device_mesh["tp"].get_local_rank() == 0:
            await self.inference_engine.release_memory_occupation()

    @GPUMemoryLogger(role="FSDPSGLangShardingManager enter", logger=logger)
    async def wake_up(self):
        if self.offload_param:
            load_megatron_model_to_gpu(self.actor_module, load_grad=False)
        per_tensor_param = per_tensor_generator(
            self.actor_module,
            self.model_config,
            self.weight_converter,
            self.transformer_config,
            self.layer_name_mapping,
        )
        await self.update_weights(per_tensor_param)
        if self.offload_param:
            offload_megatron_model_to_cpu(self.actor_module)
        # important: need to manually set the random states of each tp to be identical.
        if self.device_mesh is not None:
            self.torch_random_states = torch.cuda.get_rng_state()
            torch.cuda.set_rng_state(self.gen_random_states)

    @GPUMemoryLogger(role="FSDPSGLangShardingManager exit", logger=logger)
    async def sleep(self):
        log_gpu_memory_usage("Before SGLang offload in sharding manager", logger=logger)
        await self.release_memory()
        log_gpu_memory_usage("After SGLang offload in sharding manager", logger=logger)

        for model in self.actor_module:
            model.train()
        # add empty cache after each compute
        torch.cuda.empty_cache()

        # restore random states
        if self.device_mesh is not None:
            self.gen_random_states = torch.cuda.get_rng_state()
            torch.cuda.set_rng_state(self.torch_random_states)

    @GPUMemoryLogger(role="megatron sglang sharding_manager", logger=logger)
    def preprocess_data(self, data: DataProto) -> DataProto:
        # DP_COMPUTE_PROTO: all training ranks are dp, the same as fsdp
        if self.infer_tp_size == 1:
            return data
        all_gather_data_proto(data, self.device_mesh["tp"].get_group())
        return data

    @GPUMemoryLogger(role="megatron sglang sharding_manager", logger=logger)
    def postprocess_data(self, data: DataProto) -> DataProto:
        # DP_COMPUTE_PROTO: all training ranks are dp, the same as fsdp
        if self.infer_tp_size == 1:
            return data
        return data.chunk(chunks=self.infer_tp_size)[self.device_mesh["tp"].get_local_rank()]

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
