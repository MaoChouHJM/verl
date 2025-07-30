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
import gc
import torch

from omegaconf import DictConfig
from sglang.srt.entrypoints.engine import Engine
from torch import nn
from torch.distributed.device_mesh import DeviceMesh
from contextlib import contextmanager

from verl.protocol import DataProto, all_gather_data_proto
from verl.utils.device import get_torch_device
from verl.utils.megatron_utils import load_megatron_model_to_gpu, offload_megatron_model_to_cpu, per_tensor_generator
from verl.utils.profiler import GPUMemoryLogger, log_gpu_memory_usage, simple_timer, get_most_used_gpu_memory, calculate_string_md5

from .base import BaseShardingManager

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_PPO_LOGGING_LEVEL", "WARN"))


"""
Megatron Hybrid Engine:
- During training, only the current pp stage holds the parameters
- Before inference, broadcast the parameters of the current pp rank to all other pp ranks (all pp ranks holds all 
  the parameters)
- Bind the parameters to the inference engine
- Do inference in tp. pp is treated as additional dp
- After inference, all the parameters that doesn't belong to this pp rank is freed.
"""


class MegatronSGLangShardingManager(BaseShardingManager):
    """A sharding manager for Megatron-style training & inference with SGLang.

    This class manages the sharding of model parameters between training and inference
    phases in a Megatron-style parallel setup. It handles:
    - Loading/offloading parameters between CPU/GPU
    - Updating inference engine weights
    - Managing random states for reproducibility
    - Data preprocessing for distributed inference

    Args:
        actor_module (nn.ModuleList): The actor model modules
        inference_engine (Engine): The SGLang inference engine
        model_config: Configuration for the actor's model
        rollout_config: Configuration for rollout generation
        transformer_config: Transformer-specific configuration
        layer_name_mapping: Mapping between layer names and parameters
        weight_converter: Utility for converting weights between formats
        device_mesh (DeviceMesh | None): PyTorch device mesh for distributed training
        offload_param (bool): Whether to offload parameters to CPU when not in use
    """

    def __init__(
        self,
        actor_module: nn.ModuleList,
        inference_engine: Engine,
        model_config: DictConfig,
        rollout_config: DictConfig,
        transformer_config,
        layer_name_mapping,
        weight_converter,
        device_mesh: DeviceMesh | None = None,
        offload_param: bool = False,
        bridge=None,
    ):
        self.actor_module = actor_module
        self.inference_engine = inference_engine
        self.model_config = model_config
        self.rollout_config = rollout_config
        self.transformer_config = transformer_config
        self.layer_name_mapping = layer_name_mapping
        self.weight_converter = weight_converter
        self.offload_param = offload_param
        self.device_mesh = device_mesh
        self.bridge = bridge
        self.offload_param = offload_param

        if self.device_mesh is not None:
            self.infer_tp_size = self.device_mesh["tp"].mesh.size()[0]
        else:
            self.infer_tp_size = self.inference_engine._tp_size

        # Note that torch_random_states may be different on each dp rank
        self.torch_random_states = get_torch_device().get_rng_state()
        # get a random rng states
        if self.device_mesh is not None:
            gen_dp_rank = self.device_mesh["dp"].get_local_rank()
            get_torch_device().manual_seed(gen_dp_rank + 1000)  # make sure all tp ranks have the same random states
            self.gen_random_states = get_torch_device().get_rng_state()
            get_torch_device().set_rng_state(self.torch_random_states)
        else:
            self.gen_random_states = None
        self.timing_data = {}

    @GPUMemoryLogger(role="MegatronSGLangShardingManager enter", logger=logger)
    def __enter__(self):
        self.timing = {}
        with simple_timer("reshard", self.timing):
            loop = asyncio.get_event_loop()
            loop.run_until_complete(self.wake_up())

    @GPUMemoryLogger(role="MegatronSGLangShardingManager exit", logger=logger)
    def __exit__(self, exc_type, exc_value, traceback):
        loop = asyncio.get_event_loop()
        loop.run_until_complete(self.sleep())

    async def update_weights(self, params):
        def is_reach_max_packed_usage(packed_name_tensor, free_mem):
            size = sum([t.numel() * t.element_size() for _, t in packed_name_tensor])
            # mcore : 1; sglang: 1 (deserialize) + 1 (broadcast)
            total_size = size * 5 / (1024**3)
            return  total_size > free_mem

        timing = {}
        if self.device_mesh["tp"].get_local_rank() == 0 and self.rollout_config.free_cache_engine:
            await self.inference_engine.resume_memory_occupation()

        # Most naive implementation, can optimize a lot if it is bottleneck from sglang Engine weight update
        # named_tensors = [(k, v) for k, v in params.items()]
        named_tensors = params
        load_format = None
        with simple_timer("weight_update_total", timing):
            with simple_timer("get_gpu_info", timing):
                gpu_info = get_most_used_gpu_memory()
            free_mem = gpu_info["free_memory_mb"] / 1024 if gpu_info is not None else -1  
            start_time = time.time()
            packed_name_tensor = []
            for tensor_index, (param_name, tensor) in enumerate(named_tensors):
                packed_name_tensor.append((param_name, tensor.detach()))
                if not is_reach_max_packed_usage(packed_name_tensor, free_mem):
                    continue
                cur_time = time.time()
                name = ",".join([n for (n, _) in packed_name_tensor])
                if "," in name:
                    name = calculate_string_md5(name)
                timing[f'generate_weight_{name}'] = cur_time - start_time
                timing[f'packed_tensor_size_{name}'] = sum([t.numel() * t.element_size() for _, t in packed_name_tensor]) / (1024**3)
                timing[f'max_mem_allocated_gb_{name}'] = get_torch_device().max_memory_allocated() / (1024**3) 
                timing[f'max_mem_reserved_gb_{name}'] = get_torch_device().max_memory_reserved() / (1024**3) 
                with simple_timer(f"update_tensor_{name}", timing):
                    if self.device_mesh["tp"].get_local_rank() == 0:
                        success, sglang_time = await self.inference_engine.update_weights_from_tensor(
                            named_tensors=packed_name_tensor,
                            load_format=load_format,
                            flush_cache=True,
                        )
                        timing[f'sglang_cost_time_{name}'] = sglang_time
                with simple_timer(f"flush_cache_{name}", timing):
                    if self.device_mesh["tp"].get_local_rank() == 0:
                        await self.inference_engine.flush_cache()
                with simple_timer(f'empty_mcore_cahce_{name}', timing):
                    packed_name_tensor = []
                    gc.collect()
                    torch.cuda.empty_cache()
                start_time = time.time()
            else:
                if len(packed_name_tensor) != 0:
                    cur_time = time.time()
                    name = ",".join([n for (n, _) in packed_name_tensor])
                    if "," in name:
                        name = calculate_string_md5(name)
                    timing[f'generate_weight_{name}'] = cur_time - start_time
                    timing[f'packed_tensor_size_{name}'] = sum([t.numel() * t.element_size() for _, t in packed_name_tensor]) / (1024**3)
                    timing[f'max_mem_allocated_gb_{name}'] = get_torch_device().max_memory_allocated() / (1024**3) 
                    timing[f'max_mem_reserved_gb_{name}'] = get_torch_device().max_memory_reserved() / (1024**3) 
                    with simple_timer(f"update_tensor_{name}", timing):
                        if self.device_mesh["tp"].get_local_rank() == 0:
                            success, sglang_time = await self.inference_engine.update_weights_from_tensor(
                                named_tensors=packed_name_tensor,
                                load_format=load_format,
                                flush_cache=True,
                            )
                            timing[f'sglang_cost_time_{name}'] = sglang_time
                    with simple_timer(f"flush_cache_{name}", timing):
                        if self.device_mesh["tp"].get_local_rank() == 0:
                            await self.inference_engine.flush_cache()
                    with simple_timer(f'empty_mcore_cahce_{name}', timing):
                        packed_name_tensor = []
                        gc.collect()
                        torch.cuda.empty_cache()
            assert len(packed_name_tensor) == 0

            with simple_timer(f"post_update_weight", timing):
                if self.device_mesh["tp"].get_local_rank() == 0:
                    await self.inference_engine.post_load_weights_from_tensor()



        if not hasattr(self, '_first_call_update_weights') or not self._first_call_update_weights:
            self._first_call_update_weights = True
            if self.rollout_config.get("debug_dump_sglang_tensor", None):
                if self.device_mesh.get_rank() == 0:
                    import pickle
                    from datetime import datetime
                    now = datetime.now()
                    formatted_date_time = now.strftime("%Y-%m-%d_%H_%M_%S")
                    dump_file = os.path.dirname(os.path.abspath(__file__)) + f"/cost_{formatted_date_time}.pkl"
                    dump_pth_file = os.path.dirname(os.path.abspath(__file__)) + f"/dumped_tensor_{formatted_date_time}/"
                    await self.inference_engine.dump_weights(output_path=dump_pth_file)
                    with open(dump_file, 'wb') as f:
                        pickle.dump(timing, f) 
                        print(f'update_weights_from_tensor perf file: {dump_file} has dumped')

            
    async def release_memory(self):
        if self.device_mesh["tp"].get_local_rank() == 0 and self.rollout_config.free_cache_engine:
            await self.inference_engine.release_memory_occupation()

    @GPUMemoryLogger(role="MegatronSGLangShardingManager enter", logger=logger)
    async def wake_up(self):
        if self.offload_param:
            load_megatron_model_to_gpu(self.actor_module, load_grad=False)
        if self.bridge is not None:
            per_tensor_param = self.bridge.export_weights(self.actor_module)
        else:
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
        get_torch_device().empty_cache()
        # important: need to manually set the random states of each tp to be identical.
        if self.device_mesh is not None:
            self.torch_random_states = get_torch_device().get_rng_state()
            get_torch_device().set_rng_state(self.gen_random_states)

    @GPUMemoryLogger(role="MegatronSGLangShardingManager exit", logger=logger)
    async def sleep(self):
        if self.rollout_config.free_cache_engine:
            log_gpu_memory_usage("Before SGLang offload in sharding manager", logger=logger)
            await self.release_memory()
            log_gpu_memory_usage("After SGLang offload in sharding manager", logger=logger)

        for model in self.actor_module:
            model.train()
        # add empty cache after each compute
        get_torch_device().empty_cache()

        # restore random states
        if self.device_mesh is not None:
            self.gen_random_states = get_torch_device().get_rng_state()
            get_torch_device().set_rng_state(self.torch_random_states)

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
