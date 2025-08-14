#  Copyright 2024 Bytedance Ltd. and/or its affiliates
#
#  Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import asyncio
import copy
import json
import logging
import os
import random
from abc import ABC, abstractmethod
# Combined functionality from both branches
from re import T
from typing import Any, Optional

import hydra
import numpy as np
import torch
import torch.distributed as dist
from cachetools import LRUCache
from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel, ConfigDict
from tensordict import TensorDict
# Combined imports from both branches
from transformers import AutoProcessor, AutoTokenizer, AutoConfig

from verl.protocol import DataProto
from verl.single_controller.ray.base import RayWorkerGroup
from verl.utils.huggingface_utils import (copy_to_local, get_dtype,
                                           hf_model_from_config,
                                           init_hf_tokenizer, model_load)
from verl.utils.import_utils import import_version, is_package_available
from verl.utils.model import normalize_logits
from verl.utils.torch_functional import compute_position_id_with_mask
from verl.workers.rollout.schemas import AgentOutputSchema
from verl.workers.rollout.tokenizer import TokenizerHelper


logger = logging.getLogger(__file__)


class AgentOutput(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    input_ids: torch.Tensor
    response_ids: torch.Tensor
    input_mask: torch.Tensor
    response_mask: torch.Tensor
    multi_modal_data: Optional[dict] = None


class TrajectoryInfo(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    trajectory_id: str
    step_id: int
    is_finished: bool = False


async def get_trajectory_info(global_step: int, index: int, validate: bool) -> TrajectoryInfo:
    # This is a mock implementation. In practice, this would interact with a database or external system.
    trajectory_id = f"traj_{global_step}_{index}"
    step_id = global_step
    return TrajectoryInfo(trajectory_id=trajectory_id, step_id=step_id)


class AgentLoopBase(ABC):
    def __init__(self, config: DictConfig) -> None:
        self.config = config

    @abstractmethod
    def chat(
        self,
        input_ids: torch.Tensor,
        response_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        input_mask: torch.Tensor,
        response_mask: torch.Tensor,
        meta_info: Optional[dict] = None,
        **kwargs,
    ) -> AgentOutput:
        """Chat with the agent. Return AgentOutput object

        Args:
            input_ids: torch.Tensor. Shape: (batch_size, seq_len)
            response_ids: torch.Tensor. Shape: (batch_size, response_len)
            attention_mask: torch.Tensor. Shape: (batch_size, seq_len)
            input_mask: torch.Tensor. Shape: (batch_size, seq_len)
            response_mask: torch.Tensor. Shape: (batch_size, response_len)
            meta_info: Optional[dict]. Meta information for the agent.
        """
        raise NotImplementedError


class AgentLoopManager:
    def __init__(self, config: DictConfig) -> None:
        self.config = config
        self.agent_loop_mapping: dict[str, AgentLoopBase] = {}

    def register_agent_loop(self, name: str, agent_loop: AgentLoopBase) -> None:
        self.agent_loop_mapping[name] = agent_loop

    def get_agent_loop(self, name: str) -> AgentLoopBase:
        return self.agent_loop_mapping[name]


class AgentLoopWorker:
    def __init__(self, config: DictConfig, role: str = "") -> None:
        self.config = config
        self.role = role

        # initialize tokenizer and processor
        local_path = copy_to_local(config.actor_rollout_ref.model.path)
        self.tokenizer = init_hf_tokenizer(local_path, trust_remote_code=True)
        self.processor = AutoProcessor.from_pretrained(local_path, trust_remote_code=True) if is_package_available(
            "transformers", ">=4.40.0"
        ) else None
        # Functionality from HEAD branch
        self.hf_config = AutoConfig.from_pretrained(local_path, trust_remote_code=True)

        # Configuration from HEAD branch
        agent_loop_config_path = config.actor_rollout_ref.rollout.agent.get("agent_loop_config_path", None)
        # Alternative configuration from main branch
        # agent_loop_config_path = config.actor_rollout_ref.rollout.agent.agent_loop_config_path
        
        if agent_loop_config_path:
            agent_loop_configs = OmegaConf.load(agent_loop_config_path)
            for agent_loop_config in agent_loop_configs:
                hydra.initialize_config_dir(config_dir=os.path.dirname(agent_loop_config_path), version_base=None)
                agent_loop_cfg = hydra.compose(config_name=os.path.basename(agent_loop_config_path))
                agent_loop = hydra.utils.instantiate(agent_loop_cfg)
                self.agent_loop_manager.register_agent_loop(agent_loop_cfg.name, agent_loop)

        self.tokenizer_helper = TokenizerHelper(tokenizer=self.tokenizer)

        # offload model to CPU to save memory
        self.enable_cpu_offload = config.actor_rollout_ref.actor.enable_cpu_offload_during_generation

    async def generate_async(self, batch: DataProto) -> DataProto:
        logger.info(f"AgentLoopWorker.generate_async receive {len(batch)} examples...")
        batch = batch.to("cuda", non_blocking=True)

        input_ids = batch.batch["input_ids"]
        attention_mask = batch.batch["attention_mask"]

        # by default, we assume it's a single turn agent
        if "agent_name" not in batch.non_tensor_batch:
            # Configuration from HEAD branch
            batch.non_tensor_batch["agent_name"] = np.array(["tool_agent"] * len(batch), dtype=object)
            # Alternative configuration from main branch
            # batch.non_tensor_batch["agent_name"] = np.array(["single_turn_agent"] * len(batch), dtype=object)

        # Configuration from HEAD branch
        if "index" in batch.non_tensor_batch:
            index = batch.non_tensor_batch["index"]
        else:
            index = np.arange(len(batch))
        
        trajectory_info = await get_trajectory_info(
            batch.meta_info.get("global_steps", -1), index, batch.meta_info.get("validate", False)
        )
        # Alternative configuration from main branch
        # if "index" in batch.non_tensor_batch:
        #     index = batch.non_tensor_batch["index"]
        # else:
        #     index = np.arange(len(batch))
        # 
        # trajectory_info = await get_trajectory_info(
        #     batch.meta_info.get("global_steps", -1), index, batch.meta_info.get("validate", False)
        # )

        tasks = []
        for i in range(len(batch)):
            kwargs = {k: v[i] for k, v in batch.non_tensor_batch.items()}
            input_ids_i = input_ids[i : i + 1]
            attention_mask_i = attention_mask[i : i + 1]
            trajectory_info_i = trajectory_info[i] if isinstance(trajectory_info, list) else trajectory_info

            task = asyncio.create_task(
                self._generate_single_async(
                    input_ids_i, attention_mask_i, trajectory_info_i=trajectory_info_i, **kwargs
                )
            )
            tasks.append(task)

        outputs: list[AgentOutput] = await asyncio.gather(*tasks)

        batch_size = len(outputs)
        max_response_len = max([output.response_ids.shape[1] for output in outputs])

        padded_response_ids = []
        padded_response_masks = []

        for output in outputs:
            response_ids = output.response_ids
            response_mask = output.response_mask

            pad_len = max_response_len - response_ids.shape[1]
            if pad_len > 0:
                response_ids = torch.cat(
                    [response_ids, torch.zeros(1, pad_len, dtype=response_ids.dtype, device=response_ids.device)],
                    dim=1,
                )
                response_mask = torch.cat(
                    [response_mask, torch.zeros(1, pad_len, dtype=response_mask.dtype, device=response_mask.device)],
                    dim=1,
                )
            padded_response_ids.append(response_ids)
            padded_response_masks.append(response_mask)

        batch_response_ids = torch.cat(padded_response_ids, dim=0)
        batch_response_mask = torch.cat(padded_response_masks, dim=0)

        # remove eos_token_id from response_ids if exist
        for i in range(batch_size):
            response_ids = batch_response_ids[i : i + 1]
            response_mask = batch_response_mask[i : i + 1]

            eos_token_id = self.tokenizer.eos_token_id
            if eos_token_id is not None:
                eos_positions = (response_ids == eos_token_id).nonzero()
                if eos_positions.numel() > 0:
                    # Find the first eos position
                    first_eos_position = eos_positions[0]
                    # Set all tokens after the first eos to pad_token_id
                    response_ids[0, first_eos_position[1] :] = self.tokenizer.pad_token_id
                    response_mask[0, first_eos_position[1] :] = 0

        # construct a new DataProto with the response_ids and attention_mask
        response_data_proto = DataProto(
            batch={
                "response_ids": batch_response_ids,
                "response_mask": batch_response_mask,
            },
            non_tensor_batch={
                "multi_modal_data": [output.multi_modal_data for output in outputs],
            },
            meta_info=batch.meta_info,
        )
        logger.info(f"AgentLoopWorker.generate_async finished {len(batch)} examples")
        return response_data_proto

    async def _generate_single_async(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        trajectory_info: TrajectoryInfo,
        **kwargs,
    ) -> AgentOutput:
        # This is a simplified implementation. In practice, this would interact with the actual agent loop.
        device = input_ids.device
        
        # Functionality from HEAD branch
        if self.processor is not None:
            # Combined functionality for different processor types
            if "Qwen2VLImageProcessor" in self.processor.image_processor.__class__.__name__:
                from verl.models.transformers.qwen2_vl import get_rope_index
            elif self.hf_config.architectures[0] == "KeyeForConditionalGeneration":
                 from examples.keye.processors.utils_slowfast import get_rope_index_slowfast as get_rope_index
            else:
                raise NotImplementedError(f"not implement get_rope_index for {self.hf_config['architectures'][0]}")
            
            images = kwargs.get("images", None)
            videos = kwargs.get("videos", None)
            
            current_text = self.tokenizer.decode(input_ids.squeeze(0), skip_special_tokens=True)
            multi_modal_inputs = self.processor(text=[current_text], images=images, videos=videos, return_tensors="pt")
        # Alternative functionality from main branch
        # if (
        #     self.processor is not None
        #     and "Qwen2VLImageProcessor" in self.processor.image_processor.__class__.__name__
        # ):
        #     from verl.models.transformers.qwen2_vl import get_rope_index
        # 
        #     images = kwargs.get("images", [])
        #     current_text = self.tokenizer.decode(input_ids.squeeze(0), skip_special_tokens=True)
        #     multi_modal_inputs = self.processor(text=[current_text], images=images, return_tensors="pt")
        # End of alternative functionality
        
        if self.processor is not None and multi_modal_inputs is not None:
            multi_modal_inputs.pop("input_ids", None)
            multi_modal_inputs.pop("attention_mask", None)

            image_grid_thw = multi_modal_inputs.get("image_grid_thw")
            video_grid_thw = multi_modal_inputs.get("video_grid_thw")
            second_per_grid_ts = multi_modal_inputs.get("second_per_grid_ts")
            # Functionality from HEAD branch
            fast_video_grid_thw = multi_modal_inputs.get("fast_video_grid_thw")

            if "Qwen2VLImageProcessor" in self.processor.image_processor.__class__.__name__:
                position_ids = get_rope_index(
                    self.processor,
                    input_ids=input_ids.squeeze(0),
                    image_grid_thw=image_grid_thw,
                    video_grid_thw=video_grid_thw,
                    second_per_grid_ts=second_per_grid_ts,
                    attention_mask=attention_mask.squeeze(0),
                ).unsqueeze(0)  # (1, 3, seq_len)
            elif self.hf_config.architectures[0] == "KeyeForConditionalGeneration":
                position_ids = get_rope_index(
                    input_ids=input_ids,
                    image_grid_thw=image_grid_thw,
                    video_grid_thw=video_grid_thw,
                    fast_video_grid_thw=fast_video_grid_thw,
                    spatial_merge_size=self.hf_config.vision_config.spatial_merge_size,
                    image_token_id=self.hf_config.image_token_id,
                    video_token_id=self.hf_config.video_token_id,
                    vision_start_token_id=self.hf_config.vision_start_token_id,
                    fast_video_token_id=self.hf_config.fast_video_token_id
                ).transpose(0,1)  # (bs, 3, seq_len)
            else:
                raise NotImplementedError(f"not implement get_rope_index for {self.hf_config['architectures'][0]}")
            # Alternative functionality from main branch
            # position_ids = get_rope_index(
            #     self.processor,
            #     input_ids=input_ids.squeeze(0),
            #     image_grid_thw=image_grid_thw,
            #     video_grid_thw=video_grid_thw,
            #     second_per_grid_ts=second_per_grid_ts,
            #     attention_mask=attention_mask.squeeze(0),
            # ).unsqueeze(0)  # (1, 3, seq_len)
            # End of alternative functionality
        else:
            position_ids = compute_position_id_with_mask(attention_mask)  # (1, seq_len)

        response_ids = torch.randint(0, 1000, (1, 10), device=device)  # Mock response
        response_mask = torch.ones_like(response_ids)

        return AgentOutput(
            input_ids=input_ids,
            response_ids=response_ids,
            input_mask=attention_mask,
            response_mask=response_mask,
            multi_modal_data=kwargs.get("multi_modal_data", None),
        )
