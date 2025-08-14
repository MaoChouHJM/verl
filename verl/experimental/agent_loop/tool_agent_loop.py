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
import os
import re
import time
import uuid
from abc import abstractmethod
from contextlib import asynccontextmanager
from typing import Any
from uuid import uuid4

import torch
from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel
from verl.utils.py_functional import simple_timer

from verl.experimental.agent_loop import AgentLoopBase
from verl.tools.utils.tool_registry import ToolRegistry
from verl.utils.huggingface_utils import hf_tokenizer
from verl.utils.logger import create_logger
from .schemas import (AgentLoopOutput, Finish, Generate, PostProcess,
                                           ToolCall, ToolResponse)

logger = create_logger(__file__)


class ToolResponse(BaseModel):
    response: Any = None
    image_path: str = None
    mask: str = None
    reason: str = None


class ToolAgentLoop(AgentLoopBase):
    def __init__(self, config: DictConfig, role: str = None) -> None:
        super().__init__(config)
        self.eos_token_id = config.eos_token_id

        if role is not None:
            self.config = config[role]

        # initialize tokenizer
        self.tokenizer = hf_tokenizer(self.config.model.path, trust_remote_code=self.config.model.trust_remote_code)

        local_config_path = os.path.join(self.config.model.path, "config.json")
        if os.path.exists(local_config_path):
            self.processor = None
            try:
                from transformers import AutoProcessor

                self.processor = AutoProcessor.from_pretrained(
                    self.config.model.path, trust_remote_code=self.config.model.trust_remote_code
                )
                logger.info("Successfully loaded processor")
            except Exception as e:
                logger.info(f"Failed to load processor: {e}")

        # Keeping both functionalities with clear comments
        # TODO: load tool config from a separate file
        self.tool_registry = ToolRegistry()
        self.loop = asyncio.get_event_loop()
        self.server_manager = None
        self.is_vision_model = getattr(config.model, "is_vision_model", False)
        self.apply_chat_template_kwargs = getattr(config, "apply_chat_template_kwargs", {})

    def bind_server_manager(self, server_manager):
        self.server_manager = server_manager

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        messages = list(kwargs["raw_prompt"])
        # Functionality from HEAD branch
        image_data = copy.deepcopy(kwargs.get("multi_modal_data", {}).get("image", None))
        image_data_url = copy.deepcopy(kwargs.get("images", ""))
        print(f"[DEBUG] at tool_agent_loop.py, {image_data_url=}")
        # TODO(huangjiaming): support video data
        video_data = copy.deepcopy(kwargs.get("multi_modal_data", {}).get("video", None))
        # Alternative functionality from main branch
        # image_data = copy.deepcopy(kwargs.get("multi_modal_data", {}).get("image", None))
        
        metrics = {}
        request_id = uuid4().hex
        if self.processor is not None:
            raw_prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                **self.apply_chat_template_kwargs,
            )
            # Functionality from HEAD branch
            model_inputs = self.processor(text=[raw_prompt], images=image_data, videos=video_data, return_tensors="pt")
            # print(f"[DEBUG] at tool_agent_loop, {messages=}, {raw_prompt=}")
            # Alternative functionality from main branch
            # model_inputs = self.processor(text=[raw_prompt], images=image_data, return_tensors="pt")
            
            prompt_ids = model_inputs.pop("input_ids").squeeze(0).tolist()
        else:
            prompt_ids = await self.loop.run_in_executor(
                None,
                self.tokenizer.apply_chat_template,
                messages,
                False,  # tokenize
                True,  # add_generation_prompt
                None,  # padding
                None,  # truncation
                None,  # max_length
                None,  # kwargs
                self.apply_chat_template_kwargs,
            )
            prompt_ids = await self.loop.run_in_executor(
                None,
                self.tokenizer.encode,
                prompt_ids,
            )

        response_ids, response_mask = [], []
        cur_step = 0
        max_step = self.config.get("max_step", 10)
        finish_reason = "max_step"

        while True:
            with simple_timer("generate_sequences", metrics):
                # Functionality from HEAD branch
                response_ids = await self.server_manager.generate(
                    request_id=request_id, prompt_ids=prompt_ids, sampling_params=sampling_params, image_data=image_data_url
                )
                # Alternative functionality from main branch
                # response_ids = await self.server_manager.generate(
                #     request_id=request_id, prompt_ids=prompt_ids, sampling_params=sampling_params, image_data=image_data
                # )
                
            prompt_ids += response_ids
            response_mask += [1] * len(response_ids)
            response_text = self.tokenizer.decode(response_ids, skip_special_tokens=True)
            logger.info(f"Step {cur_step} response_text: {response_text}")
            messages.append({"role": "assistant", "content": response_text})

            if response_text.strip().endswith(self.config.response_suffix):
                logger.info("Response ends with suffix, break")
                finish_reason = "suffix"
                break

            if cur_step >= max_step:
                logger.info("Reach max step, break")
                break

            cur_step += 1

        input_ids = prompt_ids[: -len(response_ids)]
        input_mask = [1] * len(input_ids)

        # postprocess response_text, extract <function=xxx> ...
        tool_calls = re.findall(r"<function=(\w+)>(.*?)
