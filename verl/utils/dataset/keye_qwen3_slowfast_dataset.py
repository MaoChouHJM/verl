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

import copy
import json
import os
import re
from collections import defaultdict
from tkinter import NE, N
from typing import List, Optional, Union

import datasets
import numpy as np
import torch
from omegaconf import DictConfig, ListConfig
from transformers import AutoConfig, PreTrainedTokenizer, ProcessorMixin

import verl.utils.torch_functional_vl as verl_F
from verl.utils.dataset import RLHFDataset


def collate_fn(data_list: list[dict]) -> dict:
    tensors = defaultdict(list)
    non_tensors = defaultdict(list)

    for data in data_list:
        for key, val in data.items():
            if isinstance(val, torch.Tensor):
                tensors[key].append(val)
            else:
                non_tensors[key].append(val)

    for key, val in tensors.items():
        tensors[key] = torch.stack(val, dim=0)

    for key, val in non_tensors.items():
        non_tensors[key] = np.array(val, dtype=object)

    return {**tensors, **non_tensors}


class KeyeQwen3SlowFastDataset(RLHFDataset):
    """
    We assume the dataset contains a column that contains prompts and other information
    """

    def __init__(
        self,
        data_files: Union[str, List[str]],
        tokenizer: PreTrainedTokenizer,
        config: DictConfig,
        processor: Optional[ProcessorMixin] = None,
    ):
        if not isinstance(data_files, (List, ListConfig)):
            data_files = [data_files]

        self.data_files = copy.deepcopy(data_files)
        self.original_data_files = copy.deepcopy(data_files)  # use for resume
        self.tokenizer = tokenizer
        self.processor = processor
        base_model_dir = config.get("base_model_dir", None)
        assert self.tokenizer is not None
        assert self.processor is not None
        assert base_model_dir is not None
        self.config = config
        self.hf_config = AutoConfig.from_pretrained(base_model_dir, trust_remote_code=True)
        
        if os.environ.get("USE_SLOW_FAST", "false").lower() == "true":
            from examples.keye.processors.utils_slowfast import (
                get_rope_index_slowfast, process_vision_info)
            self.process_vision_info_func = process_vision_info
            self.get_rope_index_func = get_rope_index_slowfast
        else:
            from .keye_utils.keye_vl_utils import process_vision_info, get_rope_index
            self.process_vision_info_func = process_vision_info
            self.get_rope_index_func = get_rope_index

        self.cache_dir = os.path.expanduser(config.get("cache_dir", "~/.cache/verl/rlhf"))
        self.prompt_key = config.get("prompt_key", "conversations")
        self.image_key = config.get("image_key", "images")
        self.video_key = config.get("video_key", "videos")
        self.max_prompt_length = config.get("max_prompt_length", 8*1024)

        self.return_raw_chat = config.get("return_raw_chat", False)
        self.truncation = config.get("truncation", "error")

        self.num_workers = config.get("filter_overlong_prompts_workers", max(1, os.cpu_count() // 4))
        self.num_workers = min(self.num_workers, os.cpu_count())
        # whether to store the dataset in state_dict()
        # default not store
        self.serialize_dataset = False
        self._read_files_and_tokenize()

    def _read_files_and_tokenize(self):
        dataframes = []
        for parquet_file in self.data_files:
            # read parquet files and cache
            dataframe = datasets.load_dataset("parquet", data_files=parquet_file)["train"]
            dataframes.append(dataframe)
        self.dataframe: datasets.Dataset = datasets.concatenate_datasets(dataframes)

        print(f"dataset len: {len(self.dataframe)}")

    def resume_dataset_state(self):
        self.serialize_dataset = not hasattr(self, "original_data_files")
        # resume dataframe if not it's serialized in data.pt
        if not self.serialize_dataset:
            self._download(use_origin_parquet=True)  # download and resume from original parquet files
            self._read_files_and_tokenize()
        else:
            print(r"old dataloader ckpt file is used, please train from scratch for better ckpt performance")

    def __len__(self):
        return len(self.dataframe)

    def _build_messages(self, example: dict):
        if "messages" in example:
            messages: list = example.pop("messages")
        elif "conversations" in example:
            messages: list = example.pop("conversations")
        else:
            raise ValueError(f'"messages" or "conversations" must be in the example.')

        def remove_response(messages) -> Optional[str]:
            last_role = messages[-1]['role'] if messages else None
            if last_role == 'assistant':
                return messages.pop()['content']

        remove_response(messages)

        if (self.image_key in example and example[self.image_key] != None) or (self.video_key in example and example[self.video_key] != None):
            image_idx = 0
            video_idx = 0
            for message in messages:
                content = message["content"]
                content_list = []
                for segment in re.split("(<image>|<video>)", content):
                    if segment == "<image>":
                        content_list.append({"type": "image", "image": example[self.image_key][image_idx]})
                        image_idx += 1
                    elif segment == "<video>":
                        content_list.append({"type": "video", "video": example[self.video_key][video_idx]})
                        video_idx += 1
                    else:
                        content_list.append({"type": "text", "text": segment})

                message["content"] = content_list

            # here we add default multi-modal token, if no <image>|<video>
            if image_idx == 0 and video_idx == 0:
                for message in messages:
                    role = message["role"]
                    if role == "user":
                        if self.image_key in example and example[self.image_key] is not None:
                            for idx, image_path in enumerate(example[self.image_key]):
                                message["content"].insert(0 + idx, {"type": "image", "image": image_path})

                        if self.video_key in example and example[self.video_key] is not None:
                            for idx, video_path in enumerate(example[self.video_key]):
                                message["content"].insert(0 + idx, {"type": "video", "video": video_path})
        return messages

    def __getitem__(self, item):
        """
        Note that we also return the raw_input_ids so that it can be combined with other chat template
        """
        row_dict: dict = self.dataframe[item]
        # origin_messages can be  "conversations" or "messages"
        if "messages" in row_dict:
            origin_messages = copy.deepcopy(row_dict["messages"])
        elif "conversations" in row_dict:
            origin_messages = copy.deepcopy(row_dict["conversations"])
        else:
            raise ValueError(f'"messages" or "conversations" must be in the sample.')

        messages = self._build_messages(row_dict)
        row_dict["messages"] = origin_messages

        model_inputs = {}
        raw_prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        multi_modal_data = {}

        images, videos = self.process_vision_info_func(messages)

        #if videos is not None:
        #    torch.save(videos, "/nlp_group/huangjiaming/logits-distill/videos.pkl")

        if self.image_key in row_dict and row_dict[self.image_key] != None:
            multi_modal_data["image"] = images

        if self.video_key in row_dict and row_dict[self.video_key] != None:
            multi_modal_data["video"] = videos

        model_inputs = self.processor(text=[raw_prompt], images=images, videos=videos, return_tensors="pt")
        #if hasattr(model_inputs, 'image_grid_thw'):
        #    print(f'{model_inputs.image_grid_thw=}', flush=True)

        input_ids = model_inputs.pop("input_ids")
        attention_mask = model_inputs.pop("attention_mask")

        if "second_per_grid_ts" in model_inputs:
            model_inputs.pop("second_per_grid_ts")

        # There's a trap here, multi_modal_inputs has to be a dict, not BatchFeature
        row_dict["multi_modal_data"] = multi_modal_data
        row_dict["multi_modal_inputs"] = dict(model_inputs)

        # second_per_grid_ts isn't used for training, just for mrope
        row_dict["multi_modal_inputs"].pop("second_per_grid_ts", None)

        if os.environ.get("USE_SLOW_FAST", "false").lower() == "true":
            position_ids = self.get_rope_index_func(
                    input_ids=input_ids,
                    image_grid_thw=model_inputs.get("image_grid_thw"),
                    video_grid_thw=model_inputs.get("video_grid_thw"),
                    fast_video_grid_thw=model_inputs.get("fast_video_grid_thw"),
                    spatial_merge_size=self.hf_config.vision_config.spatial_merge_size,
                    image_token_id=self.hf_config.image_token_id,
                    video_token_id=self.hf_config.video_token_id,
                    vision_start_token_id=self.hf_config.vision_start_token_id,
                    fast_video_token_id=self.hf_config.fast_video_token_id
                ).transpose(0,1)  # (bs, 3, seq_len)
        else:
            position_ids = self.get_rope_index_func(
                    input_ids=input_ids,
                    image_grid_thw=model_inputs.get("image_grid_thw"),
                    video_grid_thw=model_inputs.get("video_grid_thw"),
                    spatial_merge_size=self.hf_config.vision_config.spatial_merge_size,
                    image_token_id=self.hf_config.image_token_id,
                    video_token_id=self.hf_config.video_token_id,
                    vision_start_token_id=self.hf_config.vision_start_token_id,
                    tokens_per_second=self.hf_config.vision_config.tokens_per_second,
                ).transpose(0,1)  # (bs, 3, seq_len)


        input_ids, attention_mask, position_ids = verl_F.postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )


        row_dict["input_ids"] = input_ids[0]
        row_dict["attention_mask"] = attention_mask[0]
        row_dict["position_ids"] = position_ids[0]

        raw_prompt_ids = self.tokenizer.encode(raw_prompt, add_special_tokens=False)
        if len(raw_prompt_ids) > self.max_prompt_length:
            if self.truncation == "left":
                raw_prompt_ids = raw_prompt_ids[-self.max_prompt_length :]
            elif self.truncation == "right":
                raw_prompt_ids = raw_prompt_ids[: self.max_prompt_length]
            elif self.truncation == "error":
                raise RuntimeError(f"Prompt length {len(raw_prompt_ids)} is longer than {self.max_prompt_length}.")

        row_dict["raw_prompt_ids"] = raw_prompt_ids
        # encode prompts without chat template
        if self.return_raw_chat:
            row_dict["raw_prompt"] = messages

        # add index for each prompt
        index = row_dict.get("extra_info", {}).get("index", 0)
        row_dict["index"] = index
        # print(f"[DEBUG] at keye dataset line 278: {row_dict.keys()=} {row_dict=}")

        return row_dict

    def __getstate__(self):
        if not self.serialize_dataset:
            state = self.__dict__.copy()

            if "dataframe" in state:
                del state["dataframe"]
            return state

        return self.__dict__.copy()
        return self.__dict__.copy()
