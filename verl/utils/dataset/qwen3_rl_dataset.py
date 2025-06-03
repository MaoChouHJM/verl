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
import os
import re
from collections import defaultdict
from typing import List, Optional, Union

import datasets
import numpy as np
import torch
from omegaconf import DictConfig, ListConfig
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, ProcessorMixin

import verl.utils.torch_functional_vl as verl_F
from verl.utils.model import compute_position_id_with_mask
from verl.utils.dataset import RLHFDataset


from recovlm.data.datasets import ChatCompletionVisionParquetDataset_keye, get_rope_index
from recovlm.utils.qwen_vl_utils import process_vision_info



class SkipBuildSourceDataset(ChatCompletionVisionParquetDataset_keye):
  # NOTE(huangjiaming): here we dont build dataset, we build outside
  def _build_source_dataset(self, sources):
      return None, -1


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


class Qwen3RLHFDataset(RLHFDataset):
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
        # we use the tokenizer and processor in qwen3dataset
        #self.qwen3_dataset.tokenizer = tokenizer
        #self.qwen3_dataset.processor = processor
        self.config = config

        self.cache_dir = os.path.expanduser(config.get("cache_dir", "~/.cache/verl/rlhf"))
        self.prompt_key = config.get("prompt_key", "messages")
        self.image_key = config.get("image_key", "images")
        self.video_key = config.get("video_key", "videos")
        self.max_prompt_length = config.get("max_prompt_length", 5*1024)

        self.return_raw_chat = config.get("return_raw_chat", False)
        self.truncation = config.get("truncation", "error")
        self.filter_overlong_prompts = config.get("filter_overlong_prompts", True)

        self.num_workers = config.get("filter_overlong_prompts_workers", max(1, os.cpu_count() // 4))
        self.num_workers = min(self.num_workers, os.cpu_count())

        # whether to store the dataset in state_dict()
        # default not store
        self.serialize_dataset = False
        dataset_config = {"sources": "",
                          "num_workers": 8,
                          "base_model_dir": config['base_model_dir']
                         }
        #with open(config['hf_dataset_config'], encoding="utf-8") as f:
        #    import json
        #    dataset_config = json.loads(f.read())
        self.qwen3_dataset = SkipBuildSourceDataset(**dataset_config) 
        self._read_files_and_tokenize()

    def _read_files_and_tokenize(self):
        dataframes = []
        for parquet_file in self.data_files:
            # read parquet files and cache
            dataframe = datasets.load_dataset("parquet", data_files=parquet_file)["train"]
            dataframes.append(dataframe)
        self.dataframe: datasets.Dataset = datasets.concatenate_datasets(dataframes)

        print(f"dataset len: {len(self.dataframe)}")

        # filter out too long prompts
        if self.filter_overlong_prompts:
            tokenizer = self.qwen3_dataset.tokenizer
            prompt_key = self.prompt_key
            self.dataframe = self.dataframe.filter(
                lambda doc: len(tokenizer.apply_chat_template(doc[prompt_key], add_generation_prompt=True))
                <= self.max_prompt_length,
                num_proc=self.num_workers,
                desc=f"Filtering prompts longer than {self.max_prompt_length} tokens",
            )

            print(f"filter dataset len: {len(self.dataframe)}")

        #if self.filter_overlong_prompts:
        #    tokenizer = self.qwen3_dataset.tokenizer
        #    prompt_key = self.prompt_key
        #    self.dataframe = self.dataframe.filter(
        #        lambda doc: self.image_key in doc,
        #        num_proc=self.num_workers,
        #        desc=f"Filtering prompts have {self.image_key}",
        #    )

        #    print(f"filter dataset len: {len(self.dataframe)}")


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
        messages: list = example.pop(self.prompt_key)

        if (self.image_key in example and example[self.image_key] != None) or (self.video_key in example and example[self.video_key] != None):
            image_idx = 0
            video_idx = 0
            #print(f'{example=}', flush=True)
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
                        if self.image_key in example:
                            try:
                                for idx, image_path in enumerate(example[self.image_key]):
                                    message["content"].insert(0+ idx, {"type": "image", "image": image_path})
                            except Exception as e:
                                raise ValueError(f'{example=}')

                        if self.video_key in example:
                            for idx, video_path in enumerate(example[self.video_key]):
                                message["content"].insert(0+ idx, {"type": "video", "video": image_path})

        return messages

    def __getitem__(self, item):
        """
        Note that we also return the raw_input_ids so that it can be combined with other chat template
        """
        row_dict: dict = self.dataframe[item]
        origin_messages = copy.deepcopy(row_dict.get("messages", []))

        messages = self._build_messages(row_dict)
        row_dict["messages"] = origin_messages

        model_inputs = {}

        from verl.utils.dataset.vision_utils import process_image, process_video

        raw_prompt = self.qwen3_dataset.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

        multi_modal_data = {}

        images = None
        videos = None
        # NOTE(huangjiaming): not work for qwen3
        #if self.image_key in row_dict:
        #    images = [process_image(image) for image in row_dict.pop(self.image_key)]
        #    multi_modal_data["image"] = images

        #if self.video_key in row_dict:
        #    videos = [process_video(video) for video in row_dict.pop(self.video_key)]
        #    multi_modal_data["video"] = [video.numpy() for video in videos]

        images, videos = process_vision_info(messages)

        if self.image_key in row_dict:
            #images = [process_image(image) for image in row_dict.pop(self.image_key)]
            multi_modal_data["image"] = images

        if self.video_key in row_dict:
            #videos = [process_video(video) for video in row_dict.pop(self.video_key)]
            multi_modal_data["video"] = [video.numpy() for video in videos]

        # here we pad fake image
        if images == None:
            from PIL import Image
            images = [Image.fromarray(np.zeros((50,50, 3), dtype=np.uint8)).convert("RGB")]

        model_inputs = self.qwen3_dataset.processor(text=[raw_prompt], images=images, videos=videos, return_tensors="pt")

        input_ids = model_inputs.pop("input_ids")
        attention_mask = model_inputs.pop("attention_mask")

        if "second_per_grid_ts" in model_inputs:
            model_inputs.pop("second_per_grid_ts")

        # There's a trap here, multi_modal_inputs has to be a dict, not BatchFeature
        row_dict["multi_modal_data"] = multi_modal_data
        row_dict["multi_modal_inputs"] = dict(model_inputs)

        assert "pixel_values" in row_dict["multi_modal_inputs"], f"{raw_prompt=} {images=}"

        # second_per_grid_ts isn't used for training, just for mrope
        row_dict["multi_modal_inputs"].pop("second_per_grid_ts", None)

        position_ids = get_rope_index(
                input_ids=input_ids,
                image_grid_thw=model_inputs.get("image_grid_thw"),
                video_grid_thw=model_inputs.get("video_grid_thw"),
                spatial_merge_size=self.qwen3_dataset.spatial_merge_size,
                image_token_id=self.qwen3_dataset.image_token_id,
                video_token_id=self.qwen3_dataset.video_token_id,
                vision_start_token_id=self.qwen3_dataset.vision_start_token_id
            ).transpose(0,1)  # (bs, 3, seq_len)

        input_ids, attention_mask, position_ids = verl_F.postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            max_length=self.max_prompt_length,
            pad_token_id=self.qwen3_dataset.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )


        row_dict["input_ids"] = input_ids[0]
        row_dict["attention_mask"] = attention_mask[0]
        row_dict["position_ids"] = position_ids[0]

        raw_prompt_ids = self.qwen3_dataset.tokenizer.encode(raw_prompt, add_special_tokens=False)
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

        return row_dict

    def __getstate__(self):
        if not self.serialize_dataset:
            state = self.__dict__.copy()

            if "dataframe" in state:
                del state["dataframe"]
            return state

        return self.__dict__.copy()
