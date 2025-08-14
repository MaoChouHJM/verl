# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2024 The HuggingFace Team. All rights reserved.
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
A script to convert from any supported huggingface transformers model to a megatron-lm or mcore style checkpoint.

The checkpoints will be loaded into a model using the checkpoint loader of verl.models.mcore.loader.
"""

import argparse
import os
import warnings
from contextlib import contextmanager
# Functionality from main branch
from importlib.metadata import version
from typing import Any, Callable, ContextManager, Optional

import numpy as np
import torch
import torch.distributed as dist

try:
    # NPU patch
    import mindspeed.megatron_adaptor  # noqa: F401
except ImportError:
    pass

from accelerate import init_empty_weights
from megatron.core import dist_checkpointing
from megatron.core import parallel_state as mpu
from megatron.core.dist_checkpointing.mapping import ShardedStateDict
from megatron.core.models.gpt import GPTModel
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.utils import get_model_config
from torch import nn
from transformers import AutoConfig, AutoModelForCausalLM

from verl.models.mcore.config_converter import (
    get_dynamic_pipeline_shards,
    hf_to_mcore_config,
    support_distributed_convert,
)
from verl.models.mcore.model_initializer import (
    DeepseekV3Model,
    DenseModel,
    KeyeQwen3SlowFastModel,
    MixtralModel,
    Qwen25VLModel,
    Qwen2MoEModel,
    Qwen3MoEModel,
)
from verl.models.mcore.weight_converter import (
    McoreToHFWeightConverterDense,
    McoreToHFWeightConverterDpskv3,
    McoreToHFWeightConverterKeyeQwen3SlowFast,
    McoreToHFWeightConverterMixtral,
    McoreToHFWeightConverterQwen2_5_VL,
    McoreToHFWeightConverterQwen2Moe,
    McoreToHFWeightConverterQwen3Moe,
)


def add_arguments(parser: argparse.ArgumentParser):
    group = parser.add_argument_group(title="HuggingFace to MCore converter")

    group.add_argument(
        "--hf-model-path",
        type=str,
        required=True,
        help="path to the huggingface model checkpoint",
    )

    group.add_argument(
        "--save-path",
        type=str,
        required=True,
        help="path to save the converted MCore checkpoint",
    )

    group.add_argument(
        "--target-tensor-parallel-size",
        type=int,
        default=1,
        help="target TP degree for conversion (default: 1)",
    )

    group.add_argument(
        "--target-pipeline-parallel-size",
        type=int,
        default=1,
        help="target PP degree for conversion (default: 1)",
    )

    group.add_argument(
        "--use-cpu",
        action="store_true",
        help="use cpu for conversion. (default: False)",
    )

    return parser


@contextmanager
def no_init(override=True):
    if not override:
        yield
        return

    original_init = nn.init.kaiming_uniform_
    original_normal_init = nn.init.normal_

    def no_op_init(tensor, *args, **kwargs):
        return tensor

    try:
        nn.init.kaiming_uniform_ = no_op_init
        nn.init.normal_ = no_op_init
        yield
    finally:
        nn.init.kaiming_uniform_ = original_init
        nn.init.normal_ = original_normal_init


def _sharded_to_regular_dict(sharded_dict):
    # TODO: remove this function. It's a hack of mcore's bug.
    # Currently, DistributedFSDPModel returns a list, where the last element stores the distributed
    # state_dict from FSDP. We here just take the state dict from FSDP to facilitate loading.
    from megatron.core.dist_checkpointing.dict_utils import nested_values
    from megatron.core.dist_checkpointing.mapping import ShardedTensor

    regular_dict = {}
    for k, v in sharded_dict.items():
        if isinstance(v, ShardedTensor):
            regular_dict[k] = v.data
        elif isinstance(v, list) and len(v) > 0 and isinstance(v[-1], ShardedTensor):
            regular_dict[k] = v[-1].data
        elif isinstance(v, list) and len(v) > 0 and isinstance(v[0], ShardedTensor):
            regular_dict[k] = v[0].data
        else:
            # recursively convert nested values
            if hasattr(v, "items"):
                regular_dict[k] = _sharded_to_regular_dict(v)
            else:
                regular_dict[k] = v
    return regular_dict


def _copy_param(src: torch.Tensor, dst: torch.Tensor, name: str = "") -> int:
    src_numel = src.numel()
    dst_numel = dst.numel()
    assert (
        src_numel <= dst_numel
    ), f"src {name} has {src_numel} elements, dst {name} has {dst_numel} elements"
    dst.data.flatten()[:src_numel].copy_(src.data.flatten())
    return src_numel


def convert_checkpoint_from_transformers_to_megatron(
    hfllm: nn.Module,
    mcore_model: nn.Module,
    hf_config: AutoConfig,
) -> int:
    """Convert a huggingface transformers model to a megatron-lm model.

    Args:
        hfllm: the huggingface transformers model
        mcore_model: the megatron-lm model without loaded weights
        hf_config: the huggingface config

    Returns:
        the number of copied parameters
    """
    # the hf_to_mcore_config should define the relation between hf and mcore config
    from verl.models.mcore.weight_converter import McoreToHFWeightConverterDense

    converter_cls = McoreToHFWeightConverterDense
    converter: Callable[[str, torch.Tensor, Any], tuple[str, torch.Tensor]] = converter_cls(
        hf_config, get_model_config(mcore_model), torch.bfloat16
    ).mcore_to_hf_weight

    # This function always pad the vocab size to the padded vocab size of the model
    # Thus, we need to shrink the embedding table to the actual vocab size
    # when padding_vocab is set to True.
    with torch.no_grad():
        copied_numel = 0
        for name, param in hfllm.named_parameters():
            # convert the weight name and weight
            try:
                converted_name, converted_param = converter(name, param)
                print(f"Converting {name} -> {converted_name}, {param.shape} -> {converted_param.shape}")
            except NotImplementedError as e:
                print(f"Skipping {name} due to {e.args[0]}")
                continue

            if converted_name == "":
                continue

            assert hasattr(
                mcore_model, converted_name
            ), f"mcore model doesn't have attribute {converted_name}. Please check your converter."

            mcore_param = getattr(mcore_model, converted_name)

            # copy the weight
            copied_numel += _copy_param(converted_param, mcore_param, converted_name)

    n_params = sum([t.numel() for t in hfllm.state_dict().values()])

    assert n_params == copied_numel, f"n_params={n_params} != copied_numel={copied_numel}"

    return copied_numel


# Functionality from main branch - empty lines kept
def convert_checkpoint_from_transformers_to_megatron_dpskv3(
    hf_model: nn.Module,
    mcore_model: nn.Module,
    hf_config: AutoConfig,
    *,
    tfconfig=None,
):
    from verl.models.mcore.weight_converter import McoreToHFWeightConverterDpskv3

    converter = McoreToHFWeightConverterDpskv3(hf_config, tfconfig, torch.bfloat16)
    return converter.load_into_mcore_model(hf_model, mcore_model)


def convert_checkpoint_from_transformers_to_megatron_keye_qwen3(
    hf_model: nn.Module,
    mcore_model: nn.Module,
    hf_config: AutoConfig,
):
    from verl.models.mcore.weight_converter import McoreToHFWeightConverterKeyeQwen3SlowFast

    converter = McoreToHFWeightConverterKeyeQwen3SlowFast(hf_config, get_model_config(mcore_model), torch.bfloat16)
    return converter.load_into_mcore_model(hf_model, mcore_model)


def convert_checkpoint_from_transformers_to_megatron_qwen2_5_vl(
    hf_model: nn.Module,
    mcore_model: nn.Module,
    hf_config: AutoConfig,
):
    from verl.models.mcore.weight_converter import McoreToHFWeightConverterQwen2_5_VL

    converter = McoreToHFWeightConverterQwen2_5_VL(hf_config, get_model_config(mcore_model), torch.bfloat16)
    return converter.load_into_mcore_model(hf_model, mcore_model)


def verify_model(hf_model, model, input_ids, attention_mask):
    from verl.utils.torch_functional import masked_mean

    # verify the logits
    hf_logits = hf_model(input_ids=input_ids, attention_mask=attention_mask).logits
    print(f"HF logits: {hf_logits.shape}")
    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    print(f"Converted logits: {logits.shape}")
    diff = (logits - hf_logits).abs()
    print(f"Logits diff: {diff.mean()}")

    # check if the logits are the same
    assert diff.mean() < 1e-3, f"logits diff is too large: {diff.mean()}"  # check if the logits are the same

    # check if the model can generate text
    hf_generated = hf_model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=10,
        num_return_sequences=1,
        do_sample=False,
    )
    print(f"HF generated: {hf_generated}")

    with torch.inference_mode():
        generated = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=10,
            num_return_sequences=1,
            do_sample=False,
        )
    print(f"Converted generated: {generated}")

    # check if the generated tokens are the same
    assert torch.allclose(hf_generated, generated), "generated tokens are different"


@torch.inference_mode()
def convert_hf_to_mcore(hf_model_path, save_path, target_tp, target_pp, use_cpu=False):
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    assert world_size == target_tp * target_pp, f"world_size {world_size} != tp {target_tp} * pp {target_pp}"

    if world_size > 1:
        # distributed conversion
        dist.init_process_group(backend="nccl" if not use_cpu else "gloo")
        mpu.initialize_model_parallel(target_tp, target_pp)
        torch.cuda.set_device(local_rank)
        use_cpu_initialization = use_cpu
    elif world_size == 1:
        # cpu conversion
        use_cpu_initialization = True
    else:
        raise ValueError(f"Invalid world_size: {world_size}")

    model_parallel_cuda_manual_seed(0)

    # init hf config
    hf_config = AutoConfig.from_pretrained(hf_model_path, trust_remote_code=True)
    hf_config._attn_implementation = "flash_attention_2"
    print(hf_config, flush=True)

    if world_size > 1 and not support_distributed_convert(hf_config):
        raise NotImplementedError(f"distributed conversion is not supported for {hf_config.architectures} yet.")

    pipeline_shards = get_dynamic_pipeline_shards(hf_config.num_hidden_layers, world_size)
    print(f"Pipeline shards: {pipeline_shards}", flush=True)

    tfconfig = hf_to_mcore_config(
        hf_config,
        torch.bfloat16,
        num_layers_in_first_pipeline_stage=pipeline_shards[0] if len(pipeline_shards) > 1 else None,
        num_layers_in_last_pipeline_stage=pipeline_shards[-1] if len(pipeline_shards) > 2 else None,
    )
    tfconfig.use_cpu_initialization = use_cpu_initialization
    tie_word_embeddings = getattr(hf_config, "tie_word_embeddings", False)

    # build model
    with no_init(), init_empty_weights():
        architectures = hf_config.architectures
        if "LlamaForCausalLM" in architectures or "Qwen2ForCausalLM" in architectures:
            model = DenseModel(tfconfig, hf_config, hf_model_path).initialize()
        elif "Qwen2MoeForCausalLM" in architectures:
            model = Qwen2MoEModel(tfconfig, hf_config, hf_model_path).initialize()
        elif "MixtralForCausalLM" in architectures:
            model = MixtralModel(tfconfig, hf_config, hf_model_path).initialize()
        elif "Qwen2_5_VLForConditionalGeneration" in architectures:
            model = Qwen25VLModel(tfconfig, hf_config, hf_model_path).initialize()
        elif "KeyeForConditionalGeneration" in architectures:
            model = KeyeQwen3SlowFastModel(tfconfig, hf_config, hf_model_path).initialize()
        elif "DeepseekV3ForCausalLM" in architectures:
            model = DeepseekV3Model(tfconfig, hf_config, hf_model_path).initialize()
        elif "Qwen3MoeForCausalLM" in architectures:
            model = Qwen3MoEModel(tfconfig, hf_config, hf_model_path).initialize()
        else:
            raise NotImplementedError(f"Unsupported architectures: {architectures}")

    model = [model]

    if rank == 0:
        # load hf model
        # We should load hf model on cpu because some weights will be directly copied to gpu
        hf_model = AutoModelForCausalLM.from_pretrained(hf_model_path, trust_remote_code=True, device_map="cpu")

        # copy weight
        if (
            "LlamaForCausalLM" in hf_config.architectures
            or "Qwen2ForCausalLM" in hf_config.architectures
            or "Qwen2MoeForCausalLM" in hf_config.architectures
            or "MixtralForCausalLM" in hf_config.architectures
            or "Qwen3MoeForCausalLM" in hf_config.architectures
        ):
            convert_checkpoint_from_transformers_to_megatron(hf_model, model[0].module, hf_config)
        elif "Qwen2_5_VLForConditionalGeneration" in hf_config.architectures:
            convert_checkpoint_from_transformers_to_megatron_qwen2_5_vl(hf_model, model[0].module, hf_config)
        elif "KeyeForConditionalGeneration" in hf_config.architectures:
            convert_checkpoint_from_transformers_to_megatron_keye_qwen3(hf_model, model[0].module, hf_config)
        elif "DeepseekV3ForCausalLM" in hf_config.architectures:
            convert_checkpoint_from_transformers_to_megatron_dpskv3(hf_model, model[0].module, hf_config, tfconfig=tfconfig)
        elif "Qwen3MoeForCausalLM" in hf_config.architectures:
            convert_checkpoint_from_transformers_to_megatron(hf_model, model[0].module, hf_config)
        else:
            raise NotImplementedError(f"Unsupported architectures: {hf_config.architectures}")

    if world_size > 1:
        dist.barrier()

    # save checkpoint
    sharded_state_dict: ShardedStateDict = model[0].sharded_state_dict(prefix="")
    # TODO: remove this hack. It's a bug of mcore that the DistributedFSDPModel returns a list, where the last element stores the distributed state_dict from FSDP
    sharded_state_dict = _sharded_to_regular_dict(sharded_state_dict)
    print(f"Saving checkpoint to {save_path} with rank {rank}...")
    dist_checkpointing.save(sharded_state_dict, save_path)
    print(f"Done saving checkpoint to {save_path} with rank {rank}...")

    # verify model
    if rank == 0:
        input_ids = torch.randint(0, hf_config.vocab_size, (2, 128), dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
        verify_model(hf_model, model[0], input_ids, attention_mask)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = add_arguments(parser)
    args = parser.parse_args()
    convert_hf_to_mcore(
        args.hf_model_path,
        args.save_path,
        args.target_tensor_parallel_size,
        args.target_pipeline_parallel_size,
        args.use_cpu,
    )
