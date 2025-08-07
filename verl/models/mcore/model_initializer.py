# Copyright 2025 Bytedance Ltd. and/or its affiliates
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

# use mcore transformer config to initialize the model
import os
from abc import ABC, abstractmethod
from token import OP
from typing import List, Optional, Tuple, Union

from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_with_transformer_engine_spec, get_gpt_mtp_block_spec)
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.transformer import (MegatronModule, ModuleSpec,
                                       TransformerConfig)
from megatron.core.transformer.transformer_block import \
    TransformerBlockSubmodules
from numpy.dtypes import BoolDType
from pandas._libs.lib import fast_multiget

from verl.models.mcore.qwen2_5_vl import vision_config
from verl.utils.logger import print_rank_0

from .config_converter import PretrainedConfig, TransformerConfig


def get_gpt_decoder_block_spec(config : TransformerConfig,
                               use_transformer_engine : bool,
                               vp_stage: Optional[int] = None,
) -> TransformerBlockSubmodules:
    from megatron.core.models.gpt.gpt_layer_specs import \
        get_gpt_decoder_block_spec as get_gpt_decoder_block_spec_mcore
    if os.environ["MEGATRON_EA_VERSION"].lower() == "true":
        return get_gpt_decoder_block_spec_mcore(config, use_transformer_engine=use_transformer_engine, vp_stage=vp_stage)
    else:
        return get_gpt_decoder_block_spec_mcore(config, use_transformer_engine=use_transformer_engine)
class BaseModelInitializer(ABC):
    """Base class for model initializers."""

    def __init__(self, tfconfig: TransformerConfig, hf_config: PretrainedConfig, model_path: str):
        self.tfconfig = tfconfig
        self.hf_config = hf_config
        self.model_path = model_path

    @abstractmethod
    def get_transformer_layer_spec(self):
        """Get the transformer layer specification.
        https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/models/gpt/gpt_layer_specs.py"""
        pass

    def get_rope_scaling_args(self) -> dict:
        """Get rope scaling args."""
        rope_scaling_args = {}
        if "rope_scaling" in self.hf_config:
            if self.hf_config.rope_scaling is not None:
                # assert self.hf_config.rope_scaling["type"] == "linear", "only linear scaling is supported for now"
                rope_scaling_args["seq_len_interpolation_factor"] = self.hf_config.rope_scaling["factor"]
        return rope_scaling_args

    def initialize(
        self,
        pre_process: bool = True,
        post_process: bool = True,
        share_embeddings_and_output_weights: bool = False,
        value: bool = False,
        vp_stage: Optional[int] =None,
        **extra_kwargs,
    ) -> GPTModel:
        """Initialize a GPT model with the given configuration.
        https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/models/gpt/gpt_model.py

        Args:
            pre_process (bool): include embedding layer.
            post_process (bool): including an output layer.
            share_embeddings_and_output_weights (bool): input embeddings and output logit weights are shared.
            value (bool): add an extra linear layer for classification or regression.

        Returns:
            GPTModel: An initialized GPT model instance
        """
        transformer_layer_spec = self.get_transformer_layer_spec(vp_stage=vp_stage)
        rope_scaling_args = self.get_rope_scaling_args()
        mtp_block_spec = extra_kwargs.get("mtp_block_spec", None)
        

        if os.environ["MEGATRON_EA_VERSION"].lower() == "true":
            model = GPTModel(
                config=self.tfconfig,
                transformer_layer_spec=transformer_layer_spec,
                vocab_size=self.hf_config.vocab_size,
                max_sequence_length=self.hf_config.max_position_embeddings,
                pre_process=pre_process,
                post_process=post_process,
                share_embeddings_and_output_weights=share_embeddings_and_output_weights,
                position_embedding_type="rope",
                rotary_base=self.hf_config.rope_theta,
                **rope_scaling_args,
                mtp_block_spec=mtp_block_spec,
                vp_stage=vp_stage,
            )
        else:
            model = GPTModel(
                config=self.tfconfig,
                transformer_layer_spec=transformer_layer_spec,
                vocab_size=self.hf_config.vocab_size,
                max_sequence_length=self.hf_config.max_position_embeddings,
                pre_process=pre_process,
                post_process=post_process,
                share_embeddings_and_output_weights=share_embeddings_and_output_weights,
                position_embedding_type="rope",
                rotary_base=self.hf_config.rope_theta,
                **rope_scaling_args,
                mtp_block_spec=mtp_block_spec,
            )

        if post_process and value:
            from verl.models.llama.megatron.layers.parallel_linear import \
                LinearForLastLayer

            model.output_layer = LinearForLastLayer(
                input_size=self.tfconfig.hidden_size, output_size=1, config=self.tfconfig
            )

        return model


class DenseModel(BaseModelInitializer):
    """Initializer for dense models like Llama and Qwen2."""

    def get_transformer_layer_spec(self, vp_stage: Optional[int] = None):
        assert self.tfconfig.normalization == "RMSNorm", "only RMSNorm is supported for now"
        return get_gpt_decoder_block_spec(self.tfconfig, use_transformer_engine=True, vp_stage=vp_stage)


class Qwen2MoEModel(BaseModelInitializer):
    """Initializer for Qwen2 MoE models."""

    def get_transformer_layer_spec(self, vp_stage: Optional[int] = None):
        assert self.tfconfig.normalization == "RMSNorm", "only RMSNorm is supported for now"
        transformer_layer_spec = get_gpt_decoder_block_spec(self.tfconfig, use_transformer_engine=True, vp_stage=vp_stage)

        # Patch layer spec for shared experts
        for i in range(len(transformer_layer_spec.layer_specs)):
            transformer_layer_spec.layer_specs[i].submodules.mlp.submodules.shared_experts.params["gate"] = True

        return transformer_layer_spec

    def initialize(self, **kwargs):
        # Qwen default freeze_moe_router: true
        model = super().initialize(**kwargs)
        freeze_moe_router = kwargs.get("freeze_moe_router", True)
        if freeze_moe_router:
            for layer in model.decoder.layers:
                layer.mlp.router.weight.requires_grad = False
        return model


class MixtralModel(BaseModelInitializer):
    """Initializer for Mixtral models."""

    def get_transformer_layer_spec(self, vp_stage: Optional[int] = None):
        assert self.tfconfig.normalization == "RMSNorm", "only RMSNorm is supported for now"
        transformer_layer_spec = get_gpt_decoder_block_spec(self.tfconfig, use_transformer_engine=True, vp_stage=vp_stage)
        return transformer_layer_spec

    def initialize(self, **kwargs):
        model = super().initialize(**kwargs)
        freeze_moe_router = kwargs.get("freeze_moe_router", False)
        if freeze_moe_router:
            for layer in model.decoder.layers:
                layer.mlp.router.weight.requires_grad = False
        return model


class Qwen3MoEModel(BaseModelInitializer):
    """Initializer for Qwen3 MoE models."""

    def get_transformer_layer_spec(self, vp_stage: Optional[int] = None):
        assert self.tfconfig.normalization == "RMSNorm", "only RMSNorm is supported for now"
        transformer_layer_spec = get_gpt_decoder_block_spec(self.tfconfig, use_transformer_engine=True, vp_stage=vp_stage)
        return transformer_layer_spec

    def initialize(self, **kwargs):
        # Qwen default freeze_moe_router: true
        model = super().initialize(**kwargs)
        freeze_moe_router = kwargs.get("freeze_moe_router", True)
        if freeze_moe_router:
            for layer in model.decoder.layers:
                layer.mlp.router.weight.requires_grad = False
        return model


class DeepseekV3Model(BaseModelInitializer):
    """Initializer for DeepseekV3 models."""

    def get_transformer_layer_spec(self, vp_stage: Optional[int] = None):
        transformer_layer_spec = get_gpt_decoder_block_spec(self.tfconfig, use_transformer_engine=True, vp_stage=vp_stage)
        return transformer_layer_spec

    def get_rope_scaling_args(self) -> dict:
        """Get rope scaling args."""
        rope_scaling_args = {}
        return rope_scaling_args

    def initialize(
        self,
        **kwargs,
    ):
        freeze_moe_router = kwargs.get("freeze_moe_router", True)
        if freeze_moe_router:
            self.tfconfig.moe_router_load_balancing_type = "none"
        # MTP
        if self.tfconfig.mtp_num_layers is not None:
            transformer_layer_spec = self.get_transformer_layer_spec()
            mtp_block_spec = get_gpt_mtp_block_spec(self.tfconfig,
                 transformer_layer_spec, use_transformer_engine=True, vp_stage=kwargs.get("vp_stage", None))
            kwargs["mtp_block_spec"] = mtp_block_spec

        model = super().initialize(**kwargs)
        if freeze_moe_router:
            for layer in model.decoder.layers:
                if hasattr(layer.mlp, "router"):
                    layer.mlp.router.weight.requires_grad = False
        return model


class Qwen25VLModel(BaseModelInitializer):
    """Initializer for Qwen2.5 VL models."""

    def get_transformer_layer_spec(self, vp_stage: Optional[int] = None):
        transformer_layer_spec = get_gpt_decoder_block_spec(self.tfconfig, use_transformer_engine=True, vp_stage=vp_stage)
        return transformer_layer_spec

    def initialize(
        self,
        pre_process=None,
        post_process=None,
        share_embeddings_and_output_weights=False,
        value=False,
        **extra_kwargs,
    ):
        tfconfig = self.tfconfig
        hf_config = self.hf_config
        # Qwen2_5_VLForConditionalGeneration
        from copy import deepcopy

        transformer_layer_spec = self.get_transformer_layer_spec()

        from megatron.core.extensions.transformer_engine import (
            TEColumnParallelLinear, TERowParallelLinear)
        from megatron.core.models.gpt.moe_module_specs import MLPSubmodules
        from megatron.core.models.vision.vit_layer_specs import \
            get_vit_layer_with_transformer_engine_spec

        from .qwen2_5_vl import (Qwen2_5VLModel, get_vision_model_config,
                                 get_vision_projection_config)

        vision_transformer_config = get_vision_model_config(deepcopy(tfconfig))
        vision_transformer_config.pipeline_model_parallel_size = 1
        vision_transformer_config.first_pipeline_num_layers = None

        vision_projection_config = get_vision_projection_config(
            deepcopy(tfconfig),
            vision_transformer_config.hidden_size,
            spatial_merge_size=hf_config.vision_config.spatial_merge_size,
        )
        vision_projection_layer_spec = MLPSubmodules(
            linear_fc1=TEColumnParallelLinear,
            linear_fc2=TERowParallelLinear,
        )
        vision_transformer_layer_spec = get_vit_layer_with_transformer_engine_spec()

        qwen25_vl_model = Qwen2_5VLModel(
            language_transformer_config=tfconfig,
            language_transformer_layer_spec=transformer_layer_spec,
            language_vocab_size=hf_config.vocab_size,
            language_max_sequence_length=hf_config.max_position_embeddings,
            vision_transformer_config=vision_transformer_config,
            vision_transformer_layer_spec=vision_transformer_layer_spec,
            vision_projection_config=vision_projection_config,
            vision_projection_layer_spec=vision_projection_layer_spec,
            vision_projection_type="mlp",
            language_rotary_base=hf_config.rope_theta,
            pre_process=pre_process,
            post_process=post_process,
            add_decoder=True,
            add_encoder=True,
            parallel_output=True,
            language_share_embeddings_and_output_weights=share_embeddings_and_output_weights,
        )

        if post_process and value:
            from verl.models.llama.megatron.layers.parallel_linear import \
                LinearForLastLayer

            qwen25_vl_model.language_model.output_layer = LinearForLastLayer(
                input_size=tfconfig.hidden_size, output_size=1, config=tfconfig
            )

        return qwen25_vl_model

class KeyeQwen3SlowFastModel(BaseModelInitializer):
    """Initializer for KeyeSlowFast models."""

    def get_transformer_layer_spec(self, vp_stage):
        if self.tfconfig.num_moe_experts:
            transformer_layer_spec = get_gpt_decoder_block_spec(self.tfconfig,
             use_transformer_engine=True,
             normalization=self.tfconfig.normalization,
             vp_stage=vp_stage)
        else:
            transformer_layer_spec = get_gpt_layer_with_transformer_engine_spec(
                num_experts=self.tfconfig.num_moe_experts,
                moe_grouped_gemm=self.tfconfig.moe_grouped_gemm,
                qk_layernorm=self.tfconfig.qk_layernorm,
                multi_latent_attention=self.tfconfig.multi_latent_attention,
                moe_use_legacy_grouped_gemm=self.tfconfig.moe_use_legacy_grouped_gemm,
            )
        return transformer_layer_spec

    def initialize(
        self,
        pre_process: bool = None,
        post_process: bool = None,
        vp_stage: Optional[int] = None,
        **extra_kwargs,
    ):
        tfconfig = self.tfconfig
        transformer_layer_spec = self.get_transformer_layer_spec(
            vp_stage
        )
        from megatron.core.models.keye.keye_config import (
            VisionTransformerConfig, get_vision_model_config)
        from megatron.core.models.keye.keye_layer_specs import \
            get_vision_model_spec
        from megatron.core.models.keye.keye_model_slowfast import (
            KeyeModelSlowFast, Projector, SiglipVisionModel)
        
        args = self.hf_config
        if not hasattr(self.hf_config.vision_config, "rope_thea"):
            args.vision_rope_theta = 10000
        else:
            args.vision_rope_theta = self.hf_config.vision_config.rope_theta

        vision_config = get_vision_model_config(args, tfconfig)
        vision_transformer_layer_spec = get_vision_model_spec()

        mtp_block_spec = None
        if tfconfig.mtp_num_layers is not None:
            mtp_block_spec = get_gpt_mtp_block_spec(tfconfig, transformer_layer_spec, use_transformer_engine=True)

        print_rank_0(f"in KeyeQwen3SlowFastModel initialize\n\ntransformer_config={tfconfig}\n\nvision_config={vision_config}")

        def monkey_patch_init(
            self,
            transformer_config: TransformerConfig,
            hf_config: PretrainedConfig,
            model_path: str,
            transformer_layer_spec: ModuleSpec,
            vision_config: VisionTransformerConfig,
            vision_layer_spec: ModuleSpec,
            fast_vision_config: Optional[VisionTransformerConfig] = None,
            fast_vision_layer_spec: Optional[ModuleSpec] = None,
            pre_process: bool = True,
            post_process: bool = True,
            mtp_block_spec: Optional[ModuleSpec] = None,
            vp_stage: Optional[int] = None,
        ):
           super(KeyeModelSlowFast, self).__init__(config=transformer_config)
           self.hf_config = hf_config
           self.config = transformer_config
           self.vision_config = vision_config
           self.model_path = model_path
           self.pre_process = pre_process
           self.post_process = post_process
           self.vp_stage = vp_stage
           if self.pre_process:
               self.visual = SiglipVisionModel(vision_config, vision_layer_spec, vp_stage)
               self.mlp_AR = Projector(vision_config, transformer_config.hidden_size)

               # self.fast_visual = SiglipVisionModel(fast_vision_config, fast_vision_layer_spec, vp_stage)
               # self.fast_mlp_AR = Projector(fast_vision_config, transformer_config.hidden_size)

           self.language_model = GPTModel(
               config=transformer_config,
               transformer_layer_spec=transformer_layer_spec,
               vocab_size=hf_config.vocab_size,
               #vocab_size=155136, # only for test
               #max_sequence_length=hf_config.max_position_embeddings,
               max_sequence_length=327680, # can remove
               pre_process=pre_process,
               post_process=post_process,
               parallel_output=True,
               position_embedding_type="mrope",
               rotary_base=hf_config.rope_theta,
               #rotary_base=10000, # only for test
               rope_scaling=False,
               mtp_block_spec=mtp_block_spec,
               vp_stage=vp_stage,
           )
           self.share_embeddings_and_output_weights = (
               self.language_model.share_embeddings_and_output_weights
           )
        KeyeModelSlowFast.__init__ = monkey_patch_init

        keye_model = KeyeModelSlowFast(
        transformer_config=tfconfig,
        hf_config=self.hf_config,
        model_path=self.model_path,
        transformer_layer_spec=transformer_layer_spec,
        vision_config=vision_config,
        vision_layer_spec=vision_transformer_layer_spec,
        fast_vision_config=vision_config,
        fast_vision_layer_spec=vision_transformer_layer_spec,
        pre_process=pre_process,
        post_process=post_process,
        mtp_block_spec=mtp_block_spec,
        vp_stage=vp_stage,
        )
        print_rank_0(f'initialized\n\n{keye_model=}')

        return keye_model

class KeyeQwen3Model(BaseModelInitializer):
    """Initializer for Keye models."""

    def get_transformer_layer_spec(self, vp_stage):
        if self.tfconfig.num_moe_experts:
            transformer_layer_spec = get_gpt_decoder_block_spec(self.tfconfig,
             use_transformer_engine=True,
             normalization=self.tfconfig.normalization,
             vp_stage=vp_stage)
        else:
            transformer_layer_spec = get_gpt_layer_with_transformer_engine_spec(
                num_experts=self.tfconfig.num_moe_experts,
                moe_grouped_gemm=self.tfconfig.moe_grouped_gemm,
                qk_layernorm=self.tfconfig.qk_layernorm,
                multi_latent_attention=self.tfconfig.multi_latent_attention,
                moe_use_legacy_grouped_gemm=self.tfconfig.moe_use_legacy_grouped_gemm,
            )
        return transformer_layer_spec

    def initialize(
        self,
        pre_process: bool = None,
        post_process: bool = None,
        vp_stage: Optional[int] = None,
        **extra_kwargs,
    ):
        tfconfig = self.tfconfig
        transformer_layer_spec = self.get_transformer_layer_spec(
            vp_stage
        )
        from megatron.core.models.keye.keye_config import (
            VisionTransformerConfig, get_vision_model_config)
        from megatron.core.models.keye.keye_layer_specs import \
            get_vision_model_spec
        from megatron.core.models.keye.keye_model import (KeyeModel, Projector,
                                                          SiglipVisionModel)

        args = self.hf_config
        if not hasattr(self.hf_config.vision_config, "rope_thea"):
            args.vision_rope_theta = 10000
        else:
            args.vision_rope_theta = self.hf_config.vision_config.rope_theta
        vision_config = get_vision_model_config(args, tfconfig)
        vision_transformer_layer_spec = get_vision_model_spec()

        mtp_block_spec = None
        if tfconfig.mtp_num_layers is not None:
            mtp_block_spec = get_gpt_mtp_block_spec(tfconfig, transformer_layer_spec, use_transformer_engine=True)

        print_rank_0(f"in KeyeQwen3Model initialize\n\ntransformer_config={tfconfig}\n\nvision_config={vision_config}")

        def monkey_patch_init(
            self,
            transformer_config: TransformerConfig,
            hf_config: PretrainedConfig,
            model_path: str,
            transformer_layer_spec: ModuleSpec,
            vision_config: VisionTransformerConfig,
            vision_layer_spec: ModuleSpec,
            pre_process: bool = True,
            post_process: bool = True,
            mtp_block_spec: Optional[ModuleSpec] = None,
            vp_stage: Optional[int] = None,
        ):
           super(KeyeModel, self).__init__(config=transformer_config)
           self.hf_config = hf_config
           self.config = transformer_config
           self.vision_config = vision_config
           self.model_path = model_path
           self.pre_process = pre_process
           self.post_process = post_process
           self.vp_stage = vp_stage
           if self.pre_process:
               self.visual = SiglipVisionModel(vision_config, vision_layer_spec, vp_stage)
               self.mlp_AR = Projector(vision_config, transformer_config.hidden_size)

               # self.fast_visual = SiglipVisionModel(fast_vision_config, fast_vision_layer_spec, vp_stage)
               # self.fast_mlp_AR = Projector(fast_vision_config, transformer_config.hidden_size)

           self.language_model = GPTModel(
               config=transformer_config,
               transformer_layer_spec=transformer_layer_spec,
               vocab_size=hf_config.vocab_size,
               #vocab_size=155136, # only for test
               #max_sequence_length=hf_config.max_position_embeddings,
               max_sequence_length=327680, # can remove
               pre_process=pre_process,
               post_process=post_process,
               parallel_output=True,
               position_embedding_type="mrope",
               rotary_base=hf_config.rope_theta,
               #rotary_base=10000, # only for test
               rope_scaling=False,
               mtp_block_spec=mtp_block_spec,
               vp_stage=vp_stage,
           )
           self.share_embeddings_and_output_weights = (
               self.language_model.share_embeddings_and_output_weights
           )
        KeyeModel.__init__ = monkey_patch_init

        keye_model = KeyeModel(
        transformer_config=tfconfig,
        hf_config=self.hf_config,
        model_path=self.model_path,
        transformer_layer_spec=transformer_layer_spec,
        vision_config=vision_config,
        vision_layer_spec=vision_transformer_layer_spec,
        pre_process=pre_process,
        post_process=post_process,
        mtp_block_spec=mtp_block_spec,
        vp_stage=vp_stage,
        )
        print_rank_0(f'initialized\n\n{keye_model=}')

        return keye_model
