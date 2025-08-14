set -x

# Functionality from HEAD branch - DeepSeek-V3 configuration with MTP disabled
# 0. download the config
# only need to download the configuration_deepseek.py and config.json # remove the `quantization_config` in the `config.json`
# set `num_nextn_predict_layers=0` to disable MTP, which is not currently supported
#huggingface-cli download deepseek-ai/DeepSeek-V3-0324 configuration_deepseek.py config.json

# 1. download the dist_ckpt format model from https://huggingface.co/BearBiscuit05/dpsk-v3-671B-BF16-dist_ckpt/tree/main
# change the HF_MODEL_PATH and DIST_CKPT_PATH to your own path
# DIST_CKPT_PATH="/nlp_group/huangjiaming/DeepSeek-R1-bf16-5/"
# DIST_CKPT_PATH="/nlp_group/huangjiaming/hjm_dbg/"
DIST_CKPT_PATH="/nlp_group/yuanjiawei05/new_logits_distill/converted_params"
LLM="/nlp_group/huangjiaming/deepseek_v3"
HOME=/nlp_group/huangjiaming/
timestamp=$(date +"%Y-%m-%d-%H:%M:%S")""

# Functionality from main branch - HF checkpoint with FP8 weights
# # 0. download HF checkpoint
# # remove the `quantization_config` in the `config.json`
# # set `num_nextn_predict_layers=0` to disable MTP, which is not currently supported
# huggingface-cli download deepseek-ai/DeepSeek-V3-0324

# no offline dist checkpoint needed, now with mbridge>=0.13.0, we can directly init model from huggingface downloaded fp8 weights
# tested on docker://verlai/verl:app-verl0.5-vllm0.10.0-mcore0.13.0-te2.2
# LLM="<path_to_dsv3_config>"

# 2. run the script
gsm8k_train_path=$HOME/data/gsm8k/train.parquet
gsm8k_test_path=/nlp_group/huangjiaming/logits-distill/random_row.parquet

# Alternative paths from main branch
# gsm8k_train_path=/root/data/gsm8k/train.parquet
# gsm8k_test_path=/root/data/gsm8k/test.parquet

train_files=$gsm8k_train_path
test_files=$gsm8k_test_path

CKPT_DIR=/mnt/nfs/sunqidong/seedsave/grpo_test/dpo_ckpts

# offload
COMMON_PARAM_OFFLOAD=True
COMMON_GRAD_OFFLOAD=True
COMMON_OPTIMIZER_OFFLOAD=True
ACTOR_PARAM_OFFLOAD=${ACTOR_PARAM_OFFLOAD:-$COMMON_PARAM_OFFLOAD}
ACTOR_GRAD_OFFLOAD=${ACTOR_GRAD_OFFLOAD:-$COMMON_GRAD_OFFLOAD}
ACTOR_OPTIMIZER_OFFLOAD=${ACTOR_OPTIMIZER_OFFLOAD:-$COMMON_OPTIMIZER_OFFLOAD}
REF_PARAM_OFFLOAD=${REF_PARAM_OFFLOAD:-$COMMON_PARAM_OFFLOAD}
CRITIC_PARAM_OFFLOAD=${CRITIC_PARAM_OFFLOAD:-$COMMON_PARAM_OFFLOAD}
CRITIC_GRAD_OFFLOAD=${CRITIC_GRAD_OFFLOAD:-$COMMON_GRAD_OFFLOAD}
CRITIC_OPTIMIZER_OFFLOAD=${CRITIC_OPTIMIZER_OFFLOAD:-$COMMON_OPTIMIZER_OFFLOAD}
RM_PARAM_OFFLOAD=${RM_PARAM_OFFLOAD:-$COMMON_PARAM_OFFLOAD}

# Configuration from HEAD branch - 512 H20(96GB)
NODES=4
PP=2
VPP=4

# Alternative configuration from main branch - 256 H100(80GB)
# NODES=32
# PP=16

TP=1
EP=16
ETP=1
INFER_TP=1
# consider TP/ETP, and enable recompute if short of memory

# full recompute
# +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=uniform \
# +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=full \
# +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=1 \

n_resp_per_prompt=4
max_prompt_length=2048
max_response_length=4096
use_dynamic_bsz=True
actor_ppo_max_token_len=$(((max_prompt_length + max_response_length) * 1))
infer_ppo_max_token_len=$(((max_prompt_length + max_response_length) * 3))

rollout_mode="async"
rollout_name="sglang" # sglang or vllm
if [ "$rollout_mode" = "async" ]; then
    export VLLM_USE_V1=1
    return_raw_chat="True"
fi

#actor_rollout_ref.actor.megatron.override_transformer_config.pipeline_model_parallel_layout="\"[['embedding','decoder','decoder','decoder','decoder','decoder','decoder']]+[['decoder','decoder','decoder','decoder','decoder','decoder','decoder','loss']]\"" \
#actor_rollout_ref.actor.megatron.override_transformer_config.pipeline_model_parallel_layout="\"[['embedding','decoder']]+[['decoder','decoder']]*6+[['loss']]\""

#actor_rollout_ref.actor.megatron.virtual_pipeline_model_parallel_size=$VPP \

#    +actor_rollout_ref.rollout.override_config.dp_size=$INFER_TP \
#    +actor_rollout_ref.rollout.override_config.enable_dp_attention=True \
#    +actor_rollout_ref.rollout.override_config.enable_dp_lm_head=True \

# RAY_ADDRESS='auto' ray job submit --working-dir . --
python3 -m verl.trainer.main_ppo --config-path=./config --config-name='ppo_megatron_trainer'\
    algorithm.adv_estimator=grpo \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    # Configuration from HEAD branch
    data.return_raw_chat=$return_raw_chat \
    data.train_batch_size=32 \
    data.max_prompt_length=1024 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=False \
    # Alternative configuration from main branch
    # data.train_batch_size=512 \
    # data.max_prompt_length=$max_prompt_length \
    # data.max_response_length=$max_response_length \
    # data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=$LLM \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.total_training_steps=100 \
    actor_rollout_ref.rollout.n=$n_resp_per_prompt \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.top_k=-1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$INFER_TP \
    actor_rollout_ref.rollout.free_cache_engine=True \
    +actor_rollout_ref.rollout.override_config.chunked_prefill_size=32768 \
    +actor_rollout_ref.rollout.override_config.moe_dense_tp_size=1 \
    +actor_rollout_ref.rollout.override_config.enable_deepep_moe=True \
    +actor_rollout_ref.rollout.override_config.deepep_mode=normal \
    +actor_rollout_ref.rollout.override_config.dp_size=$INFER_TP \
    +actor_rollout_ref.rollout.override_config.enable_dp_attention=True \
    +actor_rollout_ref.rollout.override_config.enable_dp_lm_head=True \
    algorithm.use_kl_in_reward=False \
    # Configuration from HEAD branch
    trainer.logger=['console'] \
    # Alternative configuration from main branch
    # trainer.logger='["console","tensorboard"]' \
    trainer.project_name='verl_megatron_gsm8k_examples' \
    trainer.experiment_name='dsv3-32nodes' \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=$NODES \
    trainer.save_freq=-1 \
    trainer.test_freq=5 \
    # Configuration from HEAD branch
    actor_rollout_ref.actor.profile.use_profile=False \
    actor_rollout_ref.actor.profile.profile_ranks=[0,1] \
    actor_rollout_ref.actor.profile.step_start=5 \
    actor_rollout_ref.actor.profile.step_end=6 \
    actor_rollout_ref.actor.profile.save_path=/nlp_group/huangjiaming/logits-distill/ \
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_enable_deepep=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_layer_freq=[0,0,0,1,1] \
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_router_group_topk=4 \
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_router_num_groups=8 \
    +actor_rollout_ref.actor.megatron.override_transformer_config.pipeline_model_parallel_layout="\"Et|(tt|)*2L\"" \
    +actor_rollout_ref.actor.megatron.override_transformer_config.combined_1f1b=False \
    +actor_rollout_ref.actor.megatron.override_transformer_config.delay_wgrad_compute=False \
    +actor_rollout_ref.actor.megatron.override_transformer_config.fp8='e4m3' \
    +actor_rollout_ref.actor.megatron.override_transformer_config.fp8_param_gather=False \
    +actor_rollout_ref.actor.megatron.override_transformer_config.fp8_recipe='blockwise' \
    # Alternative configuration from main branch
    # actor_rollout_ref.model.use_fused_kernels=True \
    # actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    # actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    # actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    # actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len} \
    # actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    # actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    # +actor_rollout_ref.actor.megatron.override_transformer_config.num_layers_in_first_pipeline_stage=4 \
    # +actor_rollout_ref.actor.megatron.override_transformer_config.num_layers_in_last_pipeline_stage=1 \
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=$PP \
    actor_rollout_ref.actor.megatron.virtual_pipeline_model_parallel_size=$VPP \
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=$TP \
    actor_rollout_ref.actor.megatron.expert_model_parallel_size=$EP \
    actor_rollout_ref.actor.megatron.expert_tensor_parallel_size=$ETP \
    actor_rollout_ref.actor.megatron.param_offload=${ACTOR_PARAM_OFFLOAD} \
    actor_rollout_ref.actor.megatron.optimizer_offload=${ACTOR_OPTIMIZER_OFFLOAD} \
    actor_rollout_ref.actor.megatron.grad_offload=${ACTOR_GRAD_OFFLOAD} \
    # Configuration from HEAD branch
    actor_rollout_ref.actor.megatron.use_dist_checkpointing=True \
    actor_rollout_ref.actor.megatron.dist_checkpointing_path=$DIST_CKPT_PATH \
    trainer.default_hdfs_dir=null \
    # Alternative configuration from main branch
    # actor_rollout_ref.ref.megatron.param_offload=${REF_PARAM_OFFLOAD} \
    # +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=uniform \
    # +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=full \
    # +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=1 \
    # actor_rollout_ref.actor.megatron.use_mbridge=True \
    trainer.default_local_dir=$CKPT_DIR \
    trainer.val_before_train=False \
    trainer.total_epochs=100 2>&1 | tee dpsk_$timestamp.log
