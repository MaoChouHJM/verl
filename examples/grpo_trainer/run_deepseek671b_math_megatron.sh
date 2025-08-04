set -x

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

# 2. run the script
gsm8k_train_path=$HOME/data/gsm8k/train.parquet
gsm8k_test_path=/nlp_group/huangjiaming/logits-distill/random_row.parquet
train_files=$gsm8k_train_path
test_files=$gsm8k_test_path

ALL_OFFLOAD=${ALL_OFFLOAD:-True}
COMMON_PARAM_OFFLOAD=${COMMON_PARAM_OFFLOAD:-$ALL_OFFLOAD}
COMMON_GRAD_OFFLOAD=${COMMON_GRAD_OFFLOAD:-$ALL_OFFLOAD}
COMMON_OPTIMIZER_OFFLOAD=${COMMON_OPTIMIZER_OFFLOAD:-$ALL_OFFLOAD}

ACTOR_PARAM_OFFLOAD=${ACTOR_PARAM_OFFLOAD:-$COMMON_PARAM_OFFLOAD}
ACTOR_GRAD_OFFLOAD=${ACTOR_GRAD_OFFLOAD:-$COMMON_GRAD_OFFLOAD}
ACTOR_OPTIMIZER_OFFLOAD=${ACTOR_OPTIMIZER_OFFLOAD:-$COMMON_OPTIMIZER_OFFLOAD}

# 512 H20(96GB)
NODES=4
PP=2
VPP=4
TP=1
EP=16
ETP=1
INFER_TP=16
# consider TP/ETP, and enable recompute if short of memory

# full recompute
# +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=uniform \
# +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=full \
# +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=1 \

n_resp_per_prompt=4

rollout_mode="async"
rollout_name="sglang" # sglang or vllm
if [ "$rollout_mode" = "async" ]; then
    export VLLM_USE_V1=1
    return_raw_chat="True"
fi

#+actor_rollout_ref.actor.megatron.override_transformer_config.pipeline_model_parallel_layout="\"[['embedding','decoder','decoder','decoder','decoder','decoder','decoder']]+[['decoder','decoder','decoder','decoder','decoder','decoder','decoder','loss']]\"" \
#+actor_rollout_ref.actor.megatron.override_transformer_config.pipeline_model_parallel_layout="\"[['embedding','decoder']]+[['decoder','decoder']]*6+[['loss']]\""

#actor_rollout_ref.actor.megatron.virtual_pipeline_model_parallel_size=$VPP \


#    +actor_rollout_ref.rollout.override_config.dp_size=$INFER_TP \
#    +actor_rollout_ref.rollout.override_config.enable_dp_attention=True \
#    +actor_rollout_ref.rollout.override_config.enable_dp_lm_head=True \






# RAY_ADDRESS='auto' ray job submit --working-dir . --
python3 -m verl.trainer.main_ppo --config-path=./config --config-name='ppo_megatron_trainer'\
    algorithm.adv_estimator=grpo \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.return_raw_chat=$return_raw_chat \
    data.train_batch_size=32 \
    data.max_prompt_length=1024 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=False \
    data.truncation='error' \
    actor_rollout_ref.model.path=$LLM \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.use_torch_compile=False \
    actor_rollout_ref.actor.load_weight=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.mode=$rollout_mode \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.60 \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
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
    trainer.logger=['console'] \
    trainer.project_name='verl_megatron_gsm8k_examples' \
    trainer.experiment_name='dsv3-32nodes' \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=$NODES \
    trainer.save_freq=-1 \
    trainer.test_freq=5 \
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
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=$PP \
    actor_rollout_ref.actor.megatron.virtual_pipeline_model_parallel_size=$VPP \
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=$TP \
    actor_rollout_ref.actor.megatron.expert_model_parallel_size=$EP \
    actor_rollout_ref.actor.megatron.expert_tensor_parallel_size=$ETP \
    actor_rollout_ref.actor.megatron.param_offload=${ACTOR_PARAM_OFFLOAD} \
    actor_rollout_ref.actor.megatron.optimizer_offload=${ACTOR_OPTIMIZER_OFFLOAD} \
    actor_rollout_ref.actor.megatron.grad_offload=${ACTOR_GRAD_OFFLOAD} \
    actor_rollout_ref.actor.megatron.use_dist_checkpointing=True \
    actor_rollout_ref.actor.megatron.dist_checkpointing_path=$DIST_CKPT_PATH \
    trainer.default_hdfs_dir=null \
    trainer.default_local_dir=$CKPT_DIR \
    trainer.val_before_train=False \
    trainer.total_epochs=100 2>&1 | tee dpsk_$timestamp.log
