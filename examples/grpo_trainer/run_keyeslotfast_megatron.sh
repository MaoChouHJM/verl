set -x

DIST_CKPT_PATH="/mmu_mllm_hdd_2/lilaiyi/718/kai-megatron/output/save/dist_ckpt_step27000/iter_0000001"
LLM="/mmu_mllm_hdd_2/zhouyang12/output1/Keye/0.9.3/Stage2/8b/slowfast-0721-0717-v2/step27000/global_step27000/converted"
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
NODES=1
PP=1
INFER_TP=1

n_resp_per_prompt=4
project_name='verl_megatron_gsm8k_examples'
experiment_name='dsv3-32nodes'


rollout_mode="async"
rollout_name="sglang" # sglang or vllm
if [ "$rollout_mode" = "async" ]; then
    export VLLM_USE_V1=1
    return_raw_chat="True"
fi

export HYDRA_FULL_ERROR=1

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
    actor_rollout_ref.model.trust_remote_code=True \
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
    trainer.project_name=$project_name \
    trainer.experiment_name=$experiment_name \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=$NODES \
    trainer.save_freq=20 \
    trainer.test_freq=5 \
    actor_rollout_ref.actor.profile.use_profile=False \
    actor_rollout_ref.actor.profile.profile_ranks=[0,1] \
    actor_rollout_ref.actor.profile.step_start=5 \
    actor_rollout_ref.actor.profile.step_end=6 \
    actor_rollout_ref.actor.profile.save_path=/nlp_group/huangjiaming/logits-distill/ \
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=$PP \
    actor_rollout_ref.actor.megatron.param_offload=${ACTOR_PARAM_OFFLOAD} \
    actor_rollout_ref.actor.megatron.optimizer_offload=${ACTOR_OPTIMIZER_OFFLOAD} \
    actor_rollout_ref.actor.megatron.grad_offload=${ACTOR_GRAD_OFFLOAD} \
    actor_rollout_ref.actor.megatron.use_dist_checkpointing=True \
    actor_rollout_ref.actor.megatron.dist_checkpointing_path=$DIST_CKPT_PATH \
    trainer.default_hdfs_dir=null \
    trainer.default_local_dir=$CKPT_DIR \
    trainer.val_before_train=False \
    trainer.total_epochs=100 2>&1 | tee keye_$timestamp.log
