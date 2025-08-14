set -x

DIST_CKPT_PATH=~/models/Qwen3-30B-MoE
LLM=~/models/Qwen3-30B-MoE
gsm8k_train_path=~/data/gsm8k/train.parquet
gsm8k_test_path=~/data/gsm8k/test.parquet

train_files=$gsm8k_train_path
test_files=$gsm8k_test_path

CKPT_DIR=~/checkpoints

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

rollout_mode="async"
rollout_name="sglang"
return_raw_chat="True"

project_name="verl_grpo_example_gsm8k_math"
exp_name="qwen3_30b_moe_megatron"

PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo --config-path=./config --config-name='ppo_megatron_trainer' \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.train_batch_size=256 \
    data.max_prompt_length=512 \
    data.max_response_length=512 \
    data.truncation='error' \
    data.return_raw_chat=$return_raw_chat \
    actor_rollout_ref.model.path=$LLM \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.total_training_steps=200 \
    actor_rollout_ref.actor.ppo_epochs=1 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=4096 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.cliprange=0.2 \
    actor_rollout_ref.actor.cliprange_value=0.2 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean \
    actor_rollout_ref.actor.megatron.tp_size=4 \
    actor_rollout_ref.actor.megatron.pp_size=4 \
    actor_rollout_ref.actor.megatron.vpp_size=8 \
    actor_rollout_ref.actor.megatron.ep_size=4 \
    actor_rollout_ref.actor.megatron.etp_size=1 \
    actor_rollout_ref.actor.megatron.sequence_parallel_size=1 \
    actor_rollout_ref.actor.megatron.use_distributed_optimizer=True \
    actor_rollout_ref.actor.megatron.param_offload=${ACTOR_PARAM_OFFLOAD} \
    actor_rollout_ref.actor.megatron.grad_offload=${ACTOR_GRAD_OFFLOAD} \
    actor_rollout_ref.actor.megatron.optimizer_offload=${ACTOR_OPTIMIZER_OFFLOAD} \
    actor_rollout_ref.actor.megatron.use_dist_checkpointing=True \
    actor_rollout_ref.actor.megatron.dist_checkpointing_path=$DIST_CKPT_PATH \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    # Configuration from HEAD branch
    actor_rollout_ref.rollout.tensor_model_parallel_size=8 \
    actor_rollout_ref.rollout.name=$rollout_name \
    actor_rollout_ref.rollout.mode=$rollout_mode \
    # Alternative configuration from main branch
    # actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    # actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    # Configuration from HEAD branch
    actor_rollout_ref.rollout.temperature=0.0 \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.rollout.free_cache_engine=True \
    # Alternative configuration from main branch
    # actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    # Configuration from HEAD branch
    actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=4 \
    actor_rollout_ref.ref.megatron.tensor_model_parallel_size=1 \
    actor_rollout_ref.ref.megatron.expert_model_parallel_size=8 \
    # Alternative configuration from main branch
    # actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=2 \
    # actor_rollout_ref.ref.megatron.tensor_model_parallel_size=4 \
    # actor_rollout_ref.ref.megatron.expert_model_parallel_size=4 \
    actor_rollout_ref.ref.megatron.use_dist_checkpointing=True \
    actor_rollout_ref.ref.megatron.dist_checkpointing_path=$DIST_CKPT_PATH \
    +reward_model.enable_reward_workers=False \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    # Configuration from HEAD branch
    trainer.val_before_train=False \
    trainer.logger=['console'] \
    # Alternative configuration from main branch
    # trainer.logger='["console","wandb"]' \
    # Project configuration from HEAD branch
    trainer.project_name='verl_grpo_example_gsm8k_math' \
    trainer.experiment_name='qwen3_30b_moe_megatron' \
    # Alternative project configuration from main branch
    # trainer.project_name="${project_name}" \
    # trainer.experiment_name="${exp_name}" \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=4 \
    trainer.save_freq=20 \
    trainer.test_freq=5 \
    trainer.default_local_dir=$CKPT_DIR \
    ++reward_function._target_=verl.utils.reward_score.gsm8k.compute_math_reward \
    algorithm.adv_estimator=grpo \
    algorithm.gamma=1.0 \
    algorithm.lam=0.95 \
    algorithm.adv_norm=False \
    2>&1 | tee run.log
