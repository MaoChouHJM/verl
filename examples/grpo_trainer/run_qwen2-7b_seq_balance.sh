set -x

LLM=~/models/Qwen2-7B-Instruct
gsm8k_train_path=~/data/gsm8k/train.parquet
gsm8k_test_path=~/data/gsm8k/test.parquet

train_files=$gsm8k_train_path
test_files=$gsm8k_test_path

CKPT_DIR=~/checkpoints

# offload
COMMON_PARAM_OFFLOAD=True
COMMON_GRAD_OFFLOAD=False
COMMON_OPTIMIZER_OFFLOAD=False
ACTOR_PARAM_OFFLOAD=${ACTOR_PARAM_OFFLOAD:-$COMMON_PARAM_OFFLOAD}
ACTOR_GRAD_OFFLOAD=${ACTOR_GRAD_OFFLOAD:-$COMMON_GRAD_OFFLOAD}
ACTOR_OPTIMIZER_OFFLOAD=${ACTOR_OPTIMIZER_OFFLOAD:-$COMMON_OPTIMIZER_OFFLOAD}
REF_PARAM_OFFLOAD=${REF_PARAM_OFFLOAD:-$COMMON_PARAM_OFFLOAD}
CRITIC_PARAM_OFFLOAD=${CRITIC_PARAM_OFFLOAD:-$COMMON_PARAM_OFFLOAD}
CRITIC_GRAD_OFFLOAD=${CRITIC_GRAD_OFFLOAD:-$COMMON_GRAD_OFFLOAD}
CRITIC_OPTIMIZER_OFFLOAD=${CRITIC_OPTIMIZER_OFFLOAD:-$COMMON_OPTIMIZER_OFFLOAD}
RM_PARAM_OFFLOAD=${RM_PARAM_OFFLOAD:-$COMMON_PARAM_OFFLOAD}

python3 -m verl.trainer.main_ppo --config-path=./config --config-name='ppo_trainer' \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.train_batch_size=512 \
    data.val_batch_size=64 \
    data.max_prompt_length=512 \
    data.max_response_length=512 \
    data.total_epochs=10 \
    actor_rollout_ref.model.path=$LLM \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.total_training_steps=150 \
    actor_rollout_ref.actor.ppo_epochs=1 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=8192 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.cliprange=0.2 \
    actor_rollout_ref.actor.cliprange_value=0.2 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.enable_activation_offload=True \
    actor_rollout_ref.actor.megatron.tp_size=2 \
    actor_rollout_ref.actor.megatron.sequence_parallel_size=1 \
    actor_rollout_ref.actor.megatron.param_offload=${ACTOR_PARAM_OFFLOAD} \
    actor_rollout_ref.actor.megatron.grad_offload=${ACTOR_GRAD_OFFLOAD} \
    actor_rollout_ref.actor.megatron.optimizer_offload=${ACTOR_OPTIMIZER_OFFLOAD} \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    +reward_model.enable_reward_workers=False \
    trainer.critic_warmup=0 \
    # Logger configuration from HEAD branch
    trainer.logger=['console'] \
    # Alternative logger configuration from main branch
    # trainer.logger='["console","wandb"]' \
    trainer.project_name='verl_grpo_example_gsm8k' \
    trainer.experiment_name='qwen2_7b_function_rm_kl1e-3' \
    trainer.val_before_train=False \
    trainer.test_freq=5 \
    trainer.save_freq=5 \
    trainer.default_local_dir=$CKPT_DIR \
    ++reward_function._target_=verl.utils.reward_score.gsm8k.compute_math_reward \
    algorithm.adv_estimator=grpo \
    algorithm.gamma=1.0 \
    algorithm.lam=0.95 \
    algorithm.adv_norm=False \
    2>&1 | tee run.log
