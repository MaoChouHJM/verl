set -x

LLM=~/models/Qwen2.5-VL-7B-Instruct
geo3k_train_path=~/data/geo3k/train.parquet
geo3k_test_path=~/data/geo3k/test.parquet

train_files=$geo3k_train_path
test_files=$geo3k_test_path

CKPT_DIR=~/checkpoints

# offload
COMMON_PARAM_OFFLOAD=False
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

PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo --config-path=./config --config-name='ppo_megatron_trainer' \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.train_batch_size=256 \
    data.max_prompt_length=512 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=True \
    data.return_multi_modal_inputs=True \
    actor_rollout_ref.model.path=$LLM \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.total_training_steps=500 \
    actor_rollout_ref.actor.ppo_epochs=1 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=4096 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.cliprange=0.2 \
    actor_rollout_ref.actor.cliprange_value=0.2 \
    actor_rollout_ref.actor.use_dynamic_bsz=False \
    actor_rollout_ref.actor.loss_agg_mode=token-mean \
    actor_rollout_ref.actor.megatron.tp_size=2 \
    actor_rollout_ref.actor.megatron.pp_size=2 \
    actor_rollout_ref.actor.megatron.vpp_size=4 \
    actor_rollout_ref.actor.megatron.ep_size=1 \
    actor_rollout_ref.actor.megatron.etp_size=1 \
    actor_rollout_ref.actor.megatron.sequence_parallel_size=1 \
    actor_rollout_ref.actor.megatron.use_distributed_optimizer=True \
    actor_rollout_ref.actor.megatron.param_offload=${ACTOR_PARAM_OFFLOAD} \
    actor_rollout_ref.actor.megatron.grad_offload=${ACTOR_GRAD_OFFLOAD} \
    actor_rollout_ref.actor.megatron.optimizer_offload=${ACTOR_OPTIMIZER_OFFLOAD} \
    actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=uniform \
    actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=full \
    actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=1 \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.top_k=-1 \
    actor_rollout_ref.rollout.infer_tp=2 \
    actor_rollout_ref.rollout.load_format=dummy_megatron \
    actor_rollout_ref.ref.megatron.tp_size=2 \
    actor_rollout_ref.ref.megatron.pp_size=2 \
    actor_rollout_ref.ref.megatron.vpp_size=4 \
    actor_rollout_ref.ref.megatron.ep_size=1 \
    actor_rollout_ref.ref.megatron.etp_size=1 \
    actor_rollout_ref.ref.megatron.sequence_parallel_size=1 \
    actor_rollout_ref.ref.megatron.use_distributed_optimizer=True \
    actor_rollout_ref.ref.megatron.param_offload=${REF_PARAM_OFFLOAD} \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    # Logger configuration from HEAD branch
    trainer.logger=['console'] \
    # Alternative logger configuration from main branch
    # trainer.logger='["console","wandb"]' \
    trainer.project_name='verl_grpo_example_geo3k' \
    trainer.experiment_name='qwen2_5_vl_7b_megatron' \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.val_before_train=False \
    trainer.test_freq=10 \
    trainer.save_freq=10 \
    trainer.default_local_dir=$CKPT_DIR \
    ++reward_function._target_=verl.utils.reward_score.geo3k.compute_geo3k_reward_batch \
    algorithm.adv_estimator=grpo \
    algorithm.gamma=1.0 \
    algorithm.lam=0.95 \
    algorithm.adv_norm=False \
    2>&1 | tee run.log
