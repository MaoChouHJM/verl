set -x
ENGINE=${1:-vllm}
# If you are using vllm<=0.6.3, you might need to set the following environment variable to avoid bugs:
# export VLLM_ATTENTION_BACKEND=XFORMERS
export http_proxy=http://oversea-squid1.jp.txyun:11080 https_proxy=http://oversea-squid1.jp.txyun:11080 no_proxy=localhost,127.0.0.1,localaddress,localdomain.com,internal,corp.kuaishou.com,test.gifshow.com,staging.kuaishou.com 
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_SOCKET_IFNAME=eth
export NCCL_IB_HCA=mlx5
export WANDB_API_KEY=173c259f71eff84cef8c20b35fcdfa0aff803073

wandb online

HOME=/nlp_group/huangjiaming

# env json
project_name='verl_grpo_keye_8node'
exp_name='hjm_test'

CKPTS_DIR=${CKPTS_DIR:-"${HOME}/ckpts/${project_name}/${exp_name}"}
timestamp=$(date +"%Y-%m-%d-%H:%M:%S")""
runtime_env_json="{\"env_vars\": {\"TIMESTAMP\":\"${timestamp}\", \"MONDB_PROJECT_NAME\": \"${project_name}\"}}"

export TIMESTAMP=${timestamp}
export MONDB_PROJECT_NAME=${project_name}
export HYDRA_FULL_ERROR=1



#    custom_reward_function.path=$HOME/kai-verl/keye_reward/__init__.py \
#    custom_reward_function.name=keye_compute_reward \



/opt/conda/envs/py310/bin/python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.custom_cls.name=Qwen3RLHFDataset \
    data.custom_cls.path=$HOME/kai-verl/verl/utils/dataset/qwen3_rl_dataset.py \
    ++data.hf_dataset_config=/llm_reco/lingzhixin/recovlm_qw0510/recovlm/examples/vlm/keye/debug_keye_8B256.json \
    data.train_files=$HOME/kai-verl/converted.parquet \
    data.val_files=$HOME/kai-verl/single.parquet \
    data.train_batch_size=192 \
    ++data.dataloader_num_workers=8 \
    data.max_prompt_length=5120 \
    data.max_response_length=3072 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.prompt_key=messages \
    data.image_key=images \
    data.reward_fn_key=swift_reward_type \
    actor_rollout_ref.model.path=/nlp_group/huangjiaming/20250519 \
    ++actor_rollout_ref.actor.freeze_vision_tower=True \
    actor_rollout_ref.actor.optim.lr=5e-7 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.01 \
    actor_rollout_ref.actor.optim.warmup_style=cosine \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=192 \
    actor_rollout_ref.actor.ppo_epochs=2 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.grad_clip=0.5 \
    actor_rollout_ref.model.enable_gradient_checkpointing=False \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    ++actor_rollout_ref.actor.fsdp_config.use_orig_params=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=24 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.n=8 \
    ++actor_rollout_ref.rollout.top_k=50 \
    ++actor_rollout_ref.rollout.top_p=0.9 \
    ++actor_rollout_ref.rollout.temperature=1.0 \
    ++actor_rollout_ref.rollout.repetition_penalty=1.1 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    reward_model.reward_manager=keye \
    ++reward_model.enable_reward_workers=True \
    trainer.balance_batch=True \
    trainer.critic_warmup=0 \
    trainer.val_before_train=False \
    trainer.logger=['console','wandb'] \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=8 \
    trainer.save_freq=10 \
    trainer.test_freq=5 \
    trainer.default_local_dir="${CKPTS_DIR}" \
    trainer.total_epochs=1 $@
