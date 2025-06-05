set -x
ENGINE=${1:-vllm}
export http_proxy=http://oversea-squid1.jp.txyun:11080 https_proxy=http://oversea-squid1.jp.txyun:11080 no_proxy=localhost,127.0.0.1,localaddress,localdomain.com,internal,corp.kuaishou.com,test.gifshow.com,staging.kuaishou.com 
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_SOCKET_IFNAME=eth
export NCCL_IB_HCA=mlx5
# 使用自己的wandb_api_key
export WANDB_API_KEY=2a51ee77ead415a63b08e6c2955ff2383f2d7fab

wandb online

# 下面的路径记得修改为自己的
HOME=/nlp_group/huangjiaming
project_name='verl_grpo_keye_8node_for_long_cot_full_0603_tianlin_perf'
exp_name='htl_test'

CKPTS_DIR=${CKPTS_DIR:-"${HOME}/ckpts/${project_name}/${exp_name}"}
# 务必保留,用于分布式group打点做区分
timestamp=$(date +"%Y-%m-%d-%H:%M:%S")""
runtime_env_json="{\"env_vars\": {\"TIMESTAMP\":\"${timestamp}\", \"MONDB_PROJECT_NAME\": \"${project_name}\"}}"

export TIMESTAMP=${timestamp}
export MONDB_PROJECT_NAME=${project_name}
export HYDRA_FULL_ERROR=1


/opt/conda/envs/py310/bin/python3 -m verl.trainer.main_ppo \
    ++user_custom_env.MIN_PIXELS=1024 \
    ++user_custom_env.MAX_PIXELS=1310720 \
    algorithm.adv_estimator=grpo \
    data.custom_cls.name=Qwen3RLHFDataset \
    data.custom_cls.path=$HOME/kai-verl/verl/utils/dataset/qwen3_rl_dataset.py \
    ++data.base_model_dir=/hetu_group/jky/misc/tools/swift_20250508/playground/keye_8b/rl/20250604.1.cot_mix_nowarmup_newthinktoken__vllm__dapo/output/v1-20250604-205443/checkpoint-10 \
    data.train_files=[$HOME/kai-verl/dataset_MMPR_K12_nn_addTokenLen__mmpr1.1_minlen30_sample5w__fixsystem__instuctnothink__new_think_token__fixnothink.parquet,$HOME/kai-verl/OpenR1_Math_220k_rule_long_cot_new_think_token.parquet,$HOME/kai-verl/ocr_qwen2vl7b_sampled_jsonl_rule_selected_fromcyy_cotmixrlformat.parquet,$HOME/kai-verl/VQAv2_sample2w__cotmixrlformat.parquet] \
    data.val_files=$HOME/kai-verl/single.parquet \
    data.train_batch_size=32 \
    ++data.dataloader_num_workers=8 \
    data.max_prompt_length=5120 \
    data.max_response_length=3072 \
    data.filter_overlong_prompts=False \
    data.truncation='error' \
    data.prompt_key=messages \
    data.image_key=images \
    data.reward_fn_key=swift_reward_type \
    actor_rollout_ref.model.path=/hetu_group/jky/misc/tools/swift_20250508/playground/keye_8b/rl/20250604.1.cot_mix_nowarmup_newthinktoken__vllm__dapo/output/v1-20250604-205443/checkpoint-10 \
    +actor_rollout_ref.model.trust_remote_code=True \
    ++actor_rollout_ref.actor.freeze_vision_tower=True \
    actor_rollout_ref.actor.optim.lr=5e-7 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.0 \
    actor_rollout_ref.actor.optim.warmup_style=cosine \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_epochs=2 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.grad_clip=0.5 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    ++actor_rollout_ref.actor.fsdp_config.use_orig_params=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
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
    ++reward_model.reward_kwargs.reward_fn_types=\'ModelBaseAccuracy,MyFormat\' \
    ++reward_model.reward_kwargs.model_api_address=\'10.82.121.34,10.82.122.98,10.82.120.218\' \
    ++reward_model.reward_kwargs.model_api_port=\'8000\' \
    ++reward_model.enable_reward_workers=True \
    trainer.balance_batch=True \
    trainer.critic_warmup=0 \
    trainer.val_before_train=False \
    trainer.logger=['console','wandb'] \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=4 \
    trainer.save_freq=10 \
    trainer.test_freq=5 \
    trainer.default_local_dir="${CKPTS_DIR}" \
    trainer.total_epochs=1 $@
