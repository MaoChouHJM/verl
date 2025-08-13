set -x
ln -s /mmu_mllm_hdd /mllm_hdd
DIST_CKPT_PATH="/nlp_group/yuanjiawei05/new_logits_distill/new_converted_hf"
LLM="/mmu_mllm_hdd_2/wenbin/SFT/Keye-8B/AutoThink/20250801.new_pretrain_mpo_cotmix_addmore_256gpu/output/v0-20250731-203710/checkpoint-2544"
HOME=/nlp_group/huangjiaming/
VAL_DUMP_DIR="/nlp_group/yuanjiawei05/new_logits_distill/val_dir"
TRAIN_DUMP_DIR="/nlp_group/yuanjiawei05/new_logits_distill/train_dir"
timestamp=$(date +"%Y-%m-%d-%H:%M:%S")""

# 2. run the script
gsm8k_train_path=$HOME/data/gsm8k/train.parquet
gsm8k_test_path=/nlp_group/huangjiaming/logits-distill/random_row.parquet
# train_files=$gsm8k_train_path
# few data
# test_files=/nlp_group/huangjiaming/data/keye_text_image_rl_data/verl_dataset_debug_text_img.parquet
# val data with swift
test_files=/nlp_group/huangjiaming/data/keye_text_image_rl_data/val.parquet


bad_cases=/nlp_group/yuanjiawei05/new_logits_distill/single_badcase.parquet
train_files=$bad_cases

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

n_resp_per_prompt=8
project_name='verl_megatron_gsm8k_examples'
experiment_name='dsv3-32nodes'


rollout_mode="async"
rollout_name="sglang" # sglang or vllm
if [ "$rollout_mode" = "async" ]; then
    export VLLM_USE_V1=1
    return_raw_chat="True"
fi

export HYDRA_FULL_ERROR=1

# rm data.gen_batch_size
# modify data.train_batch_size to 32

python3 -m verl.trainer.main_ppo --config-path=./config --config-name='ppo_megatron_trainer'\
    ++user_custom_env.USE_SLOW_FAST=True \
    ++user_custom_env.MIN_PIXELS=102400 \
    ++user_custom_env.MAX_PIXELS=3010560 \
    ++user_custom_env.KEYE_IMAGE_FACTOR=28 \
    ++user_custom_env.FPS_MAX_FRAMES=32 \
    ++user_custom_env.VIDEO_TOTAL_PIXELS=6422528 \
    algorithm.adv_estimator=grpo \
    data.custom_cls.name=KeyeQwen3SlowFastDataset \
    data.custom_cls.path=$PWD/verl/utils/dataset/keye_qwen3_slowfast_dataset.py \
    ++data.base_model_dir=$LLM \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.return_raw_chat=$return_raw_chat \
    data.trust_remote_code=True \
    data.train_batch_size=1 \
    data.max_prompt_length=40960 \
    data.max_response_length=5120 \
    data.filter_overlong_prompts=False \
    data.truncation='error' \
    data.prompt_key=messages \
    +data.image_key=images \
    data.reward_fn_key=swift_reward_type \
    +data.validation_shuffle=False \
    +data.gen_batch_size=1 \
    actor_rollout_ref.model.path=$LLM \
    actor_rollout_ref.model.trust_remote_code=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=1 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.use_torch_compile=False \
    actor_rollout_ref.actor.load_weight=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.mode=$rollout_mode \
    actor_rollout_ref.rollout.load_format=auto \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.60 \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    actor_rollout_ref.rollout.temperature=0.0 \
    actor_rollout_ref.rollout.top_p=0.9 \
    actor_rollout_ref.rollout.top_k=1 \
    +actor_rollout_ref.rollout.repetition_penalty=1.0 \
    ++actor_rollout_ref.rollout.debug_dump_val_rollout_result=True \
    ++actor_rollout_ref.rollout.debug_dump_train_rollout_result=True \
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.9 \
    actor_rollout_ref.rollout.val_kwargs.top_k=50 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=False \
    +actor_rollout_ref.rollout.val_kwargs.repetition_penalty=1.0 \
    actor_rollout_ref.rollout.val_kwargs.n=${n_resp_per_prompt} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$INFER_TP \
    actor_rollout_ref.rollout.free_cache_engine=True \
    +actor_rollout_ref.rollout.override_config.chunked_prefill_size=32768 \
    algorithm.use_kl_in_reward=False \
    reward_model.reward_manager=keye \
    reward_model.launch_reward_fn_async=True \
    ++reward_model.reward_kwargs.reward_fn_types=\'ModelBaseAccuracyV2,MyFormat\' \
    ++reward_model.reward_kwargs.model_api_address=\'10.48.47.83\' \
    ++reward_model.reward_kwargs.model_api_port=\'8001,8002,8003,8004\' \
    ++reward_model.enable_reward_workers=True \
    trainer.logger=['console'] \
    trainer.project_name=$project_name \
    trainer.experiment_name=$experiment_name \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=$NODES \
    trainer.save_freq=20 \
    trainer.test_freq=5 \
    +trainer.validation_data_dir=$VAL_DUMP_DIR \
    +trainer.val_result_dump_dir=$VAL_DUMP_DIR \
    +trainer.train_result_dump_dir=$TRAIN_DUMP_DIR \
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
