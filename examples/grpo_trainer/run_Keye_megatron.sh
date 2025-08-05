set -x

DIST_CKPT_PATH="/mmu_mllm_hdd_2/lilaiyi/718/kai-megatron/output/save/dist_ckpt_step27000/iter_0000001"
LLM="/mmu_mllm_hdd_2/zhouyang12/output1/Keye/0.9.3/Stage2/8b/slowfast-0721-0717-v2/step27000/global_step27000/converted"
HOME=/nlp_group/huangjiaming/
timestamp=$(date +"%Y-%m-%d-%H:%M:%S")""

# 2. run the script
gsm8k_train_path=$HOME/data/gsm8k/train.parquet
train_files=$gsm8k_train_path

ALL_OFFLOAD=${ALL_OFFLOAD:-True}
COMMON_PARAM_OFFLOAD=${COMMON_PARAM_OFFLOAD:-$ALL_OFFLOAD}
COMMON_GRAD_OFFLOAD=${COMMON_GRAD_OFFLOAD:-$ALL_OFFLOAD}
COMMON_OPTIMIZER_OFFLOAD=${COMMON_OPTIMIZER_OFFLOAD:-$ALL_OFFLOAD}

ACTOR_PARAM_OFFLOAD=${ACTOR_PARAM_OFFLOAD:-$COMMON_PARAM_OFFLOAD}
ACTOR_GRAD_OFFLOAD=${ACTOR_GRAD_OFFLOAD:-$COMMON_GRAD_OFFLOAD}
ACTOR_OPTIMIZER_OFFLOAD=${ACTOR_OPTIMIZER_OFFLOAD:-$COMMON_OPTIMIZER_OFFLOAD}

# 512 H20(96GB)
NODES=8
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

python3 -m verl.trainer.main_ppo --config-path=./config --config-name='ppo_megatron_trainer'\
    ++user_custom_env.USE_SLOW_FAST=False \
    ++user_custom_env.WANDB_API_KEY=local-e7f5c4b994f551dce31a51b20daac7f1eb528d88 \
    ++user_custom_env.MIN_PIXELS=102400 \
    ++user_custom_env.MAX_PIXELS=3010560 \
    ++user_custom_env.KEYE_IMAGE_FACTOR=28 \
    ++user_custom_env.FPS_MAX_FRAMES=32 \
    ++user_custom_env.VIDEO_TOTAL_PIXELS=6422528 \
    algorithm.adv_estimator=grpo \
    data.train_files="$train_files" \
    data.return_raw_chat=$return_raw_chat \
    data.train_batch_size=128 \
    data.max_prompt_length=3072 \
    data.max_response_length=5120 \
    data.filter_overlong_prompts=False \
    data.truncation='error' \
    actor_rollout_ref.model.custom_chat_template="{% set image_count = namespace(value=0) %}{% set video_count = namespace(value=0) %}{% for message in messages %}<|im_start|>{{ message['role'] }}\n{% if message['content'] is string %}{{ message['content'] }}<|im_end|>\n{% else %}{% for content in message['content'] %}{% if content['type'] == 'image' or 'image' in content or 'image_url' in content %}{% set image_count.value = image_count.value + 1 %}{% if add_vision_id %}Picture {{ image_count.value }}: {% endif %}<|vision_start|><|image_pad|><|vision_end|>{% elif content['type'] == 'video' or 'video' in content %}{% set video_count.value = video_count.value + 1 %}{% if add_vision_id %}Video {{ video_count.value }}: {% endif %}<|vision_start|><|video_pad|><|vision_end|>{% elif 'text' in content %}{{ content['text'] }}{% endif %}{% endfor %}<|im_end|>\n{% endif %}{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}" \
    actor_rollout_ref.model.path=$LLM \
    actor_rollout_ref.model.trust_remote_code=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=1024 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.use_torch_compile=False \
    actor_rollout_ref.actor.load_weight=True \
    actor_rollout_ref.actor.ppo_epochs=4 \
    actor_rollout_ref.actor.grad_clip=0.5 \
    actor_rollout_ref.actor.optim.lr=5e-7 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.005 \
    actor_rollout_ref.actor.optim.warmup_style=cosine \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.mode=$rollout_mode \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.60 \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=0.9 \
    actor_rollout_ref.rollout.top_k=50 \
    ++actor_rollout_ref.rollout.repetition_penalty=1.0 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$INFER_TP \
    actor_rollout_ref.rollout.free_cache_engine=True \
    algorithm.use_kl_in_reward=False \
    trainer.logger=['console','wandb'] \
    trainer.project_name=$project_name \
    trainer.experiment_name=$experiment_name \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=$NODES \
    trainer.save_freq=2 \
    trainer.test_freq=-1 \
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
    trainer.total_epochs=1 2>&1 | tee keye_$timestamp.log
