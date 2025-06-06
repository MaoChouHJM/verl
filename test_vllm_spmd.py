from vllm import LLM, SamplingParams
import os



os.environ["RANK"] ='0'
os.environ["LOCAL_RANK"] ='0'
os.environ["MASTER_ADDR"]="127.0.0.1"
os.environ["MASTER_PORT"]="8000"
 
inference_engine = LLM(
    model="/hetu_group/jky/misc/tools/swift_20250508/playground/keye_8b/rl/20250604.1.cot_mix_nowarmup_newthinktoken__vllm__dapo/output/v1-20250604-205443/checkpoint-10",
    enable_sleep_mode=True,
    tensor_parallel_size=1,
    distributed_executor_backend="external_launcher",
    enforce_eager=False,
    gpu_memory_utilization=0.6,
    disable_custom_all_reduce=True,
    disable_mm_preprocessor_cache=True,
    limit_mm_per_prompt={"image": 1, "video": 0},
    skip_tokenizer_init=False,
    max_model_len=8192,
    load_format="dummy",
    disable_log_stats=True,
    max_num_batched_tokens=8192,
    enable_chunked_prefill=False,
    enable_prefix_caching=True,
    trust_remote_code=True,
    seed=0,
)
