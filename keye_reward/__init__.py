import os
import re
from typing import Dict, List, Union

import json

import asyncio
import httpx
from openai import OpenAI

from latex2sympy2_extended import NormalizationConfig
from math_verify import LatexExtractionConfig, parse, verify


# prompt_template = """
# ### question
# {{question}}
# ### ground truth
# {{real_answer}}
# ### answer
# {{model_answer}}
# ## Requirement
# Evaluate the provided answer based on the question and ground truth.
# Provide your response as follows:
# - Respond with **’Yes’** if the answer is right.
# - Respond with **’No’** if the answer is wrong.
# Only respond with a single word Yes or No."""

prompt_template = """You are given a question, the correct ground truth answer, and a candidate answer.
### Question
{{question}}
### Ground Truth Answer
{{real_answer}}
### Candidate Answer
{{model_answer}}
## Task
- Determine whether the candidate answer is factually correct and fully consistent with the ground truth answer.
- Respond with:
    - Yes — if the candidate answer is correct.
    - No — if the candidate answer is incorrect.
- Output only a single word: Yes or No. Do not provide any explanation."""


class Singleton(object):
    _instances = {}

    def __new__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__new__(cls, *args, **kwargs)
        return cls._instances[cls]


async def llm_openai_api(
    messages,
    ip="10.82.120.86",
    host="1222",
    temperature=0.7,
    max_tokens=2048,
    top_p=0.7,
    openai_api_key="EMPTY",
    n=1,  # from lxy
):
    openai_api_base = f"http://{ip}:{host}/v1"
    # 使用异步 HTTP 客户端
    async with httpx.AsyncClient(timeout=httpx.Timeout(600.0)) as client:
        # 获取可用模型
        model_list = await client.get(
            f"{openai_api_base}/models",
            headers={"Authorization": f"Bearer {openai_api_key}"},
        )
        model_list.raise_for_status()
        models = model_list.json()
        model = models["data"][0]["id"]

        # 发起 chat.completions 异步请求
        resp = await client.post(
            f"{openai_api_base}/chat/completions",
            headers={"Authorization": f"Bearer {openai_api_key}"},
            json={
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "top_p": top_p,
                "n": n,
            },
        )
        resp.raise_for_status()
        response_data = resp.json()
        if n == 1:
            return response_data["choices"][0]["message"]["content"]
        else:
            return [choice["message"]["content"] for choice in response_data["choices"]]


class compute_model_base_reward(Singleton):
    def __init__(self):
        self.rank = int(os.getenv("RANK", -1))
        self.world_size = int(os.getenv("WORLD_SIZE", 1))
        self.api_address_list = list(
            map(str, os.getenv("MODEL_API_ADDRESS", "10.82.120.86").split(","))
        )
        self.api_port_list = list(
            map(int, os.getenv("MODEL_API_PORT", "1222").split(","))
        )
        self.loop = asyncio.get_event_loop()

    def __call__(self, completions, solution_str, **kwargs) -> List[float]:
        """
        Reward function that checks if the completion is correct.
        Args:
            completions (list[str]): Generated outputs
            solution (list[str]): Ground Truths.

        Returns:
            list[float]: Reward scores
        """
        cur_task = self.loop.create_task(
            self.async_call(completions, solution_str, **kwargs)
        )
        return self.loop.run_until_complete(cur_task)

    async def async_call(self, completions, solution_str, **kwargs) -> float:
        # NOTE(huangjiaming): completions  batch_size = 1
        message = kwargs.get("messages", [])
        if_longcot = message[0]["role"] == "system"
        #if_longcots = [m[0]["role"] == "system" for m in messages]
        question = message[1]["content"] if if_longcot else message[0]["content"]
        prompt_type = kwargs.get("prompt_type", "")

        swift_reward_type = kwargs.get("swift_reward_type", "")
        length = 1
        tasks = []
        cur_address = self.api_address_list[
            (self.rank * length) % len(self.api_address_list)
        ]
        cur_port = self.api_port_list[
            (self.rank * length) % len(self.api_port_list)
        ]
        reward = asyncio.create_task(
            self.evaluate(
                completions, solution_str, question, cur_address, cur_port, swift_reward_type, prompt_type
            )
        )
        tasks.append(reward)
        rewards = await asyncio.gather(*tasks)
        # only need float reward
        reward = rewards[0]
        return reward

    async def evaluate(
        self,
        cur_completion,
        cur_solution,
        question,
        cur_address,
        cur_port,
        swift_reward_type,
        prompt_type,
        **kwargs,
    ) -> float:
        """
        Reward function that checks if the completion is correct.
        Args:
            completions (list[str]): Generated outputs
            solution (list[str]): Ground Truths.

        Returns:
            list[float]: Reward scores
        """
        reward = 0.0
        if swift_reward_type != "model_base":
            return reward
        try:
            if prompt_type == "longcot":
                content_match = re.search(r"<answer>(.*?)</answer>", cur_completion)
                student_answer = (
                    content_match.group(1).strip()
                    if content_match
                    else cur_completion.strip()
                )
                cur_completion = student_answer
            
            prompt = (
                prompt_template.replace("{{question}}", question)
                .replace("{{real_answer}}", cur_solution)
                .replace("{{model_answer}}", cur_completion)
            )
            messages = [{"role": "user", "content": prompt}]
            max_try = 50
            for _ in range(max_try):
                try:
                    completion = await llm_openai_api(
                        messages, ip=cur_address, host=cur_port
                    )
                    completion = (
                        completion[0] if isinstance(completion, list) else completion
                    )
                    try:
                        with open(
                            f"./plugin_completion_log/rank{self.rank}.jsonl",
                            "a",
                            encoding="utf-8",
                        ) as f:
                            dump_data = {
                                "prompt": prompt,
                                "completion": completion,
                            }
                            f.write(json.dumps(dump_data, ensure_ascii=False) + "\n")
                            f.flush()
                    except:
                        pass
                    break
                except Exception as e:
                    print(f"xx-{e}")
                    print(dump_data)
                    continue
            if completion.lower() == "yes":
                reward = 1.0
        except Exception as e:
            print(e)
        return reward




def compute_math_reward(completion, solution_str, **kwargs) -> float:
    swift_reward_type = kwargs.get("swift_reward_type", "rule_base")
    pattern = r"^<think>.*?</think>\s*<answer>.*?</answer>(?![\s\S])"
    match = re.match(pattern, completion, re.DOTALL | re.MULTILINE)
    gold_parsed = parse(
        solution_str,
        extraction_mode="first_match",
        extraction_config=[LatexExtractionConfig()],
    )
    if not match or swift_reward_type == "model_base":
        reward = 0.0
    else:
        if len(gold_parsed) != 0:
            content_match = re.search(r"<answer>(.*?)</answer>", content)
            student_answer = (
                content_match.group(1).strip()
                if content_match
                else content.strip()
            )
            # We require the answer to be provided in correct latex (no malformed operators)
            answer_parsed = parse(
                "<answer>" + student_answer + "</answer>",
                extraction_config=[
                    LatexExtractionConfig(
                        normalization_config=NormalizationConfig(
                            nits=False,
                            malformed_operators=False,
                            basic_latex=True,
                            equations=True,
                            boxed=True,
                            units=True,
                        ),
                        # Ensures that boxed is tried first
                        boxed_match_priority=0,
                        try_extract_without_anchor=False,
                    )
                ],
                extraction_mode="first_match",
            )
            # Reward 1 if the content is the same as the ground truth, 0 otherwise
            reward = float(verify(answer_parsed, gold_parsed))
        else:
            # If the gold solution is not parseable, we reward 1 to skip this example
            reward = 1.0
    return reward


def keye_compute_reward(data_source, solution_str, ground_truth, **kwargs):
    if data_source == "model_base":
        return compute_model_base_reward()(solution_str, ground_truth, **kwargs)
    elif data_source == "rule_base":
        return compute_math_reward(solution_str, ground_truth, **kwargs)
    else:
        raise NotImplementedError(f'keye_compute_reward : not implemented for {data_source}')
