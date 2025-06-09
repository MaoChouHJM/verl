import os
import re
from typing import Dict, List, Union

import json
import numpy as np

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

prompt_template = """
You are a expert assistant. I have a question, and a model will respond based on the question and the image provided (you can not see image here). 
The model will first output its thought process, followed by the final answer. I need your help to evaluate the correctness of the model's output and its thought process.
Though you can not see the provided image, I will provide the ground truth for your reference. 

#### Question: \n{Question}\n\n
#### Ground Truth: \n{Ground_Truth}\n\n
#### Model Thought Process: \n{Model_Thought_Process}\n\n
#### Model Output: \n{Model_Output}\n\n

Please provide your evaluation based on the following criteria:
1. Is the thought process related to the question? If the model's thought process contains very irrelevant information, please mark it as "Unrelated".
2. Is the model's output correct based on the ground truth? If the model's output is correct, please mark it as "Yes". Otherwise, please mark it as "No".
3. Evaluate the quality of the thought process. If the thought process is well-structured, logical, and comprehensive, mark it as "High". If it is somewhat logical but lacks details or has minor flaws, mark it as "Medium". If it is poorly structured or contains significant errors, mark it as "Low".
4. You fisrt need to read the question and the ground truth, then read the model's thought process and output.
5. Before you provide your evaluation, please give a brief comment to help us understand your evaluation better.
6. Output your evaluation in the json format: 
```
{
    "Comments": "Your brief comment.",
    "Correctness": "Yes/No",
    "Thought_Process": "Related/Unrelated",
    "Thought_Process_Quality": "High/Medium/Low"
}
```
"""


#prompt_template = """You are given a question, the correct ground truth answer, and a candidate answer.
#### Question
#{{question}}
#### Ground Truth Answer
#{{real_answer}}
#### Candidate Answer
#{{model_answer}}
### Task
#- Determine whether the candidate answer is factually correct and fully consistent with the ground truth answer.
#- Respond with:
#    - Yes — if the candidate answer is correct.
#    - No — if the candidate answer is incorrect.
#- Output only a single word: Yes or No. Do not provide any explanation."""


class Singleton(object):
    _instances = {}

    def __new__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__new__(cls, *args, **kwargs)
        return cls._instances[cls]


async def llm_openai_api(
    messages,
    ip="127.0.0.1",
    host="1222",
    temperature=0.7,
    max_tokens=2048,
    top_p=0.8,
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


class ModelBaseAccuracy(object):
    def __init__(self, **reward_kwargs):
        self.rank = int(os.getenv("RANK", -1))
        self.world_size = int(os.getenv("WORLD_SIZE", 1))
        self.api_address_list = list(
            map(str, reward_kwargs.get("model_api_address", "127.0.0.1").split(","))
        )
        self.api_port_list = list(
            map(int, reward_kwargs.get("model_api_port", "1222").split(","))
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
        messages = kwargs.get("messages", [])
        if_longcots = [m[0]["role"] == "system" for m in messages]
        #if_longcots = [m[0]["role"] == "system" for m in messages]
        questions = [m[1]["content"] if if_longcot else m[0]["content"] for m, if_longcot in zip(messages, if_longcots)]
        prompt_types = kwargs.get("prompt_type", ["instruct"] * len(completions))

        swift_reward_types = kwargs.get("swift_reward_type", ["model_base"] * len(completions))
        length = len(completions)
        tasks = []
        for cur_idx, (
            content,
            sol,
            question,
            swift_reward_type,
            prompt_type,
        ) in enumerate(
            zip(completions, solution_str, questions, swift_reward_types, prompt_types)):
            cur_address = self.api_address_list[
                (self.rank * length + cur_idx) % len(self.api_address_list)
            ]
            cur_port = self.api_port_list[
                (self.rank * length + cur_idx) % len(self.api_port_list)
            ]
            reward = asyncio.create_task(
                self.evaluate(
                    content, sol, question, cur_address, cur_port, swift_reward_type, prompt_type
                )
            )
            tasks.append(reward)
        rewards = await asyncio.gather(*tasks)
        # only need float reward
        #reward = rewards[0]
        return rewards


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
        # hardcode model_base
        swift_reward_type="model_base"
        reward = 0.0
        # if swift_reward_type != "model_base":
        #     return reward
        try:
            if prompt_type == "longcot":
                # longcot 格式错误直接返回 0
                pattern = r"^<think>.*?</think>\s*<answer>.*?</answer>(?![\s\S])"
                match = re.match(pattern, cur_completion, re.DOTALL | re.MULTILINE)
                if not match:
                    return reward
                think_match = re.search(
                    r"<think>(.*?)</think>", cur_completion, re.DOTALL | re.MULTILINE
                )
                answer_match = re.search(
                    r"<answer>(.*?)</answer>", cur_completion, re.DOTALL | re.MULTILINE
                )
                if not think_match or not answer_match:
                    return reward
                think_content = think_match.group(1).strip()
                answer_content = answer_match.group(1).strip()
            else:
                think_content = ""
                answer_content = cur_completion

            prompt = (
                prompt_template.replace("{Question}", question)
                .replace("{Ground_Truth}", cur_solution)
                .replace("{Model_Thought_Process}", think_content)
                .replace("{Model_Output}", answer_content)
            )
            messages = [{"role": "user", "content": prompt}]
            max_try = 30
            for i in range(max_try):
                try:
                    completion = await llm_openai_api(
                        messages, ip=cur_address, host=cur_port
                    )
                    completion = (
                        completion[0] if isinstance(completion, list) else completion
                    )
                    
                    # # print(f'lcy0318-completion-{prompt}-{completion}')
                    # json_str = completion[
                    #     completion.rfind("{") : completion.rfind("}") + 1
                    # ]
                    # json_str = json_str.replace("\\", "\\\\")
                    # completion = json.loads(json_str)
                    
                    pattern = r'\{\s*[^{}]*"Comments":.*?"\n\}'
                    json_strings = re.findall(pattern, completion, re.DOTALL)
                    json_str = json_strings[-1]
                    try:
                        completion = json.loads(json_str)
                    except:
                        completion = eval(json_str)

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
                    
                    correctness = completion["Correctness"]
                    thought_process = completion["Thought_Process"]
                    thought_Process_quality = completion["Thought_Process_Quality"] # "High/Medium/Low"
                    break
                except Exception as e:
                    print(f"xx-retry_{i}-{e}")
                    try:
                        print(completion)
                    except:
                        pass
                    continue
            if prompt_type == "longcot":
                if (
                    correctness.lower() == "yes"
                    and thought_process.lower() == "related"
                ):
                    if swift_reward_type == "model_base":
                        reward += 1.0
                elif thought_process.lower() == "unrelated":
                    reward -= 0.5
                if thought_Process_quality.lower() == "medium":
                    reward -= 0.3
                elif thought_Process_quality.lower() == "low":
                    reward -= 0.8
                # 最低 -1.0
                reward = max(-1.0, reward)
            else:
                if correctness.lower() == "yes":
                    if swift_reward_type == "model_base":
                        reward += 1.0
        except Exception as e:
            print(e)
        return reward


class MyMathAccuracy(object):
    def __init__(self, **reward_kwargs):
        pass

    def __call__(self, completion, solution_str, **kwargs) -> float:
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
                content_match = re.search(r"<answer>(.*?)</answer>", completion)
                student_answer = (
                    content_match.group(1).strip()
                    if content_match
                    else completion.strip()
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

class MyFormat(object):
    def __init__(self, **reward_kwargs):
        pass

    def __call__(self, completion, solution_str, **kwargs) -> float:
        """Reward function that checks if the completion has a specific format."""
        prompt_types = kwargs.get("prompt_type", ["instruct"] * len(completion))
        swift_reward_types = kwargs.get(
            "swift_reward_type", ["model_base"] * len(completion))
        pattern = r"^<think>.*?</think>\s*<answer>.*?</answer>(?![\s\S])"
        matches =  [(re.match(pattern, content, re.DOTALL | re.MULTILINE)
                  if prompt_type == "longcot"  # longcot
                  else not re.match(pattern, content, re.DOTALL | re.MULTILINE))
                  for content, prompt_type in zip(completion, prompt_types)]
        
        return [0.0 if match else -1.0 for match in matches]


class KeyeComputeReward(object):
    def __init__(self, **reward_kwargs):
        print(f'{reward_kwargs=}', flush=True)
        self.reward_fns = []
        reward_fn_types = reward_kwargs.get("reward_fn_types", "")
        assert reward_fn_types != ""
        reward_fn_types = reward_fn_types.split(',')
        for reward_fn_type in reward_fn_types:
            self.reward_fns.append(eval(reward_fn_type.strip())(**reward_kwargs))
        reward_sum_weights = reward_kwargs.get("reward_sum_weights",[1.0] * len(self.reward_fns))
        if isinstance(reward_sum_weights, str):
            reward_sum_weights = [float(x.strip()) for x in reward_sum_weights.split(',')]
        assert len(reward_sum_weights) == len(self.reward_fns)
        self.reward_sum_weights = reward_sum_weights
    
    def __call__(self, solution_str, ground_truth, **kwargs):
        #rewards = np.array([fn(solution_str, ground_truth, **kwargs) for fn in self.reward_fns])
        rewards = list(zip(*[fn(solution_str, ground_truth, **kwargs) for fn in self.reward_fns]))
        print(f"{rewards=}")
        return [np.dot(reward, np.array(self.reward_sum_weights)) for reward in rewards]


if __name__ == "__main__":
    kwargs={'swift_reward_type': ['model_base', 'model_base'],
            'prompt_type': ['longcot', 'longcot'], 
            'messages': [np.array([{'content': '\nQuestion:\n<image [1]> Question: Add 6 matte cylinders. How many matte cylinders are left?/think', 'role': 'user'}], dtype=object),
                         np.array([{'content': '\nQuestion:\nWithin quadrilateral ABCD, with midpoints E and F on sides AB and AD respectively, and EF = 6, BC = 13, and CD = 5, what is the area of triangle DBC?\nChoices:\nA: 60\nB: 30\nC: 48\nD: 65', 'role': 'user'}], dtype=object)]}
    reward_cls = KeyeComputeReward(
        reward_fn_types="ModelBaseAccuracy,MyFormat",
        model_api_address="10.82.121.34,10.82.122.98,10.82.120.218",
        model_api_port="8000")
    reward = reward_cls(["<think>The problem that users need to solve now is calculating the number of original matte cylinders after adding 6 more. First, let's look at the matte cylinders in the scene: there are two in the original image, one gray large cylinder and one red small cylinder. Then, add 6 more, so the total is 2 + 6 = 8. It is necessary to confirm whether the count of original matte cylinders is correct. Look at each object:\nGray large cylinder: matte\nRed small cylinder: matte\nSo originally there were 2, then add 6, resulting in a total of 8.</think><answer>\boxed{8}</answer>",
                         "<think>this my thinking result.</think><answer>B</answer>"], ["$8$", "$B$"], **kwargs)
    print(reward)
