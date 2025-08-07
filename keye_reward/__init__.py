import os
import re
from typing import Dict, List, Union
import random
import time
import json
import numpy as np
import inspect
import threading
import traceback

import asyncio
import httpx
from openai import OpenAI

from latex2sympy2_extended import NormalizationConfig
import nest_asyncio
#from math_verify import LatexExtractionConfig, parse, verify


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

prompt_template_nothink = """You are a expert assistant. I have a question, and a model will respond based on the question and the image provided (you can not see image here). 
The model will output its answer. I need your help to evaluate the correctness of the model's output.
Though you can not see the provided image, I will provide the ground truth for your reference. 
#### Question: \n{Question}\n\n
#### Ground Truth: \n{Ground_Truth}\n\n
#### Model Output: \n{Model_Output}\n\n
Please provide your evaluation based on the following criteria:
1. Is the model's output correct based on the ground truth? If the model's output is correct, please mark it as "Yes". Otherwise, please mark it as "No".
2. Before you provide your evaluation, please give a brief comment to help us understand your evaluation better.
3. Output your evaluation in the json format: 
```
{
    "Comments": "Your brief comment.",
    "Correctness": "Yes/No"
}
```"""

prompt_thinkmode_think_system = "You are a helpful assistant. I will give you a question(###Question), an answer(###Answer), and the reasoning process(###Reasoning Process) for the question. You need to judge whether the reasoning process has enough information to arrive at the correct answer based on the question and the answer. Please output the score as 0 or 1, where 0 means the reasoning process does not have enough information, and 1 means the reasoning process has enough information. Please output the final score directly."
prompt_thinkmode_think_template = """###Question\n{{Question}}\n
###Answer\n{{Answer}}\n
###Reasoning Process\n{{Reasoning Process}}\n
Please directly judge whether the reasoning process of the question has enough information to arrive at the correct answer. Answer with numbers only. Limit to 5 words."""

prompt_thinkmode_answer_system = "You are a helpful assistant. I will provide you with a question(###Question), an answer(###Answer), and the model's predicted result(###Predict Result). You need to determine whether the model's prediction is correct based on the answer. Please output the score as 1 or -1, where 1 means the model's prediction is correct, and -1 means the model's prediction is incorrect. Please directly output the final score."
prompt_thinkmode_answer_template = """###Question\n{{Question}}\n
###Answer\n{{Answer}}\n
###Predict Result\n{{Predict Result}}\n
Please directly determine whether the model's prediction is correct and output the score. Answer with numbers only. Limit to 5 words."""

prompt_nothinkmode_answer_system = "You are a helpful assistant. I will provide you with a question(###Question), an answer(###Answer), and the model's predicted result(###Predict Result). You need to determine whether the model's prediction is correct based on the answer. Please output the score as 1 or -1, where 1 means the model's prediction is correct, and -1 means the model's prediction is incorrect. Please directly output the final score."
prompt_nothinkmode_answer_template = """###Question\n{{Question}}\n
###Answer\n{{Answer}}\n
###Predict Result\n{{Predict Result}}\n
Please directly determine whether the model's prediction is correct and output the score. Answer with numbers only. Limit to 5 words."""


class Singleton(object):
    _instances = {}

    def __new__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__new__(cls, *args, **kwargs)
        return cls._instances[cls]


NUM_CLIENTS = 20
NUM_MAX_CONNENCTIONS_PER_CLIENT = 20
NUM_MAX_KEEPALIVE_CONNECTIONS_PER_CLIENT = 5

DUMMY_RESPONSE = False


client = httpx.AsyncClient(timeout=httpx.Timeout(600.0), limits=httpx.Limits(
    max_connections=NUM_MAX_CONNENCTIONS_PER_CLIENT, 
    max_keepalive_connections=NUM_MAX_KEEPALIVE_CONNECTIONS_PER_CLIENT))

clients = [httpx.AsyncClient(timeout=httpx.Timeout(600.0), limits=httpx.Limits(
    max_connections=NUM_MAX_CONNENCTIONS_PER_CLIENT, 
    max_keepalive_connections=NUM_MAX_CONNENCTIONS_PER_CLIENT))
    for _ in range(NUM_CLIENTS)]


async def llm_openai_api_multiple_clients(
    messages,
    ip="127.0.0.1",
    host="1222",
    temperature=0.7,
    max_tokens=2048,
    top_p=0.8,
    openai_api_key="EMPTY",
    n=1,  # from lxy
    idx=-1,
):
    my_client = clients[idx%NUM_CLIENTS]
    openai_api_base = f"http://{ip}:{host}/v1"
    model_list = await my_client.get(
            f"{openai_api_base}/models",
            headers={"Authorization": f"Bearer {openai_api_key}"},
        )
    model_list.raise_for_status()
    models = model_list.json()
    model = models["data"][0]["id"]
    
    resp = await my_client.post(
            f"{openai_api_base}/chat/completions",
            headers={"Authorization": f"Bearer {openai_api_key}"},
            json={
                "model": model,
                "messages": messages,
                "temperature": temperature,
                #"max_tokens": max_tokens,
                #"top_p": top_p,
                #"n": n,
            },
        )
    resp.raise_for_status()
    response_data = resp.json()
    if n == 1:
        return response_data["choices"][0]["message"]["content"]
    else:
        return [choice["message"]["content"] for choice in response_data["choices"]]



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
    model_list = await client.get(
            f"{openai_api_base}/models",
            headers={"Authorization": f"Bearer {openai_api_key}"},
        )
    model_list.raise_for_status()
    models = model_list.json()
    model = models["data"][0]["id"]
    
    resp = await client.post(
            f"{openai_api_base}/chat/completions",
            headers={"Authorization": f"Bearer {openai_api_key}"},
            json={
                "model": model,
                "messages": messages,
                "temperature": temperature,
                #"max_tokens": max_tokens,
                #"top_p": top_p,
                #"n": n,
            },
        )
    resp.raise_for_status()
    response_data = resp.json()
    if n == 1:
        return response_data["choices"][0]["message"]["content"]
    else:
        return [choice["message"]["content"] for choice in response_data["choices"]]

def get_boxed(response, bb="\\boxed{"):
    resp = response.split(bb)[-1]
    lt = len(resp)
    counter, end = 1, None
    for i in range(lt):
        if resp[i] == "{":
            counter += 1
        elif resp[i] == "}":
            counter -= 1
        if counter == 0:
            end = i
            break
        elif i == lt - 1:
            end = lt
            break
    if end is not None:
        response = resp[:end]
    return response


def post_process(prediction):
    import re

    if "</analysis>" in prediction:
        prediction = prediction.split("</analysis>")[-1]
    if "</think>" in prediction:
        prediction = prediction.split("</think>")[-1].lstrip("\n").strip()
    if "<answer>" not in prediction:
        boxed_matches = get_boxed(prediction, bb=r"\boxed{")
        if len(boxed_matches) != len(prediction):
            return boxed_matches
        else:
            boxed_matches = get_boxed(prediction, bb="\boxed{")
            return (
                boxed_matches if len(boxed_matches) != len(prediction) else prediction
            )
        return prediction
    matches = re.findall(r"<answer>(.*?)</answer>", prediction, re.DOTALL)
    if matches:
        content_match = matches[-1]
        boxed_matches = get_boxed(content_match, bb=r"\boxed{")
        if len(boxed_matches) != len(content_match):
            return boxed_matches
        else:
            boxed_matches = get_boxed(content_match, bb="\boxed{")
            return (
                boxed_matches
                if len(boxed_matches) != len(content_match)
                else content_match
            )
    else:
        return prediction


class ModelBaseAccuracyV2(object):
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
        self.reward_fn_loop = None
        self.reward_fn_ready = threading.Event()
        self.reward_fn_thread = threading.Thread(target=self._init_reward_fn_loop, daemon=True)
        self.reward_fn_thread.start()
        self.reward_fn_ready.wait()

    def _init_reward_fn_loop(self):
        self.reward_fn_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.reward_fn_loop)
        self.reward_fn_ready.set()
        self.reward_fn_loop.run_forever()

    def __call__(self, completions, solution_str, **kwargs) -> List[float]:
        """
        Reward function that checks if the completion is correct.
        Args:
            completions (list[str]): Generated outputs
            solution (list[str]): Ground Truths.

        Returns:
            list[float]: Reward scores
        """
        print("*" * 50, "at keye __init__.py reward_fn call", len(completions), "*" * 50)
        print(f"[DEBUG] at keye reward init ModelBaseAccuracyV2 __call__, {solution_str=}")
        # cur_task = self.loop.create_task(
        #     self.async_call(completions, solution_str, **kwargs)
        # )
        # result = self.robust_run_until_complete(cur_task)
        assert self.reward_fn_loop is not None, "reward_fn_loop is not initialized."
        future = asyncio.run_coroutine_threadsafe(
            self.async_call(completions, solution_str, **kwargs),
            self.reward_fn_loop,
        )
        result = future.result()
        # print(f"[DEBUG] at keye __init__.py reward_fn call get result")
        return result

    async def async_call(self, completions, solution_str, **kwargs) -> float:
        # NOTE(huangjiaming): completions  batch_size = 1
        messages = kwargs.get("messages", [])
        # print(f"[DEBUG] at async_call, {messages=}")
        if_longcots = [m[0]["role"] == "system" for m in messages]
        #if_longcots = [m[0]["role"] == "system" for m in messages]
        questions = [m[1]["content"] if if_longcot else m[0]["content"] for m, if_longcot in zip(messages, if_longcots)]
        prompt_types = kwargs.get("prompt_type", ["instruct"] * len(completions))

        swift_reward_types = kwargs.get("swift_reward_type", ["model_base"] * len(completions))
        length = len(completions)
        tasks = []
        #print(f"keye reward io request number: {length}")
        for cur_idx, (
            content,
            sol,
            question,
            swift_reward_type,
            prompt_type,
        ) in enumerate(
            zip(completions, solution_str, questions, swift_reward_types, prompt_types)):
            #cur_address = self.api_address_list[
            #    (self.rank * length + cur_idx) % len(self.api_address_list)
            #]
            #cur_port = self.api_port_list[
            #    (self.rank * length + cur_idx) % len(self.api_port_list)
            #]
            cur_address = random.choice(self.api_address_list)
            cur_port = random.choice(self.api_port_list)
            reward = asyncio.create_task(
                self.evaluate(
                    content, sol, question, cur_address, cur_port, swift_reward_type, prompt_type, cur_idx
                )
            )
            tasks.append(reward)
        for_end = time.perf_counter()
        rewards = await asyncio.gather(*tasks)
        result_end = time.perf_counter()
        ModelBaseAccuracy_Call_Asyncio_gather_time = (result_end - for_end) * 1000
        print(f"ModelBaseAccuracy_Call_Asyncio_gather_time: {ModelBaseAccuracy_Call_Asyncio_gather_time}")  
        # only need float reward
        #reward = rewards[0]
        return rewards

    async def get_score(self, system_prompt, prompt, cur_idx, retry=50):
        cur_address = random.choice(self.api_address_list)
        cur_port = random.choice(self.api_port_list)
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}]
        for i in range(retry):
            try:
                completion = await llm_openai_api_multiple_clients(
                    messages, ip=cur_address, host=cur_port, temperature=0.3, idx=cur_idx
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
                            "system_prompt": system_prompt,
                            "prompt": prompt,
                            "messages": messages,
                            "completion": completion,
                        }
                        f.write(json.dumps(dump_data, ensure_ascii=False) + "\n")
                        f.flush()
                except:
                    pass
                score = float(completion.split("\n")[0].split(" ")[0])
                assert score in [-1, 0, 1], f"Invalid score: {score}. Must be one of -1, 0, or 1."

                return score
            except Exception as e:
                print(f"modelreward-retry_{i}-{e} {cur_adress=} {cur_port}")
                try:
                    pass
                except:
                    pass
                continue

        return -1.0


    async def evaluate(
        self,
        cur_completion,
        cur_solution,
        question,
        cur_address,
        cur_port,
        swift_reward_type,
        prompt_type,
        cur_idx,
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
        swift_reward_type = "model_base"
        reward = 0.0
        # if swift_reward_type != "model_base":
        #     return reward
        try:
            # format judge
            if prompt_type == "longcot":
                # longcot 格式错误直接返回 0
                pattern = r"^<think>.*?</think>\s*<answer>.*?</answer>(?![\s\S])"
                match = re.match(pattern, cur_completion, re.DOTALL | re.MULTILINE)
                if not match:
                    reward = -1.0

                    return reward
                think_match = re.search(
                    r"<think>(.*?)</think>", cur_completion, re.DOTALL | re.MULTILINE
                )
                answer_match = re.search(
                    r"<answer>(.*?)</answer>", cur_completion, re.DOTALL | re.MULTILINE
                )
                if not think_match or not answer_match:
                    reward = -1.0

                    return reward
                think_content = think_match.group(1).strip()
                answer_content = answer_match.group(1).strip()
            elif prompt_type == "auto_think":
                pattern_auto_think = r"^<analysis>.*?</analysis>\s*<think>.*?</think>\s*<answer>.*?</answer>(?![\s\S])"
                pattern_auto_nothink = r"^<analysis>.*?</analysis>"
                pattern_think_token = r"<think>|</think>|<answer>|</answer>"

                match_auto = (
                    # think
                    re.match(
                        pattern_auto_think, cur_completion, re.DOTALL | re.MULTILINE
                    )
                    # no think
                    or (
                        re.match(
                            pattern_auto_nothink,
                            cur_completion,
                            re.DOTALL | re.MULTILINE,
                        )
                        and not re.search(pattern_think_token, cur_completion)
                    )
                )

                if not match_auto:
                    reward = -1.0

                    return reward

                match = re.match(
                    pattern_auto_think, cur_completion, re.DOTALL | re.MULTILINE
                )
                # auto no think
                if not match:
                    think_content = ""
                    answer_content = cur_completion
                # auto think
                else:
                    think_match = re.search(
                        r"<think>(.*?)</think>",
                        cur_completion,
                        re.DOTALL | re.MULTILINE,
                    )
                    answer_match = re.search(
                        r"<answer>(.*?)</answer>",
                        cur_completion,
                        re.DOTALL | re.MULTILINE,
                    )
                    if not think_match or not answer_match:
                        reward = -1.0

                        return reward
                    think_content = think_match.group(1).strip()
                    answer_content = answer_match.group(1).strip()
            else:
                think_content = ""
                answer_content = cur_completion

            answer_content = post_process(answer_content)

            if prompt_type == "longcot":
                prompt_answer = (
                    prompt_thinkmode_answer_template.replace("{{Question}}", question)
                    .replace("{{Answer}}", cur_solution)
                    .replace("{{Predict Result}}", answer_content)
                )
                score_answer = await self.get_score(prompt_thinkmode_answer_system, prompt_answer, cur_idx)
                
                prompt_think = (
                    prompt_thinkmode_think_template.replace("{{Question}}", question)
                    .replace("{{Answer}}", cur_solution)
                    .replace("{{Reasoning Process}}", think_content)
                )
                score_think = max(0., await self.get_score(prompt_thinkmode_think_system, prompt_think, cur_idx))
                
                reward += score_answer + score_think
            elif prompt_type == "auto_think":
                prompt_answer = (
                    prompt_thinkmode_answer_template.replace("{{Question}}", question)
                    .replace("{{Answer}}", cur_solution)
                    .replace("{{Predict Result}}", answer_content)
                )
                score_answer = await self.get_score(prompt_thinkmode_answer_system, prompt_answer, cur_idx)
                reward += score_answer

                if think_content != "":
                    # autothink - think
                    if score_answer == 1:
                        prompt_think = (
                            prompt_thinkmode_think_template.replace("{{Question}}", question)
                            .replace("{{Answer}}", cur_solution)
                            .replace("{{Reasoning Process}}", think_content)
                        )
                        score_think = max(0., await self.get_score(prompt_thinkmode_think_system, prompt_think, cur_idx))

                        reward += score_think
                else:
                    # autothink - no think
                    if score_answer == 1:
                        reward += 1
            else:
                prompt_answer = (
                    prompt_thinkmode_answer_template.replace("{{Question}}", question)
                    .replace("{{Answer}}", cur_solution)
                    .replace("{{Predict Result}}", answer_content)
                )
                score_answer = await self.get_score(prompt_thinkmode_answer_system, prompt_answer, cur_idx)
                reward += score_answer * 2

        except Exception as e:
            print(e)

        return reward


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
        start = time.perf_counter()
        cur_task = self.loop.create_task(
            self.async_call(completions, solution_str, **kwargs)
        )
        result = self.loop.run_until_complete(cur_task)
        sync_time = (time.perf_counter() - start) * 1000

        print(f"ModelBaseAccuracy Sync task time: {sync_time}")
        return result

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
        print(f"keye reward io request number: {length}")
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
                    content, sol, question, cur_address, cur_port, swift_reward_type, prompt_type, cur_idx
                )
            )
            tasks.append(reward)
        for_end = time.perf_counter()
        rewards = await asyncio.gather(*tasks)
        result_end = time.perf_counter()
        ModelBaseAccuracy_Call_Asyncio_gather_time = (result_end - for_end) * 1000
        print(f"ModelBaseAccuracy_Call_Asyncio_gather_time: {ModelBaseAccuracy_Call_Asyncio_gather_time}")  
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
        cur_idx,
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

        if DUMMY_RESPONSE:
            delay = 80.0
            await asyncio.sleep(delay)
        else:
            if swift_reward_type != "model_base":
                return reward
            try:
                # setting_start = time.perf_counter()
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
                max_try = 3
                
                # tryloop_start = time.perf_counter()
                # api_time = 0
                for i in range(max_try):
                    try:
                        completion = await llm_openai_api_multiple_clients(
                            messages, ip=cur_address, host=cur_port, idx=cur_idx
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
                            print(f"enter exception msg area")
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
        pattern_auto_think = r"^<analysis>.*?</analysis>\s*<think>.*?</think>\s*<answer>.*?</answer>(?![\s\S])"
        pattern_auto_nothink = r"^<analysis>.*?</analysis>"
        pattern_think_token = r"<think>|</think>|<answer>|</answer>"

        matches = []
        for content, prompt_type in zip(completion, prompt_types):
            if prompt_type == "longcot":  # longcot
                matches.append(re.match(pattern, content, re.DOTALL | re.MULTILINE))
            elif prompt_type == "auto_think":
                matches.append(
                    # think
                    re.match(pattern_auto_think, content, re.DOTALL | re.MULTILINE)
                    # no think
                    or (
                        re.match(
                            pattern_auto_nothink, content, re.DOTALL | re.MULTILINE
                        )
                        and not re.search(pattern_think_token, content)
                    )
                )
            else:
                matches.append(
                    not re.match(pattern, content, re.DOTALL | re.MULTILINE)
                    and not re.search(pattern_think_token, content)
                )

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
        rewards = [fn(solution_str, ground_truth, **kwargs) for fn in self.reward_fns]
        rewards0 = rewards[0]
        rewards1 = rewards[1]
        rewards = zip(*rewards)
        rewards = list(rewards)
        res = [np.dot(reward, np.array(self.reward_sum_weights)) for reward in rewards]
        if isinstance(rewards0, list):
            res0 = [x * self.reward_sum_weights[0] for x in rewards0]
        else:
            res0 = rewards0 * self.reward_sum_weights[0]
        if isinstance(rewards1, list):
            res1 = [x * self.reward_sum_weights[1] for x in rewards1]
        else:
            res1 = rewards0 * self.reward_sum_weights[1]

        return res, res0, res1


if __name__ == "__main__":
    kwargs={'swift_reward_type': ['model_base', 'model_base'],
            'prompt_type': ['longcot', 'longcot'], 
            'messages': [np.array([{'content': '\nQuestion:\n<image [1]> Question: Add 6 matte cylinders. How many matte cylinders are left?/think', 'role': 'user'}], dtype=object),
                         np.array([{'content': '\nQuestion:\nWithin quadrilateral ABCD, with midpoints E and F on sides AB and AD respectively, and EF = 6, BC = 13, and CD = 5, what is the area of triangle DBC?\nChoices:\nA: 60\nB: 30\nC: 48\nD: 65', 'role': 'user'}], dtype=object)]}
    
    reward_cls = KeyeComputeReward(
        reward_fn_types="ModelBaseAccuracyV2,MyFormat",
    # model_api_address="10.82.121.34,10.82.122.98,10.82.120.218",
        model_api_address="10.48.47.84,10.48.47.83",
    # model_api_port="8000")
        model_api_port="8000,8001,8002,8003")

    
    # message = [{'role': 'user', 'content': '\nYou are a expert assistant. I have a question, and a model will respond based on the question and the image provided (you can not see image here). \nThe model will first output its thought process, followed by the final answer. I need your help to evaluate the correctness of the model\'s output and its thought process.\nThough you can not see the provided image, I will provide the ground truth for your reference. \n\n#### Question: \n\nQuestion:\nWithin quadrilateral ABCD, with midpoints E and F on sides AB and AD respectively, and EF = 6, BC = 13, and CD = 5, what is the area of triangle DBC?\nChoices:\nA: 60\nB: 30\nC: 48\nD: 65\n\n\n#### Ground Truth: \n$B$\n\n\n#### Model Thought Process: \nthis my thinking result.\n\n\n#### Model Output: \nB\n\n\n\nPlease provide your evaluation based on the following criteria:\n1. Is the thought process related to the question? If the model\'s thought process contains very irrelevant information, please mark it as "Unrelated".\n2. Is the model\'s output correct based on the ground truth? If the model\'s output is correct, please mark it as "Yes". Otherwise, please mark it as "No".\n3. Evaluate the quality of the thought process. If the thought process is well-structured, logical, and comprehensive, mark it as "High". If it is somewhat logical but lacks details or has minor flaws, mark it as "Medium". If it is poorly structured or contains significant errors, mark it as "Low".\n4. You fisrt need to read the question and the ground truth, then read the model\'s thought process and output.\n5. Before you provide your evaluation, please give a brief comment to help us understand your evaluation better.\n6. Output your evaluation in the json format: \n```\n{\n    "Comments": "Your brief comment.",\n    "Correctness": "Yes/No",\n    "Thought_Process": "Related/Unrelated",\n    "Thought_Process_Quality": "High/Medium/Low"\n}\n```\n'}]
    
    reward = reward_cls(["<think>The problem that users need to solve now is calculating the number of original matte cylinders after adding 6 more. First, let's look at the matte cylinders in the scene: there are two in the original image, one gray large cylinder and one red small cylinder. Then, add 6 more, so the total is 2 + 6 = 8. It is necessary to confirm whether the count of original matte cylinders is correct. Look at each object:\nGray large cylinder: matte\nRed small cylinder: matte\nSo originally there were 2, then add 6, resulting in a total of 8.</think><answer>\boxed{8}</answer>",
                         "<think>this my thinking result.</think><answer>B</answer>"], ["$8$", "$B$"], **kwargs)
    print(reward)
