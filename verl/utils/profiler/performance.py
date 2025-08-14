# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import datetime
import inspect
import logging
import subprocess
import hashlib
import xml.etree.ElementTree as ET

from contextlib import contextmanager
from typing import Any, Optional

import torch
import torch.distributed as dist
from codetiming import Timer

from verl.utils.device import get_device_id, get_torch_device
from verl.utils.logger import DecoratorLoggerBase


def _get_current_mem_info(unit: str = "GB", precision: int = 2) -> tuple[str]:
    """Get current memory usage.

    Note that CPU device memory info is always 0.

    Args:
        unit (str, optional): The unit of memory measurement. Defaults to "GB".
        precision (int, optional): The number of decimal places to round memory values. Defaults to 2.

    Returns:
        tuple[str]: A tuple containing memory allocated, memory reserved, memory used, and memory total
        in the specified unit.
    """
    assert unit in ["GB", "MB", "KB"]
    device = get_torch_device()
    # torch.cpu.memory_allocated() does not exist
    if device == torch.cpu:
        return "0.00", "0.00", "0.00", "0.00"

    divisor = 1024**3 if unit == "GB" else 1024**2 if unit == "MB" else 1024
    mem_allocated = get_torch_device().memory_allocated()
    mem_reserved = get_torch_device().memory_reserved()
    # use get_torch_device().mem_get_info to profile device memory
    # since vllm's sleep mode works below pytorch
    # see https://github.com/vllm-project/vllm/pull/11743#issuecomment-2754338119
    mem_free, mem_total = get_torch_device().mem_get_info()
    mem_used = mem_total - mem_free
    mem_allocated = f"{mem_allocated / divisor:.{precision}f}"
    mem_reserved = f"{mem_reserved / divisor:.{precision}f}"
    mem_used = f"{mem_used / divisor:.{precision}f}"
    mem_total = f"{mem_total / divisor:.{precision}f}"
    return mem_allocated, mem_reserved, mem_used, mem_total


def log_gpu_memory_usage(head: str, logger: logging.Logger = None, level=logging.DEBUG, rank: int = 0):
    """Log GPU memory usage information.

    Args:
        head (str): A descriptive header for the memory usage log message.
        logger (logging.Logger, optional): Logger instance to use for logging. If None, prints to stdout.
        level: Logging level to use. Defaults to logging.DEBUG.
        rank (int): The rank of the process to log memory for. Defaults to 0.
    """
    if (not dist.is_initialized()) or (rank is None) or (dist.get_rank() == rank):
        mem_allocated, mem_reserved, mem_used, mem_total = _get_current_mem_info()
        message = (
            f"{head}, memory allocated (GB): {mem_allocated}, memory reserved (GB): {mem_reserved}, "
            f"device memory used/total (GB): {mem_used}/{mem_total}"
        )

        if logger is None:
            print(message)
        else:
            logger.log(msg=message, level=level)


class GPUMemoryLogger(DecoratorLoggerBase):
    """A decorator class to log GPU memory usage.

    Example:
        >>> from verl.utils.profiler.performance import GPUMemoryLogger
        >>> @GPUMemoryLogger(role="actor")
        >>> def update_actor(self, batch):
        ...     # real actor update logics
        ...     return
    """

    def __init__(self, role: str, logger: logging.Logger = None, level=logging.DEBUG, log_only_rank_0: bool = True):
        if dist.is_initialized() and dist.get_world_size() > 1:
            rank = dist.get_rank()
        else:
            rank = 0
        super().__init__(role, logger, level, rank, log_only_rank_0)

    def __call__(self, decorated_function: callable):
        def f(*args, **kwargs):
            return self.log(decorated_function, *args, **kwargs)

        return f

    def log(self, func, *args, **kwargs):
        name = func.__name__
        mem_allocated, mem_reserved, mem_used, mem_total = _get_current_mem_info()
        message = (
            f"Before {name}, memory allocated (GB): {mem_allocated}, memory reserved (GB): {mem_reserved}, "
            f"device memory used/total (GB): {mem_used}/{mem_total}"
        )
        self.logging_function(message)

        output = func(*args, **kwargs)

        mem_allocated, mem_reserved, mem_used, mem_total = _get_current_mem_info()
        message = (
            f"After {name}, memory allocated (GB): {mem_allocated}, memory reserved (GB): {mem_reserved}, "
            f"device memory used/total (GB): {mem_used}/{mem_total}"
        )

        self.logging_function(message)
        return output


def log_print(ctn: Any):
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    frame = inspect.currentframe().f_back
    function_name = frame.f_code.co_name
    line_number = frame.f_lineno
    file_name = frame.f_code.co_filename.split("/")[-1]
    print(f"[{current_time}-{file_name}:{line_number}:{function_name}]: {ctn}")


def _timer(name: str, timing_raw: dict[str, float]):
    """Inner function that handles the core timing logic.

    Args:
        name (str): The name/identifier for this timing measurement.
        timing_raw (Dict[str, float]): Dictionary to store timing information.
    """
    with Timer(name=name, logger=None) as timer:
        yield
    if name not in timing_raw:
        timing_raw[name] = 0
    timing_raw[name] += timer.last


@contextmanager
def simple_timer(name: str, timing_raw: dict[str, float]):
    """Context manager for basic timing without NVTX markers.

    This utility function measures the execution time of code within its context
    and accumulates the timing information in the provided dictionary.

    Args:
        name (str): The name/identifier for this timing measurement.
        timing_raw (Dict[str, float]): Dictionary to store timing information.

    Yields:
        None: This is a context manager that yields control back to the code block.
    """
    yield from _timer(name, timing_raw)


@contextmanager
def marked_timer(
    name: str,
    timing_raw: dict[str, float],
    color: str = None,
    domain: Optional[str] = None,
    category: Optional[str] = None,
):
    """Context manager for timing with platform markers.

    This utility function measures the execution time of code within its context,
    accumulates the timing information, and adds platform markers for profiling.
    This function is a default implementation when hardware profiler is not available.

    Args:
        name (str): The name/identifier for this timing measurement.
        timing_raw (Dict[str, float]): Dictionary to store timing information.
        color (Optional[str]): Color for the marker. Defaults to None.
        domain (Optional[str]): Domain for the marker. Defaults to None.
        category (Optional[str]): Category for the marker. Defaults to None.

    Yields:
        None: This is a context manager that yields control back to the code block.
    """
    yield from _timer(name, timing_raw)


def reduce_timing(
    timing_raw: dict[str, float], reduce_op: torch.distributed.ReduceOp = torch.distributed.ReduceOp.AVG
) -> dict[str, float]:
    """Reduce timing information across all processes.

    This function uses distributed communication to gather and sum the timing
    information from all processes in a distributed environment.

    Args:
        timing_raw (Dict[str, float]): Dictionary containing timing information.

    Returns:
        Dict[str, float]: Reduced timing information.
    """
    if not dist.is_initialized():
        return timing_raw

    key_list, timing_list = [], []
    for key in sorted(timing_raw.keys()):
        key_list.append(key)
        timing_list.append(timing_raw[key])
    timing_list = torch.tensor(timing_list, dtype=torch.float32, device=get_device_id())
    torch.distributed.all_reduce(timing_list, op=reduce_op)
    timing_list = [tensor.item() for tensor in timing_list.to("cpu")]
    timing_generate = {key_list[i]: timing_list[i] for i in range(len(key_list))}
    return timing_generate


<<<<<<< HEAD
def get_most_used_gpu_memory():
    """
    通过执行 nvidia-smi -q -x 命令获取单机多卡环境中显存使用最多的GPU卡信息。
    只输出总显存、空余显存和已用显存。
    """
    try:
        # 执行 nvidia-smi -q -x 命令，获取完整的XML格式输出
        command = "nvidia-smi -q -x"
        result = subprocess.run(command, shell=True, capture_output=True, text=True, check=True)
        xml_output = result.stdout

        # 使用 ElementTree 解析 XML
        root = ET.fromstring(xml_output)

        max_used_memory_mb = -1
        most_used_gpu_info = None

        # 遍历所有 <gpu> 元素，并使用 enumerate 获取其在列表中的索引
        for i, gpu_elem in enumerate(root.findall('gpu')):
            gpu_id = i # 使用循环变量 i 作为 GPU 的逻辑索引

            # 获取显存信息
            fb_memory_usage = gpu_elem.find('fb_memory_usage')
            if fb_memory_usage is not None:
                total_memory_text = fb_memory_usage.find('total').text if fb_memory_usage.find('total') is not None else '0 MiB'
                used_memory_text = fb_memory_usage.find('used').text if fb_memory_usage.find('used') is not None else '0 MiB'
                free_memory_text = fb_memory_usage.find('free').text if fb_memory_usage.find('free') is not None else '0 MiB'
                
                # 移除单位 " MiB" 并转换为整数
                try:
                    total_memory_mb = int(total_memory_text.replace(' MiB', ''))
                    used_memory_mb = int(used_memory_text.replace(' MiB', ''))
                    free_memory_mb = int(free_memory_text.replace(' MiB', ''))
                except ValueError as ve:
                    print(f"警告: GPU {gpu_id} 显存值 '{total_memory_text}, {used_memory_text}, {free_memory_text}' 无法转换为整数: {ve}。跳过此GPU。")
                    continue
            else:
                print(f"警告: GPU {gpu_id} 未找到显存信息，跳过。")
                continue # 如果没有显存信息，跳过此GPU

            # 获取GPU名称 (保留名称以便识别是哪张卡)
            gpu_name_elem = gpu_elem.find('product_name')
            gpu_name = gpu_name_elem.text if gpu_name_elem is not None else "Unknown GPU"

            # 仅比较已用显存，找出使用最多的GPU
            if used_memory_mb > max_used_memory_mb:
                max_used_memory_mb = used_memory_mb
                most_used_gpu_info = {
                    'index': gpu_id,
                    'name': gpu_name,
                    'total_memory_mb': total_memory_mb,
                    'free_memory_mb': free_memory_mb,
                    'used_memory_mb': used_memory_mb,
                }
        
        return most_used_gpu_info

    except FileNotFoundError:
        print("错误: nvidia-smi 命令未找到。请确保NVIDIA驱动已正确安装并配置了PATH。")
        return None
    except subprocess.CalledProcessError as e:
        print(f"执行 nvidia-smi 命令时发生错误: {e}")
        print(f"标准输出: {e.stdout}")
        print(f"标准错误: {e.stderr}")
        return None
    except ET.ParseError as e:
        print(f"解析 nvidia-smi XML 输出时发生错误: {e}")
        print(f"原始XML输出:\n{xml_output}") # 打印原始XML以便调试
        return None
    except Exception as e:
        print(f"获取或解析GPU信息时发生未知错误: {e}")
        return None



def calculate_string_md5(text_string):
    """
    计算给定字符串的 MD5 哈希值。
    """
    # 将字符串编码为字节序列（通常使用 UTF-8）
    encoded_string = text_string.encode('utf-8')

    # 创建 MD5 哈希对象
    md5_hasher = hashlib.md5()

    # 更新哈希对象，传入字节序列
    md5_hasher.update(encoded_string)

    # 获取十六进制表示的 MD5 值
    md5_hex = md5_hasher.hexdigest()

    return md5_hex
=======
def topk_reduce_ratio_min_max(timing: float, k: int = 10) -> tuple[float, float, float]:
    """Calculate topk items take-up ratio, and min/max timing across all ranks."""
    if not dist.is_initialized():
        return -1.0, -1.0, -1.0

    world_size = dist.get_world_size()
    timing_tensor = torch.tensor(timing, dtype=torch.float32, device=get_device_id())
    tensor_list = [torch.zeros(1, dtype=torch.float32, device=get_device_id()) for _ in range(world_size)]
    torch.distributed.all_gather(tensor_list, timing_tensor)
    tensor_stack = torch.stack(tensor_list)
    timing_min = tensor_stack.min().cpu().item()
    timing_max = tensor_stack.max().cpu().item()
    top_k_percentile = torch.quantile(tensor_stack, 1 - k / 100)
    tail_ratio = torch.mean((tensor_stack > top_k_percentile).float()).cpu().item()
    return tail_ratio, timing_min, timing_max
>>>>>>> main
