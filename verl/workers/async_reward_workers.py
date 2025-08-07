import logging
import os
import pickle
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from abc import ABC, abstractmethod
import ray
import zmq
import numpy as np
from omegaconf import DictConfig
from verl.protocol import DataProto


class AsyncRewardWorkerBase(ABC):
    """Base class for AsyncServer."""

    def __init__(self):
        pass

    @abstractmethod
    def init_worker(self):
        """Init async reward worker"""
        raise NotImplementedError



@ray.remote(num_cpus=1)
class AsyncRewardWorker(AsyncRewardWorkerBase):
    def __init__(self, config: DictConfig, reward_fn, val_reward_fn):
        """
        Initialize reward workers.

        Args:
            config: Configuration object containing training parameters.
            reward_fn: Function for computing rewards during training.
            val_reward_fn: Function for computing rewards during validation.
        """
        self.config = config
        self.reward_fn = reward_fn()
        self.val_reward_fn = val_reward_fn()
   
    def init_worker(self):
        pass

    def compute_reward(self, data):
        res_dict = self.reward_fn(data, True)
        print(f"[DEBUG] at async reward workers, {res_dict.keys()=}")
        res = DataProto.from_dict({'reward_tensor': res_dict['reward_tensor'], 'fn1_score': np.array(res_dict['fn1_score']), 'fn2_score': np.array(res_dict['fn2_score'])},
                                      non_tensors=res_dict['reward_extra_info'])
        return res