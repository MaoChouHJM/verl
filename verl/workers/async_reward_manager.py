import asyncio
import logging
import os
import socket
import threading

from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional, Tuple, Type

import fastapi
import ray
import uvicorn
from omegaconf import DictConfig
from starlette.requests import Request

from verl.protocol import DataProto
from verl.single_controller.ray.base import RayWorkerGroup
from verl.workers.async_reward_workers import AsyncRewardWorker


class AsyncRewardWorkerManager:
    """AsyncRewardServerManager manage a group of reward worker instances."""

    def __init__(self, config: DictConfig, worker_group: RayWorkerGroup, reward_fn, val_reward_fn, reward_worker_node_id):
        """Initialize AsyncRewardServerManage.

        Args:
            config: DictConfig.
            worker_group: RayWorkerGroup, worker group of reward workers.
        """
        self.full_config = config
        self.config = config.reward_model
        self.worker_group = worker_group
        self.worker_num = worker_group.world_size


        self.async_reward_workers = [None] * self.worker_num

        # worker_class = AsyncRewardWorker()
        
        workers_info = reward_worker_node_id
        assert len(workers_info) == self.worker_group.world_size

        
        # Start all server instances, restart if address already in use.
        unready_ranks = set(range(self.worker_num))
        while len(unready_ranks) > 0:
            workers = {
                rank: AsyncRewardWorker.options(
                    # make sure AsyncRewardWorker colocates with its corresponding workers
                    scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
                        node_id=workers_info[rank],
                        soft=False,
                    ),
                    name=f"async_reward_worker_{rank}",
                ).remote(config, reward_fn, val_reward_fn)
                for rank in unready_ranks
            }

            for rank, worker in workers.items():
                try:
                    # test_data = DataProto.from_dict({'test': 1})
                    # test_result = ray.get(worker.compute_reward.remote(test_data))
                    self.async_reward_workers[rank] = worker
                    unready_ranks.remove(rank)
                    # worker.init_worker.remote()
                    print(f"Async Reward worker {rank} initialized successfully")
                except Exception:
                    ray.kill(worker)
                    print(f"Reward worker {rank} failed to initialize: {e}, restarting...")

    def compute_reward_async(self, data: DataProto):
        chunks = data.chunk(self.worker_num)
        futures = []
        for worker, chunk in zip(self.async_reward_workers, chunks):
            futures.append(worker.compute_reward.remote(chunk))

        return futures