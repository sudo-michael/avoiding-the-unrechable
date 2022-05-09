from itertools import accumulate
import time
from collections import deque
from typing import Optional

import numpy as np
import gym


class RecordEpisodeStatisticsWithCost(gym.Wrapper):
    def __init__(self, env, deque_size=100):
        super().__init__(env)
        self.num_envs = getattr(env, "num_envs", 1)
        self.t0 = time.perf_counter()
        self.total_cost = 0
        self.episode_count = 0
        self.episode_returns = None
        self.episode_lengths = None
        self.return_queue = deque(maxlen=deque_size)
        self.length_queue = deque(maxlen=deque_size)
        self.is_vector_env = getattr(env, "is_vector_env", False)

    def reset(self, **kwargs):
        observations = super().reset(**kwargs)
        self.episode_returns = np.zeros(self.num_envs, dtype=np.float32)
        self.episode_costs = np.zeros(self.num_envs, dtype=np.float32)
        self.episode_lengths = np.zeros(self.num_envs, dtype=np.int32)
        return observations

    def step(self, action):
        observations, rewards, dones, infos = super().step(action)
        costs = np.zeros_like(rewards)
        self.episode_returns += rewards
        self.episode_lengths += 1
        if not self.is_vector_env:
            costs = 0.0 if infos["safe"] else 1.0
            infos = [infos]
            dones = [dones]
            self.total_cost += costs
        else:
            infos = list(infos)  # Convert infos to mutable type
            for i in range(len(infos)):
                costs[i] = 0.0 if infos[i]["safe"] else 1.0
            self.total_cost += accumulate(costs, 0)

        self.episode_costs += costs
        for i in range(len(dones)):
            if dones[i]:
                infos[i] = infos[i].copy()
                episode_return = self.episode_returns[i]
                episode_cost = self.episode_costs[i]
                episode_length = self.episode_lengths[i]
                episode_info = {
                    "r": episode_return,
                    "l": episode_length,
                    "c": episode_cost,
                    "tc": self.total_cost,
                    "t": round(time.perf_counter() - self.t0, 6),
                }
                infos[i]["episode"] = episode_info
                self.return_queue.append(episode_return)
                self.length_queue.append(episode_length)
                self.episode_count += 1
                self.episode_returns[i] = 0
                self.episode_costs[i] = 0
                self.episode_lengths[i] = 0
        if self.is_vector_env:
            infos = tuple(infos)
        return (
            observations,
            rewards,
            dones if self.is_vector_env else dones[0],
            infos if self.is_vector_env else infos[0],
        )