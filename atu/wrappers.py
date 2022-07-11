from itertools import accumulate
from re import U
import time
from collections import deque
from typing import Optional

import numpy as np
import gym
from gym import Env, spaces


class RecordEpisodeStatisticsWithCost(gym.Wrapper):
    def __init__(self, env, deque_size=100):
        super().__init__(env)
        self.num_envs = getattr(env, "num_envs", 1)
        self.t0 = time.perf_counter()
        # cost := reward for constraint
        # safe: boolean := indicate where state is safe or not
        self.total_cost = 0
        self.total_unsafe = 0
        self.total_reach_goal = 0
        self.total_use_hj = 0
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
        self.episode_unsafes = np.zeros(self.num_envs, dtype=np.float32)
        self.episode_lengths = np.zeros(self.num_envs, dtype=np.int32)
        return observations

    def step(self, action):
        observations, rewards, dones, infos = super().step(action)
        costs = np.zeros_like(rewards)
        uses_hj = np.zeros_like(rewards)
        safes = np.ones_like(rewards)
        self.episode_returns += rewards
        self.episode_lengths += 1
        if not self.is_vector_env:
            costs = infos.get("cost", 0)
            uses_hj = infos.get("use_hj", False)
            safes = infos.get("safe", True)
            infos = [infos]
            dones = [dones]
            self.total_cost += costs
            self.total_unsafe += not safes
            self.total_use_hj += not uses_hj
        else:
            infos = list(infos)  # Convert infos to mutable type
            for i in range(len(infos)):
                costs[i] = infos[i].get("cost", 0)
                safes[i] = infos[i].get("safe", True)
                uses_hj[i] = infos[i].get("used_hj", False)
            self.total_cost += sum(costs)
            self.total_unsafe += sum(np.invert(safes))
            self.total_use_hj += sum(uses_hj)
        self.episode_costs += costs
        self.episode_unsafes += np.invert(safes)

        for i in range(len(dones)):
            if dones[i]:
                self.episode_count += 1

                infos[i] = infos[i].copy()
                self.total_reach_goal += infos[i].get("reach_goal", False)
                episode_return = self.episode_returns[i]
                episode_cost = self.episode_costs[i]
                episode_unsafe = self.episode_unsafes[i]
                episode_length = self.episode_lengths[i]
                episode_info = {
                    "r": episode_return,
                    "l": episode_length,
                    "c": episode_cost,
                    "us": episode_unsafe,
                    "prg": self.total_reach_goal / self.episode_count,
                    "trg": self.total_reach_goal,
                    "tc": self.total_cost,
                    "tus": self.total_unsafe,
                    "thj": self.total_use_hj,
                    "t": round(time.perf_counter() - self.t0, 6),
                }
                infos[i]["episode"] = episode_info
                self.return_queue.append(episode_return)
                self.length_queue.append(episode_length)
                self.episode_returns[i] = 0
                self.episode_costs[i] = 0
                self.episode_unsafes[i] = 0
                self.episode_lengths[i] = 0

        if self.is_vector_env:
            infos = tuple(infos)
        return (
            observations,
            rewards,
            dones if self.is_vector_env else dones[0],
            infos if self.is_vector_env else infos[0],
        )


class SauteWrapper(gym.Wrapper):
    def __init__(
        self,
        env: Env,
        safety_budget: float = 1.0,
        use_reward_shaping: bool = True,
        use_state_augmentation: bool = True,
        mode="train",
        min_rel_budget: float = 1.0,
        max_rel_budget: float = 1.0,
        test_rel_budget: float = 1.0,
        saute_discount_factor: float = 0.99,
        unsafe_reward: float = 0.0,
    ):
        super().__init__(env)

        assert safety_budget > 0, "Please specify a positive safety budget"
        assert (
            saute_discount_factor > 0 and saute_discount_factor <= 1
        ), "Please specify a discount factor in (0, 1]"
        assert (
            min_rel_budget <= max_rel_budget
        ), "Minimum relative budget should be smaller or equal to maximum relative budget"
        assert isinstance(env.observation_space, spaces.Box)

        self.use_reward_shaping = use_reward_shaping
        self.use_state_augmentation = use_state_augmentation
        self.test_rel_budget = test_rel_budget  # test relative budget
        self.min_rel_budget = (
            min_rel_budget  # minimum relative (with respect to safety_budget) budget
        )
        self.max_rel_budget = (
            max_rel_budget  # maximum relative (with respect to safety_budget) budget
        )
        self.mode = mode

        # what does this do?
        if saute_discount_factor < 1.0:
            # sum of finite geometric sequence /
            safety_budget = (
                safety_budget
                * (1 - saute_discount_factor ** env.spec.max_episode_steps)
                / (1 - saute_discount_factor)
                / env.spec.max_episode_steps
            )

        self.safety_state = 1.0  # z in paper
        self.safety_budget = safety_budget  # d
        self.saute_discount_factor = saute_discount_factor
        self.unsafe_reward = unsafe_reward

        if self.use_state_augmentation:
            self.obs_low = np.array(
                np.hstack([env.observation_space.low, -np.inf]), dtype=np.float32
            )
            self.obs_high = np.array(
                np.hstack([env.observation_space.high, np.inf]), dtype=np.float32
            )
            self.observation_space = spaces.Box(low=self.obs_low, high=self.obs_high)

    def reset(self, **kwargs):
        return_info = kwargs.get("return_info", False)
        if return_info:
            obs, info = self.env.reset(**kwargs)
        else:
            obs = self.env.reset(**kwargs)

        if self.mode == "train":
            self.safety_state = np.random.uniform(
                low=self.min_rel_budget, high=self.max_rel_budget
            )
        elif self.mode in ["test", "deterministic"]:
            self.safety_state = self.test_rel_budget
        else:
            raise NotImplementedError(f"Unknown mode: {self.mode}")

        augmented_obs = self.augment_obs(obs, self.safety_state)
        if return_info:
            return augmented_obs, info
        else:
            return augmented_obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        safety_state = self.safety_step(info["cost"])
        info["reward"] = reward
        info["safety_state"] = safety_state

        if self.use_reward_shaping:
            reward = reward if safety_state > 0 else self.unsafe_reward
        augmented_obs = self.augment_obs(obs, safety_state)
        return augmented_obs, reward, done, info

    def safety_step(self, cost: np.ndarray) -> np.ndarray:
        """ Update the normalized safety state z' = (z - l / d) / gamma. """
        self.safety_state -= cost / self.safety_budget
        self.safety_state /= self.saute_discount_factor
        return self.safety_state

    def augment_obs(self, obs: np.ndarray, safety_state: float):
        augmented_obs = (
            np.hstack([obs, safety_state]) if self.use_state_augmentation else obs
        )
        return augmented_obs
