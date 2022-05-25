import numpy as np
import gym
from gym import Env
from gym import spaces


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
                * (1 - saute_discount_factor ** env._max_episode_steps)
                / (1 - saute_discount_factor)
                / env._max_episode_steps
            )

        self.safety_state = 1.0  # z in paper
        self.safety_budget = safety_budget  # d
        self.saute_discount_facctor = saute_discount_factor
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
        self._safety_state -= cost / self.safety_budget
        self._safety_state /= self.saute_discount_factor
        return self._safety_state

    def augment_obs(self, obs: np.ndarray, safety_state: float):
        augmented_obs = (
            np.hstack([obs, safety_state]) if self.use_state_augmentation else obs
        )
        return augmented_obs


if __name__ == "__main__":
    import gym
    env = gym.make("Pendulum-v1")
    env = SauteWrapper(env)
    env.reset()
    env.step(env.action_space.sample())
