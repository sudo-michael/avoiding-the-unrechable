import gym


class SauteWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)

    def step(self, action):
        return super().step(action)

    def reset(self, **kwargs):
        return super().reset(**kwargs)

    @property
    def safety_budget(self):
        return self._safety_budget

    @property
    def saute_discount_factor(self):
        return self._saute_discount_factor

    @property
    def unsafe_reward(self):
        return self._unsafe_reward
