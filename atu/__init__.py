from gym.envs.registration import register
import numpy as np

register(
    id="Safe-Pendulum-v1",
    entry_point="atu.envs:SafePendulumEnv",
    max_episode_steps=200,
)


register(
    id="Safe-Pendulum-Hard-v1",
    entry_point="atu.envs:SafePendulumEnv",
    max_episode_steps=200,
    kwargs={"unsafe_lower": np.pi / 2, "unsafe_upper": 0},
)

register(
    id="Safe-DubinsHallway-v1",
    entry_point="atu.envs:DubinsHallwayEnv",
    max_episode_steps=500,
)
