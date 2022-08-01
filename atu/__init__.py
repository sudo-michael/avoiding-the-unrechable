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
    max_episode_steps=600,
)

register(
    id="Safe-SingleNarrowPassage-v0",
    entry_point="atu.envs:SingleNarrowPassageEnv",
    max_episode_steps=600,
)

# register(
#     id="Safe-DubinsHallway-Flipped-v1",
#     entry_point="atu.envs:DubinsHallwayEnv",
#     max_episode_steps=500,
#     goal_location=np.array([-2, -2.3, 0.5]),
# )
