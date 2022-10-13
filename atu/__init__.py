from gym.envs.registration import register
import numpy as np

register(
    id="Safe-DubinsHallway-v2",
    entry_point="atu.envs:DubinsHallwayEnv",
    max_episode_steps=400,
)

register(
    id="Safe-DubinsHallway4D-v1",
    entry_point="atu.envs:DubinsHallway4DEnv",
    max_episode_steps=400,
)

register(
    id="Safe-SingleNarrowPassage-v0",
    entry_point="atu.envs:SingleNarrowPassageEnv",
    max_episode_steps=200,
)