from gym.envs.registration import register

register(
    id='dubins3d-v0',
    entry_point='gym_dubins.envs:Dubins3DEnv',
)
