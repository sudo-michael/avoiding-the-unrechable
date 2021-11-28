from gym.envs.registration import register

register(
    id='dubins3d-v0',
    entry_point='gym_dubins.envs:Dubins3DEnv',
)

register(
    id='dubins3d-sparse-v0',
    entry_point='gym_dubins.envs:Dubins3DSparseEnv',
)

register(
    id='dubins3d-discrete-v0',
    entry_point='gym_dubins.envs:Dubins3DDiscreteEnv',
)

