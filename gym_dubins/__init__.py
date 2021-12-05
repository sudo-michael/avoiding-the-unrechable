from gym.envs.registration import register

register(
    id='dubins3d-v0',
    max_episode_steps=150,
    entry_point='gym_dubins.envs:Dubins3DEnv',
)

register(
    id='dubins3d-v1',
    max_episode_steps=150,
    entry_point='gym_dubins.envs:Dubins3DEnv1',
)

register(
    id='dubins3d-sparse-v0',
    max_episode_steps=150,
    entry_point='gym_dubins.envs:Dubins3DSparseEnv',
)

register(
    id='dubins3d-discrete-v0',
    max_episode_steps=150,
    entry_point='gym_dubins.envs:Dubins3DDiscreteEnv',
)
