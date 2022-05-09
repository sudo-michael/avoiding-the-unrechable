from gym.envs.registration import register

register(
    id="Safe-Pendulum-v1",
    entry_point="atu.envs:SafePendulumEnv",
    max_episode_steps=200,
)

