import gym

import gym_dubins
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

# Parallel environments
env = make_vec_env("CartPole-v1", n_envs=4)
env = gym.wrappers.TimeLimit(env, 125)

model = PPO("MlpPolicy", env, verbose=1, learning_rate=1e-3, tensorboard_log="./dubins3d-ppo_tensorboard/")
model.learn(total_timesteps=500_000, log_interval=1_000,)
model.save("ppo_dubins3d-0")