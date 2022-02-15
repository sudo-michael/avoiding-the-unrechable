import gym
from gym.wrappers import Monitor
import numpy as np
import gym_dubins


from stable_baselines3 import DQN
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise

env = gym.make("dubins3d-discrete-v0")
# env = Monitor(env, './videos_1')

# REMEMBER TO CHANGE BACK REWARDS
model = DQN("MlpPolicy", env, verbose=1, learning_rate=1e-3, tensorboard_log="./dubins3d-discrete-_tensorboard/", batch_size=1024, tau=0.99)
model.learn(total_timesteps=700_000, log_interval=1000)
model.save("dqn_dubins3d-0")

# model.load("dqn_dubins3d-0")

# obs = env.reset()
# r = 0
# while True:
#     actions, _states = model.predict(obs, deterministic=True)
#     next_obs, reward, done, info = env.step(actions)
#     r += reward
#     if done:
#         print(info)
#         print(r)
#         r = 0
#         obs = env.reset()

