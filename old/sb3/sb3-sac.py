import gym

import gym_dubins
from stable_baselines3 import SAC
# from stable_baselines3.common.env_util import make_vec_env

# Parallel environments
env = gym.make("dubins3d-v1")
env = gym.wrappers.TimeLimit(env, 125)

print('model')
model = SAC("MlpPolicy", env, verbose=1, learning_rate=1e-3, tensorboard_log="./dubins3d-sac_tensorboard/")
print('learning')
model.learn(total_timesteps=1_000_000, log_interval=100,)
model.save("sac_dubins3d-1-normal-pen-for-collide")

# model = SAC.load("sac_dubins3d-1-penalty-for-collide")
# obs = env.reset()
# while True:
#     action, _states = model.predict(obs, deterministic=True)
#     obs, reward, done, info = env.step(action)
#     env.render()
#     if done:
#       obs = env.reset()
