# # %%
# import numpy as np
# x = np.linspace(-1 ,1, 40)
# y = np.linspace(-1 ,1, 40)
# theta = np.linspace(-np.pi, np.pi, 40)

# X, Y, T = np.meshgrid(x, y, theta)
# # %%

# X = np.expand_dims(X.flatten(), axis=0).T
# Y = np.expand_dims(Y.flatten(), axis=0).T
# # T = np.expand_dims(T.flatten(), axis=0).T
# T = np.ones_like(X) * 1.5863
# # %%
# data = np.append(X, Y, axis=1)
# data = np.append(data, T, axis=1)

# # %%

# import gym
# from gym.wrappers import RecordVideo

# env = gym.make("Pendulum-v1")
# env = RecordVideo(env, "./test_record")

# env.reset()
# for _ in range(100):
#     env.step(env.action_space.sample())
#     # env.render()
# env.close()
import gym
import gym_dubins

import safe_controller

env = gym.make("dubins3d-v0")

safe_controller = safe_controller.SafeController()
action = env.action_space.sample()
obs = env.reset()
done = False
while not done:
    action = env.action_space.sample()
    opt_action = [safe_controller.opt_ctrl(obs)]

    print(action, opt_action)
    exit()

    next_obs, reward, done, _ = env.step(action)
