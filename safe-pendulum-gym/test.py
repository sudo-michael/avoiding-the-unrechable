# %%
import gym
from safe_pendulum import PendulumEnv
from helper_oc_controller import HelperOCController
import time
import numpy as np


done = False
controller = HelperOCController()

opt_ctrl, value = controller.opt_ctrl_value([0, 0])
print(value)
exit()
env = PendulumEnv(matlab_controller=controller)
obs = env.reset()
from pprint import pprint
import random


t = 0
states = []
neg_val_counter = 0
from tqdm import tqdm

# %%
neg_val_counter = 0
obs = env.reset()
for _ in tqdm(range(100)):
    state = env.state
    states.append(state)
    opt_ctrl, value = controller.opt_ctrl_value(state)
    use_opt = False
    if value <= 0.3:
        use_opt = True
        action = opt_ctrl
    else:
        # action = env.action_space.sample()
        # action = [-2]
        action = -opt_ctrl

    # action = opt_ctrl
    next_obs, reward, done, info = env.step(action)
    next_state = env.state
    pprint(f"{t=} {value=} {state=} {next_state=} {action=} {use_opt=}")
    if value < 0:
        neg_val_counter += not info["safe"]
    # env.render()
    obs = next_obs
    t += 1
env.close()
print(f"{neg_val_counter=}")

from pprint import pprint

pprint(states)
# # %%
# import matplotlib.pyplot as plt

# tau = range(len(states))
# plt.plot(tau, [s[0] for s in states])
# plt.show()

# tau = range(len(states))
# plt.plot(tau, [s[1] for s in states])
# plt.show()

# # %%
# x_init = np.array([-0.0875, -0.9786])
# x_next = np.array([-0.14559409, -1.35069142])

# # %%
# env.state = x_init


# # %%
# opt_ctrl, value = controller.opt_ctrl_value(x_init)
# print(opt_ctrl, value)
# # %%
# env.step([-2])

# # %%
# env.state
# # %%

# %%
