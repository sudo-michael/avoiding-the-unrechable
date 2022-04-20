import argparse
import datetime
import gym
import numpy as np
import itertools
import torch

import gym
from gym.wrappers import TimeLimit
import gym_dubins


import numpy as np
from Grid.GridProcessing import Grid
from dynamics.DubinsCar import *
from spatialDerivatives.first_order_generic import spa_deriv

dubins_car = DubinsCar(uMode='max', dMode='min')
V = np.load('V_r1.15_grid100.npy')
g = Grid(np.array([-4.0, -4.0, -np.pi]), np.array([4.0, 4.0, np.pi]), 3, np.array(V.shape), [2])

env = gym.make('dubins3d-v0')

def opt_ctrl(state):
    spa_derivatives = spa_deriv(g.get_index(state), V, g, periodic_dims=[2])
    opt_w = dubins_car.wMax
    if spa_derivatives[2] > 0:
        if dubins_car.uMode == "min":
            opt_w = -opt_w
    elif spa_derivatives[2] < 0:
        if dubins_car.uMode == "max":
            opt_w = -opt_w
    return opt_w

bad = 0
for j in range(10000):
    done = False
    state = env.reset()
    i = 0
    while not done:
        # print(f"state={state}, v={g.get_value(V, state)}")
        # print(f"{np.linalg.norm(state)}")
        if g.get_value(V, state) < 0.26:
            # print('using opt ctrl')
            action = [opt_ctrl(state)]
            next_state, reward, done, info = env.step(action)
        elif i < 8:
            next_state, reward, done, info = env.step([1])
        else:
            next_state, reward, done, info = env.step(env.action_space.sample())
        state = next_state

        i += 1
        if done:
            break
    state = env.reset()
    done = False
    if info['collide_with_obs']:
        bad += 1
    print(j, bad)
print(bad, 10000)