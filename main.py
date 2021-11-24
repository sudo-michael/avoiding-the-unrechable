import gym
from gym.wrappers import TimeLimit
import gym_dubins


import numpy as np
from Grid.GridProcessing import Grid
from dynamics.DubinsCar import *
from spatialDerivatives.first_order_generic import spa_deriv

g = Grid(np.array([-4.0, -4.0, -np.pi]), np.array([4.0, 4.0, np.pi]), 3, np.array([40, 40, 40]), [2])
dubins_car = DubinsCar(uMode='max', dMode='min')
V = np.load('V.npy')


def opt_ctrl(state):
    spa_derivatives = spa_deriv(g.get_index(state), V, g, periodic_dims=[2])
    return dubins_car.opt_ctrl_non_hcl(0, state, spa_derivatives)


env = gym.make('dubins3d-v0')
env = gym.wrappers.TimeLimit(env, 150)

done = False
state = env.reset()
while not done:
    env.render()
    if g.get_value(V, state) < 0.1:
        action = [opt_ctrl(state)]
    else:
        action = env.action_space.sample()
    next_state, reward, done, _ = env.step(action)
    state = next_state
    