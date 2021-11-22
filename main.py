import gym
from gym.wrappers import TimeLimit
import gym_dubins

env = gym.make('dubins3d-v0')
env = gym.wrappers.TimeLimit(env, 150)

done = False
state = env.reset()
while not done:
    env.render()
    next_state, reward, done, _ = env.step(env.action_space.sample())
    state = next_state
    