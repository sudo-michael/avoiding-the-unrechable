import gym
from safe_pendulum import PendulumEnv
from helper_oc_controller import HelperOCController
import time

env = PendulumEnv()
obs = env.reset()
done = False
controller = HelperOCController()
from pprint import pprint
import random

t = 0
while not done:
    state = env.state
    opt_ctrl, value = controller.opt_ctrl_value(state)
    use_opt = False
    if value <= 0.5:
        use_opt = True
        action = opt_ctrl
    else:
        # action = env.action_space.sample()
        action = [-2]
        # action = -opt_ctrl

    next_obs, reward, done, info = env.step(action)
    print(f"{t=} {value=} {state=}")
    if not info["safe"]:
        pprint(f"{value=} {use_opt=} {state=} {action=}")
    env.render()
    time.sleep(0.05)
    obs = next_obs
    t += 1

