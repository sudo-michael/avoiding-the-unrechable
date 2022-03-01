import gym
from safe_pendulum import PendulumEnv
from helper_oc_controller import HelperOCController
import time

env = PendulumEnv()
# env = gym.make('Pendulum-v1')
obs = env.reset()
done = False
controller = HelperOCController()
from pprint import pprint

while not done:
    state = env.state
    opt_ctrl, value = controller.opt_ctrl_value(state)
    use_opt = False
    if value <= 0.5:
        use_opt = True
        action = opt_ctrl
    else:
        action = env.action_space.sample()
    pprint(f"{value=} {use_opt=} {state=}")
    next_obs, reward, done, info = env.step(action)
    obs = next_obs
    env.render()
    time.sleep(0.05)

