# adapted from https://github.com/sasayesh/safe_rl/main/env/sim.py?token=AGK4GPUHCPGHORLMFKXV733BUUBVE

import gym
from gym.spaces import Discrete, Box
from gym.utils import seeding
import numpy as np
import matplotlib.pyplot as plt

class Dubins3DEnv(gym.Env):
    """
    Description:
        The agent (a car) starts at (-3, -3) and needs to reach a goal at (3, 3) while avoiding
        an obstacle at (0, 0) of radius 0.75
    Observation:
        Type: Box(4)
        Num    Observation               Min            Max
        0      x
        1      y
        2      theta                    -pi             pi
    Actions:
        Type: Box(1)
        Num    Action                    Min            Max
        0      turn rate                 -1.0           1.0
    Reward:
         Episode length is greater than 150
    """
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.inital_position = np.array([-3.0, -3.0, 0], dtype=np.float32)
        self.robot_radius = 0.4

        self.goal_position = np.array([3, 3])
        self.goal_radius = 0.5

        self.obstacle_position = np.array([0, 0])
        self.obstacle_radius = 0.75

        # dynamics
        self.max_w = 1.0
        self.velocity = 1
        self.dt = 0.1

        self.action_space = Box(low=-self.max_w, high=self.max_w, dtype=np.float32,
                                     shape=(1,))
        self.observation_space = Box(low=-np.inf, high=np.inf, dtype=np.float32,
                                     shape=(3,))

        self.state = None
        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.state = self.inital_position.copy()
        return self.state

    def step(self, action):
        x, y, theta = self.state
        w = np.clip(action[0], -self.max_w, self.max_w)

        x_dot = self.velocity * np.cos(theta)
        y_dot = self.velocity * np.sin(theta)
        theta_dot = w

        # forward euler
        self.state += self.dt * np.array([x_dot, y_dot, theta_dot])

        # normalize angle to be in range [-pi, pi]
        self.state[2] = ((self.state[2] + np.pi) % (2 * np.pi)) - np.pi

        done = False
        if np.linalg.norm(self.state[:2]) <= self.obstacle_radius + self.robot_radius:
            # collide with obstacle
            done = True
            reward = -100
        elif np.linalg.norm(self.state[:2] - self.goal_position) <= self.goal_radius + self.robot_radius:
            # reach goal
            done = True
            reward = 100
        else:
            reward = -np.linalg.norm(self.state[:2] - self.goal_position)

        return self.state, reward, done, {}
        
    def render(self, mode='human'):
        plt.cla()
        # for stopping simulation with the esc key.
        plt.gcf().canvas.mpl_connect(
            'key_release_event',
            lambda event: [exit(0) if event.key in ['escape', 'q'] else None])
        plt.plot(self.inital_position[0], self.inital_position[1], "xr")
        plt.plot(self.goal_position[0], self.goal_position[1], "xb")
        obs = plt.Circle(self.obstacle_position, self.obstacle_radius, color='black')
        plt.gcf().gca().add_artist(obs)

        robot = plt.Circle(self.state[:2], self.robot_radius, color='blue')
        plt.gcf().gca().add_artist(robot)
        dir = self.state[:2] + np.array([np.cos(self.state[2]), np.sin(self.state[2])]) * self.robot_radius
        plt.plot([self.state[0], dir[0]], [self.state[1], dir[1]], "-k")

        plt.axis("equal")
        plt.grid(True)
        plt.pause(0.001)

        # if mode == 'rgb_array':
        #     image_from_plot = np.frombuffer(plt.gcf().canvas.tostring_rgb(), dtype=np.uint8)
        #     image_from_plot = image_from_plot.reshape(plt.gcf().canvas.get_width_height()[::-1] + (3,))
        #     return image_from_plot


def main():
    import random
    random.seed(0)

    import time
    env = Dubins3DEnv()
    state = env.reset()
    env.render()
    done = False
    for i in range(100):
        if i % 2 == 0:
            state, reward, done, _ = env.step(1)
        else:
            state, reward, done, _ = env.step(-1)
        print(state, reward)
        env.render()


if __name__ == "__main__":
    main()