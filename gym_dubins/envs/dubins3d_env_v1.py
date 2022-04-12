# adapted from https://github.com/sasayesh/safe_rl/main/env/sim.py?token=AGK4GPUHCPGHORLMFKXV733BUUBVE

import gym
from gym.spaces import Discrete, Box
from gym.utils import seeding
import numpy as np
import matplotlib

# matplotlib.use("tkagg")

import matplotlib.pyplot as plt


class Dubins3DEnv1(gym.Env):
    """
    Description:
        The agent (a car) starts at (-3, -3) and needs to reach a goal at (3, 3) while avoiding
        an obstacle at (0, 0) of radius 0.75. The episode ends when the robot either
        1. reaches the goal
        2. reaches the obstacles
        3. leaves the boundary of the grid [-4, -4] x [4, 4]
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

    metadata = {"render.modes": ["human", "rgb_array"]}

    def __init__(self):
        self.inital_position = np.array([-3.0, -3.0, 0], dtype=np.float32)
        self.robot_radius = 0.4

        self.goal_position = np.array([3, 3])
        self.goal_radius = 0.5

        self.obstacle_position = np.array([0, 0])
        self.obstacle_radius = 0.75

        self.boundary = np.array(
            [
                [-4, 4, -4, -4],
                [-4, 4, 4, 4],
                [-4, -4, -4, 4],
                [4, 4, -4, 4],
            ]
        )

        # dynamics
        self.max_w = 1.0
        self.velocity = 1
        self.dt = 0.1

        self.action_space = Box(
            low=-self.max_w, high=self.max_w, dtype=np.float32, shape=(1,)
        )
        self.observation_space = Box(
            low=-np.inf, high=np.inf, dtype=np.float32, shape=(3,)
        )

        self.hj_used_states = []
        self.state = None
        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.hj_used_states.clear()
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
        reward = -np.linalg.norm(self.state[:2] - self.goal_position)
        info = {"reach_goal": False, "collide_with_obs": False, "out_of_bounds": False}

        if np.linalg.norm(self.state[:2]) <= self.obstacle_radius + self.robot_radius:
            # collide with obstacle
            done = True
            reward = 0
            info["collide_with_obs"] = True
        elif (
            np.linalg.norm(self.state[:2] - self.goal_position)
            <= self.goal_radius + self.robot_radius
        ):
            # reach goal
            done = True
            reward = 1000
            info["reach_goal"] = True
        elif (
            self.state[0] < -4
            or self.state[0] > 4
            or self.state[1] < -4
            or self.state[1] > 4
        ):
            done = True
            reward = -1000
            info["out_of_bounds"] = True

        return self.state, reward, done, info

    def render(self, mode="human"):
        plt.cla()
        # for stopping simulation with the esc key.
        plt.gcf().canvas.mpl_connect(
            "key_release_event",
            lambda event: [exit(0) if event.key in ["escape", "q"] else None],
        )
        plt.plot(self.inital_position[0], self.inital_position[1], "xr")
        plt.plot(self.goal_position[0], self.goal_position[1], "xb")
        obs = plt.Circle(self.obstacle_position, self.obstacle_radius, color="black")
        plt.gcf().gca().add_artist(obs)

        goal = plt.Circle(self.goal_position, self.goal_radius, color="green")
        plt.gcf().gca().add_artist(goal)

        robot = plt.Circle(self.state[:2], self.robot_radius, color="blue")
        plt.gcf().gca().add_artist(robot)
        dir = (
            self.state[:2]
            + np.array([np.cos(self.state[2]), np.sin(self.state[2])])
            * self.robot_radius
        )
        plt.plot([self.state[0], dir[0]], [self.state[1], dir[1]], "-k")

        for i in range(len(self.boundary)):
            plt.plot(
                [self.boundary[i][0], self.boundary[i][1]],
                [self.boundary[i][2], self.boundary[i][3]],
                "-k",
            )

        for state in self.hj_used_states:
            plt.plot(state[0], state[1], "xr")

        plt.axis("equal")
        plt.grid(True)
        plt.gcf().tight_layout(pad=0)

        if mode == "rgb_array":
            plt.gcf().canvas.draw()
            data = np.frombuffer(plt.gcf().canvas.tostring_rgb(), dtype=np.uint8)
            w, h = plt.gcf().canvas.get_width_height()
            im = data.reshape((h, w, -1))
            return im
        else:
            plt.pause(0.001)

    def close(self):
        pass


def main():
    import random

    random.seed(0)

    import time

    env = Dubins3DEnv()
    state = env.reset()
    env.render()
    done = False
    for i in range(100):
        state, reward, done, _ = env.step(env.action_space.sample())
        print(state, reward)
        env.render()

        if done:
            break


if __name__ == "__main__":
    main()
