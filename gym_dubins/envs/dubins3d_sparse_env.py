# adapted from https://github.com/sasayesh/safe_rl/main/env/sim.py?token=AGK4GPUHCPGHORLMFKXV733BUUBVE

from random import randint
import gym
from gym.spaces import Discrete, Box
from gym.utils import seeding
import numpy as np
import matplotlib.pyplot as plt

class Dubins3DSparseEnv(gym.Env):
    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 30}

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
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.inital_position = np.array([-3.0, -3.0, np.pi/4], dtype=np.float32)
        # self.inital_position = np.array([1.8, 1.8, 0], dtype=np.float32)
        self.robot_radius = 0.4

        self.goal_position = np.array([3, 3])
        self.goal_radius = 0.75

        self.obstacle_position = np.array([0, 0])
        self.obstacle_radius = 0.75

        self.boundary = np.array([
            [-4, 4, -4, -4], 
            [-4, 4,  4, 4], 
            [-4, -4, -4, 4], 
            [4, 4, -4, 4], 
            ]
        )


        self.viewer = None
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
        self.last_state = self.inital_position.copy()

        return self.state

    def step(self, action):
        x, y, theta = self.state
        w = np.clip(action[0], -self.max_w, self.max_w)

        x_dot = self.velocity * np.cos(theta)
        y_dot = self.velocity * np.sin(theta)
        theta_dot = w

        # forward euler
        self.last_state = self.state.copy()
        self.state += self.dt * np.array([x_dot, y_dot, theta_dot])

        # normalize angle to be in range [-pi, pi]
        self.state[2] = ((self.state[2] + np.pi) % (2 * np.pi)) - np.pi

        done = False
        reward = 0.0
        info = {'reach_goal': False, 'collide_with_obs': False, 'out_of_bounds': False}

        if np.linalg.norm(self.state[:2]) <= self.obstacle_radius + self.robot_radius:
            # collide with obstacle
            done = True
            reward = -1
            info['collide_with_obs'] = True
        elif np.linalg.norm(self.state[:2] - self.goal_position) <= self.goal_radius + self.robot_radius:
            # reach goal
            done = True
            reward = 1
            info['reach_goal'] = True
        elif self.state[0] < -4 or self.state[0] > 4 or self.state[1] < -4 or self.state[1] > 4:
            done = True
            reward = -1
            info['out_of_bounds'] = True

        return self.state, reward, done, info
        
    def render(self, mode='human'):
        if not self.viewer:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(500, 500)
            self.viewer.set_bounds(-5, 5 ,-5, 5)

            boundary = rendering.PolyLine(v=[(-4, 4), (-4, -4), (4, -4), (4, 4)], close=True)
            boundary.set_color(0, 0, 0)
            self.viewer.add_geom(boundary)

            obs = rendering.make_circle(radius=self.obstacle_radius)
            obs.set_color(0,0,0)
            self.viewer.add_geom(obs)

            goal = rendering.make_circle(radius=self.goal_radius)
            goal.set_color(0,1,0)
            goal.add_attr(
                rendering.Transform(
                    translation=(self.goal_position[0], self.goal_position[1])
                )
            )
            self.viewer.add_geom(goal)
            
            
            self.robot_render = rendering.make_circle(radius=self.robot_radius)
            self.robot_transform = rendering.Transform()
            self.robot_render.add_attr(
                rendering.Transform(
                    translation=(self.inital_position[0], self.inital_position[1])
                )
            )

            self.robot_render2 = rendering.make_circle(radius=self.robot_radius)
            self.robot_render2.set_color(0, 1, 1)
            self.robot_render2.add_attr(
                rendering.Transform(
                    translation=(1, 1)
                )
            )
            self.viewer.add_geom(self.robot_render2)
            self.robot_render3 = rendering.make_circle(radius=self.robot_radius)
            self.robot_render3.set_color(0, 1, 1)
            self.robot_render3.add_attr(
                rendering.Transform(
                    translation=(-1, 1)
                )
            )
            self.viewer.add_geom(self.robot_render3)



            self.robot_render.add_attr(
                self.robot_transform
            )
            self.robot_render.set_color(0.0, 0.0, 1.0)
            self.viewer.add_geom(self.robot_render)


        self.robot_transform.set_translation(
            # self.state[0] - self.inital_position[0],  self.state[0] - self.inital_position[0]
            1, 1
        )

        return self.viewer.render(return_rgb_array=mode == "rgb_array")

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

def main():
    import random
    random.seed(0)

    env = Dubins3DSparseEnv()
    state = env.reset()
    env.render()
    done = False
    for i in range(100):
        # if i % 2 == 0:
        #     state, reward, done, _ = env.step([1])
        # else:
        #     state, reward, done, _ = env.step([-1])
        state, reward, done, _ = env.step(env.action_space.sample())
        print(state, reward)
        env.render()


if __name__ == "__main__":
    main()