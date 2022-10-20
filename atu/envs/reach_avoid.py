import gym
import os
from gym.spaces import Box
from atu.brt.brt_air_3d import g as grid
from atu.brt.brt_air_3d import car_brt
from atu.brt.brt_air_3d import cylinder_r
from atu.utils import spa_deriv
import numpy as np
import matplotlib.pyplot as plt

from scipy.integrate import odeint

# import matplotlib matplotlib.use('tkagg')


class ReachAvoid3DEnv(gym.Env):

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render.modes": ["human", "rgb_array"],
        "render_fps": 30,
    }

    def __init__(
        self,
        goal_location=np.array([-2, 2.3, 0.5]),
        speed=1,
        eval=False,
    ) -> None:
        self.eval = eval

        path = os.path.abspath(__file__)
        dir_path = os.path.dirname(path)
        self.brt = np.load(os.path.join(dir_path, "assets/brts/air3d_brt_test.npy"))
        self.car = car_brt
        self.dt = 0.05
        self.action_space = Box(low=-1 * self.car.we_max, high=self.car.we_max, dtype=np.float32, shape=(1,))
        self.world_boundary = np.array([4.5, 4.5, np.pi], dtype=np.float32)

        self.observation_space = Box(
            low=-self.world_boundary,
            high=self.world_boundary,
            shape=(3,),
            dtype=np.float32,
        )

        self.world_width = 10
        self.world_height = 10

        self.left_wall = -4.5
        self.right_wall = 4.5
        self.bottom_wall = -4.5
        self.top_wall = 4.5

        self.grid = grid
        self.fig, self.ax = plt.subplots(figsize=(5, 5))

        # yes
        self.evader_state = np.array([0, 0, 0])
        self.persuer_state = np.array([2, 0, np.pi])
        print(self.relative_state(self.persuer_state, self.evader_state))
        index = self.grid.get_index(self.relative_state(self.persuer_state, self.evader_state))
        print(self.grid.grid_points[0][index[0]], self.grid.grid_points[1][index[1]], self.grid.grid_points[2][index[2]])
        
        # no
        self.evader_state = np.array([0, 0, -np.pi])
        self.persuer_state = np.array([-2, 0, 0])
        print(self.relative_state(self.persuer_state, self.evader_state))
        index = self.grid.get_index(self.relative_state(self.persuer_state, self.evader_state))
        print(self.grid.grid_points[0][index[0]], self.grid.grid_points[1][index[1]], self.grid.grid_points[2][index[2]])

        self.evader_state = np.array([0, 0, np.pi/2])
        self.persuer_state = np.array([-2, 0, 0])
        print(self.relative_state(self.persuer_state, self.evader_state))
        index = self.grid.get_index(self.relative_state(self.persuer_state, self.evader_state))

        exit()
        

        self.goal_location = np.array([2, 2, 0.2])

        # self.hist = []

    def reset(self, seed=None):
        pass

    def step(self, action: np.array):
        opt_ctrl = self.opt_ctrl()
        # self.evader_state = self.car.dynamics_non_hcl(0, self.evader_state, opt_ctrl, is_evader=True) * self.dt + self.evader_state
        # self.evader_state[2] = self.normalize_angle(self.evader_state[2])
        opt_dstb = self.opt_dstb()
        self.persuer_state = self.car.dynamics_non_hcl(0, self.persuer_state, opt_dstb, is_evader=False) * self.dt + self.persuer_state
        self.persuer_state[2] = self.normalize_angle(self.persuer_state[2])

        print(f"{self.evader_state=}")
        print(f"{self.persuer_state=}")
        print(f"{self.relative_state(self.persuer_state, self.evader_state)}")
        done = False

        if np.linalg.norm(self.relative_state(self.persuer_state, self.evader_state)[:2]) <= cylinder_r:
            print('capture')
            done = True

        relative_state = self.relative_state(self.persuer_state, self.evader_state)
        # print(f"rel state: {relative_state}, {np.linalg.norm(relative_state)}")
        index = self.grid.get_index(relative_state)
        # print(f'value: = {self.brt[index]}')

        return {
            'persuer_state': np.copy(self.persuer_state),
            'evader_state': np.copy(self.evader_state),
            'goal_location': np.copy(self.goal_location)
        }, 0, done, {}
        

    def render(self, mode='human'):
        self.ax.clear()
        def add_robot(state, color='green'):
            self.ax.add_patch(
                plt.Circle(state[:2], radius=self.car.r, color=color)
            )

            dir = state[:2] + self.car.r * np.array(
                [np.cos(state[2]), np.sin(state[2])]
            )

            self.ax.plot([state[0], dir[0]], [state[1], dir[1]], color="c")

        add_robot(self.evader_state, color='blue')
        add_robot(self.persuer_state, color='red')

        goal = plt.Circle(
            self.goal_location[:2], radius=self.goal_location[2], color="g"
        )
        self.ax.add_patch(goal)

        
        relative_state = self.relative_state(self.persuer_state, self.evader_state)
        relative_state[2] = self.normalize_angle(relative_state[2])

        index = self.grid.get_index(relative_state)
        print(self.grid.grid_points[0][index[0]], self.grid.grid_points[1][index[1]], self.grid.grid_points[2][index[2]])
        X, Y = np.meshgrid(
            np.linspace(self.grid.min[0], self.grid.max[0], self.grid.pts_each_dim[0]),
            np.linspace(self.grid.min[1], self.grid.max[1], self.grid.pts_each_dim[1]),
            indexing='ij'
        )

        self.ax.contour(
            X + self.evader_state[0],
            Y + self.evader_state[1],
            self.brt[:, :, index[2]],
            levels=[0.2],
        )

        

        # walls
        # self.ax.hlines(y=[-4.5, 4.5], xmin=[-4.5, -4.5], xmax=[4.5, 4.5], color="k")
        # self.ax.vlines(x=[-4.5, 4.5], ymin=[-4.5, -4.5], ymax=[4.5, 4.5], color="k")

        # self.ax.set_xlim(-5, 5)
        # self.ax.set_ylim(-5, 5)
        self.ax.set_aspect("equal")
        self.ax.set_xlabel("x")
        self.ax.set_ylabel("y")
        self.ax.autoscale_view()

        self.fig.canvas.draw()
        img = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(self.fig.canvas.get_width_height()[::-1] + (3,))


        if mode == "human":
            self.fig.canvas.flush_events()
            plt.pause(1 / self.metadata["render_fps"])
            # plt.show()
        return img

    def close(self):
        plt.close()
        return

    def normalize_angle(self, theta):
        """normalize theta to be in range (-pi, pi]"""
        return ((-theta + np.pi) % (2.0 * np.pi) - np.pi) * -1.0

    def relative_state(self, persuer_state, evader_state):
        relative_state = persuer_state - evader_state
        relative_state[2] = self.normalize_angle(relative_state[2])
        return relative_state

    def opt_dstb(self):
        relative_state = self.relative_state(self.persuer_state, self.evader_state)
        index = self.grid.get_index(relative_state)
        spat_deriv = spa_deriv(index, self.brt, self.grid)
        opt_dstb = self.car.opt_dstb_non_hcl(spat_deriv)
        print(f"spa deriv: {spat_deriv[2]}, {opt_dstb}")
        
        return opt_dstb

    def opt_ctrl(self):
        relative_state = self.relative_state(self.persuer_state, self.evader_state)
        index = self.grid.get_index(relative_state)
        spat_deriv = spa_deriv(index, self.brt, self.grid)
        opt_dstb = self.car.opt_ctrl_non_hcl(relative_state, spat_deriv)
        return opt_dstb

if __name__ in "__main__":
    env = ReachAvoid3DEnv()
    env.render()
    import time
    for _ in range(200):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        if done:
            print('done')
            break
        env.render()
