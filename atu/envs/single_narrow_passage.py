import gym
import os
from gym.spaces import Box
from atu.brt.brt_5D import g as grid
from atu.brt.brt_5D import car_brt
from atu.utils import spa_deriv
import numpy as np
import matplotlib.pyplot as plt

import atu.brt.brt_5D as brt_config
from atu.brt.brt_5D import L, CURB_POSITION, STRANDED_CAR_POS, STRANDED_R2_POS, GOAL_POS
from scipy.integrate import odeint


class SingleNarrowPassageEnv(gym.Env):

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render.modes": ["human", "rgb_array"],
        "render_fps": 30,
    }

    def __init__(
        self,
        done_if_unsafe=True,
        use_disturbances=False,
        dist=np.array([0.1, 0.1, 0.1, 0.1, 0.1]),
        eval=False,
    ) -> None:
        self.done_if_unsafe = done_if_unsafe
        self.use_disturbances = use_disturbances
        self.eval = eval

        path = os.path.abspath(__file__)
        dir_path = os.path.dirname(path)

        if self.use_disturbances and np.any(dist) > 0:
            print("using disturbances")
            print(f"{dist=}")
            assert len(dist) == 5
            self.car = car_brt
            self.car.dMax = dist
            self.car.dMin = -dist
            self.max_over_min_brt = np.load(
                os.path.join(
                    dir_path,
                    "assets/brts/max_over_min_single_narrow_passage_brt_dist.npy",
                )
            )
            self.brt = np.load(
                os.path.join(
                    dir_path, "assets/brts/min_single_narrow_passage_brt_dist.npy"
                )
            )

        else:
            print("not using disturbances")
            self.car = car_brt
            self.car.dMax = np.zeros(5)
            self.car.dMin = np.zeros(5)
            self.max_over_min_brt = np.load(
                os.path.join(
                    dir_path, "assets/brts/max_over_min_single_narrow_passage_brt.npy"
                )
            )
            self.brt = np.load(
                os.path.join(
                    dir_path, "assets/brts/min_single_narrow_passage_brt_dist.npy"
                )
            )
        print(f"{self.car.alpha_max=} {self.car.psi_max}")
        self.state = None
        self.dt = 0.05
        self.action_space = Box(
            low=np.array([self.car.alpha_min, self.car.psi_min]),
            high=np.array([self.car.alpha_max, self.car.psi_max]),
            dtype=np.float32,
            shape=(2,),
        )

        self.observation_space = Box(
            low=grid.min,
            high=grid.max,
            shape=(5,),
            dtype=np.float32,
        )

        self.goal_location = GOAL_POS

        self.grid = grid
        self.fig, self.ax = plt.subplots(figsize=(5, 5))

        # self.hist = []

    def reset(self, seed=None):
        # while True:
        #     self.car.x = np.random.uniform(
        #         low=,
        #         high=
        #     )

        #     if self.grid.get_value(self.max_over_min_brt, self.car.x) > 0.5:
        #         break
        self.car.x = np.array([-6, -1.4, 0, 1, 0], dtype=np.float32)
        self.state = np.copy(self.car.x)
        return np.copy(self.car.x)

    def step(self, action: np.array):
        if isinstance(action, dict):
            used_hj = action["used_hj"]
            action = action["actions"]
        else:
            used_hj = False

        if action.shape != (2,):
            action = action[0]

        if self.use_disturbances and self.eval:
            dist = self.opt_dist()
            sol = odeint(
                self.car.dynamics_non_hcl,
                y0=self.car.x,
                t=np.linspace(0, self.dt, 4),
                args=(action, dist),
                tfirst=True,
            )
        elif self.use_disturbances and not self.eval:
            dist = np.array(
                [
                    np.random.uniform(self.car.dMin[i], self.car.dMax[i])
                    for i in range(len(self.car.dMax))
                ]
            )
            sol = odeint(
                self.car.dynamics_non_hcl,
                y0=self.car.x,
                t=np.linspace(0, self.dt, 4),
                args=(action, dist),
                tfirst=True,
            )
        else:
            dist = np.zeros(5)
            sol = odeint(
                self.car.dynamics_non_hcl,
                y0=self.car.x,
                t=np.linspace(0, self.dt, 4),
                args=(action, dist),
                tfirst=True,
            )
        self.car.x = sol[-1]
        self.car.x[2] = self.normalize_angle(self.car.x[2])
        self.car.x = np.clip(self.car.x, self.g.min, self.g.max)
        self.state = np.copy(self.car.x)

        reward = -np.linalg.norm(self.state[:2] - self.goal_location)

        done = False
        info = {}
        info["used_hj"] = used_hj
        if self.collision_curb_or_bounds():
            print(f"colide curb or bounds, {self.state=}")
            if self.done_if_unsafe:
                done = True
            info["cost"] = 1
            info["safe"] = False
            info["collision"] = "curb or oob"
        elif self.collision_car():
            print(f"collision car: {self.state=}")
            if self.done_if_unsafe:
                done = True
            info["cost"] = 1
            info["safe"] = False
            info["collision"] = "car"
        elif self.near_goal():
            print("reach goal")
            done = True
            reward = 100.0
            info["reach_goal"] = True

        # cost is based on distance to obstacle
        info["hj_value"] = self.grid.get_value(self.brt, self.state)

        # self.hist.append(np.copy(self.state))

        return np.copy(self.state), reward, done, info

    def simulate_step(self, state, action):
        if isinstance(action, dict):
            used_hj = action["used_hj"]
            action = action["actions"]
        else:
            used_hj = False

        if action.shape != (2,):
            action = action[0]

        if self.use_disturbances and self.eval:
            dist = self.opt_dist()
            sol = odeint(
                self.car.dynamics_non_hcl,
                y0=state,
                t=np.linspace(0, self.dt, 4),
                args=(action, dist),
                tfirst=True,
            )
        elif self.use_disturbances and not self.eval:
            dist = np.array(
                [
                    np.random.uniform(self.car.dMin[i], self.car.dMax[i])
                    for i in range(len(self.car.dMax))
                ]
            )
            sol = odeint(
                self.car.dynamics_non_hcl,
                y0=state,
                t=np.linspace(0, self.dt, 4),
                args=(action, dist),
                tfirst=True,
            )
        else:
            dist = np.zeros(5)
            sol = odeint(
                self.car.dynamics_non_hcl,
                y0=state,
                t=np.linspace(0, self.dt, 4),
                args=(action, dist),
                tfirst=True,
            )
        state = sol[-1]
        state[2] = self.normalize_angle(state)
        state = np.clip(state, self.g.min, self.g.max)

        reward = -np.linalg.norm(state[:2] - self.goal_location)

        done = False
        info = {}
        info["used_hj"] = used_hj
        if self.collision_curb_or_bounds(state):
            if self.done_if_unsafe:
                done = True
            info["cost"] = 1
            info["safe"] = False
            info["collision"] = "curb or oob"
        elif self.collision_car(state):
            if self.done_if_unsafe:
                done = True
            info["cost"] = 1
            info["safe"] = False
            info["collision"] = "car"
        elif self.near_goal():
            print("reach goal")
            done = True
            reward = 100.0
            info["reach_goal"] = True

        # cost is based on distance to obstacle
        info["hj_value"] = self.grid.get_value(self.brt, self.state)

        # self.hist.append(np.copy(self.state))

        return np.copy(self.state), reward, done, info

    def render(self, mode="human"):
        # self.state = np.array([0, 1, np.pi/2, 0, 0])
        self.ax.clear()

        # world  boundary
        self.ax.hlines(y=[-4.0, 4.0], xmin=[-8, -8], xmax=[8, 8], color="k")
        self.ax.vlines(x=[-8, 8], ymin=[-4.0, -4.0], ymax=[4.0, 4.0], color="k")

        # middle of road
        self.ax.hlines(
            y=[0], xmin=[-8], xmax=[8], color="k", linestyles="dashed", alpha=0.1
        )

        # curb
        self.ax.hlines(
            y=[brt_config.CURB_POSITION[0], brt_config.CURB_POSITION[1]],
            xmin=[-8, -8],
            xmax=[8, 8],
            color="y",
        )

        robot = plt.Circle(self.state[:2], radius=brt_config.L / 2, color="blue")
        self.ax.add_patch(robot)

        dir = self.state[:2] + brt_config.L / 2 * np.array(
            [np.cos(self.state[2]), np.sin(self.state[2])]
        )
        self.ax.plot([self.state[0], dir[0]], [self.state[1], dir[1]], color="c")

        stranded_car = plt.Circle(
            brt_config.STRANDED_CAR_POS, radius=brt_config.L / 2, color="r"
        )

        self.ax.add_patch(stranded_car)

        stranded_car_unsafe = plt.Circle(
            brt_config.STRANDED_R2_POS,
            radius=brt_config.L / 2,
            color="r",
        )
        self.ax.add_patch(stranded_car_unsafe)

        goal = plt.Circle(
            brt_config.GOAL_POS, radius=brt_config.L, color="g", alpha=0.5
        )

        self.ax.add_patch(goal)
        # brt
        X, Y = np.meshgrid(
            np.linspace(self.grid.min[0], self.grid.max[0], self.grid.pts_each_dim[0]),
            np.linspace(self.grid.min[1], self.grid.max[1], self.grid.pts_each_dim[1]),
        )

        index = self.grid.get_index(self.state)
        # by convention dim[0] == row == y
        #               dim[1] == col == x
        # want x to be dim[0], y to be dim[1] so need tranpose
        # without it, contour is flipped along x and y axis
        self.ax.contour(
            X,
            Y,
            self.brt[:, :, index[2], index[3], index[4]].transpose(),
            levels=[0],
        )

        self.ax.set_xlim(-8, 8)
        self.ax.set_ylim(-4, 4)
        self.ax.set_aspect("equal")
        self.ax.set_xlabel("x")
        self.ax.set_ylabel("y")
        self.ax.autoscale_view()

        self.fig.canvas.draw()
        img = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(self.fig.canvas.get_width_height()[::-1] + (3,))

        if mode == "human":
            # plt.show()
            self.fig.canvas.flush_events()
            plt.pause(1 / self.metadata["render_fps"])
        return img

    def close(self):
        plt.close()
        return

    def normalize_angle(self, theta):
        """normalize theta to be in range (-pi, pi]"""
        return ((-theta + np.pi) % (2.0 * np.pi) - np.pi) * -1.0

    def collision_curb_or_bounds(self, state=None):
        if not isinstance(state, np.ndarray):
            state = self.state
        if not (
            CURB_POSITION[0] + 0.5 * self.car.length
            <= state[1]
            <= CURB_POSITION[1] - 0.5 * self.car.length
        ):
            return True
        elif not (
            self.g.min[0] + 0.5 * self.car.length
            <= state[0]
            <= self.g.max[0] - 0.5 * self.car.length
        ):
            return True
        return False

    def collision_car(self, state=None):
        if not isinstance(state, np.ndarray):
            state = self.state
        return (
            min(
                np.linalg.norm(state[:2] - brt_config.STRANDED_CAR_POS),
                np.linalg.norm(state[:2] - brt_config.STRANDED_R2_POS),
            )
            <= self.car.length
        )

    def near_goal(self, state=None):
        if not isinstance(state, np.ndarray):
            state = self.state

        return np.linalg.norm(self.goal_location[:2] - state[:2]) <= (
            self.car.length + self.car.length
        )

    def opt_ctrl(self):
        index = self.grid.get_index(self.state)
        spat_deriv = spa_deriv(index, self.brt, self.grid)
        opt_ctrl = self.car.opt_ctrl_non_hcl(0, self.state, spat_deriv)
        return opt_ctrl

    def opt_dist(self, state=None):
        if state is None:
            state = self.state
        index = self.grid.get_index(state)
        spat_deriv = spa_deriv(index, self.brt, self.grid)
        opt_dist = self.car.opt_dist_non_hcl(0, state, spat_deriv)
        return opt_dist

    def use_opt_ctrl(self, threshold=0.2):
        return self.grid.get_value(self.brt, self.state) < threshold

    def reward_penalty(self, state: np.array, action: np.array):
        """
        calculate grad V dot f(x, u)
        """
        # assert len(state.shape) == 1
        # assert len(action.shape) == 1

        index = self.grid.get_index(state)
        spat_deriv = spa_deriv(index, self.brt, self.grid)
        dstb = self.opt_dist(state)

        gradVdotFxu = self.car.gradVdotFxu(state, action, dstb, spat_deriv)

        return gradVdotFxu


if __name__ in "__main__":

    import time
    from atu.wrappers import RecordEpisodeStatisticsWithCost
    import atu
    from gym.wrappers import TransformObservation
    import gym

    env = SingleNarrowPassageEnv(use_disturbances=False, reach_avoid=False)
    obs = env.reset()
    done = False
    while not done:
        if env.use_opt_ctrl():
            print("using opt")
            action = env.opt_ctrl()
        else:
            action = env.action_space.sample()
        # print(opt_ctrl)
        # action = env.opt_ctrl()
        obs, reward, done, info = env.step(action)
        env.render()
        # obs, reward, done, info = env.step([-2, opt_ctrl[1]])
        # obs, reward, done, info = env.step(env.action_space.sample())
        # time.sleep(0.4)
