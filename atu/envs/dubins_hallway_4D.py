import gym
import os
from gym.spaces import Box
from atu.brt.brt_4D import g as grid
from atu.brt.brt_4D import car_brt
from atu.utils import spa_deriv
import numpy as np
import matplotlib.pyplot as plt

from scipy.integrate import odeint


class DubinsHallway4DEnv(gym.Env):

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render.modes": ["human", "rgb_array"],
        "render_fps": 30,
    }

    def __init__(
        self,
        done_if_unsafe=True,
        use_disturbances=True,
        goal_location=np.array([-2, 2.3, 0.5]),
        dist=np.array([0.1, 0.1, 0.1, 0.1]),
        speed=1,
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
            assert len(dist) == 4
            self.car = car_brt
            self.car.speed = speed
            self.car.dMax = dist
            self.car.dMix = -dist

            self.max_over_min_brt = np.load(
                os.path.join(
                    dir_path, "assets/brts/max_over_min_hallway_4D_brt_dist.npy"
                )
            )
            self.brt = np.load(
                os.path.join(dir_path, "assets/brts/min_hallway_4D_brt_dist.npy")
            )
        else:
            print("not using disturbances")
            self.car = car_brt
            self.car.speed = speed
            self.car.dMax = np.zeros(4)
            self.car.dMin = np.zeros(4)

            self.max_over_min_brt = np.load(
                os.path.join(dir_path, "assets/brts/max_over_min_hallway_4D_brt.npy")
            )
            self.brt = np.load(
                os.path.join(dir_path, "assets/brts/min_hallway_4D_brt.npy")
            )

        print(f"{self.car.uMax=}")
        print(f"{self.car.uMin=}")
        self.state = None
        self.dt = 0.05
        self.action_space = Box(
            low=self.car.uMin, high=self.car.uMax, dtype=np.float32, shape=(2,)
        )

        self.low_world_boundary = np.array([-4.5, -4.5, -1, -np.pi], dtype=np.float32)
        self.world_boundary = np.array([4.5, 4.5, 5, np.pi], dtype=np.float32)

        self.observation_space = Box(
            low=self.low_world_boundary,
            high=self.world_boundary,
            shape=(4,),
            dtype=np.float32,
        )

        self.goal_location = goal_location  # x y r
        self.obstacle_location = np.array([-4.5, -0.5, 6.5, 1.0])  # x y w h

        self.world_width = 10
        self.world_height = 10

        self.left_wall = -4.5
        self.right_wall = 4.5
        self.bottom_wall = -4.5
        self.top_wall = 4.5

        self.grid = grid
        self.fig, self.ax = plt.subplots(figsize=(5, 5))

        # self.hist = []

    def reset(self, seed=None):
        # self.hist.clear()
        while True:
            self.car.x = np.random.uniform(
                low=self.low_world_boundary,
                # reset in bottom left-half of map
                high=np.array([0.0, -0.5, 5, np.pi]),
            )

            if (
                self.grid.get_value(self.max_over_min_brt, self.car.x) > 0.5
            ):  # place car in location that is always possible to avoid obstacle
                break

        self.car.x = np.array(self.car.x, dtype=np.float32)
        self.state = np.copy(self.car.x)
        return np.copy(self.state)

    def step(self, action: np.array):
        if isinstance(action, dict):
            used_hj = action["used_hj"]
            action = action["actions"]
        else:
            used_hj = False

        if action.shape != (2,) and action.shape != (4,):  # opt_ctrl returns (4, ) policy network is (2, )
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
            dist = np.zeros(4)
            sol = odeint(
                self.car.dynamics_non_hcl,
                y0=self.car.x,
                t=np.linspace(0, self.dt, 4),
                args=(action, dist),
                tfirst=True,
            )
        self.car.x = sol[-1]
        self.car.x[3] = self.normalize_angle(self.car.x[3])
        self.state = np.copy(self.car.x)
        # print(f"{self.state=}")

        reward = -np.linalg.norm(self.state[:2] - self.goal_location[:2])

        done = False
        info = {}
        info["used_hj"] = used_hj
        info["safe"] = True
        info["cost"] = 0
        if self.collision_rect_circle(
            self.obstacle_location[0],
            self.obstacle_location[1],
            self.obstacle_location[2],
            self.obstacle_location[3],
            self.car.x[0],
            self.car.x[1],
            self.car.r,
        ):
            if self.done_if_unsafe:
                done = True
            info["safe"] = False
            info["cost"] = 1
            info["collision"] = "lava"
        elif not (
            self.left_wall + self.car.r <= self.car.x[0] <= self.right_wall - self.car.r
        ):
            done = True
            info["safe"] = False
            info["cost"] = 1
            info["collision"] = "wall"
        elif not (
            self.bottom_wall + self.car.r <= self.car.x[1] <= self.top_wall - self.car.r
        ):
            done = True
            info["safe"] = False
            info["cost"] = 1
            info["collision"] = "wall"
        elif self.near_goal():
            done = True
            reward = 100.0
            info["reach_goal"] = True

        # cost is based on distance to obstacle
        info["hj_value"] = self.grid.get_value(self.brt, self.state)

        # if done and not info.get('reach_goal', False):
        #     print(info['hj_value'])

        # self.hist.append(np.copy(self.state))

        return np.copy(self.state), reward, done, info

    def simulate_step(self, state: np.array, action: np.array):
        if isinstance(action, dict):
            used_hj = action["used_hj"]
            action = action["actions"]
        else:
            used_hj = False

        if action.shape != (2,):
            action = action[0]

        if self.use_disturbances and self.eval:
            dist = self.opt_dist(state)
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
            dstb = np.zeros(4)
            sol = odeint(
                self.car.dynamics_non_hcl,
                y0=state,
                t=np.linspace(0, self.dt, 4),
                args=(action, dist),
                tfirst=True,
            )
        state = sol[-1]
        state[3] = self.normalize_angle(state[3])
        # print(f"{self.state=}")

        reward = -np.linalg.norm(state[:2] - self.goal_location[:2])

        done = False
        info = {}
        info["used_hj"] = used_hj
        info["safe"] = True
        info["cost"] = 0
        if self.collision_rect_circle(
            self.obstacle_location[0],
            self.obstacle_location[1],
            self.obstacle_location[2],
            self.obstacle_location[3],
            state[0],
            state[1],
            self.car.r,
        ):
            if self.done_if_unsafe:
                done = True
            info["safe"] = False
            info["cost"] = 1
            info["collision"] = "lava"
        elif not (
            self.left_wall + self.car.r <= state[0] <= self.right_wall - self.car.r
        ):
            done = True
            info["safe"] = False
            info["cost"] = 1
            info["collision"] = "wall"
        elif not (
            self.bottom_wall + self.car.r <= state[1] <= self.top_wall - self.car.r
        ):
            done = True
            info["safe"] = False
            info["cost"] = 1
            info["collision"] = "wall"
        elif self.near_goal(state):
            done = True
            reward = 100.0
            info["reach_goal"] = True

        # cost is based on distance to obstacle
        info["hj_value"] = self.grid.get_value(self.brt, state)

        return np.copy(state), reward, done, info

    def reward_penalty(self, state: np.array, action: np.array):
        """
        calculate grad V dot f(x, u)
        """
        assert len(state.shape) == 1
        assert len(action.shape) == 1

        index = self.grid.get_index(state)
        spat_deriv = spa_deriv(index, self.brt, self.grid)
        dstb = self.opt_dist(state)

        gradVdotFxu = self.car.gradVdotFxu(state, action, dstb, spat_deriv)

        return gradVdotFxu

    def render(self, mode="human"):
        self.ax.clear()
        robot = plt.Circle(self.state[:2], radius=self.car.r, color="blue")
        self.ax.add_patch(robot)

        dir = self.state[:2] + self.car.r * np.array(
            [np.cos(self.state[3]), np.sin(self.state[3])]
        )

        self.ax.plot([self.state[0], dir[0]], [self.state[1], dir[1]], color="c")

        goal = plt.Circle(
            self.goal_location[:2], radius=self.goal_location[2], color="g"
        )
        self.ax.add_patch(goal)

        lava = plt.Rectangle(
            self.obstacle_location[:2],
            self.obstacle_location[2],
            self.obstacle_location[3],
            color="r",
        )
        self.ax.add_patch(lava)
        self.ax.hlines(y=[-4.5, 4.5], xmin=[-4.5, -4.5], xmax=[4.5, 4.5], color="k")
        self.ax.vlines(x=[-4.5, 4.5], ymin=[-4.5, -4.5], ymax=[4.5, 4.5], color="k")

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
            self.brt[:, :, index[2], index[3]].transpose(),
            levels=[0],
        )

        self.ax.set_xlim(-5, 5)
        self.ax.set_ylim(-5, 5)
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
        return img

    def close(self):
        plt.close()
        return

    def normalize_angle(self, theta):
        """normalize theta to be in range (-pi, pi]"""
        return ((-theta + np.pi) % (2.0 * np.pi) - np.pi) * -1.0

    def collision_rect_circle(
        self, rleft, rtop, width, height, center_x, center_y, radius
    ):
        """
        Detect collision between a rectangle and circle.
        https://stackoverflow.com/questions/21089959/detecting-collision-of-rectangle-with-circle
        """
        rect_x = rleft
        rect_y = rtop

        dist_x = np.abs(center_x - rect_x - width / 2)
        dist_y = np.abs(center_y - rect_y - height / 2)

        if dist_x > (width / 2 + radius):
            return False
        if dist_y > (height / 2 + radius):
            return False

        if dist_x <= width / 2:
            return True
        if dist_y <= height / 2:
            return True

        dx = dist_x - width / 2
        dy = dist_y - height / 2

        return dx**2 + dy**2 <= radius**2

    def near_goal(self, state=None):
        if not isinstance(state, np.ndarray):
            state = self.state

        return (
            np.linalg.norm(self.goal_location[:2] - state[:2])
            <= self.goal_location[2] + self.car.r
        )

    def opt_ctrl(self):
        index = self.grid.get_index(self.state)
        brt = self.max_over_min_brt
        spat_deriv = spa_deriv(index, brt, self.grid)
        opt_ctrl = self.car.opt_ctrl_non_hcl(0, self.state, spat_deriv)
        return opt_ctrl

    def opt_dist(self, state=None):
        if state is None:
            state = self.state
        index = self.grid.get_index(state)
        spat_deriv = spa_deriv(index, self.max_over_min_brt, self.grid)
        opt_dist = self.car.opt_dstb_non_hcl(0, state, spat_deriv)
        return opt_dist

    def use_opt_ctrl(self, threshold=0.1, threshold_ra=0.0):
        return self.grid.get_value(self.max_over_min_brt, self.state) < threshold


if __name__ in "__main__":
    import gym

    # def run_one_episode():
    #     env = gym.make(
    #         "Safe-DubinsHallway4D-v0", use_reach_avoid=False, use_disturbances=True
    #     )

    #     # env = gym.wrappers.RecordVideo(env, 'tmp/')
    #     obs = env.reset()
    #     done = False
    #     while not done:
    #         env.render()
    #         if env.use_opt_ctrl():
    #             action = env.safe_ctrl()
    #         else:
    #             action = env.action_space.sample()
    #         next_obs, reward, done, info = env.step(action)
    #         if done:
    #             break
    #     steps = env._elapsed_steps
    #     env.close()
    #     return steps

    # from timeit import default_timer as timer

    # start = timer()
    # steps = run_one_episode()
    # end = timer()
    # print((end - start) / steps)

    env = DubinsHallway4DEnv(use_reach_avoid=False, use_disturbances=True)
    obs = env.reset()
    # print(obs)
    done = False
    while not done:
        # print(obs)
        # if env.use_opt_ctrl():
        action = env.safe_ctrl()
        # else:
        # action = env.action_space.sample()
        next_obs, reward, done, info = env.step(action)
        print(info["hj_value"])
        obs = next_obs
        env.render()