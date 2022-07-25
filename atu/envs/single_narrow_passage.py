import gym
import os
from gym.spaces import Box
from atu.optimized_dp.Grid.GridProcessing import Grid
import atu.optimized_dp.brt_single_narrow_passage as brt_config
import numpy as np
import math
import matplotlib.pyplot as plt


def spa_deriv(index, V, g, periodic_dims=[]):
    """
    Calculates the spatial derivatives of V at an index for each dimension

    Args:
        index:
        V:
        g:
        periodic_dims:

    Returns:
        List of left and right spatial derivatives for each dimension

    """
    # print('idx:' ,index)
    spa_derivatives = []
    for dim, idx in enumerate(index):
        if dim == 0:
            left_index = []
        else:
            left_index = list(index[:dim])

        if dim == len(index) - 1:
            right_index = []
        else:
            right_index = list(index[dim + 1 :])

        next_index = tuple(left_index + [index[dim] + 1] + right_index)
        prev_index = tuple(left_index + [index[dim] - 1] + right_index)

        if idx == 0:
            if dim in periodic_dims:
                left_periodic_boundary_index = tuple(
                    left_index + [V.shape[dim] - 1] + right_index
                )
                left_boundary = V[left_periodic_boundary_index]
            else:
                left_boundary = V[index] + np.abs(V[next_index] - V[index]) * np.sign(
                    V[index]
                )
            left_deriv = (V[index] - left_boundary) / g.dx[dim]
            right_deriv = (V[next_index] - V[index]) / g.dx[dim]
        elif idx == V.shape[dim] - 1:
            if dim in periodic_dims:
                right_periodic_boundary_index = tuple(left_index + [0] + right_index)
                right_boundary = V[right_periodic_boundary_index]
            else:
                right_boundary = V[index] + np.abs(V[index] - V[prev_index]) * np.sign(
                    V[index]
                )
            left_deriv = (V[index] - V[prev_index]) / g.dx[dim]
            right_deriv = (right_boundary - V[index]) / g.dx[dim]
        else:
            left_deriv = (V[index] - V[prev_index]) / g.dx[dim]
            right_deriv = (V[next_index] - V[index]) / g.dx[dim]

        spa_derivatives.append((left_deriv + right_deriv) / 2)

    return np.array(spa_derivatives)


class SingleNarrowPassage:
    def __init__(self, x=[0,0,0,0,0], alpha_max = 2.0, alpha_min=-4.0, psi_max = np.pi/6, psi_min= -np.pi/6, length=2.0, u_mode="min", d_mode="max"):
        self.x = x
        self.alpha_max = alpha_max
        self.alpha_min = alpha_min
        self.psi_max = psi_max
        self.psi_min = psi_min
        self.length = length

        if u_mode == 'min' and d_mode == 'max':
            print('Control for reaching target set')
        elif u_mode =='max' and d_mode == 'min':
            print('Control for avoiding target set')
        else:
            raise ValueError(f'u_mode: {u_mode} and d_mode: {d_mode} are not opposite of each other!')

        self.u_mode = u_mode
        self.d_mode = d_mode

    def opt_ctrl(self, t, state, spat_deriv):
        """_summary_

        Args:
            t (_type_): _description_
            state (_type_): _description_
            spat_deriv (_type_): _description_

        Returns:
            u_alpha, u_psi: scalar
        """
        if self.u_mode == "max":
            if spat_deriv[3] >= 0:
                opt_u_alpha = self.alpha_max
            else:
                opt_u_alpha = self.alpha_min
        else: # u_mode == 'min
            if spat_deriv[3] > 0: # to match sign for deepreach
                opt_u_alpha = self.alpha_min
            else:
                opt_u_alpha = self.alpha_max

        if self.u_mode == "max":
            if spat_deriv[4] >= 0:
                opt_u_psi = self.psi_max
            else: 
                opt_u_psi = self.psi_min
        else: # u_mode == 'min'
            if spat_deriv[4] > 0:
                opt_u_psi = -self.psi_min
            else:
                opt_u_psi = self.psi_max

        return np.array([opt_u_alpha, opt_u_psi])

    def opt_dstb(self, t, state, spat_deriv):
        return np.zeros(5)

    def dynamics(self, t, state, u_opt, d_opt=np.zeros(5)):
        """
        u_opt[0] = alpha
        u_opt[1] = psi
        """
        """
        \dot{x_1} = x_4 \sin(x_3) # x
        \dot{x_2} = x_4 \cos(x_3) # y
        \dot{x_3} = x_4 \tan(x_5) / L # theta (heading)
        \dot{x_4} = a  # theta  # v 
        \dot{x_5} = psi # phi (steering angle)
        """
        x0_dot = state[3] * np.cos(state[2])
        x1_dot = state[3] * np.sin(state[2])
        x2_dot = state[3] * np.tan(state[4]) / self.length
        x3_dot = u_opt[0]
        x4_dot = u_opt[1]

        return np.array([x0_dot, x1_dot, x2_dot, x3_dot, x4_dot])


class SingleNarrowPassageEnv(gym.Env):

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
        self,done_if_unsafe=True, use_disturbances=False, reach_avoid=True
    ) -> None:
        self.car = SingleNarrowPassage(u_mode='min', d_mode='max')
        self.done_if_unsafe = done_if_unsafe
        self.use_disturbances = use_disturbances
        self.use_reach_avoid = reach_avoid

        path = os.path.abspath(__file__)
        dir_path = os.path.dirname(path)
        if self.use_reach_avoid:
            self.car = SingleNarrowPassage(u_mode="min", d_mode="max")  # ra
            self.brt = np.load(
                os.path.join(dir_path, "assets/brts/single_narrow_passage_brt.npy")
            )

        self.state = None
        self.dt = 0.05

        self.action_space = Box(low=np.array([self.car.alpha_min, self.car.psi_min]), high=np.array([self.car.alpha_max, self.car.psi_max]), dtype=np.float32, shape=(2,)
        )

        self.observation_space = Box(
            low=brt_config.grid_low,
            high=brt_config.grid_high,
            shape=(5,),
            dtype=np.float32,
        )

        self.goal_location = brt_config.GOAL_POS 

        self.grid = brt_config.grid
        self.fig, self.ax = plt.subplots(figsize=(5, 5))

        self.hist = []

    def reset(self, seed=None):
        # self.hist.clear()
        # self.car.x = np.array(self.car.x, dtype=np.float32)
        self.car.x = np.array([-6, -1.5, 0, 1, 0.5])
        self.state = np.copy(self.car.x)
        # # self.hist.append(np.copy(self.state))
        return np.array(self.car.x)

    def step(self, action: np.array):
        if self.use_disturbances:
            self.car.x = (
                self.car.dynamics(0, self.car.x, action, disturbance=self.opt_dist())
                * self.dt
                + self.car.x
            )
        else:
            print(f'before: {self.car.x}')
            print(f'dyn:    {self.car.dynamics(0, self.car.x, action)}')
            self.car.x = self.car.dynamics(0, self.car.x, action) * self.dt + self.car.x
        self.car.x[2] = self.normalize_angle(self.car.x[2])
        self.car.x = np.clip(self.car.x, brt_config.grid_low, brt_config.grid_high)
        print(f'after:  {self.car.x}')

        reward = -np.linalg.norm(self.car.x[:2] - self.goal_location)


        done = False
        info = {}
        if self.collision_curb_or_bounds():
            print('colide curb or bounds')
            if self.done_if_unsafe:
                done = True
            elif self.penalize_unsafe:
                reward = self.min_reward * 2
            info["cost"] = 1
            info["safe"] = False
        elif self.collision_car():
            print('collision car')
            if self.done_if_unsafe:
                done = True
            elif self.penalize_unsafe:
                reward = self.min_reward * 2
            info["cost"] = 1
            info["safe"] = False
        elif self.near_goal():
            print('reach goal')
            done = True
            reward = 100.0
            info["reach_goal"] = True

        # cost is based on distance to obstacle
        # info["hj_value"] = self.grid.get_value(self.brt, self.car.x)

        self.state = np.copy(self.car.x)

        # self.hist.append(np.copy(self.state))

        return np.copy(self.state), reward, done, info

    def render(self, mode="human"):
        # self.state = np.array([0, 1, np.pi/2, 0, 0])
        self.ax.clear()

        # world  boundary
        self.ax.hlines(y=[-4.0, 4.0], xmin=[-8, -8], xmax=[8, 8], color="k")
        self.ax.vlines(x=[-8, 8], ymin=[-4.0, -4.0], ymax=[4.0, 4.0], color="k")

        # middle of road
        self.ax.hlines(y=[0], xmin=[-8], xmax=[8], color="k", linestyles='dashed', alpha=0.1)

        # curb
        # self.ax.hlines(y=[-2.8 + 0.5 * brt_config.L * 0.5, 2.8 - 0.5 * brt_config.L * 0.5], xmin=[-8, -8], xmax=[8, 8], color="y")
        self.ax.hlines(y=[brt_config.CURB_POSITION[0], brt_config.CURB_POSITION[1]], xmin=[-8, -8], xmax=[8, 8], color="y")

        robot = plt.Circle(self.state[:2],  radius=brt_config.L / 2, color="blue")
        self.ax.add_patch(robot)

        dir = self.state[:2] + 1 * np.array(
            [np.cos(self.state[2]), np.sin(self.state[2])]
        )
        self.ax.plot([self.state[0], dir[0]], [self.state[1], dir[1]], color="c")


        stranded_car = plt.Circle(
               brt_config.STRANDED_CAR_POS, radius=brt_config.L / 2, color='r' 
        )

        self.ax.add_patch(stranded_car)

        # stranded_car_unsafe = plt.Circle(
        #        brt_config.STRANDED_CAR_POS, radius=brt_config.L, color='r' , alpha=0.5
        # )
        # self.ax.add_patch(stranded_car_unsafe)

        goal = plt.Circle(
           brt_config.GOAL_POS, radius=brt_config.L, color='g', alpha=0.5
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
            X, Y, self.brt[:, :, index[2], index[3], index[4]].transpose(), levels=[0],
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

    def collision_curb_or_bounds(self):
        if not (brt_config.CURB_POSITION[0] + 0.5 * self.car.length <= self.state[1] <= brt_config.CURB_POSITION[1] - 0.5 * self.car.length):
            return True
        elif not (brt_config.grid_low[0] + 0.5 * self.car.length <= self.state[0] <= brt_config.grid_high[0] - 0.5 * self.car.length):
            return True
        return False

    def collision_car(self):
        return np.linalg.norm(self.state[:2] - brt_config.STRANDED_CAR_POS) <= self.car.length
 
    def near_goal(self):
        return (
            np.linalg.norm(self.goal_location[:2] - self.car.x[:2])
            <= self.car.length
        )

    def opt_ctrl(self):
        index = self.grid.get_index(self.state)
        spat_deriv = spa_deriv(index, self.brt, self.grid, periodic_dims=[2, 4])
        opt_ctrl = self.car.opt_ctrl(0, self.state, spat_deriv)
        return opt_ctrl

    def opt_dist(self):
        index = self.grid.get_index(self.state)
        spat_deriv = spa_deriv(index, self.brt, self.grid, periodic_dims=[2, 4])
        opt_dist = self.car.opt_dist(0, self.state, spat_deriv)
        return opt_dist

    def safe_ctrl(self):
        index = self.grid.get_index(self.state)
        spat_deriv = spa_deriv(index, self.brt, self.grid, periodic_dims=[2, 4])
        return self.car.safe_ctrl(0, self.state, spat_deriv)

    def use_opt_ctrl(self, threshold=0.3, threshold_ra=0.2):
        if self.use_reach_avoid:
            return self.grid.get_value(self.brt, self.state) > threshold_ra
        else:
            return self.grid.get_value(self.brt, self.state) < threshold

    def unsafe_ctrl(self):
        index = self.grid.get_index(self.state)
        spat_deriv = spa_deriv(index, self.brt, self.grid, periodic_dims=[2, 4])
        return self.car.unsafe_ctrl(0, self.state, spat_deriv)

    @property
    def min_reward(self):
        return -np.linalg.norm(
            self.goal_location[:2] - np.array([4.5, -4.5])
        )  # -9.4069


if __name__ in "__main__":

    import time
    from atu.wrappers import RecordEpisodeStatisticsWithCost
    import atu
    from gym.wrappers import TransformObservation
    import gym

    # env = gym.make("Safe-DubinsHallway-v1", use_reach_avoid=False)
    # env = TransformObservation(env, lambda obs: obs / env.world_boundary)
    # env = RecordEpisodeStatisticsWithCost(env)
    env = SingleNarrowPassageEnv(use_disturbances=False)
    obs = env.reset()
    env.render('human')
    done = False
    while not done:
        env.render()
        opt_ctrl = env.opt_ctrl()
        obs, reward, done, info = env.step([1.0,0.5])
        # obs, reward, done, info = env.step(env.action_space.sample())
        # time.sleep(0.4)
