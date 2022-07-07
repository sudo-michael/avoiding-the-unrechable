from asyncore import read
from tkinter import W
from turtle import pen
import gym
import os
import pygame
from gym.spaces import Box
from atu.optimized_dp.Grid.GridProcessing import Grid
import atu.optimized_dp.brt_single_narrow_passage as brt_config
import numpy as np
import math


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
                    [V[index]]
                )
            left_deriv = (V[index] - V[prev_index]) / g.dx[dim]
            right_deriv = (right_boundary - V[index]) / g.dx[dim]
        else:
            left_deriv = (V[index] - V[prev_index]) / g.dx[dim]
            right_deriv = (V[next_index] - V[index]) / g.dx[dim]

        spa_derivatives.append((left_deriv + right_deriv) / 2)

    return np.array(spa_derivatives)


class SingleNarrowPassage:
    def __init__(self, x=[0,0,0,0,0], alpha_max = 2.0, alpha_min=-4.0, psi_max = 3.0 * np.pi, psi_min= -3.0 * np.pi, length=2.0, u_mode="min", d_mode="max"):
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

        return opt_u_alpha, opt_u_psi

    def opt_dstb(self, t, state, spat_deriv):
        return np.zeros(5)

    def dynamics(self, t, state, u_opt, d_opt):
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
        x1_dot = state[3] * np.sin(state[2])
        x2_dot = state[3] * np.cos(state[2])
        x3_dot = state[3] * np.tan(4) / self.length
        x4_dot = u_opt[0]
        x5_dot = u_opt[1]

        return x1_dot, x2_dot, x3_dot, x4_dot, x5_dot


class SingleNarrowPassageEnv(gym.Env):

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
        self,done_if_unsafe=True, use_disturbances=False, reach_avoid=True
    ) -> None:
        self.car = SingleNarrowPassage(u_mode='min', d_mode='max')
        self.screen = None
        self.clock = None
        self.isopen = True
        self.done_if_unsafe = done_if_unsafe
        self.use_disturbances = use_disturbances
        self.use_reach_avoid = reach_avoid

        path = os.path.abspath(__file__)
        dir_path = os.path.dirname(path)
        # if self.use_reach_avoid:
        #     self.car = SingleNarrowPassage(u_mode="min", d_mode="max")  # ra
        #     self.ra = np.load(
        #         os.path.join(dir_path, "assets/brts/single_narrow_passage_ra.npy")
        #     )

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

        self.hist = []

    def reset(self, seed=None):
        # self.hist.clear()
        # self.car.x = np.array(self.car.x, dtype=np.float32)
        # self.state = self.car.x
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
            self.car.x = self.car.dynamics(0, self.car.x, action) * self.dt + self.car.x
        self.car.x[2] = self.normalize_angle(self.car.x[2])

        reward = -np.linalg.norm(self.car.x[:2] - self.goal_location)


        done = False
        info = {}
        if self.collision_curb_or_bounds():
            if self.done_if_unsafe:
                done = True
            elif self.penalize_unsafe:
                reward = self.min_reward * 2
            info["cost"] = 1
            info["safe"] = False
        elif self.collision_car():
            if self.done_if_unsafe:
                done = True
            elif self.penalize_unsafe:
                reward = self.min_reward * 2
            info["cost"] = 1
            info["safe"] = False
        elif self.near_goal():
            done = True
            reward = 100.0
            info["reach_goal"] = True

        # cost is based on distance to obstacle
        # info["hj_value"] = self.grid.get_value(self.brt, self.car.x)

        self.state = np.copy(self.car.x)

        # self.hist.append(np.copy(self.state))

        return np.copy(self.state), reward, done, info

    def render(self, mode="human"):
        try:
            import pygame
            from pygame import gfxdraw
        except ImportError:
            raise gym.DependencyNotInstalled(
                "pygame is not installed, run `pip install gym[classic_control]`"
            )

        screen_width = 400
        screen_height = 800

        self.world_height = 8
        self.world_width =  16

        world_to_screen = lambda x: int(x * screen_height / self.world_height)

        def ww2sw(x):
            """world width to screen width"""
            return int(x * (screen_width / self.world_width) + screen_width // 2)

        def wh2sh(y):
            """world height to screen height"""
            return int(y * (screen_height / self.world_height) + screen_height // 2)

        if not self.screen:
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode((screen_width, screen_height))

        if not self.clock:
            self.clock = pygame.time.Clock()

        self.surf = pygame.Surface((screen_width, screen_height))
        self.surf.fill((255, 255, 255))

        # boundary
        # gfxdraw.hline(self.surf, ww2sw(brt_config.grid_low), ww2sw(4.5), wh2sh(-4.5), (0, 0, 0))
        # gfxdraw.hline(self.surf, ww2sw(-4.5), ww2sw(4.5), wh2sh(4.5), (0, 0, 0))
        # gfxdraw.vline(self.surf, ww2sw(-4.5), ww2sw(-4.5), wh2sh(4.5), (0, 0, 0))
        # gfxdraw.vline(self.surf, ww2sw(4.5), ww2sw(-4.5), wh2sh(4.5), (0, 0, 0))

        gfxdraw.filled_circle(
            self.surf,
            ww2sw(self.goal_location[0]),
            wh2sh(self.goal_location[1]),
            world_to_screen(self.car.r),
            (0, 255, 0),
        )

        # obstacle = pygame.Rect(
        #     ww2sw(self.obstacle_location[0]),
        #     wh2sh(self.obstacle_location[1]),
        #     world_to_screen(self.obstacle_location[2]),
        #     world_to_screen(self.obstacle_location[3]),
        # )

        # gfxdraw.box(self.surf, obstacle, (255, 0, 0))

        # gfxdraw.filled_circle(
        #     self.surf,
        #     ww2sw(self.car.x[0]),
        #     wh2sh(self.car.x[1]),
        #     world_to_screen(self.car.r),
        #     (255, 255, 0) if self.used_hj else (0, 0, 255),
        # )

        # gfxdraw.line(
        #     self.surf,
        #     ww2sw(self.car.x[0]),
        #     wh2sh(self.car.x[1]),
        #     ww2sw(np.cos(self.car.x[2]) * self.car.r + self.car.x[0]),
        #     wh2sh(np.sin(self.car.x[2]) * self.car.r + self.car.x[1]),
        #     (255, 255, 255),
        # )

        self.surf = pygame.transform.flip(self.surf, False, True)
        self.screen.blit(self.surf, (0, 0))
        if mode == "human":
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()

        if mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )
        else:
            return self.isopen

    def close(self):
        if not self.screen:
            import pygame

            pygame.display.quit()
            pygame.quit()
            self.isopen = False

    def normalize_angle(self, theta):
        """normalize theta to be in range (-pi, pi]"""
        return ((-theta + np.pi) % (2.0 * np.pi) - np.pi) * -1.0

    def collision_curb_or_bounds(self):
        if not (brt_config.grid_low[1] + 0.5 * self.car.r <= self.state[1] <= brt_config.grid_high[1] - 0.5 * self.car.r):
            return True
        elif not (brt_config.grid_low[0] + self.car.r <= self.state[0] <= brt_config.grid_high[0] - self.car.r):
            return True
        return False

    def colision_car(self):
        return np.norm(self.state[:2] - brt_config.STRANDED_CAR_POS) <= self.car.r 
 
    def near_goal(self):
        return (
            np.linalg.norm(self.goal_location[:2] - self.car.x[:2])
            <= self.car.r
        )

    def opt_ctrl(self):
        index = self.grid.get_index(self.state)
        spat_deriv = spa_deriv(index, self.brt, self.grid, periodic_dims=[2])
        opt_ctrl = self.car.opt_ctrl(0, self.state, spat_deriv)
        return opt_ctrl

    def opt_dist(self):
        index = self.grid.get_index(self.state)
        spat_deriv = spa_deriv(index, self.brt, self.grid, periodic_dims=[2])
        opt_dist = self.car.opt_dist(0, self.state, spat_deriv)
        return opt_dist

    def safe_ctrl(self):
        index = self.grid.get_index(self.state)
        spat_deriv = spa_deriv(index, self.brt, self.grid, periodic_dims=[2])
        return self.car.safe_ctrl(0, self.state, spat_deriv)

    def use_opt_ctrl(self, threshold=0.3, threshold_ra=0.2):
        if self.use_reach_avoid:
            return self.grid.get_value(self.brt, self.state) > threshold_ra
        else:
            return self.grid.get_value(self.brt, self.state) < threshold

    def unsafe_ctrl(self):
        index = self.grid.get_index(self.state)
        spat_deriv = spa_deriv(index, self.brt, self.grid, periodic_dims=[2])
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
    env = SingleNarrowPassageEnv()
    obs = env.reset()
    done = False
    while not done:
        env.render()
        time.sleep(0.4)
