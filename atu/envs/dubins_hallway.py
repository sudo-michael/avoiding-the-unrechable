import gym
import os
import pygame
from gym.spaces import Box
from atu.optimized_dp.Grid.GridProcessing import Grid
from atu.optimized_dp.dubin_hallway import g as grid
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


class DubinsCar:
    def __init__(
        self,
        x_0=np.array([0, 0, 0]),
        w_max=1,
        speed=1,
        d_max=[0, 0, 0],
        u_mode="min",
        d_mode="max",
    ):
        self.x = x_0
        self.w_max = w_max  # turn rate
        self.speed = speed
        self.d_max = d_max
        self.u_mode = u_mode
        self.d_mode = d_mode
        self.r = 0.3

    def opt_ctrl(self, t, state, spat_deriv):
        opt_w = self.w_max
        if spat_deriv[2] > 0:
            if self.u_mode == "min":
                opt_w = -opt_w
        elif spat_deriv[2] < 0:
            if self.u_mode == "max":
                opt_w = -opt_w
        return np.array([opt_w])

    def dynamics(self, t, state, u_opt: np.array):
        x_dot = self.speed * np.cos(state[2])
        y_dot = self.speed * np.sin(state[2])
        theta_dot = u_opt[0]

        return np.array([x_dot, y_dot, theta_dot], dtype=np.float32)


class DubinsHallwayEnv(gym.Env):

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, use_reach_avoid=True) -> None:
        self.car = DubinsCar()
        self.goal = 0
        self.screen = None
        self.clock = None
        self.isopen = True
        self.use_reach_avoid = use_reach_avoid

        path = os.path.abspath(__file__)
        dir_path = os.path.dirname(path)
        if self.use_reach_avoid:
            self.car = DubinsCar(u_mode="min", d_mode="max")  # ra
            self.brt = np.load(
                os.path.join(dir_path, "assets/brts/reach_avoid_hallway.npy")
            )
        else:
            self.car = DubinsCar(u_mode="max", d_mode="min")  # avoid obstacle
            self.brt = np.load(
                os.path.join(dir_path, "assets/brts/max_over_min_brt.npy")
            )

        self.dt = 0.05

        self.action_space = Box(
            low=-self.car.w_max, high=self.car.w_max, dtype=np.float32, shape=(1,)
        )

        self.word_boundary = np.array([4.5, 4.5, np.pi], dtype=np.float32)

        self.observation_space = Box(
            low=-self.word_boundary,
            high=self.word_boundary,
            shape=(3,),
            dtype=np.float32,
        )

        self.goal_location = np.array([-2, 2.3, 0.5])  # x y r
        self.obstacle_location = np.array([-4.5, -0.5, 6.5, 1.0])  # x y w h

        self.world_width = 10
        self.world_height = 10

        self.left_wall = -4.5
        self.right_wall = 4.5
        self.bottom_wall = -4.5
        self.top_wall = 4.5

        self.grid = grid
        # self.brt = np.load("./atu/envs/brt.npy")

    def reset(self, seed=None):
        # self.car.x = np.array([-1.5, 2.5, -np.pi / 2])
        # self.car.x = np.array([-1.5, -1.5, np.pi / 2])
        # self.car.x = np.array([-3, -2, np.pi / 2])
        # return self.car.x
        while True:
            self.car.x = np.random.uniform(
                low=-self.word_boundary, high=self.word_boundary
            )

            if self.use_reach_avoid:
                if self.grid.get_value(self.brt, self.car.x) < 0.0:
                    break
            else:
                if self.grid.get_value(self.brt, self.car.x) > 0.05:
                    break

        return np.array(self.car.x)

    def step(self, action: np.array):
        self.car.x = self.car.dynamics(0, self.car.x, action) * self.dt + self.car.x
        self.car.x[0] = min(
            max(self.left_wall + self.car.r, self.car.x[0]),
            self.right_wall - self.car.r,
        )
        self.car.x[1] = min(
            max(self.bottom_wall + self.car.r, self.car.x[1]),
            self.top_wall - self.car.r,
        )
        self.car.x[2] = self.normalize_angle(self.car.x[2])

        done = False
        if self.collision_rect_circle(
            self.obstacle_location[0],
            self.obstacle_location[1],
            self.obstacle_location[2],
            self.obstacle_location[3],
            self.car.x[0],
            self.car.x[1],
            self.car.r,
        ):
            done = True
        elif self.near_goal():
            done = True

        # calculate reward
        reward = -np.linalg.norm(self.car.x[:2] - self.goal_location[:2])

        info = {}
        info["V_brt"] = self.grid.get_value(self.brt, self.car.x)
        # print(self.grid.get_value(self.brt, self.car.x))

        next_state = self.car.x

        return next_state, reward, done, info

    def render(self, mode="human"):
        try:
            import pygame
            from pygame import gfxdraw
        except ImportError:
            raise gym.DependencyNotInstalled(
                "pygame is not installed, run `pip install gym[classic_control]`"
            )

        screen_width = 600
        screen_height = 600

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
        gfxdraw.hline(self.surf, ww2sw(-4.5), ww2sw(4.5), wh2sh(-4.5), (0, 0, 0))
        gfxdraw.hline(self.surf, ww2sw(-4.5), ww2sw(4.5), wh2sh(4.5), (0, 0, 0))
        gfxdraw.vline(self.surf, ww2sw(-4.5), ww2sw(-4.5), wh2sh(4.5), (0, 0, 0))
        gfxdraw.vline(self.surf, ww2sw(4.5), ww2sw(-4.5), wh2sh(4.5), (0, 0, 0))

        gfxdraw.filled_circle(
            self.surf,
            ww2sw(self.goal_location[0]),
            wh2sh(self.goal_location[1]),
            world_to_screen(self.goal_location[2]),
            (0, 255, 0),
        )

        obstacle = pygame.Rect(
            ww2sw(self.obstacle_location[0]),
            wh2sh(self.obstacle_location[1]),
            world_to_screen(self.obstacle_location[2]),
            world_to_screen(self.obstacle_location[3]),
        )

        gfxdraw.box(self.surf, obstacle, (255, 0, 0))

        gfxdraw.filled_circle(
            self.surf,
            ww2sw(self.car.x[0]),
            wh2sh(self.car.x[1]),
            world_to_screen(self.car.r),
            (0, 0, 255),
        )

        gfxdraw.line(
            self.surf,
            ww2sw(self.car.x[0]),
            wh2sh(self.car.x[1]),
            ww2sw(np.cos(self.car.x[2]) * self.car.r + self.car.x[0]),
            wh2sh(np.sin(self.car.x[2]) * self.car.r + self.car.x[1]),
            (255, 255, 255),
        )

        # index = self.grid.get_index(self.car.x)
        # # fix theta
        # brt_slice = self.brt[:, :, index[2]]

        # # get all points that are greater than 0
        # index = np.argwhere(brt_slice >= 0)

        # def to_screen(index):
        #     point = self.grid.index_to_grid_point(index)
        #     print(point)
        #     return [ww2sw(point[0]), wh2sh(point[1])]

        # # convert to screenspace
        # points = np.array(list(map(to_screen, index)))
        # print(len(points))
        # print(points)
        # # pygame.draw.polygon(self.surf, points, (0, 255, 0), width=1)

        # gfxdraw.polygon(self.surf, points, (0, 255, 0))

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

    def collision_rect_circle(
        self, rleft, rtop, width, height, center_x, center_y, radius
    ):

        """
        Detect collision between a rectangle and circle.
        https://stackoverflow.com/a/54841116
        """

        rect = pygame.Rect(rleft, rtop, width, height)
        if rect.collidepoint(center_x, center_y):
            return True

        center_point = pygame.math.Vector2(center_x, center_y)
        corner_pts = [rect.bottomleft, rect.topleft, rect.topright, rect.bottomright]
        if any(
            [
                p
                for p in corner_pts
                if pygame.math.Vector2(*p).distance_to(center_point) <= radius
            ]
        ):
            return True
        return False

    def near_goal(self):
        print(np.linalg.norm(self.goal_location[:2] - self.car.x[:2]))
        return (
            np.linalg.norm(self.goal_location[:2] - self.car.x[:2])
            <= self.goal_location[2] + self.car.r
        )


if __name__ in "__main__":

    import time

    env = DubinsHallwayEnv(use_reach_avoid=False)
    obs = env.reset(0)
    for i in range(500):
        env.render()
        value = env.grid.get_value(env.brt, obs)
        print(f"{i} {value=} {obs=}")
        if value <= 0.05:
            print("use opt")
            index = env.grid.get_index(obs)
            spat_deriv = spa_deriv(index, env.brt, env.grid, periodic_dims=[2])
            opt_ctrl = env.car.opt_ctrl(0, env.car.x, spat_deriv)
            next_obs, reward, done, info = env.step(opt_ctrl)
        # elif i % 2 == 0:
        #     next_state, reward, done, info = env.step(env.action_space.high)
        else:
            next_obs, reward, done, info = env.step(env.action_space.sample())

        obs = next_obs

        if done:
            print(f"done: {obs}")
            break

        time.sleep(0.05)
