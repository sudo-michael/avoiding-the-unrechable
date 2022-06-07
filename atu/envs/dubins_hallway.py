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
        d_min=[0, 0, 0],
        d_max=[0, 0, 0],
        u_mode="min",
        d_mode="max",
    ):
        self.x = x_0
        self.w_max = w_max  # turn rate
        self.speed = speed
        self.d_min = d_min
        self.d_max = d_max
        self.u_mode = u_mode
        self.d_mode = d_mode
        self.r = 0.25  # barely touch lava sometimes

    def opt_ctrl(self, t, state, spat_deriv):
        opt_w = self.w_max
        if spat_deriv[2] > 0:
            if self.u_mode == "min":
                opt_w = -opt_w
        elif spat_deriv[2] < 0:
            if self.u_mode == "max":
                opt_w = -opt_w
        return np.array([opt_w])

    def safe_ctrl(self, t, state, spat_deriv, uniform_sample=True):
        opt_w = self.w_max
        if spat_deriv[2] > 0:
            if self.u_mode == "min":
                opt_w = -opt_w
        elif spat_deriv[2] < 0:
            if self.u_mode == "max":
                opt_w = -opt_w
        b = (self.speed * np.cos(state[2])) * spat_deriv[0] + (
            self.speed * np.sin(state[2])
        ) * spat_deriv[1]
        # print('gradVdotF: ', b + opt_w * spat_deriv[2])
        m = spat_deriv[2]
        x_intercept = -b / (m + 1e-5)
        if np.sign(m) == 1:
            w_upper = opt_w
            w_lower = max(x_intercept, -self.w_max)
        elif np.sign(m) == -1:
            w_upper = min(x_intercept, self.w_max)
            w_lower = -self.w_max
        else:
            w_lower = opt_w
            w_upper = opt_w
        if uniform_sample:
            return np.random.uniform(w_lower, w_upper, size=(1,))
        return np.array([w_lower, w_upper])

    def unsafe_ctrl(self, t, state, spat_deriv, uniform_sample=True):
        safe_ctrl_bounds = self.safe_ctrl(t, state, spat_deriv, uniform_sample=False)
        for _ in range(100):
            ctrl = np.random.uniform(-self.w_max, self.w_max)
            if not (safe_ctrl_bounds[0] <= ctrl <= safe_ctrl_bounds[1]):
                break
        return np.array([ctrl])

    def opt_dist(self, t, state, spat_deriv):
        d_opt = np.zeros(3)
        if self.d_mode == "max":
            for i in range(3):
                if spat_deriv[i] >= 0:
                    d_opt[i] = self.d_max[i]
                else:
                    d_opt[i] = self.d_min[i]
        elif self.d_mode == "min":
            for i in range(3):
                if spat_deriv[i] >= 0:
                    d_opt[i] == self.d_min[i]
                else:
                    d_opt[i] = self.d_max[i]
        return d_opt

    def dynamics(self, t, state, u_opt: np.array, disturbance: np.array = np.zeros(3)):
        x_dot = self.speed * np.cos(state[2]) + disturbance[0]
        y_dot = self.speed * np.sin(state[2]) + disturbance[1]
        theta_dot = u_opt[0] + disturbance[2]

        return np.array([x_dot, y_dot, theta_dot], dtype=np.float32)


class DubinsHallwayEnv(gym.Env):

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
        self,
        use_reach_avoid=True,
        done_if_unsafe=False,
        use_disturbances=False,
        goal_location=np.array([-2, 2.3, 0.5]),
    ) -> None:
        self.car = DubinsCar()
        self.goal = 0
        self.screen = None
        self.clock = None
        self.isopen = True
        self.used_hj = False
        self.use_reach_avoid = use_reach_avoid
        self.done_if_unsafe = done_if_unsafe
        self.use_disturbances = use_disturbances

        path = os.path.abspath(__file__)
        dir_path = os.path.dirname(path)
        if self.use_reach_avoid:
            self.car = DubinsCar(u_mode="min", d_mode="max")  # ra
            print("using reach avoid brt")
            self.brt = np.load(
                os.path.join(dir_path, "assets/brts/reach_avoid_hallway.npy")
            )
        else:
            self.car = DubinsCar(u_mode="max", d_mode="min")  # avoid obstacle
            print("using max over min brt")
            self.brt = np.load(
                os.path.join(dir_path, "assets/brts/max_over_min_brt.npy")
            )

        self.min_brt = np.load(
            os.path.join(dir_path, "assets/brts/max_over_min_brt.npy")
        )

        self.state = None
        self.dt = 0.05

        self.action_space = Box(
            low=-self.car.w_max, high=self.car.w_max, dtype=np.float32, shape=(1,)
        )

        self.world_boundary = np.array([4.5, 4.5, np.pi], dtype=np.float32)

        self.observation_space = Box(
            low=-self.world_boundary,
            high=self.world_boundary,
            shape=(3,),
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

    def reset(self, seed=None):
        self.use_hj = False
        while True:
            self.car.x = np.random.uniform(
                low=-self.world_boundary,
                high=self.world_boundary
                # low=-self.world_boundary, high=np.array([4.5, -0.5, np.pi]) # reset in bottom half of map
            )

            if (
                self.grid.get_value(self.min_brt, self.car.x) > 0.1
            ):  # place car in location that is always possible to avoid obstacle
                break

        self.state = self.car.x
        print(f"intial state: {self.state}")
        return np.array(self.car.x)

    def step(self, action: np.array):
        if isinstance(action, dict):
            used_hj = action["used_hj"]
            action = action["action"]
        else:
            used_hj = False
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
        info = {}
        # info["cost"] = 0
        info["cost"] = used_hj
        info["safe"] = True
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
            info["cost"] = 1
            info["safe"] = False
        elif self.near_goal():
            done = True
            info["reach_goal"] = True

        # calculate reward
        reward = -np.linalg.norm(self.car.x[:2] - self.goal_location[:2])

        # cost is based on distance to obstacle
        info["hj_value"] = self.grid.get_value(self.brt, self.car.x)

        next_state = self.car.x

        self.state = self.car.x

        return next_state, reward, done, info

    def simulate_step(self, state: np.array, action: np.array):
        """ assume state is [x, y, theta] """
        next_state = self.car.dynamics(0, self.car.x, action) * self.dt + state
        next_state[0] = min(
            max(self.left_wall + self.car.r, next_state[0]),
            self.right_wall - self.car.r,
        )
        next_state[1] = min(
            max(self.bottom_wall + self.car.r, next_state[1]),
            self.top_wall - self.car.r,
        )
        next_state[2] = self.normalize_angle(next_state[2])

        done = False
        info = {}
        info["cost"] = 0
        info["safe"] = True
        if self.collision_rect_circle(
            self.obstacle_location[0],
            self.obstacle_location[1],
            self.obstacle_location[2],
            self.obstacle_location[3],
            next_state[0],
            next_state[1],
            self.car.r,
        ):
            if self.done_if_unsafe:
                done = True
            info["cost"] = 1
            info["safe"] = False
        elif self.near_goal():
            done = True
            info["reach_goal"] = True

        # calculate reward
        reward = -np.linalg.norm(next_state[:2] - self.goal_location[:2])

        # cost is based on distance to obstacle
        info["hj_value"] = self.grid.get_value(self.brt, next_state)

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
            (255, 255, 0) if self.used_hj else (0, 0, 255),
        )

        gfxdraw.line(
            self.surf,
            ww2sw(self.car.x[0]),
            wh2sh(self.car.x[1]),
            ww2sw(np.cos(self.car.x[2]) * self.car.r + self.car.x[0]),
            wh2sh(np.sin(self.car.x[2]) * self.car.r + self.car.x[1]),
            (255, 255, 255),
        )

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

        return dx ** 2 + dy ** 2 <= radius ** 2

    def near_goal(self):
        return (
            np.linalg.norm(self.goal_location[:2] - self.car.x[:2])
            <= self.goal_location[2] + self.car.r
        )

    def opt_ctrl(self):
        index = self.grid.get_index(self.state)
        spat_deriv = spa_deriv(index, self.brt, self.grid, periodic_dims=[2])
        opt_ctrl = self.car.opt_ctrl(0, self.state, spat_deriv)
        return opt_ctrl

    def safe_ctrl(self):
        index = self.grid.get_index(self.state)
        spat_deriv = spa_deriv(index, self.brt, self.grid, periodic_dims=[2])
        return self.car.safe_ctrl(0, self.state, spat_deriv)

    def use_opt_ctrl(self, threshold=0.2, threshold_ra=0.1):
        if self.use_reach_avoid:
            return self.grid.get_value(self.brt, self.state) > threshold_ra
        else:
            return self.grid.get_value(self.min_brt, self.state) < threshold

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

    env = gym.make("Safe-DubinsHallway-v1", use_reach_avoid=True)
    env = TransformObservation(env, lambda obs: obs / env.world_boundary)
    env = RecordEpisodeStatisticsWithCost(env)
    obs = env.reset()
    done = False
    while not done:
        # env.render()
        # action = env.action_space.sample()
        # if env.use_opt_ctrl():
        action = env.opt_ctrl()
        next_obs, reward, done, info = env.step(action)
        time.sleep(0.03)
        print(info["hj_value"])
    print(info)
