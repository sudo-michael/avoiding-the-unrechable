import gym
import numpy as np


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

    def opt_ctrl(self, t, state, spat_deriv):
        opt_w = self.w_max
        if spat_deriv[2] > 0:
            if self.u_mode == "min":
                opt_w = -opt_w
        elif spat_deriv[2] < 0:
            if self.u_mode == "max":
                opt_w = -opt_w
        return opt_w

    def dynamics(self, t, state, u_opt: np.array):
        x_dot = self.speed * np.cos(state[2])
        y_dot = self.speed * np.sin(state[2])
        theta_dot = u_opt[0]

        return np.array(x_dot, y_dot, theta_dot)


class DubinsHallwayEnv(gym.Env):

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self) -> None:
        self.car = DubinsCar()
        self.goal = 0
        self.screen = None
        self.clock = None
        self.isopen = True

        self.dt = 0.05

        self.car = DubinsCar(u_mode="max", d_mode="min")  # avoid obstacle

    def reset(self, seed=None):
        self.car.x = np.array([-2, -2, 0])
        self.state = self.car.x
        return 0

    def step(self, action: np.array):
        next_state = self.car.dynamics(0, self.state, action) + self.dt + self.state

        # check for things

        # calculate reward

        return None

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
        world_width = 10
        world_height = 10

        world_to_screen = lambda x: int(x * screen_height / world_height)

        def ww2sw(x):
            """ world width to screen width """
            return int(x * (screen_width / world_width) + screen_width // 2)

        def wh2sh(y):
            """ world height to screen height """
            return int(y * (screen_height / world_height) + screen_height // 2)

        if not self.screen:
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode((screen_width, screen_height))

        if not self.clock:
            self.clock = pygame.time.Clock()

        self.surf = pygame.Surface((screen_width, screen_height))
        self.surf.fill((255, 255, 255))

        gfxdraw.filled_circle(
            self.surf, ww2sw(-2), wh2sh(-2), world_to_screen(0.5), (0, 0, 255)
        )

        state = (-2, -2, np.pi / 3)
        x1 = state[0]
        y1 = state[0]
        r = 0.5
        x2 = np.cos(state[2]) * r + x1
        y2 = np.sin(state[2]) * r + y1

        print(x1, y1, x2, y2)
        print(list(map(world_to_screen, [x1, y1, x2, y2])))

        gfxdraw.line(
            self.surf, ww2sw(x1), wh2sh(y1), ww2sw(x2), wh2sh(y2), (255, 255, 255),
        )

        # boundary
        gfxdraw.hline(self.surf, ww2sw(-4.5), ww2sw(4.5), wh2sh(-4.5), (0, 0, 0))
        gfxdraw.hline(self.surf, ww2sw(-4.5), ww2sw(4.5), wh2sh(4.5), (0, 0, 0))
        gfxdraw.vline(self.surf, ww2sw(-4.5), ww2sw(-4.5), wh2sh(4.5), (0, 0, 0))
        gfxdraw.vline(self.surf, ww2sw(4.5), ww2sw(-4.5), wh2sh(4.5), (0, 0, 0))

        goal = pygame.Rect(
            ww2sw(-3), wh2sh(2), world_to_screen(1.2), world_to_screen(1.2)
        )
        gfxdraw.box(self.surf, goal, (0, 255, 0))

        obstacle = pygame.Rect(
            ww2sw(-4.5), wh2sh(-0.5), world_to_screen(6.5), world_to_screen(1)
        )
        gfxdraw.box(self.surf, obstacle, (255, 0, 0))

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
        """ normalize theta to be in range (-pi, pi]"""
        return ((-theta + np.pi) % (2.0 * np.pi) - np.pi) * -1.0


if __name__ in "__main__":
    import time

    env = DubinsHallwayEnv()
    env.reset()
    for _ in range(100):
        env.render()
        time.sleep(0.1)
