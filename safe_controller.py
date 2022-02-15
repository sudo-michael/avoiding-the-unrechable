import numpy as np
from Grid.GridProcessing import Grid
from dynamics.DubinsCar import *
from spatialDerivatives.first_order_generic import spa_deriv


class SafeController:
    def __init__(self) -> None:
        self.dubins_car = DubinsCar(uMode="max", dMode="min")
        self.V = np.load("V_r1.15_grid101.npy")
        self.grid = Grid(
            np.array([-4.0, -4.0, -np.pi]),
            np.array([4.0, 4.0, np.pi]),
            3,
            np.array(self.V.shape),
            [2],
        )

    def opt_ctrl(self, state):
        spa_derivatives = spa_deriv(
            self.grid.get_index(state), self.V, self.grid, periodic_dims=[2]
        )
        opt_w = self.dubins_car.wMax
        if spa_derivatives[2] > 0:
            if self.dubins_car.uMode == "min":
                opt_w = -opt_w
        elif spa_derivatives[2] < 0:
            if self.dubins_car.uMode == "max":
                opt_w = -opt_w
        return opt_w

    def is_safe(self, state, threshold=0.26):
        if self.grid.get_value(self.V, state) < threshold:
            return False, np.array([self.opt_ctrl(state)])
        return True, None

    def safety_layer(self, state, action):
        """
        Given action at current state, determine if action should be replaced
        with action from optimal control

        Return (True, action) := original action is safe enough
        Return (False, optimal_action) := original action is not safe
        """
        is_safe, optimal_action = self.is_safe(state)

        if is_safe:
            return True, action
        else:
            return False, optimal_action


# dubins_car = DubinsCar(uMode="max", dMode="min")
# V = np.load("V_r1.15_grid101.npy")
# g = Grid(
#     np.array([-4.0, -4.0, -np.pi]),
#     np.array([4.0, 4.0, np.pi]),
#     3,
#     np.array(V.shape),
#     [2],
# )


# def opt_ctrl(state):
#     spa_derivatives = spa_deriv(g.get_index(state), V, g, periodic_dims=[2])
#     opt_w = dubins_car.wMax
#     if spa_derivatives[2] > 0:
#         if dubins_car.uMode == "min":
#             opt_w = -opt_w
#     elif spa_derivatives[2] < 0:
#         if dubins_car.uMode == "max":
#             opt_w = -opt_w
#     return opt_w


# def grad_V_dot_f(state, action=None):
#     spa_derivatives = spa_deriv(g.get_index(state), V, g, periodic_dims=[2])
#     # print('b: ', spa_derivatives[0] * np.cos(state[2]) + spa_derivatives[1] * np.sin(state[2]))
#     # print('m: ', spa_derivatives[2])
#     if not action:
#         return (
#             spa_derivatives[0] * np.cos(state[2])
#             + spa_derivatives[1] * np.sin(state[2])
#             + spa_derivatives[2] * opt_ctrl(state)
#         )
#     else:
#         return (
#             spa_derivatives[0] * np.cos(state[2])
#             + spa_derivatives[1] * np.sin(state[2])
#             + spa_derivatives[2] * action
#         )


# def least_restr(state, action):
#     spa_derivatives = spa_deriv(g.get_index(state), V, g, periodic_dims=[2])
#     b = spa_derivatives[0] * np.cos(state[2]) + spa_derivatives[1] * np.sin(state[2])
#     m = spa_derivatives[2]
#     if m * action + b >= 0:
#         return action

#     action = b / m
#     if b < 0:
#         action *= -1

#     # print('ac: ', action)
#     return action


# def is_safe(state):
#     if g.get_value(V, state) < 0.26:
#         action = np.array([opt_ctrl(state)])
#         return False, action
#     return True, None

# dubins_car = DubinsCar(uMode="max", dMode="min")
# V = np.load("V_r1.15_grid101.npy")
# g = Grid(
#     np.array([-4.0, -4.0, -np.pi]),
#     np.array([4.0, 4.0, np.pi]),
#     3,
#     np.array(V.shape),
#     [2],
# )


# def opt_ctrl(state):
#     spa_derivatives = spa_deriv(g.get_index(state), V, g, periodic_dims=[2])
#     opt_w = dubins_car.wMax
#     if spa_derivatives[2] > 0:
#         if dubins_car.uMode == "min":
#             opt_w = -opt_w
#     elif spa_derivatives[2] < 0:
#         if dubins_car.uMode == "max":
#             opt_w = -opt_w
#     return opt_w


# def grad_V_dot_f(state, action=None):
#     spa_derivatives = spa_deriv(g.get_index(state), V, g, periodic_dims=[2])
#     # print('b: ', spa_derivatives[0] * np.cos(state[2]) + spa_derivatives[1] * np.sin(state[2]))
#     # print('m: ', spa_derivatives[2])
#     if not action:
#         return (
#             spa_derivatives[0] * np.cos(state[2])
#             + spa_derivatives[1] * np.sin(state[2])
#             + spa_derivatives[2] * opt_ctrl(state)
#         )
#     else:
#         return (
#             spa_derivatives[0] * np.cos(state[2])
#             + spa_derivatives[1] * np.sin(state[2])
#             + spa_derivatives[2] * action
#         )


# def least_restr(state, action):
#     spa_derivatives = spa_deriv(g.get_index(state), V, g, periodic_dims=[2])
#     b = spa_derivatives[0] * np.cos(state[2]) + spa_derivatives[1] * np.sin(state[2])
#     m = spa_derivatives[2]
#     if m * action + b >= 0:
#         return action

#     action = b / m
#     if b < 0:
#         action *= -1

#     # print('ac: ', action)
#     return action


# def is_safe(state):
#     if g.get_value(V, state) < 0.26:
#         action = np.array([opt_ctrl(state)])
#         return False, action
#     return True, None


# dubins_car = DubinsCar(uMode="max", dMode="min")
# V = np.load("V_r1.15_grid101.npy")
# g = Grid(
#     np.array([-4.0, -4.0, -np.pi]),
#     np.array([4.0, 4.0, np.pi]),
#     3,
#     np.array(V.shape),
#     [2],
# )


# def opt_ctrl(state):
#     spa_derivatives = spa_deriv(g.get_index(state), V, g, periodic_dims=[2])
#     opt_w = dubins_car.wMax
#     if spa_derivatives[2] > 0:
#         if dubins_car.uMode == "min":
#             opt_w = -opt_w
#     elif spa_derivatives[2] < 0:
#         if dubins_car.uMode == "max":
#             opt_w = -opt_w
#     return opt_w


# def grad_V_dot_f(state, action=None):
#     spa_derivatives = spa_deriv(g.get_index(state), V, g, periodic_dims=[2])
#     # print('b: ', spa_derivatives[0] * np.cos(state[2]) + spa_derivatives[1] * np.sin(state[2]))
#     # print('m: ', spa_derivatives[2])
#     if not action:
#         return (
#             spa_derivatives[0] * np.cos(state[2])
#             + spa_derivatives[1] * np.sin(state[2])
#             + spa_derivatives[2] * opt_ctrl(state)
#         )
#     else:
#         return (
#             spa_derivatives[0] * np.cos(state[2])
#             + spa_derivatives[1] * np.sin(state[2])
#             + spa_derivatives[2] * action
#         )


# def least_restr(state, action):
#     spa_derivatives = spa_deriv(g.get_index(state), V, g, periodic_dims=[2])
#     b = spa_derivatives[0] * np.cos(state[2]) + spa_derivatives[1] * np.sin(state[2])
#     m = spa_derivatives[2]
#     if m * action + b >= 0:
#         return action

#     action = b / m
#     if b < 0:
#         action *= -1

#     # print('ac: ', action)
#     return action


# def is_safe(state):
#     if g.get_value(V, state) < 0.26:
#         action = np.array([opt_ctrl(state)])
#         return False, action
#     return True, None
