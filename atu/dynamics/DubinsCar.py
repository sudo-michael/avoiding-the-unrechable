# TODO based on: odp
import heterocl as hcl
import numpy as np
import time


class DubinsCar:
    def __init__(
        self,
        x=[0, 0, 0],
        wMax=1.5,
        speed=1,
        dMin=[0, 0, 0],
        dMax=[0, 0, 0],
        uMode="min",
        dMode="max",
        r=0.2,
    ):
        self.x = x
        self.wMax = wMax
        self.speed = speed
        self.dMin = dMin
        self.dMax = dMax
        self.uMode = uMode
        self.dMode = dMode

        self.r = 0.2

    def opt_ctrl(self, t, state, spat_deriv):
        opt_w = hcl.scalar(self.wMax, "opt_w")
        # Just create and pass back, even though they're not used
        in3 = hcl.scalar(0, "in3")
        in4 = hcl.scalar(0, "in4")
        with hcl.if_(spat_deriv[2] > 0):
            with hcl.if_(self.uMode == "min"):
                opt_w[0] = -opt_w
        with hcl.elif_(spat_deriv[2] < 0):
            with hcl.if_(self.uMode == "max"):
                opt_w[0] = -opt_w
        return (opt_w[0], in3[0], in4[0])

    def opt_dstb(self, t, state, spat_deriv):
        """
        :param spat_deriv: tuple of spatial derivative in all dimensions
        :return: a tuple of optimal disturbances
        """
        # Graph takes in 4 possible inputs, by default, for now
        d1 = hcl.scalar(self.dMax[0], "d1")
        d2 = hcl.scalar(self.dMax[1], "d2")
        d3 = hcl.scalar(self.dMax[2], "d3")

        with hcl.if_(self.dMode == "max"):
            with hcl.if_(spat_deriv[0] >= 0):
                d1[0] = self.dMax[0]
            with hcl.else_():
                d1[0] = self.dMin[0]
            with hcl.if_(spat_deriv[1] >= 0):
                d2[0] = self.dMax[1]
            with hcl.else_():
                d2[0] = self.dMin[1]
            with hcl.if_(spat_deriv[2] >= 0):
                d3[0] = self.dMax[2]
            with hcl.else_():
                d3[0] = self.dMin[2]
        with hcl.else_():
            with hcl.if_(spat_deriv[0] >= 0):
                d1[0] = self.dMin[0]
            with hcl.else_():
                d1[0] = self.dMax[0]
            with hcl.if_(spat_deriv[1] >= 0):
                d2[0] = self.dMin[1]
            with hcl.else_():
                d2[0] = self.dMax[1]
            with hcl.if_(spat_deriv[2] >= 0):
                d3[0] = self.dMin[2]
            with hcl.else_():
                d3[0] = self.dMax[2]
        return d1, d2, d3

    def dynamics(self, t, state, uOpt, dOpt):
        x_dot = hcl.scalar(0, "x_dot")
        y_dot = hcl.scalar(0, "y_dot")
        theta_dot = hcl.scalar(0, "theta_dot")

        x_dot[0] = self.speed * hcl.cos(state[2]) + dOpt[0]
        y_dot[0] = self.speed * hcl.sin(state[2]) + dOpt[1]
        theta_dot[0] = uOpt[0] + dOpt[2]

        return (x_dot[0], y_dot[0], theta_dot[0])

    def opt_ctrl_non_hcl(self, t, state, spat_deriv):
        opt_w = None
        if spat_deriv[2] >= 0:
            if self.uMode == "max":
                opt_w = self.wMax
            else:
                opt_w = -self.wMax
        elif spat_deriv[2] <= 0:
            if self.uMode == "max":
                opt_w = -self.wMax
            else:
                opt_w = self.wMax

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

    def opt_dist_non_hcl(self, t, state, spat_deriv):
        d_opt = np.zeros(3)
        if self.dMode == "max":
            for i in range(3):
                if spat_deriv[i] >= 0:
                    d_opt[i] = self.dMax[i]
                else:
                    d_opt[i] = self.dMin[i]
        elif self.dMode == "min":
            for i in range(3):
                if spat_deriv[i] >= 0:
                    d_opt[i] = self.dMin[i]
                else:
                    d_opt[i] = self.dMax[i]
        return d_opt

    def dynamics_non_hcl(self, t, state, u_opt, disturbance=np.zeros(3)):
        x_dot = self.speed * np.cos(state[2]) + disturbance[0]
        y_dot = self.speed * np.sin(state[2]) + disturbance[1]
        theta_dot = u_opt[0] + disturbance[2] / 3
        return np.array([x_dot, y_dot, theta_dot], dtype=np.float32)
