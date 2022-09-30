import heterocl as hcl
import numpy as np

# https://easychair.org/publications/open/Dd8G
# do compute, check with heleprOc


class DubinsCarCAvoid:
    def __init__(
        self,
        x=[0, 0, 0],
        wMaxA=1,
        wMaxB=1,
        vA=1,
        vB=1,
        uMode="max",
        dMode="min",
        dist=[0, 0, 0, 0, 0],
    ):
        # u = [vA, wA]
        # d = [vB, wB, d1, d2, d3]
        # TODO modify solver
        self.x = x
        self.wMaxA = wMaxA
        self.wMaxB = wMaxB
        self.vA = vA
        self.vB = vB
        self.uMode = uMode
        self.dMode = dMode

        self.dMaxA = [0, 0]
        self.dMaxB = [0, 0]

    def opt_ctrl(self, t, state, spat_deriv):
        opt_vA = hcl.scalar(self.vA, "opt_vA")
        opt_wA = hcl.scalar(self.wMaxB, "opt_wA")
        opt_none = hcl.scalar(0, "opt_none")
        det_0 = hcl.scalar(0, "det_0")
        det_1 = hcl.scalar(0, "det_1")

        det_0[0] = -spat_deriv[0]
        det_1[0] = spat_deriv[0] * state[1] - spat_deriv[1] * state[0] - spat_deriv[2]

        with hcl.if_(self.uMode == "max"):
            with hcl.if_(det_0 >= 0):
                opt_vA[0] = self.vA
            with hcl.else_():
                opt_vA[0] = -self.vA
        with hcl.if_(self.uMode == "min"):
            with hcl.if_(det_0 >= 0):
                opt_vA[0] = -self.vA
            with hcl.else_():
                opt_vA[0] = self.vA

        with hcl.if_(self.uMode == "max"):
            with hcl.if_(det_1 >= 0):
                opt_wA[0] = self.wMaxA
            with hcl.else_():
                opt_wA[0] = -self.wMaxA
        with hcl.if_(self.uMode == "min"):
            with hcl.if_(det_1 >= 0):
                opt_wA[0] = -self.wMaxA
            with hcl.else_():
                opt_wA[0] = self.wMaxA

        return opt_vA, opt_wA, opt_none

    def opt_dstb(self, t, state, spat_deriv):
        """
        :param spat_deriv: tuple of spatial derivative in all dimensions
        :return: a tuple of optimal disturbances
        """
        # Graph takes in 4 possible inputs, by default, for now
        d0 = hcl.scalar(0, "d0")
        d1 = hcl.scalar(0, "d1")
        d2 = hcl.scalar(0, "d2")
        d3 = hcl.scalar(0, "d3")
        d4 = hcl.scalar(0, "d4")

        det0 = hcl.scalar(0, "det0")
        det1 = hcl.scalar(0, "det1")
        det4 = hcl.scalar(0, "det4")

        det0[0] = spat_deriv[0] * hcl.cos(state[2]) + spat_deriv[1] * hcl.sin(state[2])
        det1[0] = spat_deriv[2]
        det4[0] = spat_deriv[2]

        with hcl.if_(self.uMode == "max"):
            with hcl.if_(det0 >= 0):
                d0[0] = self.vA
            with hcl.else_():
                d0[0] = -self.vA
        with hcl.if_(self.uMode == "min"):
            with hcl.if_(det0 >= 0):
                d0[0] = -self.vA
            with hcl.else_():
                d0[0] = self.vA

        with hcl.if_(self.uMode == "max"):
            with hcl.if_(det1 >= 0):
                d1[0] = self.wMaxB
            with hcl.else_():
                d1[0] = -self.wmaxB
        with hcl.if_(self.uMode == "min"):
            with hcl.if_(det1 >= 0):
                d1[0] = -self.wMaxB
            with hcl.else_():
                d1[0] = self.wMaxB

        with hcl.if_(self.uMode == "max"):
            with hcl.if_(det4 >= 0):
                d4[0] = self.dMaxA[1] + self.dMaxB[1]
            with hcl.else_():
                d4[0] = -self.dMaxA[1] - self.dMaxB[1]
        with hcl.if_(self.uMode == "min"):
            with hcl.if_(det4 >= 0):
                d4[0] = -self.dMaxA[1] - self.dMaxB[1]
            with hcl.else_():
                d4[0] = self.dMaxA[1] + self.dMaxB[1]

        if self.dMode == "max":
            s = 1
        else:
            s = -1

        demon = hcl.sqrt(spat_deriv[0] ** 2 + spat_deriv[1] ** 2)
        with hcl.if_(demon > 0):
            d2[0] = s * (self.dMaxA[0] + self.dMaxB[0]) * spat_deriv[0] / demon
            d3[0] = s * (self.dMaxA[0] + self.dMaxB[0]) * spat_deriv[1] / demon

        return d0[0], d1[0], d2[0], d3[0], d4[0]

    def dynamics(self, t, state, uOpt, dOpt):
        # uOpt = [vA, wA]
        # dOpt = [vB, wB, d1, d2, d3]
        x_dot = hcl.scalar(0, "x_dot")
        y_dot = hcl.scalar(0, "y_dot")
        theta_dot = hcl.scalar(0, "theta_dot")

        x_dot[0] = -uOpt[0] + dOpt[0] * hcl.cos(state[2]) + uOpt[1] * state[1] + dOpt[2]
        y_dot[0] = dOpt[0] * hcl.sin(state[2]) - uOpt[1] * state[0] + dOpt[3]
        theta_dot[0] = dOpt[1] - uOpt[1] + dOpt[4]

        return x_dot[0], y_dot[0], theta_dot[0]
