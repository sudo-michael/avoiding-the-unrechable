import heterocl as hcl
import numpy as np


class Air3D:
    def __init__(self, we_max=1, wp_max=1, ve=1, vp=1, r=1, u_mode="max", d_mode="min") -> None:
        # state = [x_e, y_e, theta_e, x_p, y_p, theta_p]
        self.ve = ve
        self.vp = vp
        self.we_max = we_max
        self.wp_max = wp_max
        self.r = r

        self.u_mode = u_mode
        self.d_mode = d_mode

    def opt_ctrl(self, t, state, spat_deriv):
        opt_we = hcl.scalar(self.we_max, "opt_we")
        in2 = hcl.scalar(0, "in2")
        in3 = hcl.scalar(0, "in3")

        det = hcl.scalar(0, "det")
        det[0] = spat_deriv[0] * state[1] - spat_deriv[1] * state[0] - spat_deriv[2]

        with hcl.if_(self.u_mode == "max"):
            with hcl.if_(det[0] >= 0):
                opt_we[0] = self.we_max
            with hcl.else_():
                opt_we[0] = -1 * self.we_max
        with hcl.else_():
            with hcl.if_(det[0] >= 0):
                opt_we[0] = -1 * self.we_max
            with hcl.else_():
                opt_we[0] = self.we_max

        return opt_we, in2, in3

    def opt_dstb(self, t, state, spat_deriv):
        opt_wp = hcl.scalar(self.wp_max, "opt_wp")
        in2 = hcl.scalar(0, "in2")
        in3 = hcl.scalar(0, "in3")

        with hcl.if_(self.d_mode == "max"):
            with hcl.if_(spat_deriv[2] >= 0):
                opt_wp[0] = self.wp_max
            with hcl.else_():
                opt_wp[0] = -1 * self.wp_max
        with hcl.else_():
            with hcl.if_(spat_deriv[2] >= 0):
                opt_wp[0] = -1 * self.wp_max
            with hcl.else_():
                opt_wp[0] = self.wp_max

        return opt_wp, in2, in3

    def dynamics(self, t, state, u_opt, d_opt):
        # u_opt[0] = omega_e
        # d_opt[0] = omega_p
        x1_dot = hcl.scalar(0, "x1_dot")
        x2_dot = hcl.scalar(0, "x2_dot")
        x3_dot = hcl.scalar(0, "x3_dot")

        x1_dot[0] = -self.ve + self.vp * hcl.cos(state[2]) + u_opt[0] * state[1]
        x2_dot[0] = self.vp * hcl.sin(state[2]) - d_opt[0] * state[0]
        x3_dot[0] = d_opt[0] - u_opt[0]

        return x1_dot, x2_dot, x3_dot

    def opt_ctrl_non_hcl(self, state, spat_deriv):
        opt_we = 0
        det = spat_deriv[0] * state[1] - spat_deriv[1] * state[0] - spat_deriv[2]

        if self.u_mode == "max":
            if det >= 0:
                opt_we = self.we_max
            else:
                opt_we = -1 * self.we_max
        else:
            if det >= 0:
                opt_we = -1 * self.we_max
            with hcl.else_():
                opt_we = self.we_max

        return np.array([opt_we], dtype=np.float32)
        pass

    def opt_dstb_non_hcl(self, spat_deriv):
        opt_wp = 0
        if self.d_mode == "max":
            if spat_deriv[2] >= 0:
                opt_wp = self.wp_max
            else:
                opt_wp = -1 * self.wp_max
        else:
            if spat_deriv[2] >= 0:
                opt_wp = -1 * self.wp_max
            else:
                opt_wp = self.wp_max

        return np.array([opt_wp], dtype=np.float32)

    def dynamics_non_hcl(self, t, state, ctrl, is_evader=True):
        v = self.ve if is_evader else self.vp
        x_dot = v * np.cos(state[2])
        y_dot = v * np.sin(state[2])
        theta_dot = ctrl[0]
        return np.array([x_dot, y_dot, theta_dot], dtype=np.float32)

