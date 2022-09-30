import heterocl as hcl
import numpy as np


class DubinsPersuitEvade():
    def __init__(self, x, we_max, wp_max, ve, vp, u_mode='max', d_mode='min') -> None:
        self.x = x # neeed?
        # state = [x_e, y_e, theta_e, x_p, y_p, theta_p]
        self.ve = ve
        self.vp = vp
        self.we_max = we_max
        self.wp_max = wp_max

        self.u_mode = u_mode
        self.d_mode = d_mode

    def opt_ctrl(self, t, state, spat_deriv):
        opt_w = hcl.scalar(self.we_max, "opt_w")
        in2 = hcl.scalar(0, "in2")
        in3 = hcl.scalar(0, "in3")
        in4 = hcl.scalar(0, "in4")
        in5 = hcl.scalar(0, "in5")
        in6 = hcl.scalar(0, "in6")
        
        with hcl.if_(self.u_mode == 'max'):
            with hcl.if_(spat_deriv[2] >= 0):
                opt_w[0] = self.we_max
            with hcl.else_():
                opt_w[0] = -self.we_max
        with hcl.else_():
            with hcl.if_(spat_deriv[2] >= 0):
                opt_w[0] = -self.we_max
            with hcl.else_():
                opt_w[0] = self.we_max
        
        # return opt_w, in2, in3, in4, in5, in6
        return opt_w, in2, in3, in4

    def opt_dstb(self, t, state, spat_deriv):
        opt_w = hcl.scalar(self.wp_max, "opt_w")
        in2 = hcl.scalar(0, "in2")
        in3 = hcl.scalar(0, "in3")
        in4 = hcl.scalar(0, "in4")
        in5 = hcl.scalar(0, "in5")
        in6 = hcl.scalar(0, "in6")
        
        with hcl.if_(self.d_mode == 'max'):
            with hcl.if_(spat_deriv[2] >= 0):
                opt_w[0] = self.wp_max
            with hcl.else_():
                opt_w[0] = -self.wp_max
        with hcl.else_():
            with hcl.if_(spat_deriv[2] >= 0):
                opt_w[0] = -self.wp_max
            with hcl.else_():
                opt_w[0] = self.wp_max
        
        # return opt_w, in2, in3, in4, in5, in6
        return opt_w, in2, in3, in4

    def dynamics(self, t, state, u_opt, d_opt):
        xe_dot = hcl.scalar(0, "xe_dot")
        ye_dot = hcl.scalar(0, "ye_dot")
        thetae_dot = hcl.scalar(0, "thetae_dot")
        xp_dot = hcl.scalar(0, "xp_dot")
        yp_dot = hcl.scalar(0, "yp_dot")
        thetap_dot = hcl.scalar(0, "thetap_dot")

        
        xe_dot = self.ve * hcl.cos(state[2])
        ye_dot = self.ve * hcl.sin(state[2])
        thetae_dot = u_opt[0]

        xp_dot = self.vp * hcl.cos(state[5])
        yp_dot = self.vp * hcl.sin(state[5])
        thetap_dot = d_opt[0]

        return xe_dot, ye_dot, thetae_dot, xp_dot, yp_dot, thetap_dot        

    def opt_ctrl_non_hcl(self):
        pass

    def opt_dstb_non_hcl(self):
        pass

    def dynamics_non_hcl(self):
        pass