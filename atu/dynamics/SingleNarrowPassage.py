import heterocl as hcl
import numpy as np


class SingleNarrowPassage:
    def __init__(
        self,
        x=[0, 0, 0, 0, 0],
        alpha_max=2.0,
        alpha_min=-4.0,
        psi_max=3 * np.pi,
        psi_min=-3 * np.pi,
        length=2.0,
        u_mode="min",
        d_mode="max",
    ):
        self.x = x
        self.alpha_max = alpha_max
        self.alpha_min = alpha_min
        self.psi_max = psi_max
        self.psi_min = psi_min
        self.length = length

        if u_mode == "min" and d_mode == "max":
            print("Control for reaching target set")
        elif u_mode == "max" and d_mode == "min":
            print("Control for avoiding target set")
        else:
            raise ValueError(
                f"u_mode: {u_mode} and d_mode: {d_mode} are not opposite of each other!"
            )

        self.u_mode = u_mode
        self.d_mode = d_mode

    def opt_ctrl(self, t, state, spat_deriv):
        opt_u_alpha = hcl.scalar(0, "opt_u_alpha")
        opt_u_psi = hcl.scalar(0, "opt_u_psi")
        in3 = hcl.scalar(0, "in3")
        in4 = hcl.scalar(0, "in4")
        in5 = hcl.scalar(0, "in5")

        with hcl.if_(self.u_mode == "max"):
            with hcl.if_(spat_deriv[3] >= 0):
                opt_u_alpha[0] = self.alpha_max
            with hcl.elif_(spat_deriv[3] < 0):
                opt_u_alpha[0] = self.alpha_min
        with hcl.else_():  # u_mode == 'min
            with hcl.if_(spat_deriv[3] > 0):  # to match sign for deepreach
                opt_u_alpha[0] = self.alpha_min
            with hcl.elif_(spat_deriv[3] <= 0):
                opt_u_alpha[0] = self.alpha_max

        with hcl.if_(self.u_mode == "max"):
            with hcl.if_(spat_deriv[4] >= 0):
                opt_u_psi[0] = self.psi_max
            with hcl.elif_(spat_deriv[4] < 0):
                opt_u_psi[0] = self.psi_min
        with hcl.else_():
            with hcl.if_(spat_deriv[4] > 0):
                opt_u_psi[0] = self.psi_min
            with hcl.elif_(spat_deriv[4] <= 0):
                opt_u_psi[0] = self.psi_max

        return opt_u_alpha[0], opt_u_psi[0], in3, in4, in5

    def opt_dstb(self, t, state, spat_deriv):
        d1 = hcl.scalar(0, "d1")
        d2 = hcl.scalar(0, "d2")
        d3 = hcl.scalar(0, "d3")
        d4 = hcl.scalar(0, "d4")
        d5 = hcl.scalar(0, "d5")

        return d1, d2, d3, d4, d5

    def dynamics(self, t, state, u_opt, d_opt):
        """
        \dot{x_1} = x_4 \cos(x_3) # x
        \dot{x_2} = x_4 \sin(x_3) # y
        \dot{x_3} = x_4 \tan(x_5) / L # theta (heading)
        \dot{x_4} = a  # theta  # v
        \dot{x_5} = psi # phi (steering angle)
        """
        x1_dot = hcl.scalar(0, "x1_dot")
        x2_dot = hcl.scalar(0, "x2_dot")
        x3_dot = hcl.scalar(0, "x3_dot")
        x4_dot = hcl.scalar(0, "x4_dot")
        x5_dot = hcl.scalar(0, "x5_dot")

        x1_dot[0] = state[3] * hcl.cos(state[2])
        x2_dot[0] = state[3] * hcl.sin(state[2])
        x3_dot[0] = state[3] * (hcl.sin(state[4]) / hcl.cos(state[4])) / self.length
        x4_dot[0] = u_opt[0]
        x5_dot[0] = u_opt[1]

        return x1_dot[0], x2_dot[0], x3_dot[0], x4_dot[0], x5_dot[0]

    def opt_ctrl_non_hcl(self, t, state, spat_deriv):
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
        else:  # u_mode == 'min
            if spat_deriv[3] > 0:  # to match sign for deepreach
                opt_u_alpha = self.alpha_min
            else:
                opt_u_alpha = self.alpha_max

        if self.u_mode == "max":
            if spat_deriv[4] >= 0:
                opt_u_psi = self.psi_max
            else:
                opt_u_psi = self.psi_min
        else:  # u_mode == 'min'
            if spat_deriv[4] > 0:
                opt_u_psi = self.psi_min
            else:
                opt_u_psi = self.psi_max

        return np.array([opt_u_alpha, opt_u_psi])

    def opt_dstb_non_hcl(self, t, state, spat_deriv):
        return np.zeros(5)

    def dynamics_non_hcl(self, t, state, u_opt, d_opt=np.zeros(5)):
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


# # TODO
# - [ ] implement baseline 1
# - [ ] implement baseline 2
# - [ ] run experiments on 3 environments
# - contributions
#   - introduce simple reward-shpaing framework to incorporate feedback from an HJ controller to RL
#   - imperically show that our method allows policy to act safe after HJ controller is removed
#   - and show that method is robust to worse-case disturbances
