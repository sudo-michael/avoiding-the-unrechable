# %%
import matlab.engine
import numpy as np


class HelperOCController:
    def __init__(self):
        print("connecting to matlab engine")
        self.eng = matlab.engine.start_matlab()
        self.eng.addpath(
            self.eng.genpath(
                r"/local-scratch/localhome/mla233/Downloads/Pendulum/Pendulum"
            )
        )
        self.eng.addpath(
            self.eng.genpath(r"/local-scratch/localhome/mla233/hj/ToolboxLS")
        )

        # self.vars = self.eng.eval(
        #     "load('/local-scratch/localhome/mla233/Downloads/Pendulum/Pendulum/pend_brt.mat')"
        # )
        self.vars = self.eng.eval(
            "load('/local-scratch/localhome/mla233/hj/Pendulum/Pendulum/pend_brt_max_min_2.mat')"
        )
        print("done 2")

    def opt_ctrl_value(self, state):
        # state[0] = th, state[1] = thdot
        x = matlab.double([[state[0]], [state[1]]])
        opt_ctrl, value = self.eng.pendulum_opt_ctrl(x, self.vars, nargout=2)
        # opt_ctrl, value = self.eng.pendulum_opt_ctrl(x, nargout=2)
        return np.array([opt_ctrl]), value


# %%
if __name__ in "__main__":
    c = HelperOCController()
    print(c.opt_ctrl_value([0, 0]))
    c.opt_ctrl_value([-0.0875, -0.9786])


# %%
