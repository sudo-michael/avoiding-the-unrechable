import matlab.engine
import numpy as np
import pathlib
import os


class HelperOCController:
    def __init__(self):
        print("connecting to matlab engine")
        self.helperOC_path = os.path.join(pathlib.Path().resolve(), "matlab/helperOC")
        self.eng = matlab.engine.start_matlab()
        self.eng.addpath(self.eng.genpath(self.helperOC_path))
        self.eng.addpath(
            self.eng.genpath(os.path.join(pathlib.Path().resolve(), "matlab/ToolboxLS"))
        )

        self.vars = self.eng.eval(f"load('{self.helperOC_path}/pend_brt_min_max.mat')")
        print("connected to matlab engine")

    def opt_ctrl_value(self, state):
        x = matlab.double([[state[0]], [state[1]]])
        opt_ctrl, value = self.eng.pendulum_opt_ctrl(x, self.vars, nargout=2)
        return np.array([opt_ctrl]), value

    def safe_ctrl_bnds(self, state):
        x = matlab.double([[state[0]], [state[1]]])
        low, high = self.eng.pendulum_safe_ctrl_bnds(x, self.vars, nargout=2)
        return low, high


if __name__ in "__main__":
    c = HelperOCController()
    print(c.opt_ctrl_value([0, 0]))
    print(c.safe_ctrl_bnds([0, 0]))
