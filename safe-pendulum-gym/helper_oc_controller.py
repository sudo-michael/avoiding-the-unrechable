import matlab.engine
import numpy as np


class HelperOCController:
    def __init__(self):
        print("connecting to matlab engine")
        self.eng = matlab.engine.start_matlab()
        self.eng.addpath(self.eng.genpath(r"/local-scratch/localhome/mla233/hj"))
        print("done")

    def opt_ctrl_value(self, state):
        # state[0] = th, state[1] = thdot
        x = matlab.double([[state[0]], [state[1]]])
        opt_ctrl, value = self.eng.pendulum_brt(x, nargout=2)
        return np.array([opt_ctrl]), value
