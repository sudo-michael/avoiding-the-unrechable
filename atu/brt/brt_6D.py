import numpy as np
import sys

from odp.Grid.GridProcessing import Grid
from odp.Plots.plot_options import PlotOptions
from atu.dynamics.DubinsPersuitEvade import DubinsPersuitEvade

g = Grid(
    np.array([-5, -5, -np.pi, -5, -5, -np.pi]),
    np.array([5, 5, np.pi, 5, 5, np.pi]),
    6,
    np.array([20, 20, 10, 20, 20, 10]),
    [2, 5],
)

car_brt = DubinsPersuitEvade(x=np.zeros(6), we_max=1, wp_max=1, ve=1, vp=1, u_mode="max", d_mode="min")  # ra
dt = 0.05

# car_brt = DubinsCar5DAvoid()

if __name__ in "__main__":
    from odp.Shapes.ShapesFunctions import *
    from odp.Plots.plot_options import *
    from odp.solver import HJSolver

    ivf = CylinderShape(g, np.array([2, 5]), np.zeros(6), radius=1)

    def brt(d=False):
        lookback_length = 6.0
        t_step = 0.05
        small_number = 1e-5
        tau = np.arange(start=0, stop=lookback_length + small_number, step=t_step)

        compMethods = {"TargetSetMode": "minVWithV0"}

        result = HJSolver(
            car_brt,
            g,
            ivf,
            tau,
            compMethods,
            PlotOptions(
                do_plot=False, plot_type="3d_plot", plotDims=[0, 1, 3], slicesCut=[5, 5, 5]
            ),
            saveAllTimeSteps=False,
        )

        np.save("./atu/envs/assets/brts/6d.npy", result)

    brt(d=False)