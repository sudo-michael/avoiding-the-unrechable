import numpy as np
import odp
from atu.dynamics.air3d import Air3D
from odp.Grid.GridProcessing import Grid

g = Grid(
    np.array([-1, -1, -np.pi]),
    np.array([1, 1, np.pi]),
    3,
    np.array([11, 11, 21]),
    [2],
)
car_r = 0.2
car_brt = Air3D(r=car_r, u_mode="max", d_mode="min", we_max=1.0, wp_max=1.0, ve=1.0, vp=1.0)
cylinder_r = 0.25

if __name__ in "__main__":
    from odp.Shapes.ShapesFunctions import *
    from odp.Plots.plot_options import *
    from odp.solver import HJSolver

    ivf = CylinderShape(g, [2], np.zeros(3), cylinder_r)

    def brt(d=True):
        lookback_length = 0.5
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
                do_plot=True, plot_type="3d_plot", plotDims=[0, 1, 2], slicesCut=[]
            ),
            saveAllTimeSteps=False,
        )

        np.save("./odp_air3d", result)

    brt()