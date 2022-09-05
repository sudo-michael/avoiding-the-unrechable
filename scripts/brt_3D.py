import numpy as np
import odp
from atu.dynamics.DubinsCar import DubinsCar
from odp.Grid.GridProcessing import Grid

g = Grid(
    np.array([-4.5, -4.5, -np.pi]),
    np.array([4.5, 4.5, np.pi]),
    3,
    np.array([80, 80, 80]),
    [2],
)
car_r = 0.2

car_brt = DubinsCar(uMode="max", dMode="min", wMax=1.5, r=car_r)
car_ra = DubinsCar(uMode="min", dMode="max", wMax=1.5, r=car_r)

if __name__ in "__main__":
    from odp.Shapes.ShapesFunctions import *
    from odp.dynamics.DubinsCar import *
    from odp.Plots.plot_options import *
    from odp.solver import HJSolver

    # lava
    Initial_value_f = ShapeRectangle(
        g,
        np.array([-4.5, -0.5 - car_r, -np.inf]),
        np.array([-4.5 + 6.5 + car_r, -0.5 + car_r + 1.0, np.inf]),
    )

    # x walls
    Initial_value_f = Union(
        Initial_value_f,
        Union(
            Lower_Half_Space(g, 0, -4.5 + car_r), Upper_Half_Space(g, 0, 4.5 - car_r)
        ),
    )

    # y walls
    Initial_value_f = Union(
        Initial_value_f,
        Union(
            Lower_Half_Space(g, 1, -4.5 + car_r), Upper_Half_Space(g, 1, 4.5 - car_r)
        ),
    )

    Goal = CylinderShape(g, [2], [-2, 2.3], 0.5)

    def brt(dist=True):
        lookback_length = 3.0
        t_step = 0.05
        small_number = 1e-5
        tau = np.arange(start=0, stop=lookback_length + small_number, step=t_step)

        if dist:
            car_brt.dMax = np.array([0.1, 0.1, 0.1])
            car_brt.dMix = -np.array([0.1, 0.1, 0.1])

        compMethods = {"TargetSetMode": "minVWithV0"}

        result = HJSolver(
            car_brt,
            g,
            Initial_value_f,
            tau,
            compMethods,
            PlotOptions(
                do_plot=True, plot_type="3d_plot", plotDims=[0, 1, 2], slicesCut=[]
            ),
            saveAllTimeSteps=False,
        )
        if dist:
            np.save("./atu/envs/assets/brts/min_hallway_brt_dist.npy", result)
        else:
            np.save("./atu/envs/assets/brts/min_hallway_brt.npy", result)

        lookback_length = 0.7
        t_step = 0.05

        small_number = 1e-5
        tau2 = np.arange(start=0, stop=lookback_length + small_number, step=t_step)

        if dist:
            car_ra.dMax = np.array([0.1, 0.1, 0.1])
            car_ra.dMix = -np.array([0.1, 0.1, 0.1])
        compMethods = { "TargetSetMode": "minVWithV0"}
        result = HJSolver(
            car_ra,
            g,
            result,
            tau2,
            compMethods,
            PlotOptions(
                do_plot=True, plot_type="3d_plot", plotDims=[0, 1, 2], slicesCut=[]
            ),
            saveAllTimeSteps=False,
        )

        if dist:
            np.save("./atu/envs/assets/brts/max_over_min_hallway_brt_dist.npy", result)
        else:
            np.save("./atu/envs/assets/brts/max_over_min_hallway_brt.npy", result)

    brt(dist=True)
