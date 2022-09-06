import numpy as np
import atu
from odp.Grid import Grid
from atu.dynamics.DubinsCar4D import DubinsCar4D

g = Grid(
    np.array([-4.5, -4.5, -1, -np.pi]),
    np.array([4.5, 4.5, 5, np.pi]),
    4,
    # np.array([80, 80, 40, 40]),
    np.array([40, 40, 20, 20]),
    [3],
)
car_r = 0.2
dist = np.array([0.1, 0.1, 0, 0])
car_brt = DubinsCar4D(uMode="max", dMode="min", length=car_r, dMin=-dist, dMax=dist) # TODO length != car_r
car_ra = DubinsCar4D(uMode="min", dMode="max", length=car_r, dMin=-dist, dMax=dist)


if __name__ in "__main__":
    from odp.Shapes.ShapesFunctions import *
    from odp.Plots.plot_options import *
    from odp.solver import HJSolver

    # lava
    Initial_value_f = ShapeRectangle(
        g,
        np.array([-4.5, -0.5 - car_r, -np.inf, -np.inf]),
        np.array([-4.5 + 6.5 + car_r, -0.5 + car_r + 1.0, np.inf, np.inf]),
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

    # speed
    Initial_value_f = Union(Initial_value_f, Lower_Half_Space(g, 2, -0.1))
    Goal = CylinderShape(g, [2, 3], [-2, 2.3], 0.5)

    lookback_length = 3.0
    t_step = 0.05
    small_number = 1e-5
    tau = np.arange(start=0, stop=lookback_length + small_number, step=t_step)

    compMethods = {"TargetSetMode": "minVWithV0"}

    def brt():
        compMethods = {"TargetSetMode": "minVWithV0"}
        result = HJSolver(
            car_brt,
            g,
            Initial_value_f,
            tau,
            compMethods,
            PlotOptions(
                do_plot=False, plot_type="3d_plot", plotDims=[0, 1, 3], slicesCut=[5]
            ),
            saveAllTimeSteps=False,
        )
        np.save("./atu/envs/assets/brts/dubin_hallway_4D_dist_brt.npy", result)
        # if dist:
        #     np.save("min_brt_dist.npy", result)
        # else:
        #     np.save("min_brt.npy", result)

        lookback_length = 0.1
        t_step = 0.05

        small_number = 1e-5
        tau2 = np.arange(start=0, stop=lookback_length + small_number, step=t_step)

        compMethods = {"TargetSetMode": "minVWithV0"}
        result = HJSolver(
            car_ra,
            g,
            result,
            tau2,
            compMethods,
            PlotOptions(
                do_plot=True, plot_type="3d_plot", plotDims=[0, 1, 3], slicesCut=[5]
            ),
            saveAllTimeSteps=False,
        )
        np.save(
            "./atu/envs/assets/brts/dubin_hallway_4D_dist_max_over_min_brt.npy", result
        )

    brt()
