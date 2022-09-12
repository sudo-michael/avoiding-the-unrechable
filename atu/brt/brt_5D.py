import numpy as np
import sys

from odp.Grid.GridProcessing import Grid
from odp.Plots.plot_options import PlotOptions
from atu.dynamics.SingleNarrowPassage import SingleNarrowPassage
from odp.dynamics.DubinsCar5DAvoid import DubinsCar5DAvoid

# 0 1 2  3 4
# x y th v phi
g = Grid(
    np.array([-8.0, -3.0, -np.pi, -0.2, -1.4]),
    np.array([8.0, 3.0, np.pi, 7.0, 1.4]),
    5,
    np.array([40, 40, 20, 20, 20]),
    [2, 4],
)

L = 1.00
CURB_POSITION = np.array([-2.8, 2.8])
WALL_POSITION = np.array([-8.0, 8.0])
STRANDED_CAR_POS = np.array([0.0, -1.8])
GOAL_POS = np.array([6.0, -1.4])

dist = np.array([0.1, 0.1, 0.1, 0.1, 0.1])
car_brt = SingleNarrowPassage(u_mode="max", d_mode="min", length=L)  # ra
car_ra = SingleNarrowPassage(u_mode="min", d_mode="max", length=L)  # ra
dt = 0.05

# car_brt = DubinsCar5DAvoid()

if __name__ in "__main__":
    from odp.Shapes.ShapesFunctions import *
    from odp.Plots.plot_options import *
    from odp.solver import HJSolver

    curb = Union(
        Lower_Half_Space(g, 1, CURB_POSITION[0] + 0.5 * L),
        Upper_Half_Space(g, 1, CURB_POSITION[1] - 0.5 * L),
    )

    walls = Union(
        Lower_Half_Space(g, 0, WALL_POSITION[0] + 0.5 * L),
        Upper_Half_Space(g, 0, WALL_POSITION[1] - 0.5 * L),
    )


    stranded_car = CylinderShape(
        g, np.array([2, 3, 4]), center=STRANDED_CAR_POS, radius=L
    )

    obstacle = Union(stranded_car, Union(walls, curb))

    ivf = Union(obstacle, Union(Lower_Half_Space(g, 3, 0.9), Upper_Half_Space(g, 3, 5))) # v
    ivf = Union(
        ivf,
        Union(
            Lower_Half_Space(g, 4, -1.25), # psi
            Upper_Half_Space(g, 4, 1.25),  # psi
        ),
    )

    def brt(d=True):
        lookback_length = 6.0
        t_step = 0.05
        small_number = 1e-5
        tau = np.arange(start=0, stop=lookback_length + small_number, step=t_step)

        if d:
            car_brt.dMax = dist
            car_brt.dMin = -dist

        compMethods = {"TargetSetMode": "minVWithV0"}

        result = HJSolver(
            car_brt,
            g,
            ivf,
            tau,
            compMethods,
            PlotOptions(
                do_plot=False, plot_type="3d_plot", plotDims=[0, 1, 3], slicesCut=[5, 5]
            ),
            saveAllTimeSteps=False,
        )
        print(result.sum())

        if d:
            np.save("./atu/envs/assets/brts/min_single_narrow_passage_brt_dist_4.npy", result)
        else:
            np.save("./atu/envs/assets/brts/min_single_narrow_passage_brt.npy", result)

        lookback_length = 0.05
        t_step = 0.05
        small_number = 1e-5
        tau2 = np.arange(start=0, stop=lookback_length + small_number, step=t_step)

        if d:
            car_ra.dMax = dist
            car_ra.dMin = -dist
        compMethods = {"TargetSetMode": "minVWithV0"}
        result = HJSolver(
            car_ra,
            g,
            result,
            tau2,
            compMethods,
            PlotOptions(
                do_plot=True, plot_type="3d_plot", plotDims=[0, 1, 3], slicesCut=[5, 5]
            ),
            saveAllTimeSteps=False,
        )

        if d:
            np.save(
                "./atu/envs/assets/brts/max_over_min_single_narrow_passage_brt_dist_4.npy", result
            )
        else:
            np.save("./atu/envs/assets/brts/max_over_min_single_narrow_passage_brt.npy", result)

    brt(d=True)