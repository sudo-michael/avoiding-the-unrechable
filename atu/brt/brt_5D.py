import numpy as np
import sys

from odp.Grid.GridProcessing import Grid
from odp.Plots.plot_options import PlotOptions
from atu.dynamics.SingleNarrowPassage import SingleNarrowPassage

# x y th v phi
grid_low = np.array([-8, -3.0, np.pi, 0.1, -1.3])
grid_high = np.array([8, 3.0, np.pi, 6.5, 1.3])
grid_pts = np.array([20, 20, 20, 20, 20])
grid = Grid(
    grid_low,
    grid_high,
    5,
    grid_pts,
    [2, 4],
)

L = 1.25
CURB_POSITION = np.array([-2.8, 2.8])
STRANDED_CAR_POS = np.array([0.0, -1.8])
STRANDED_R2_POS = np.array([-6.0, 1.8])
GOAL_POS = np.array([6.0, -1.4])

car_ra = SingleNarrowPassage(u_mode="min", d_mode="max", length=L)  # ra
car_brt = SingleNarrowPassage(u_mode="max", d_mode="min", length=L)  # ra
dt = 0.05

if __name__ in "__main__":
    from odp.Shapes.ShapesFunctions import *
    from odp.Plots.plot_options import *
    from odp.solver import HJSolver

    curb = Union(
        Lower_Half_Space(grid, 1, CURB_POSITION[0] + 0.5 * L),
        Upper_Half_Space(grid, 1, CURB_POSITION[1] - 0.5 * L),
    )

    stranded_car = CylinderShape(
        grid, np.array([2, 3, 4]), center=STRANDED_CAR_POS, radius=L
    )

    stranded_r2_pos = CylinderShape(
        grid, np.array([2, 3, 4]), center=STRANDED_R2_POS, radius=L
    )

    obstacle = Union(curb, Union(stranded_r2_pos, stranded_car))

    ivf = Union(obstacle, Lower_Half_Space(grid, 3, -0.1))
    ivf = Union(
        obstacle,
        Union(
            Lower_Half_Space(grid, 4, -1.25),
            Upper_Half_Space(grid, 4, 1.25),
        ),
    )

    def brt(d=True):
        lookback_length = 1.0
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
            Initial_value_f,
            tau,
            compMethods,
            PlotOptions(
                do_plot=True, plot_type="3d_plot", plotDims=[0, 1, 3], slicesCut=[5]
            ),
            saveAllTimeSteps=False,
        )
        print(result.sum())

        if d:
            np.save("./atu/envs/assets/brts/min_hallway_4D_brt_dist.npy", result)
        else:
            np.save("./atu/envs/assets/brts/min_hallway_4D_brt.npy", result)

        lookback_length = 0.15
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
                do_plot=True, plot_type="3d_plot", plotDims=[0, 1, 3], slicesCut=[5]
            ),
            saveAllTimeSteps=False,
        )

        if d:
            np.save(
                "./atu/envs/assets/brts/max_over_min_hallway_4D_brt_dist.npy", result
            )
        else:
            np.save("./atu/envs/assets/brts/max_over_min_hallway_4D_brt.npy", result)

    brt(d=True)
