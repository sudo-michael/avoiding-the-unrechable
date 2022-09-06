import numpy as np
import sys

from odp.Grid.GridProcessing import Grid
from odp.Plots.plot_options import PlotOptions
from atu.dynamics.SingleNarrowPassage import SingleNarrowPassage

import atu.utils

# # x y th v phi
# grid_low = np.array(
#     [-8, -3.8, -1.1 * np.pi, 0.1, -1.2]
# )  # mul by 1.1 to allow mod to go to -pi
# grid_high = np.array([8, 3.8, 1.1 * np.pi, 6.5, 1.2])

# # for v, deepreach normalizes with (v - beta) / alpha to be in range [-1, 1]
# # (7 -  3) / 4 -> 1 (-1 -3) / 4 -> -1

# # grid_pts = np.array([40, 40, 40, 40, 40])
# grid_pts = np.array([20, 20, 20, 20, 20])
# grid = Grid(
#     grid_low,
#     grid_high,
#     5,
#     grid_pts,
#     [2, 4],
# )

# from brt_test.py
# x y th v phi
grid_low = np.array(
    [-8, -3.0, -1.1 * np.pi, 0.1, -1.3]
)  # mul by 1.1 to allow mod to go to -pi
grid_high = np.array([8, 3.0, 1.1 * np.pi, 6.5, 1.3])

# for v, deepreach normalizes with (v - beta) / alpha to be in range [-1, 1]
# (7 -  3) / 4 -> 1 (-1 -3) / 4 -> -1

# grid_pts = np.array([40, 40, 40, 40, 40])
grid_pts = np.array([20, 20, 20, 20, 20])
# grid_pts = np.array([10, 10, 10, 10, 10])
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


def reach_goal_straight():
    goal = CylinderShape(grid, np.array([2, 3, 4]), center=np.array([6.0, 0]), radius=L)
    tau = np.arange(start=0, stop=3 + dt, step=dt)
    compMethods = {"TargetSetMode": "minVWithV0"}
    po = PlotOptions(
        do_plot=True, plot_type="3d_plot", plotDims=[0, 1, 2], slicesCut=[1, 8]
    )
    result = HJSolver(car_ra, grid, goal, tau, compMethods, po, saveAllTimeSteps=False)
    np.save("./atu/envs/assets/brts/single_narrow_passage_ra_goal_straight.npy", result)
    

def reach_goal_low():
    goal = CylinderShape(grid, np.array([2, 3, 4]), center=np.array([6.0, -1.4]), radius=L)

    psi_dot_constraint = Union(
        Lower_Half_Space(grid, 4, -1.1),
        Upper_Half_Space(grid, 4, 1.1),
    )
    curb = Union(
        Lower_Half_Space(grid, 1, CURB_POSITION[0] + 0.5 * L),
        Upper_Half_Space(grid, 1, CURB_POSITION[1] - 0.5 * L),
    )
    stranded_car = CylinderShape(
        grid, np.array([2, 3, 4]), center=STRANDED_CAR_POS, radius=L
    )
    stranded_car_2 = CylinderShape(
        grid, np.array([2, 3, 4]), center=STRANDED_R2_POS, radius=L
    )


    constraint = curb
    constraint = Union(constraint, psi_dot_constraint)
    constraint = Union(constraint, stranded_car)
    constraint = Union(constraint, stranded_car_2)

    comp_method = {
        "TargetSetMode": "minVWithVTarget",
        "ObstacleSetMode": "maxVWithObstacle",
    }
    tau = np.arange(start=0, stop=6 + dt, step=dt)
    po = PlotOptions(
        do_plot=True, plot_type="3d_plot", plotDims=[0, 1, 2], slicesCut=[1, 8]
    )
    result = HJSolver(car_ra, grid, [goal, constraint], tau, comp_method, po, saveAllTimeSteps=False)
    np.save("./atu/envs/assets/brts/single_narrow_passage_ra_goal_low.npy", result)


def avoid_goal_low():
    goal = CylinderShape(grid, np.array([2, 3, 4]), center=np.array([6.0, -1.4]), radius=L)
    psi_dot_constraint = Union(
        Lower_Half_Space(grid, 4, -1.0),
        Upper_Half_Space(grid, 4, 1.0),
    )

    vel_constraint = Lower_Half_Space(grid, 3, 0.1)

    curb = Union(
        Lower_Half_Space(grid, 1, CURB_POSITION[0] + 0.5 * L),
        Upper_Half_Space(grid, 1, CURB_POSITION[1] - 0.5 * L),
    )

    bounds = Union(
        Lower_Half_Space(grid, 0, grid_low[0] + 0.5 * L),
        Upper_Half_Space(grid, 0, grid_high[0] - 0.5 * L),
    )

    stranded_car = CylinderShape(
        grid, np.array([2, 3, 4]), center=STRANDED_CAR_POS, radius=L
    )

    stranded_car_2 = CylinderShape(
        grid, np.array([2, 3, 4]), center=STRANDED_R2_POS, radius=L
    )


    constraint = curb
    constraint = Union(constraint, bounds)
    constraint = Union(constraint, vel_constraint)
    constraint = Union(constraint, psi_dot_constraint)
    constraint = Union(constraint, stranded_car)
    constraint = Union(constraint, stranded_car_2)

    tau = np.arange(start=0, stop=1.75 + dt, step=dt)
    po = PlotOptions(
        do_plot=False, plot_type="3d_plot", plotDims=[0, 1, 2], slicesCut=[1, 8]
    )

    comp_method = {"TargetSetMode": "minVWithV0"}
    result = HJSolver(car_brt, grid, constraint, tau, comp_method, po, saveAllTimeSteps=False)

    tau = np.arange(start=0, stop=0.85 + dt, step=dt)
    po = PlotOptions(
        do_plot=True, plot_type="3d_plot", plotDims=[0, 1, 2], slicesCut=[1, 8]
    )
    result = HJSolver(car_brt, grid, result, tau, comp_method, po, saveAllTimeSteps=False)
    np.save("./atu/envs/assets/brts/single_narrow_passage_brt_goal_low.npy", result)

if __name__ in "__main__":
    from Shapes.ShapesFunctions import *
    from dynamics.SingleNarrowPassage import SingleNarrowPassage
    from plot_options import *
    from solver import HJSolver
    # using env_setting =='v2'

    # 1. just compute maximum reachable tubes
    #    a. go to goal infront
    #    b. go to goal need turn left
    #    b. go to goal need turn right

    # goal = CylinderShape(grid, np.array([2, 3, 4]), center=GOAL_POS, radius=L)

    # curb = Union(
    #     Lower_Half_Space(grid, 1, CURB_POSITION[0] + 0.5 * L),
    #     Upper_Half_Space(grid, 1, CURB_POSITION[1] - 0.5 * L),
    # )

    # stranded_car = CylinderShape(
    #     grid, np.array([2, 3, 4]), center=STRANDED_CAR_POS, radius=L
    # )
    # stranded_r2_pos = CylinderShape(
    #     grid, np.array([2, 3, 4]), center=STRANDED_R2_POS, radius=L
    # )

    # obstacle = Union(curb, Union(stranded_r2_pos, stranded_car))

    # # turning_rate_constranit = Lower_Half_Space()
    # # turning_rate_constranit = Lower_Half_Space()

    # tau = np.arange(start=0, stop=10 + dt, step=dt)
    # # tau = np.arange(start=0, stop=1 + 0.05, step=0.05)

    # po = PlotOptions(
    #     do_plot=False, plot_type="3d_plot", plotDims=[0, 1, 2], slicesCut=[1, 8]
    # )

    # comp_method = {
    #     "TargetSetMode": "minVWithVTarget",
    #     "ObstacleSetMode": "maxVWithObstacle",
    # }
    # result = HJSolver(
    #     car, grid, [goal, obstacle], tau, comp_method, po, saveAllTimeSteps=False
    # )
    # plot_isosurface(
    #     grid,
    #     result,
    #     PlotOptions(
    #         do_plot=False, plot_type="3d_plot", plotDims=[0, 1, 2], slicesCut=[1, 8]
    #     ),
    # )
    # np.save("./atu/envs/assets/brts/single_narrow_passage_ra.npy", result)