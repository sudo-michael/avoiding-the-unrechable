import sys

sys.path.append("../atu/optimized_dp/")
import numpy as np

# Utility functions to initialize the problem
from atu.optimized_dp.Grid.GridProcessing import Grid
from atu.optimized_dp.Shapes.ShapesFunctions import *

# Specify the  file that includes dynamic systems
from atu.optimized_dp.dynamics.DubinsCar4D import *
from atu.optimized_dp.dynamics.DubinsCapture import *
from atu.optimized_dp.dynamics.DubinsCar4D2 import *

# Plot options
from atu.optimized_dp.plot_options import *

# Solver core
from atu.optimized_dp.solver import HJSolver
from atu.optimized_dp.Grid.GridProcessing import Grid

# g = Grid(
#     np.array([-4.0, -4.0, -math.pi]),
#     np.array([4.0, 4.0, math.pi]),
#     3,
#     np.array([40, 40, 40]),
#     [2],
# )

# # Reachable set
# goal = CylinderShape(g, [2], np.zeros(3), 0.5)

# # Avoid set
# obstacle = CylinderShape(g, [2], np.array([1.0, 1.0, 0.0]), 0.5)

# # Look-back length and time step
# lookback_length = 1.5
# t_step = 0.05

# small_number = 1e-5
# tau = np.arange(start=0, stop=lookback_length + small_number, step=t_step)

# my_car = DubinsCapture(uMode="min", dMode="max")

# po2 = PlotOptions(do_plot=True, plot_type="3d_plot", plotDims=[0, 1, 2], slicesCut=[])

# """
# Assign one of the following strings to `TargetSetMode` to specify the characteristics of computation
# "TargetSetMode":
# {
# "none" -> compute Backward Reachable Set,
# "minVWithV0" -> min V with V0 (compute Backward Reachable Tube),
# "maxVWithV0" -> max V with V0,
# "maxVWithVInit" -> compute max V over time,
# "minVWithVInit" -> compute min V over time,
# "minVWithVTarget" -> min V with target set (if target set is different from initial V0)
# "maxVWithVTarget" -> max V with target set (if target set is different from initial V0)
# }

# (optional)
# Please specify this mode if you would like to add another target set, which can be an obstacle set
# for solving a reach-avoid problem
# "ObstacleSetMode":
# {
# "minVWithObstacle" -> min with obstacle set,
# "maxVWithObstacle" -> max with obstacle set
# }
# """

# compMethods = {
#     "TargetSetMode": "minVWithVTarget",
#     "ObstacleSetMode": "maxVWithObstacle",
# }
# # HJSolver(dynamics object, grid, initial value function, time length, system objectives, plotting options)
# result = HJSolver(
#     my_car, g, [goal, obstacle], tau, compMethods, po2, saveAllTimeSteps=True
# )


g = Grid(
    np.array([-8.0, -8.0, 0.0, -math.pi]),
    np.array([8.0, 8.0, 4.0, math.pi]),
    4,
    np.array([40, 40, 20, 36]),
    [3],
)

# Define my object
my_car = DubinsCar4D2(uMode="max", dMode="min")

# Use the grid to initialize initial value function
Initial_value_f = CylinderShape(g, [2, 3], np.zeros(4), 1)

# Look-back length and time step
lookback_length = 1.0
t_step = 0.05

small_number = 1e-5

tau = np.arange(start=0, stop=lookback_length + small_number, step=t_step)

po = PlotOptions(do_plot=True, plot_type="3d_plot", plotDims=[0, 1, 3], slicesCut=[19])

# In this example, we compute a Backward Reachable Tube
# compMethods = {"TargetSetMode": "minVWithV0"}
compMethods = {"TargetSetMode": "minVWithV0"}
result = HJSolver(
    my_car, g, Initial_value_f, tau, compMethods, po, saveAllTimeSteps=False
)

# extra step
po2 = PlotOptions(do_plot=True, plot_type="3d_plot", plotDims=[0, 1, 3], slicesCut=[19])
my_car2 = DubinsCar4D2(uMode="min", dMode="max")
tau2 = np.arange(start=0, stop=0.1 + small_number, step=t_step)
# In this example, we compute a Backward Reachable Tube
compMethods = {"TargetSetMode": "minVWithV0"}
result2 = HJSolver(my_car2, g, result, tau2, compMethods, po2, saveAllTimeSteps=False)


# r = np.maximum(result, result2)
# from atu.optimized_dp.Plots.plotting_utilities import plot_isosurface

# plot_isosurface(g, r, po2)
