# %%
import numpy as np
ve = 1
vp = 1
u_opt = 1.5
d_opt = -1.5
dt = 0.05
def relative_dynamics(state):
    x1_dot = -ve + vp * np.cos(state[2]) + u_opt * state[1]
    x2_dot = vp * np.sin(state[2]) - u_opt * state[0]
    x3_dot = d_opt - u_opt

    return np.array([x1_dot, x2_dot, x3_dot])

def e_dubins_dynamics(state):
    x1_dot = ve * np.cos(state[2])
    x2_dot = ve * np.sin(state[2])
    x3_dot = u_opt
    return np.array([x1_dot, x2_dot, x3_dot])

def p_dubins_dynamics(state):
    x1_dot = vp * np.cos(state[2])
    x2_dot = vp * np.sin(state[2])
    x3_dot = d_opt
    return np.array([x1_dot, x2_dot, x3_dot])

# %%    
rel_state = np.array([1, 0, 0])
e_state = np.array([0, 0, 0])
p_state = np.array([1, 0, 0])
# %%
next_e_state = e_dubins_dynamics(e_state) * dt + e_state
next_p_state = p_dubins_dynamics(p_state) * dt + p_state
next_rel_state = relative_dynamics(rel_state) * dt + rel_state

next_e_state = e_dubins_dynamics(next_e_state) * dt + next_e_state
next_p_state = p_dubins_dynamics(next_p_state) * dt + next_p_state
next_rel_state = relative_dynamics(next_rel_state) * dt + next_rel_state
# %%
print(next_rel_state)
print(next_p_state - next_e_state)

# %%
import scipy.io
helperOC_brt = scipy.io.loadmat('art3d_brt.mat')['data']
odp_brt = np.load('./atu/envs/assets/brts/air3d_brt_test.npy')

# %%
np.max(helperOC_brt[..., 0] - odp_brt)

# %%
