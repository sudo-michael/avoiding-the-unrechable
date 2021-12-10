import numpy as np
import torch
def normalize(state, bounds=np.array([4.0, 4.0, np.pi])):
    state /= bounds
    state =  torch.Tensor(state)
    state = torch.cat((torch.ones(1), state)).reshape(1, 1, 4)
    return state


def unnormalize(model_out):
    x = model_out["model_in"]  # (meta_batch_size, num_points, 4)
    y = model_out["model_out"]  # (meta_batch_size, num_points, 1)
    batch_size = x.shape[1]

    du, status = diff_operators.jacobian(y, x)
    dudt = du[..., 0, 0].reshape(1).detach().cpu().numpy()
    dudx = du[..., 0, 1:].reshape(3).detach().cpu().numpy()

    norm_to = 0.02
    mean = 0.25
    var = 0.5
    y = ((y*var/norm_to) + mean)
    return y, dudx


state = np.array([0.1, 0.1, 0.42])
print(normalize(state))