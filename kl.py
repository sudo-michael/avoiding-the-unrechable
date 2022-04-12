# %%
import pdb
import numpy as np
import torch
from torch.autograd import grad
import torch.nn.functional as F
import matplotlib.pyplot as plt

# for animation
import matplotlib.animation
from IPython.display import Image

import warnings

from typing import Dict, List, Tuple

# %%
mu1, sigma1 = 2, 0.5 / 3
low1, high1 = 1, 2
gaussian = torch.distributions.Normal(mu1, sigma1)
u = torch.distributions.Uniform(low1, high1)

# %%
plt.figure(figsize=(14, 4))
x = torch.linspace(-2, 2, 1000)
plt.subplot(1, 2, 1)
plt.plot(x.numpy(), gaussian.log_prob(x).exp().numpy())
plt.title(f"$\mu={mu1},\sigma={sigma1}$")

x = torch.linspace(1, 2, 1000)
plt.subplot(1, 2, 2)
plt.plot(x.numpy(), u.log_prob(x).exp().numpy())
plt.title(f"low$={1},high={2}$")

plt.suptitle("Plotting the distributions")
# %%
mu = torch.tensor([0.0])
sigma = torch.tensor([1.0])

plt.figure(figsize=(14, 4))
x = torch.linspace(
    -mu1 - mu2 - 5 * sigma1 - 5 * sigma2, mu1 + mu2 + 5 * sigma1 + 5 * sigma2, 1000
)
Q = torch.distributions.Normal(mu, sigma)  # this should approximate P, eventually :-)
qx = Q.log_prob(x).exp()
plt.subplot(1, 2, 2)
plt.plot(x.numpy(), qx.detach().numpy())
plt.title("$Q(X)$")
