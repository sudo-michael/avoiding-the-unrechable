import torch
import torch.nn as nn


class Net(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.net1 = nn.Sequential(nn.Linear(3, 3), nn.Tanh(), nn.Linear(3, 1))

        self.net2 = nn.Sequential(nn.Linear(3, 3), nn.Tanh(), nn.Linear(3, 1))

    def one(self, a):
        return self.net1(a)

    def two(self, b):
        return self.net2(b)


net = Net()
optim = torch.optim.Adam(net.parameters())
loss_fn = nn.MSELoss()
inp = torch.ones(1, 3)
target = torch.ones(1, 1) * 3

print(f"pred1: {net.one(inp)} pred2: {net.two(inp)}")
for _ in range(10):
    optim.zero_grad()
    loss = loss_fn(target, net.one(inp))
    loss.backward()
    optim.step()
    print(f"pred1: {net.one(inp)} pred2: {net.two(inp)}")
