#!/usr/bin/env python
"""
a basic 1D experiment using torchdiffeq

ilker bayram, ibayram@ieee.org, 2021
"""

from scipy.integrate import solve_ivp
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from torchdiffeq import odeint_adjoint
from torch.utils.data import DataLoader
import utils


def dynamics(t: float, x):
    """
    a simple dynamics for a first order non-linear ODE
    """
    return np.sin(2 * np.pi * x)


def loss_func(y_true, y):
    """
    loss function to train the non-linear ODE parameters
    """
    return torch.mean(torch.flatten((y_true - y) ** 2))


def main():

    # figure set up
    fig_params = {"bbox_inches": "tight"}
    figsize = (6, 3)
    savefig = utils.wrap_savefig(plt.savefig, **fig_params)

    t = np.linspace(0, 2, num=500)
    eps = 1e-3
    sols = [
        solve_ivp(fun=dynamics, t_span=(t[0], t[-1]), y0=np.array([y0]), t_eval=t)
        for y0 in [-eps, eps]
    ]
    # take a look at the data
    fig, ax = plt.subplots(figsize=figsize)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for sol in sols:
        ax.plot(
            sol.t.reshape(-1), sol.y.reshape(-1), label=f"y[0]={sol.y.reshape(-1)[0]}"
        )
    ax.legend()
    plt.grid(which="major", axis="both")
    savefig("phase.png")

    x = np.linspace(-1.1, 1.1, num=200)
    z = dynamics(0, x)
    fig, ax = plt.subplots(figsize=figsize)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.plot(x, z)
    ax.scatter(np.arange(-1, 2), np.zeros(3), marker="o", color="r")
    ax.scatter([-0.5, 0.5], [0, 0], marker="o", color="k")
    ax.axhline(y=0, color="k")
    # plt.arrow(x=0.25, y=0.0, dx=0.1, dy=0.0)
    plt.grid(which="major", axis="both")

    savefig("dynamics.png")

    # construct the dataset and dataloader objects
    data_set = utils.OdeDataset(solutions=sols, win=20)
    data = DataLoader(dataset=data_set, batch_size=40, shuffle=True)

    # construct the aproximator for the derivative of x
    approximator = utils.ApproxDynamics()

    # set up the optimizer
    optimizer = optim.Adam(approximator.parameters(), lr=1e-3)

    # save the initial approximator
    x = torch.linspace(start=-1, end=1, steps=100, dtype=torch.float64)
    z = dynamics(0.0, x.detach().numpy())
    with torch.no_grad():
        g = approximator(x.view(-1, 1), x.view(-1, 1))

    fig, ax = plt.subplots(figsize=figsize)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.plot(x.detach().numpy(), z, label="true")
    ax.plot(x.detach().numpy(), g.detach().numpy(), label="estimate")
    ax.legend()
    # ax.set_title("before")
    plt.grid(which="major", axis="both")
    savefig("before.png")

    # training loop
    num_epochs = 200
    loss_list = []
    for epoch in range(num_epochs):
        avg = []
        for t, y0, y_true in data:

            optimizer.zero_grad()
            estimate = odeint_adjoint(approximator, y0, t[0, :])
            loss = loss_func(y_true, estimate)
            loss.backward()
            optimizer.step()
            avg.append(loss.detach().numpy())
        loss_list.append(np.mean(avg))
        print(f"Epoch : {epoch}, loss : {loss_list[-1]}")

    with torch.no_grad():
        g = approximator(x.view(-1, 1), x.view(-1, 1))
    fig, ax = plt.subplots(figsize=figsize)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.plot(x.detach().numpy(), z, label="true")
    ax.plot(x.detach().numpy(), g.detach().numpy(), label="estimate")
    ax.legend()
    # ax.set_title("after")
    plt.grid(which="major", axis="both")
    savefig("after.png")

    fig, ax = plt.subplots(figsize=figsize)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.plot(np.log(np.array(loss_list)))

    savefig("loss.png")
    print("Done!")
    return None


if __name__ == "__main__":
    main()
