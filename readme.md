# Experiments with `torchdiffeq`
This repository contains code demonstrating the use of
`torchdiffeq` on a simple experiment (somewhat even simpler than the one in the original repo).

To get the figures below, run `demo.py` under `./code/` -- make sure you run the script from that directory.

## Differentiating a Parametric ODE

Consider a one-dimensional nonlinear ODE of the form 
$$
\frac{d}{dt} x(t) = f_{\Theta}\,\bigl(x(t)\bigr),
$$
where $f_{\Theta} : \mathbb{R} \to \mathbb{R}$ is a parametric function that varies smoothly with respect to its parameters $\Theta$ (e.g., a neural network).

In practice, the function $x$ can be computed numerically using an integrator like `scipy`'s `solve_ivp`.

While the notation does not suggest it, notice that $x(t)$ depends on $\Theta$. Therefore, for a given initial value we can change the behavior of $x(t)$ by varying $\Theta$. Ignoring issues on existence, we can also talk about the derivative of $x(t)$ with respect to $\Theta$, for each value of $t$, namely
$$
\frac{\partial}{\partial \Theta} x(t).
$$

`torchdiffeq` provides an efficient numerical procedure that relies on the **adjoint method** to obtain the derivative above.

## `torchdiffeq` in Action
Let us now discover how we can use this derivative to approximate a nonlinear ODE.

Consider the following ODE : 
$$
\dot{x}(t)=\sin(2\pi\, x(t)).
$$
Notice that when $x$ is a multiple of $0.5$, the right hand side equals zero. These constitute the fixed points of the ODE. 

However, some of these fixed points are stable, and some are unstable. Consider the phase portrait, shown below. The stable fixed points occur whenever the dynamics crosses the $x$-axis with a negative slope.

![](./figures/dynamics.png)

Specifically, when $x =\epsilon$, the dynamics pushes $x$ away from the fixed point at $0$. This continues until $x$ reache the next stable point on the right, at $0.5$. 

The converse occurs when we start from $x = -\epsilon$. The point gets pushed away from 0 until it reaches the next stable fixed point at $x = -0.5$. These two solutions, corresponding to the two slightly different initial conditions are shown below.

![](./figures/phase.png)

Suppose now that we observe these two curves, and we would like to estimate $f(\cdot)$ in 
$$
\dot{x} = f(x).
$$
Let $X$ denote the available data. 

We will estimate $f(\cdot)$ using a simple neural network. Specifically, let $f_{\Theta}(\cdot)$ denote a multilayer perceptron with two hidden layers having 100 neurons each. We set the input and output sizes as $1$, to match our problem. We also use $\tanh$ as our nonlinearity -- there is no activation at the last layer.

To set up a loss function, 
1. we extract a segment from the available data, of the form $x(t_0), \ldots, x(t_n)$. 
2. We then simulate $x_{\Theta}(t_0), \ldots, x_{\Theta}(t_n)$ the state dynamics are defined via
$\dot{x_{\Theta}} = f_{\Theta}(x_{\Theta})$, and $x_{\Theta}(t_0)$ is set to $x(t_0)$.
3. Finally, using $x$ and $x_{\Theta}$, we form the loss function
$$\mathcal{L}(\Theta) = \frac{1}{n} \sum_{i=1}^n |x(t_i) - x_{\Theta}(t_i)|^2.
$$

Given the differentiable loss function, we obtain $\partial_{\Theta} \mathcal{L}(\Theta)$ via `torchdiffeq`, and take steps so as to minimize the average of $\mathcal{L}(\Theta)$ (averaged over the segments).

Note that we are free to form batches, which we do, to come up with less noisy gradients of the loss function.

Below are the results of this optimization procedure. 

First, before starting the initialization, the MLP $f_{\Theta}$ is an arbitrary function, far from $\sin(2\pi x)$, as shown below.
![](./figures/before.png)

But after training, we get a much closer function, as shown below. Notice that we provide data to the network only in the range $x \in [-0.5, 0.5]$. Therefore, beyond that interval, we don't really expect $f_{\Theta}$ to well-approximate the unknown non-linearity.
![](./figures/after.png)

As a check of sanity, the logarithm of the loss trajectory is shown below.
![](./figures/loss.png)

*Ilker Bayram, ibayram@ieee.org, 2021*.
