# Introduction

In this repository, I am attempting to understand diffusion models from scratch using [Step-by-Step Diffusion: An Elementary Tutorial](https://arxiv.org/abs/2406.08929).

# Experiments

The `experiments` folder contains subfolders corresponding to different experiements. Each experiment explores a small concept, performs a sanity check or creates a proof-of-concept setup to implement an idea (there are only four experiments right now but more will be added as I learn and explore diffusion models further).

There is a description at the top of a main script in each experiment that details what the experiment aims to do. There is also a `figures` subfolder in each experiment containing resulting figures that should be produced by running the experiment. These figures should give some idea on what the experiment is about. To run an experiment (check [setup](#installation) first):

```
cd experiments/<experiment-name>

python <name-of-main-script>.py
```

Below is a short description of each experiment with illustrative figures.

## Experiment-0

For a known 1D-Gaussian data distribution $p_{o}(x_{o}) = \mathcal{N}(\mu_{0}, \sigma_{o}^{2})$, this experiment demonstrates that starting from samples from a base distribution $p_{1}(x_{1}) = \mathcal{N}(0, \sigma_{q}^{2})$, reverse sampling of DDPM and DDIM (Alg. (1) and (2) respectively in the tutorial) produces a histogram that resembles the original data distribution $p_{o}(x_{o})$. The conditional expectation needed for DDPM and DDIM can be obtained manually and is given by $$E[x_{t-\Delta t}|x_{t}] = \frac{1}{\sigma_{o}^{2} + \sigma_{q}^{2}t} \Big((\sigma_{o}^{2} + \sigma_{q}^{2}(t - \Delta t)) x_{t} + \sigma_{q}^{2}\Delta t \mu_{o} \Big).$$ 

![experiment-0](experiments/experiment-0/figures/experiment-0-samples-ddm-ddim-toy-example.jpg)

## Experiment-1

For a degenerate data distribtion in 2D i.e., $p_o(\mathbf{x_{o}}) = \delta(\mathbf{x_{o}} - \mathbf{a})$, this experiment plots the trajectories of DDIM samples (Alg. (2) in the tutorial) during reverse sampling when starting from a base distribution  $p_{1}(x_{1}) = \mathcal{N}(\mathbf{0}, \sigma_{q}^{2}\mathbf{I})$. The trajectories move towards the single data point. Here $$E[x_{t-\Delta t}|x_{t}] = \frac{(t - \Delta t)}{t} \mathbf{x_{t}} + \frac{\Delta t}{t} \mathbf{a}.$$  

![experiment-1](experiments/experiment-1/figures/experiment-1-sample-trajectories-ddim.jpg)

## Experiment-2

Same as `experiment-1` but with data distribution being $p_{o}(\mathbf{x_{o}}) = 0.5 \delta(\mathbf{x_{o}} - \mathbf{a}) + 0.5 \delta(\mathbf{x_{0}} - \mathbf{b})$. The trajectories move towards one of the two data points. 

Here,

$$E[x_{t-\Delta t} | x_{t}] = p(x_{o} = a | x_{t}) * E[x_{t-\Delta t} | x_{t}, x_{o} = a] + p(x_{o} = b | x_{t}) * E[x_{t-\Delta t} | x_{t}, x_{o} = b],$$

where the $E[x_{t-\Delta t} | x_{t}, x_{o} = a]$ is the same as in `experiment-1` and 

$$p(x_{o} = a | x_{t}) = \frac{p_{o}(x_{o} = a) p(x_{t} | x_{o} = a)}{p_{t}(x_{t})}$$

by Baye's rule. The expressions for $p(x_{t} | x_{o} = a)$ and $p_{t}(x_{t})$ can be obtained easily from the forward diffusion process.


![experiment-2](experiments/experiment-2/figures/experiment-2-sample-trajectories-ddim.jpg)

## Experiment-3

For synthetic Swiss roll dataset, this experiment trains a fully connected NN with ReLU activations to learn $E[x_{0}| x_{t}]$ from data. Then $E[x_{t-\Delta t}| x_{t}]$ is obtained via its relationship to $E[x_{0}| x_{t}]$ (eq. (24) in the tutorial) which is then used in the reverse sampling of DDPM and DDIM. 

![experiment-3](experiments/experiment-3/figures/eval_ddpm_samples.jpg)


# Installation

```
git clone https://github.com/mosama182/diffusion_models.git
```
Create new conda environment and install requirements.

```
pip install requirements.txt
```

