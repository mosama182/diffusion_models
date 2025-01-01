In this repository, I am attempting to understand diffusion models from scratch using Step-by-Step diffusion tutorial [here](https://arxiv.org/abs/2406.08929).

The `experiments` folder contains subfolders: `experiment-0`, `experiment-1` and so on. Each experiment explores a small concept, do a sanity check 
to test my own understanding of a concept or something similar. For example:

- `experiment-0`: is simply demonstrating that using DDPM/DDIM sampling for a known data distribution (so known `E[x_{t-\delta_t}|x_{t}]`) produces
                  samples that match the underlying data distribution.

- `experiment-1`: is about tracking the trajectories of DDPM/DDIM samples for a dirac delta data distribution to see that the trajectories
                  converge to that single data point.

- `experiment-2`: same as `experiment-1` but the underlying data distribution contains two diract deltas. So you can see the trajectories converge
                  to either of the two points.

- `experiment-3`: train a fully connected NN with ReLU activations to learn E[x_{0}| x_{t}] from synthetic Swiss roll data and then produce samples
                  from DDPM/DDIM sampling. 

There is a description on the main script in each experiment that details what the experiment aims to do (I plan to add a `readme` file later). 
There is also a `figures` subfolder in each experiment containing resulting figures that should be produced by running the main script. 
These figures should give you some idea on what the experiment is about. 
