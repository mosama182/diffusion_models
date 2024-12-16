"""
#######
DATASET
#######
This experiment is to learn a diffusion model
for a toy Swissroll dataset. And then use the 
learned model in DDIM reverse sampler to see
the trajectories and learned data points. The data
is in R^{2}.

#######################
WHAT WE AIM TO ESTIMATE
#######################
The model will be trained to estimate E[x_t | x_{t + \delat}]. 
We know that E[Y|X] is the MMSE estimator. Here Y = x_{t} and 
X = [x_{t + \deltat}, t + \deltat].  
So the input to the model will be x_{t + \deltat}, t + \deltat 
and the output will \hat{x_t} and we will be minimizing the mean
square error.

#########################
GETTING A TRAINING SAMPLE
#########################
Given a datapoint x_{0}, we need to produce a training sample 
(x_{t}, x_{t + \deltat}, t + \deltat) and for this we will be
creating two functions/classes:
    1. Schedule: is a class that defines N fixed time points
    in interval [0, 1]. When generating a training sample, we call
    the schedule to give us value of 't' which is randomly sampled from 
    the N fixed points 
    
    2. generate_training_sample(x0, sigma2_q, \deltat, Schedule): It calls the schedule to 
    get a value of t and then produces training sample a/c to:
        t = Schedule(x0)
        \eta_{t} ~ N (0, sigma2_q * t * I)
        x_{t}  = x_{0} + \eta_{t}
        epsilon ~ N (0, \sigma2_q * \deltat) 
        x_{t + \deltat} = x_{t} + epsilon  

########
NN MODEL
########
The model is simply a fully connected NN. It has 3 layers with 128 neurons per
layer and ReLU activation. The raw input to the model x_{t+\deltat} and t+\deltat
are converted to a 4-dimensional vector. The first 2 dimensions are simply the raw 
x_{t + \deltat} itself and the next 2 dimensions is a transformed times embedding
\phi(t+\delta) = [sin(2 * pi * (t + \deltat)), cos(2 * pi * (t + \deltat))]. So the 
first fully connected layer has input dimension of 4. The output layer has dimenson 
2, and estimated \hat{x_{t}}

"""