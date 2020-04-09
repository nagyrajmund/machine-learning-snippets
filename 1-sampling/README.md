In this folder, you can find the implementation of two sampling techniques that belong to the family of __Markov chain Monte Carlo__ methods. 

## Metropolis-Hastings
The __Metropolis-Hastings algorithm__ can be used to draw samples from a distribution P(x) that is known up to a normalizing constant Z.
In machine learning, this is very useful as the normalizing constants of the posterior distributions are generally intractable as they are very high dimensional integrals.

The algorithm generates samples iteratively in such a way that the distribution of the values approximate the density P(x) as the number of samples grows.
The noise variance of the proposal distribution is a key hyperparameter that greatly affects the quality of the approximation.
This demonstration aims to show the effect of different choices for this parameter, as seen below:

![](https://raw.githubusercontent.com/nagyrajmund/machine-learning-snippets/master/1-sampling/plot_metropolis-hastings.png)
## Bootstrap filter
One of the most basic sampling techniques is __rejection sampling__, where first samples are generated from an arbitrary proposal distribution, then they are either accepted or thrown away based on how likely they are under the target distribution. 
Throwing samples away can be very expensive, so one could improve the method by assigning weights to each particle (sample).
The samples should be weighed relative to their likelihood under the proposal and the target distributions. This technique is called __importance sampling__.

If we adopt importance sampling for online approximation (i.e. computing a trajectory of particles that keeps growing), we end up with a method called __Sequential Importance sampling__.
A core problem of SIS is that as the number of timesteps grows, the distribution of the importance weights becomes more and more skewed, and we will end up with only one non-zero weight after a few timesteps.

The __Bootstrap filter__ (also known as Sequential Importance Sampling with Resampling) aims to remedy this problem by resampling particles after each timestep, based on their current importance weights.

In this demonstration, the algorithm is applied to a nonlinear, non-Gaussian model of the following form:
 
 ![x_t = \frac{1}{2}x_{t-1} + 25\frac{x_{t-1}}{1 + x_{t-1}^2} + 8\cos(1.2t) + v_t](https://render.githubusercontent.com/render/math?math=x_t%20%3D%20%5Cfrac%7B1%7D%7B2%7Dx_%7Bt-1%7D%20%2B%2025%5Cfrac%7Bx_%7Bt-1%7D%7D%7B1%20%2B%20x_%7Bt-1%7D%5E2%7D%20%2B%208%5Ccos(1.2t)%20%2B%20v_t)
  
  ![y_t = \frac{x_t^2}{20} + w_t](https://render.githubusercontent.com/render/math?math=y_t%20%3D%20%5Cfrac%7Bx_t%5E2%7D%7B20%7D%20%2B%20w_t)
 
 where w and v are mutually independent Gaussian noise terms.
 
 The results can be seen below, with the original signal in black, while the colored distributions are the estimated filtering distributions at particular timesteps:
 
 ![](https://raw.githubusercontent.com/nagyrajmund/machine-learning-snippets/master/1-sampling/plot_bootstrap.png)
