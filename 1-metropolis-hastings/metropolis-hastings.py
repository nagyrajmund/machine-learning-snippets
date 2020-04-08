import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
"""
    A small demonstration of Metropolis-Hastings sampling. 
    
    The target distribution is a simple bimodal Gaussian mixture.
    
    In this example, we explore the effect of changing the variance 
    of the (Gaussian) proposal distribution in the M-H algorithm.
"""

def p(x, a_1=0.5, mu_1=0, var_1=1, a_2=0.5, mu_2=3, var_2=0.5):
    """ The distribution of the data, known up to a normalizing constant.
    
    Returns the pdf of a simple bimodal Gaussian mixture, i.e.:
            p(x) = a_1 * N(x|mu_1, var_1) + a_2 * N(x|mu_2, var_2)
    """

    return a_1 * norm.pdf(x, mu_1, np.sqrt(var_1)) \
         + a_2 * norm.pdf(x, mu_2, np.sqrt(var_2))

def q(x, x_given, var):
    """Return the pdf of the proposal (conditional) distribution."""
    return norm.pdf(x,x_given,np.sqrt(var))

def q_sample(x, var):
    """Return a sample from the proposal (conditional) distribution."""
    return np.random.normal(x,np.sqrt(var))

def metropolis_hastings(var, N):
    u = np.random.uniform(size=N)
    x = np.empty(shape=N)
    
    x[0] = 0 # initialize x_0
    for i in range(N-1):
        x_star = q_sample(x[i], var)

        A = min(1, (p(x_star) * q(x[i],   x_star, var)) / 
                   (p(x[i])   * q(x_star, x[i],   var))    )

        if u[i] < A:
            x[i+1] = x_star
        else:
            x[i+1] = x[i]

    return x

def visualize_results(x, var, ax):
    """ Visualize the results of the sampling procedure in a 3D plot."""

    # Plot the random walk
    ys = [i for i in range(x.shape[0])]
    ax.plot(x, ys, zs=0, zdir='z', label='Random walk')

    # Plot the true distribution
    true_distr_linspace = np.linspace(-2, 5, x.shape[0])
    ax.plot(true_distr_linspace, p(true_distr_linspace), zs=0, zdir='y', label='True distribution')

    # Plot the histogram of the random walk
    hist, bins = np.histogram(x, bins=100, density=True)
    xs = (bins[:-1] + bins[1:])/2
    ax.step(xs, hist, zs=x.shape[0], zdir='y', label='Estimate')
    
    # Formatting
    plt.title('$\sigma^2=${}'.format(var))
    ax.grid(False)
    ax.set_xlabel("x", fontsize=12, labelpad=12)
    ax.set_ylabel("iterations", fontsize=12, labelpad=12)
    ax.set_zlabel("p(x)", fontsize=12, labelpad=12)
    ax.legend(loc='top left')

if __name__ == "__main__":    
    N = 5000 # Number of iterations

    fig = plt.figure()
    fig.suptitle("{} iterations of the Metropolis-Hastings"
                 "algorithm for different $\sigma$ values".format(N), fontsize=24)

    var = 0.01 # Variance of the proposal distribution
    ax = fig.add_subplot(2, 2, 1, projection='3d')
    visualize_results(metropolis_hastings(var,N), var, ax)
    
    var = 1
    ax = fig.add_subplot(2, 2, 2, projection='3d')
    visualize_results(metropolis_hastings(var,N), var, ax)
    
    var = 10
    ax = fig.add_subplot(2, 2, 3, projection='3d')
    visualize_results(metropolis_hastings(var,N), var, ax)
    
    var = 1000
    ax = fig.add_subplot(2, 2, 4, projection='3d')
    visualize_results(metropolis_hastings(var,N), var, ax)
    plt.tight_layout()
    plt.show()