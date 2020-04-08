from matplotlib import cm
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import gaussian_kde

""" A short demonstration of a sequential Monte Carlo technique called Bootstrap Filter,
    also known as "sequential importance sampling with resampling".

    In this example, we will apply the algorithm to the following 
    nonlinear, non-Gaussian model:

        x[t] = v[t] + 8 * cos(1.2t) + x[t-1] / 2 + 25 * ( x[t-1] / (1 + x[t-1]^2) ) 
        y[t] = w[t] + x[t]^2 / 20

    where w[t] and v[t] are Gaussian noise terms with a variance of 1 and 10, respectively.
"""

v_variance   = 10 # v ~ N(0,10)
w_variance   = 1  # w ~ N(0,1)
x_0_variance = 10 # x[0] ~ N(0,10)

def sample_x(x_prev, t):
    """Sample x[t] given x[t-1] according to the model."""
    v_t = np.random.normal(0, np.sqrt(v_variance))

    return  v_t + 8 * np.cos(1.2 * t) + 0.5 * x_prev + 25 * (x_prev / (1 + x_prev**2))

def p_y(x, y):
    """Return the conditional distribution p(y | x)."""
    return norm.pdf(y, x**2 / 20, 1)
   
def simulate_system(T):
    """Simulate the behaviour of the system for T timesteps."""
    X = np.empty(T)
    Y = np.empty(T)

    X[0] = np.random.normal(0,np.sqrt(x_0_variance))

    for t in range(1,T):
        X[t] = sample_x(X[t-1], t)
        Y[t] = X[t]**2 / 20 + np.random.normal(0,1)

    return X, Y

def bootstrap_filter(Y_data, T, N):
    """ Apply the bootstrap filter to the model based on the observations in Y_data.

    Y_data : The given Y values up until the T-th timestep
    T : number of timesteps
    N : number of particles

    Returns the final particles in (T,N)-shaped ndarray.
    """
    X = np.empty((T,N)) # N particles
    W = np.empty((T,N)) # Importance weights for each particle

    # Initialize x[0]
    X[0,:] = np.random.normal(0, np.sqrt(x_0_variance), X[0].shape)

    for t in range(1,T):
        # IS step
        for i in range(N):
            X[t][i] = sample_x(X[t-1][i], t)
            W[t][i] = p_y(X[t][i], Y_data[t])
        
        # Normalize the importance weights
        W[t] /= np.sum(W[t])

        # Resample with replacement using W[t] as the selection probabilities
        idx = np.random.choice(np.arange(N), N, replace=True, p=W[t])
        X[t,:] = X[t,idx]

    return X

if __name__ == "__main__":
    T = 100  # Number of timesteps
    N = 1000 # Number of particles

    X_data, Y_data = simulate_system(T)
    
    X_result = bootstrap_filter(Y_data, T, N)

    # Create a 3D figure
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    step = 8 # Plot the estimated distributions at every 8th timestep
    for t in range(0,T,step):
        x = np.linspace(-25,25,N)
        y = gaussian_kde(X_result[t, :]).evaluate(x)
        ax.plot(x, y, zs=t, zdir='y', linewidth=1.5)

    # Plot the generating signal
    ax.plot(X_data[0:T], np.arange(0,T), c='k', zs=0, zdir='z', label='Generating signal')
    plt.title("Bootstrap filtering with {} particles".format(N))
    ax.set_xlabel('$x_t$')
    ax.set_ylabel('t')
    ax.set_zlabel('p($y_t|x_t$)')
    ax.grid(False)
    ax.legend()
    plt.show()