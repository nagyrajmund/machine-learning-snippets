import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

def p(x):
    a_1 = a_2 = 0.5
    mu_1   = 0
    sig2_1 = 1
    mu_2   = 3
    sig2_2 = 0.5

    return a_1 * norm.pdf(x, mu_1, np.sqrt(sig2_1)) + a_2 * norm.pdf(x, mu_2, np.sqrt(sig2_2))

def q(x, x_given, var):
    return norm.pdf(x,x_given,np.sqrt(var))

def q_sample(x, var):
    return np.random.normal(x,np.sqrt(var))

def metropolis_hastings(var):
    N = 5000
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

def combined_plot(x, var, ax):
    # Plot the random walk
    ys = [i for i in range(x.shape[0])]
    ax.plot(x, ys, zs=0, zdir='z')

    # Plot the true distribution
    true_distr_linspace = np.linspace(-2, 5, x.shape[0])
    ax.plot(true_distr_linspace, p(true_distr_linspace), zs=0, zdir='y')

    # Plot the histogram of the random walk
    hist, bins = np.histogram(x, bins=100, density=True)
    xs = (bins[:-1] + bins[1:])/2
    ax.step(xs, hist, zs=x.shape[0], zdir='y')
    
    # Formatting
    plt.title('$\sigma^2=${}'.format(var))
    # ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    plt.tight_layout()

if __name__ == "__main__":
    fig = plt.figure()
    ax = fig.add_subplot(2, 2, 1, projection='3d')
    var = 0.01
    combined_plot(metropolis_hastings(var),var, ax)
    var = 1
    ax = fig.add_subplot(2, 2, 2, projection='3d')
    combined_plot(metropolis_hastings(var),var, ax)
    var = 10
    ax = fig.add_subplot(2, 2, 3, projection='3d')
    combined_plot(metropolis_hastings(var),var, ax)
    var = 1000
    ax = fig.add_subplot(2, 2, 4, projection='3d')
    combined_plot(metropolis_hastings(var),var, ax)
    plt.grid(False)
    plt.show()