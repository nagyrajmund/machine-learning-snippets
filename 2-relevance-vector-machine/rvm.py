giimport numpy as np 
from numpy.linalg import norm, inv
import matplotlib.pyplot as plt

""" An implementation of the Relevance Vector Machine regression (proposed by Mike Tipping in 2000), 
which can be viewed as the reinterpretation of the SVM under a Bayesian framework.

This model aims to remedy three major shortcomings of SVMs: 
    1) computational complexity
        -> the RVM needs significantly fewer support vectors for inference
        -> the hyperparameters (e.g. the slack) are automatically selected 
           by the Bayesian approach
    2) non-probabilistic outputs
        -> the RVM aims to model the complete posterior
    3) SVM kernels have to satisfy Mercer's condition
        -> with RVMs, this requirement can be relaxed

For more details, read the original paper at http://www.jmlr.org/papers/volume1/tipping01a/tipping01a.pdf.
"""

class RVM():
    float_eps = 2e-16

    def __init__(self, X, t, kernel=None, fixed_sigma=False, use_logging=False):
        """
            X           (N,D) array of inputs.
            t           (N,) vector of labels.
            kernel      Kernel function. Takes two (N,) column vectors as input.
            fixed_sigma If True, only estimate alpha in the learning process.
            use_logging Bool value, indicating whether the progress should be printed.
        """
        if kernel is None:
            kernel = self._default_kernel
    
        self.kernel = kernel
        self.N = X.shape[0]
        self.D = X.shape[1]
        self.X = X
        self.t = t
        # Indices of the kept (= relevance) vectors, used in the predictive posterior
        self.absolute_kept_idx = np.array([i for i in range(self.N+1)]) # Will be pruned!
        # Control whether the noise variance should be reestimated during training
        self.fixed_sigma = fixed_sigma
        self.use_logging = use_logging
    
    def train(self, alpha_0 = 1, sigma_sq_0 = 1, maxiter=1000):
        """ Estimate the optimal alpha and sigma² parameters for predicting new values.
        
            alpha_0     (N+1,) dimensional array of initial alpha values.
            sigma_sq_0  Initial value for sigma²
            maxiter     Maximum number of iterations.
        """
        self.Phi = self._compute_design_matrix()
        # The alpha values determine the relevance of the basis functions
        alphas = np.array([alpha_0] * (self.N + 1)).T
        sigma_sq = sigma_sq_0 # Noise variance

        converged = False
        iterc = 0
        while not converged and iterc < maxiter:
            iterc += 1

            # Approximate alpha and sigma
            alphas_new, sigma_new_sq, kept_idx \
                = self._iterate_approximation(alphas, sigma_sq)

            # Check for convergence
            alpha_diff = np.sum(abs(alphas_new - alphas[kept_idx]))
            if not self.fixed_sigma:
                sigma_diff = abs(sigma_new_sq - sigma_sq)
                converged = max(alpha_diff, sigma_diff) < self.float_eps
            else:
                converged = alpha_diff < self.float_eps

            # Update alpha and sigma
            alphas = alphas_new
            if not self.fixed_sigma:
                sigma_sq = sigma_new_sq

        self._log("Final iteration count: " + str(iterc))
        self.alphas_max = alphas_new
        self.sigma_sq_max = sigma_new_sq

    def predictive_gaussian_params(self, x_vec, point_estimate=True):
        """ Return the parameters of the predictive Gaussian for each point in x_vec. 
        
            x_vec           (N,D) array of inputs.
            point_estimate  If True, only the predictive means are returned.

            Returns the means as an (N,) array, and the 
                    variances as well (if point_estimate is false).
        """
        alphas = self.alphas_max
        sigma_sq = self.sigma_sq_max
        Sigma = self._Sigma_formula(sigma_sq, alphas)
        mu = self._mu_formula(sigma_sq, Sigma)
        
        phi_x = [self._phi_x(x) for x in x_vec]
        N = len(x_vec)
        pred_means = np.array([mu.T @ phi_x[i] for i in range(N)])
        if point_estimate:
            return pred_means

        pred_vars = [sigma_sq + phi_x[i].T @ Sigma @ phi_x[i] for i in range(N)]

        return pred_means, np.array(pred_vars).T

    def _iterate_approximation(self, alphas, sigma_sq):
        """ Perform a single iteration of approximating sigma and alpha. 
        
        Return the updated values for alpha, sigma and the kept basis indices.
        """
        # Compute the updated values for Sigma, mu and gamma
        Sigma   = self._Sigma_formula(sigma_sq, alphas)
        mu      = self._mu_formula(sigma_sq, Sigma).ravel()
        gammas  = 1 - alphas * np.diag(Sigma)

        # Prune the basis functions based on the gamma values
        # (high gammas indicate zero weights)
        kept_idx = gammas > 10e-12
        gammas = gammas[kept_idx]
        mu = mu[kept_idx]
        self.absolute_kept_idx = self.absolute_kept_idx[kept_idx]
        self.Phi = self.Phi[:, kept_idx] 
        
        # Compute the new alphas
        alphas_new = gammas / (mu ** 2)
        
        if not self.fixed_sigma:
            sigma_new_sq = norm(self.t - self.Phi @ mu) ** 2 / (self.N - np.sum(gammas))
        else:
            sigma_new_sq = sigma_sq

        return alphas_new, sigma_new_sq, kept_idx

    def _Sigma_formula(self, sigma_sq, alphas):
        """ Return the updated value of Sigma, i.e. the covariance of the 
        posterior over the weights.
        """
        return inv((1 / sigma_sq) * self.Phi.T @ self.Phi + np.diag(alphas))

    def _mu_formula(self, sigma_sq, Sigma):
        """
        Return the updated value of mu, i.e. the mean of the 
        posterior over the weights.
        """
        return (1 / sigma_sq) * Sigma @ self.Phi.T @ self.t

    def _phi_x(self, x):
        Phi_x = [1]
        Phi_x += [self.kernel(x, self.X[i - 1]) for i in self.absolute_kept_idx[1:]]
        return np.array(Phi_x)

    def _compute_design_matrix(self):
        """ Compute and return the (N, N+1) dimensional design matrix.

            The i-th row of of the design matrix (i=1..N) is
                Phi_i = [1, K(x_i, x_1)], ... , K(x_i, x_N)]

        where K is the kernel function.
        """        
        # In each row, the first element is 1 
        # and the others are kernel function values
        Phi = np.asarray([
                [1] + [self.kernel(self.X[i], self.X[j])
                    for i in range(self.N)] for j in range(self.N)])
        return Phi

    def _default_kernel(self, x, y):
        """ Gaussian kernel as defined in the paper (eq. (30)) with r = 0.5. """
        return np.exp(-(norm(x-y))**2 / (0.5**2))
    
    def _log(self, msg):
        if self.use_logging:
            print("[RVM]:" + msg)
