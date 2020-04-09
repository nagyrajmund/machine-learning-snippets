import time
import numpy as np
from numpy.linalg import norm
from rvm import RVM
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import statistics as stats

""" A short demonstration of the RVM on the Boston housing dataset.

In the follow-up paper from 2001, Tipping reports the following results: 
    
    For the RVM:
        average error:          7.46
        # of relevance vectors: 39.0
    For the SVM:
        average error:          8.04
        # of support vectors:   142.8

With a well-chosen kernel radius of 200, we actually manage 
to outperform the reference values, as the code below results in:
    
    - an average error around 4.8, and 
    - an average # of relevance vectors 18.6

which is quite a significant improvement over the SVM results. Not bad!
"""

kernel_radius = 200
kernel = lambda x,y : gaussian_kernel(x,y,kernel_radius)
 
def gaussian_kernel(x,y,r):
    """A gaussian kernel with radius r."""
    return np.exp(-(norm(x-y))**2 / (r**2))

def evaluate_RVM(x_train, y_train, x_test, y_test, max_iterations = 1000):
    """ Train an RVM and evaluate its performance with the given train/test split."""
    regressor = RVM(x_train, y_train,kernel=kernel, use_logging=False)
    regressor.train(maxiter=max_iterations)

    predictions = regressor.predictive_gaussian_params(x_test)
    # Root mean squared error
    error = mean_squared_error(predictions, y_test, squared=False)
    rel_vec_count = len(regressor.absolute_kept_idx)
    
    return error, rel_vec_count

if __name__ == "__main__":
    num_repeat = 20 # We will average the results across several runs
    X, y = load_boston(return_X_y = True)

    errors     = [] # History of root mean square errors
    vec_counts = [] # History of relevance vector counts

    start_time = time.time()
    for i in range(num_repeat):
        print("[Iteration {}]".format(i), end='\t')
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 25)

        error, num_vec = evaluate_RVM(X_train, y_train, X_test, y_test, max_iterations=2000)
        print("Error: {}\t# of relevance vectors: {}".format(error, num_vec))

        errors.append(error)
        vec_counts.append(num_vec)
    
    print('------- Final results -------')
    print('Elapsed time:', time.time() - start_time, "s")
    print('[Error statistics]\t\taverage: {}\tstd. dev.: {}\tMax: {}\tMin: {}'
          .format(stats.mean(errors), stats.stdev(errors), max(errors), min(errors)))
    print('[# of relevance vectors]\taverage: {}\tstd. dev.: {}\tMax: {}\tMin: {}'
          .format(stats.mean(vec_counts), stats.stdev(vec_counts), max(vec_counts), min(vec_counts)))

    