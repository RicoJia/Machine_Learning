import numpy as np


def mean_squared_error(estimates, targets):
    """
    Mean squared error measures the average of the square of the errors (the
    average squared difference between the estimated values and what is
    estimated. The formula is:

    MSE = (1 / n) * \sum_{i=1}^{n} (Y_i - Yhat_i)^2

    Implement this formula here, using numpy and return the computed MSE

    https://en.wikipedia.org/wiki/Mean_squared_error
    Input: 2 numpy arrays
    Output: Mean Square Error
    """
    n = estimates.shape[0]
    mse = 1.0 / n * (((targets - estimates) * (targets - estimates)).sum())
    return mse
