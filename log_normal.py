import numpy as np

def log_normal(X, mu, sigma):
    """Return log-likelihood of data given parameters"

    Computes the log-likelihood that the data X have been generated
    from the given parameters (mu, sigma) of the one-dimensional
    normal distribution.

    Args:
        X: vector of point samples
        mu: mean
        sigma: standard deviation
    Returns:
        a scalar log-likelihood
    """
    c = -1.0/(2.0*sigma**2.0)
    #b = (1.0/(2.0*np.pi*sigma**2.0))**(0.5*np.size(X))
    #loglik = np.log( b * np.exp( c * np.sum( (X-mu)**2.0 ) ) ) #using this version leads to issues with large N (ie np.size(X)) as we exponentiate a fraction by N in the previous line leading to underflow
    loglik = -0.5*np.size(X)*np.log(2.0*np.pi) - np.size(X)*np.log(sigma) + c * np.sum( (X-mu)**2.0 )
    return loglik