import numpy as np

def normal(X, mu, sigma):
    """Return likelihood of data given parameters"

    Computes the likelihood that the data X have been generated
    from the given parameters (mu, sigma) of the one-dimensional
    normal distribution.

    Args:
        X: vector of point samples
        mu: mean
        sigma: standard deviation
    Returns:
        a scalar likelihood
    """
    c = -1.0/(2.0*sigma**2.0)
    b = (1.0/(2.0*np.pi*sigma**2.0))**(0.5*np.size(X))
    lik = b * np.exp( c * np.sum( (X-mu)**2.0 ) )
    return lik