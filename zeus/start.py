import numpy as np
import logging

def jitter(x , nwalkers, ndim):

    input_shape = np.shape(x)

    if input_shape == (nwalkers, ndim):
        return x
    elif input_shape == (ndim, nwalkers):
        return x.T
    elif input_shape == (ndim,):
        return x * (1.0 + np.random.randn(nwalkers, ndim))
    else:
        logging.warning('Please provide a valid starting point e.g. (nwalkers,ndim) or (ndim,)!')
