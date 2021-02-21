import numpy as np
from scipy.fft import fft, ifft

def _autocorr_func_1d(x, norm=True):
    """
    Autocorrelation Function of 1-dimensional chain.

    Args:
        x (array) : 1-dimensional chain.
        norm (bool) : By default norm=True and the autocorrelation function will be normalized.

    Returns:
        The (normalised if norm=True) autocorrelation function of the chain x.
    """
    x = np.atleast_1d(x)
    if len(x.shape) != 1:
        raise ValueError("invalid dimensions for 1D autocorrelation function")

    # Next largest power of 2
    n = 1
    while n < len(x):
        n = n << 1

    # Compute the auto-correlation function using FFT
    #f = np.fft.fft(x - np.mean(x), n=2 * n)
    #acf = np.fft.ifft(f * np.conjugate(f))[: len(x)].real
    f = fft(x - np.mean(x), n=2 * n)
    acf = ifft(f * np.conjugate(f))[: len(x)].real
    acf /= 4 * n

    # Normalize
    if norm:
        acf /= acf[0]

    return acf


def _autocorr_time_1d(y, c=5.0, method='mk'):
    """
    Integrated Autocorrelation Time (IAT) for 1-dimensional chain.

    Args:
        y (array) : (nsteps, nwalkers) array for one parameter.
        c (float) : Truncation parameter of automated windowing procedure of Sokal (1989), default is 5.0.
        method (str) : Method to use to compute the IAT. Available options are ``mk`` (Default), ``dfm``, and ``gw``.

    Returns:
        The IAT of the chain y.
    """

    if method not in ['mk', 'dfm', 'gw']:
        raise ValueError('Please select one of the supported methods i.e. mk (Recommended), dfm, gw.')

    if method == 'mk':
        # Minas Karamanis method
        f = _autocorr_func_1d(y.reshape((-1), order='C'))
    elif method == 'dfm':
        # Daniel Forman-Mackey method
        f = np.zeros(y.shape[1])
        for yy in y:
            f += _autocorr_func_1d(yy)
        f /= len(y)
    else:
        # Goodman-Weary method
        f = _autocorr_func_1d(np.mean(y, axis=0))
    
    taus = 2.0 * np.cumsum(f) - 1.0

    # Automated windowing procedure following Sokal (1989)
    def auto_window(taus, c):
        m = np.arange(len(taus)) < c * taus
        if np.any(m):
            return np.argmin(m)
        return len(taus) - 1

    window = auto_window(taus, c)
    return taus[window]


def AutoCorrTime(samples, c=5.0, method='mk'):
    """
    Integrated Autocorrelation Time (IAT) for all the chains.

    Parameters
    ----------
        samples : array
            3-dimensional array of shape (nsteps, nwalkers, ndim)
        c : float
            Truncation parameter of automated windowing procedure of Sokal (1989), default is 5.0
        method : str
            Method to use to compute the IAT. Available options are ``mk`` (Default), ``dfm``, and ``gw``.

    Returns
    -------
    taus : array
        Array with the IAT of all the chains.
    """

    if method not in ['mk', 'dfm', 'gw']:
        raise ValueError('Please select one of the supported methods i.e. mk (Recommended), dfm, gw.')

    _, _, ndim = np.shape(samples)

    taus = np.empty(ndim)
    for i in range(ndim):
        taus[i] = _autocorr_time_1d(samples[:,:,i].T, c, method)

    return taus
