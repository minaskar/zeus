import numpy as np


def _autocorr_func_1d(x, norm=True):
    x = np.atleast_1d(x)
    if len(x.shape) != 1:
        raise ValueError("invalid dimensions for 1D autocorrelation function")

    def next_pow_two(n):
        i = 1
        while i < n:
            i = i << 1
        return i

    n = next_pow_two(len(x))

    # Compute the FFT and then (from that) the auto-correlation function
    f = np.fft.fft(x - np.mean(x), n=2 * n)
    acf = np.fft.ifft(f * np.conjugate(f))[: len(x)].real
    acf /= 4 * n

    # Optionally normalize
    if norm:
        acf /= acf[0]

    return acf


def _autocorr_time_1d(y, c=5.0):
    f = np.zeros(y.shape[1])
    for yy in y:
        f += _autocorr_func_1d(yy)
    f /= len(y)
    taus = 2.0 * np.cumsum(f) - 1.0

    # Automated windowing procedure following Sokal (1989)
    def auto_window(taus, c):
        m = np.arange(len(taus)) < c * taus
        if np.any(m):
            return np.argmin(m)
        return len(taus) - 1

    window = auto_window(taus, c)
    return taus[window]


def _autocorr_time(samples, c=5.0):

    _, _, ndim = np.shape(samples)

    taus = np.empty(ndim)
    for i in range(ndim):
        taus[i] = _autocorr_time_1d(samples[:,:,i], c)

    return taus
