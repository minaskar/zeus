import numpy as np
from collections import deque

class samples:

    def __init__(self, nsteps, nwalkers, ndim):
        self.nsteps = nsteps
        self.nwalkers = nwalkers
        self.ndim = ndim
        self.samples = deque()

    def append(self, x):
        self.samples.append(x.tolist())

    @property
    def chain(self):
        return np.swapaxes(np.array(self.samples), 0, 1)

    def flatten(self, burn=None):
        if burn is None:
            burn = int(self.nsteps/2)
        return self.chain[:,burn:,:].reshape(-1,self.ndim)
