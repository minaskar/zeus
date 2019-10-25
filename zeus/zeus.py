import numpy as np


class zeus:

    def __init__(self, logp, nwalkers, ndim):
        self.logp = logp
        self.nwalkers = nwalkers
        self.ndim = ndim


    def run(self, start, nsteps):
