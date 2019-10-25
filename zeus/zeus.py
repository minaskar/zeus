import numpy as np
from itertools import permutations
from tqdm import tqdm

class zeus:

    def __init__(self,
                 logp,
                 nwalkers,
                 ndim):

        self.logp = logp
        self.nwalkers = nwalkers
        self.ndim = ndim


    def run(self,
            start,
            nsteps=1000,
            width=1.0,
            maxsteps=1,
            thin=1):

        dummy = 1.0


    def _slice(self,
               x,
               direction):

        dummy = 1.0
