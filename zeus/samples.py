import numpy as np
from collections import deque

class samples:
    '''
    Creates object that stores the samples.
    Args:
        nsteps (int): Number of steps/generations.
        nwalkers (int): Number of walkers.
        ndim (int): Number of dimensions/paramters

    '''

    def __init__(self, nsteps, nwalkers, ndim):
        """
        Initialise a deque object to store the samples.
        """
        self.nsteps = nsteps
        self.nwalkers = nwalkers
        self.ndim = ndim
        self.samples = deque()

    def append(self, x):
        """
        Append sample into the storage.
        Args:
            x (ndarray): Sample to be appended into the storage.
        """
        self.samples.append(x.tolist())

    @property
    def chain(self):
        return np.swapaxes(np.array(self.samples), 0, 1)

    def flatten(self, burn=None, thin=1):
        """
        Flatten samples by thinning them, removing the burn in phase, and combining all the walkers.

        Args:
            burn (int): Number of burn-in steps to be removed from each walker.
            thin (int): Thinning parameter (the default value is 1).

        Returns:
            2D object containing the ndim flattened chains.
        """
        if burn is None:
            burn = int(self.nsteps/2)
        return self.chain[:,burn::thin,:].reshape(-1,self.ndim)
