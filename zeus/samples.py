import numpy as np

class samples:
    '''
    Creates object that stores the samples.
    Args:
        nsteps (int): Number of steps/generations.
        nwalkers (int): Number of walkers.
        ndim (int): Number of dimensions/paramters

    '''

    def __init__(self, ndim, nwalkers):
        """
        Initialise object to store the samples.
        """
        self.initialised = False
        self.index = 0
        self.ndim = ndim
        self.nwalkers = nwalkers


    def extend(self, n):
        if self.initialised:
            ext = np.empty((n,self.nwalkers,self.ndim))
            self.samples = np.concatenate((self.samples,ext),axis=0)
        else:
            self.samples = np.empty((n,self.nwalkers,self.ndim))
            self.initialised = True


    def save(self, x):
        """
        Save sample into the storage.
        Args:
            x (ndarray): Sample to be appended into the storage.
        """
        self.samples[self.index] = x
        self.index += 1


    @property
    def chain(self):
        return np.swapaxes(self.samples, 0, 1)


    def flatten(self, burn=None, thin=1):
        """
        Flatten samples by thinning them, removing the burn in phase, and combining all the walkers.

        Args:
            burn (int): Number of burn-in steps to be removed from each walker.
            thin (int): Thinning parameter (the default value is 1).

        Returns:
            2D object containing the ndim flattened chains.
        """

        nsteps = np.shape(self.chain)[1]

        if burn is None:
            burn = int(nsteps/2)
        return self.chain[:,burn::thin,:].reshape(-1,self.ndim)
