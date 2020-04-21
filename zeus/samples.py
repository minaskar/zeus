import numpy as np

class samples:
    '''
    Creates object that stores the samples.
    Args:
        ndim (int): Number of dimensions/paramters
        nwalkers (int): Number of walkers.

    '''

    def __init__(self, ndim, nwalkers):
        self.initialised = False
        self.index = 0
        self.ndim = ndim
        self.nwalkers = nwalkers


    def extend(self, n):
        """
        Method to extend saving space.
        Args:
            n (int) : Extend space by n slots.
        """
        if self.initialised:
            ext = np.empty((n,self.nwalkers,self.ndim))
            self.samples = np.concatenate((self.samples,ext),axis=0)
            ext = np.empty((n,self.nwalkers,))
            self.logp = np.concatenate((self.logp,ext),axis=0)
        else:
            self.samples = np.empty((n,self.nwalkers,self.ndim))
            self.logp = np.empty((n,self.nwalkers,))
            self.initialised = True


    def save(self, x, logp):
        """
        Save sample into the storage.
        Args:
            x (ndarray): Sample to be appended into the storage.
            logp (ndarray): log probability corresponding to the samples
        """
        self.samples[self.index] = x
        self.logp[self.index] = logp
        self.index += 1


    @property
    def chain(self):
        """
        Chain property.

        Returns:
            3D array of shape (nwalkers,nsteps,ndim) containing the samples.
        """
        return self.samples


    @property
    def logprob(self):
        """
        Chain property.

        Returns:
            2D array of shape (nwalkers,nsteps) containing the log-probabilities.
        """
        return self.logp


    @property
    def length(self):
        """
        Number of samples per walker.

        Returns:
            The total number of samples per walker.
        """
        length, _, _ = np.shape(self.chain)
        return length


    def flatten(self, burn=0, thin=1):
        """
        Flatten samples by thinning them, removing the burn in phase, and combining all the walkers.

        Args:
            burn (int): Number of burn-in steps to be removed from each walker (default is 0).
            thin (int): Thinning parameter (the default value is 1).

        Returns:
            2D object containing the ndim flattened chains.
        """

        return self.chain[burn::thin,:,:].reshape((-1,self.ndim), order='F')


    def get_log_prob(self, burn=0, thin=1):
        """
        Flatten log probability by thinning the chain, removing the burn in phase, and combining all the walkers.

        Args:
            burn (int): Number of burn-in steps to be removed from each walker (default is 0).
            thin (int): Thinning parameter (the default value is 1).

        Returns:
            1D object containing the logprob of the flattened chains.
        """
        
        return self.logprob[burn::thin,:].reshape((-1,), order='F')
