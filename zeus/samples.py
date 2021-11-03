import numpy as np

class samples:
    '''
    Creates object that stores the samples.
    Args:
        ndim (int): Number of dimensions/parameters
        nwalkers (int): Number of walkers.

    '''

    def __init__(self, ndim, nwalkers):
        self.initialised = False
        self.index = 0
        self.ndim = ndim
        self.nwalkers = nwalkers


    def extend(self, n, blobs):
        """
        Method to extend saving space.
        Args:
            n (int) : Extend space by n slots.
        """
        if self.initialised:
            ext = np.empty((n,self.nwalkers,self.ndim))
            self.samples = np.concatenate((self.samples,ext),axis=0)
            ext = np.empty((n,self.nwalkers))
            self.logp = np.concatenate((self.logp,ext),axis=0)
            if blobs is not None:
                dt = np.dtype((blobs[0].dtype, blobs[0].shape))
                ext = np.empty((n,self.nwalkers), dtype=dt)
                self.blobs = np.concatenate((self.blobs, ext),axis=0)
        else:
            self.samples = np.empty((n,self.nwalkers,self.ndim))
            self.logp = np.empty((n,self.nwalkers))
            if blobs is not None:
                dt = np.dtype((blobs[0].dtype, blobs[0].shape))
                self.blobs = np.empty((n,self.nwalkers), dtype=dt)
            self.initialised = True


    def save(self, x, logp, blobs):
        """
        Save sample into the storage.
        Args:
            x (ndarray): Samples to be appended into the storage.
            logp (ndarray): Logprob values to be appended into the storage.
        """
        self.samples[self.index] = x
        self.logp[self.index] = logp
        if blobs is not None:
            self.blobs[self.index] = blobs
        self.index += 1


    @property
    def chain(self):
        """
        Chain property.

        Returns:
            3D array of shape (nsteps,nwalkers,ndim) containing the samples.
        """
        return self.samples[:self.index]


    @property
    def length(self):
        """
        Number of samples per walker.

        Returns:
            The total number of samples per walker.
        """
        return self.index


    def flatten(self, discard=0, thin=1):
        """
        Flatten samples by thinning them, removing the burn in phase, and combining all the walkers.

        Args:
            discard (int): Number of burn-in steps to be removed from each walker (default is 0).
            thin (int): Thinning parameter (the default value is 1).

        Returns:
            2D object containing the ndim flattened chains.
        """
        return self.chain[discard:self.index:thin,:,:].reshape((-1,self.ndim), order='F')


    @property
    def logprob(self):
        """
        Chain property.

        Returns:
            2D array of shape (nwalkers,nsteps) containing the log-probabilities.
        """
        return self.logp[:self.index]


    def flatten_logprob(self, discard=0, thin=1):
        """
        Flatten log probability by thinning the chain, removing the burn in phase, and combining all the walkers.

        Args:
            discard (int): Number of burn-in steps to be removed from each walker (default is 0).
            thin (int): Thinning parameter (the default value is 1).

        Returns:
            1D object containing the logprob of the flattened chains.
        """
        return self.logprob[discard:self.index:thin,:].reshape((-1,), order='F')


    def flatten_blobs(self, discard=0, thin=1):
        """
        Flatten blobs by thinning the chain, removing the burn in phase, and combining all the walkers.

        Args:
            discard (int): Number of burn-in steps to be removed from each walker (default is 0).
            thin (int): Thinning parameter (the default value is 1).

        Returns:
            (structured) NumPy array containing the blobs metadata.
        """
        return self.blobs[discard:self.index:thin,:].reshape((-1,), order='F')