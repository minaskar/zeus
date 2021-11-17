import numpy as np
from .autocorr import AutoCorrTime

try:
    import h5py
except ImportError:
    h5py = None


class AutocorrelationCallback:
    """
    The Autocorrelation Time Callback class checks the integrated autocorrelation time (IAT)
    of the chain during the run and terminates sampling if the rate of change of IAT is below
    some threshold and the length of the chain is greater than some multiple of the IAT estimate.

    Args:
        ncheck (int): The number of steps after which the IAT is estimated and the tests are performed.
            Default is ``ncheck=100``.
        dact (float): Threshold of the rate of change of IAT. Sampling terminates once this threshold is
            reached along with the other criteria. Default is ``dact=0.01``.
        nact (float): Minimum lenght of the chain as a mutiple of the IAT. Sampling terminates once this threshold is
            reached along with the other criteria. Default is ``nact=10``.
        discard (float): Percentage of chain to discard prior to estimating the IAT. Default is ``discard=0.5``.
        trigger (bool): If ``True`` (default) then terminatate sampling once converged, else just monitor statistics.
        method (str): Method to use for the estimation of the IAT. Available options are ``mk`` (Default), ``dfm``, and ``gw``.
    """

    def __init__(self, ncheck=100, dact=0.01, nact=10, discard=0.5, trigger=True, method='mk'):
        self.ncheck = ncheck
        self.dact = dact 
        self.nact = nact

        self.discard = discard
        self.trigger = trigger
        self.method = method

        self.estimates = []
        self.old_tau = np.inf

    def __call__(self, i, x, y):
        """
        Method that calls the callback function.

        Args:
            i (int): Current iteration of the run.
            x (array): Numpy array containing the chain elements up to iteration i for every walker.
            y (array): Numpy array containing the log-probability values of all chain elements up to
                iteration i for every walker.
        Returns:
            True if the criteria are satisfied and sampling terminates or False if the criteria are
                not satisfied and sampling continues.
        
        """
        converged = False

        if i % self.ncheck == 0:
        
            tau = np.mean(AutoCorrTime(x[int(i * self.discard):], method=self.method))
            self.estimates.append(tau)

            # Check convergence
            converged = tau * self.nact < i
            converged &= np.abs(self.old_tau - tau) / tau < self.dact
        
            self.old_tau = tau

        if self.trigger:
            return converged
        else:
            return None


class SplitRCallback:
    """
    The Split-R Callback class checks the Gelman-Rubin criterion during the run by splitting the chain
    into multiple parts and terminates sampling if the Split-R coefficient is close to unity.

    Args:
        ncheck (int): The number of steps after which the Gelman-Rubin statistics is estimated and the tests are performed.
            Default is ``ncheck=100``.
        epsilon (float): Threshold of the Split-R value. Sampling terminates when ``|R-1|<epsilon``. Default is ``0.05``
        nsplits (int): Split each chain into this many pieces. Default is ``2``.
        discard (float): Percentage of chain to discard prior to estimating the IAT. Default is ``discard=0.5``.
        trigger (bool): If ``True`` (default) then terminatate sampling once converged, else just monitor statistics.
    """

    def __init__(self, ncheck=100, epsilon=0.05, nsplits=2, discard=0.5, trigger=True):
        self.ncheck = ncheck
        self.epsilon = epsilon
        self.nsplits = nsplits

        self.trigger = trigger

        self.discard = discard

        self.estimates = []

    def __call__(self, i, x, y):
        """
        Method that calls the callback function.

        Args:
            i (int): Current iteration of the run.
            x (array): Numpy array containing the chain elements up to iteration i for every walker.
            y (array): Numpy array containing the log-probability values of all chain elements up to
                iteration i for every walker.
        Returns:
            True if the criteria are satisfied and sampling terminates or False if the criteria are
                not satisfied and sampling continues.
        
        """
        converged = False

        if i % self.ncheck == 0:
            ndim = x.shape[-1]
            samples = np.array_split(x[int(i * self.discard):], self.nsplits)
            mean = []
            var = []
            for sample in samples:
                mean.append(np.mean(sample.reshape(-1,ndim),axis=0))
                var.append(np.var(sample.reshape(-1,ndim),axis=0))
                N = len(sample.reshape(-1,ndim))
            
            Rhat = np.mean(self.estimate_Rhat(mean, var, N))

            self.estimates.append(Rhat)

            # Check convergence
            converged = (Rhat - 1.0) <= self.epsilon

        if self.trigger:
            return converged
        else:
            return None


    def estimate_Rhat(self, means, vars, N):

        _means = [item for sublist in means for item in sublist]
        _vars = [item for sublist in vars for item in sublist]

        # Between chain variance
        B = N * np.var(_means, ddof=1, axis=0)

        # Within chain variance
        W = np.mean(_vars)

        # Weighted variance
        Var_hat = (1.0 - 1.0 / N) * W + B / N

        # Return R_hat statistic
        R_hat = np.sqrt(Var_hat / W)
    
        return R_hat


class MinIterCallback:
    """
    The Minimum Iteration Callback class ensure that sampling does not terminate early prior to a
    prespecified number of steps.

    Args:
        nmin (int): The number of minimum steps before other callbacks can terminate the run.
    """
    def __init__(self, nmin=1000):
        self.nmin = nmin 

    def __call__(self, i, x, y):
        """
        Method that calls the callback function.

        Args:
            i (int): Current iteration of the run.
            x (array): Numpy array containing the chain elements up to iteration i for every walker.
            y (array): Numpy array containing the log-probability values of all chain elements up to
                iteration i for every walker.
        Returns:
            True if the criteria are satisfied and sampling terminates or False if the criteria are
                not satisfied and sampling continues.
        
        """
        if i >= self.nmin:
            return True
        else:
            return False


class ParallelSplitRCallback:
    """
    The Parallel Split-R Callback class extends the functionality of the Split-R Callback to more than one CPUs by 
    checking the Gelman-Rubin criterion during the run by splitting the chain into multiple parts and combining different 
    parts from parallel chains and terminates sampling if the Split-R coefficient is close to unity.

    Args:
        ncheck (int): The number of steps after which the Gelman-Rubin statistics is estimated and the tests are performed.
            Default is ``ncheck=100``.
        epsilon (float): Threshold of the Split-R value. Sampling terminates when ``|R-1|<epsilon``. Default is ``0.05``
        nsplits (int): Split each chain into this many pieces. Default is ``2``.
        discard (float): Percentage of chain to discard prior to estimating the IAT. Default is ``discard=0.5``.
        trigger (bool): If ``True`` (default) then terminatate sampling once converged, else just monitor statistics.
        chainmanager (ChainManager instance): The ``ChainManager`` used to parallelise the sampling process.
    """

    def __init__(self, ncheck=100, epsilon=0.01, nsplits=2, discard=0.5, trigger=True, chainmanager=None):
        self.ncheck = ncheck
        self.epsilon = epsilon
        self.nsplits = nsplits

        self.trigger = trigger

        self.discard = discard

        self.estimates = []

        if chainmanager is None:
            raise ValueError("Please provide a ChainManager instance for this method to work.")
        self.cm = chainmanager


    def __call__(self, i, x, y):
        """
        Method that calls the callback function.

        Args:
            i (int): Current iteration of the run.
            x (array): Numpy array containing the chain elements up to iteration i for every walker.
            y (array): Numpy array containing the log-probability values of all chain elements up to
                iteration i for every walker.
        Returns:
            True if the criteria are satisfied and sampling terminates or False if the criteria are
                not satisfied and sampling continues.
        
        """
        converged = False

        if i % self.ncheck == 0:
            ndim = x.shape[-1]
            samples = np.array_split(x[int(i * self.discard):], self.nsplits)
            mean = []
            var = []
            for sample in samples:
                mean.append(np.mean(sample.reshape(-1,ndim),axis=0))
                var.append(np.var(sample.reshape(-1,ndim),axis=0))
                N = len(sample.reshape(-1,ndim))

            mean_all = self.cm.gather(mean, root=0)
            var_all = self.cm.gather(var, root=0)
            N_all = np.sum(self.cm.gather(N, root=0))

            if self.cm.get_rank == 0:
                Rhat = np.mean(self.estimate_Rhat(mean_all, var_all, N_all))
                self.estimates.append(Rhat)
                # Check convergence
                converged = (Rhat - 1.0) <= self.epsilon

            converged = self.cm.bcast(converged, root=0)
            self.estimates  = self.cm.bcast(self.estimates, root=0)

        if self.trigger:
            return converged
        else:
            return None


    def estimate_Rhat(self, means, vars, N):

        _means = [item for sublist in means for item in sublist]
        _vars = [item for sublist in vars for item in sublist]

        # Between chain variance
        B = N * np.var(_means, ddof=1, axis=0)

        # Within chain variance
        W = np.mean(_vars)

        # Weighted variance
        Var_hat = (1.0 - 1.0 / N) * W + B / N

        # Return R_hat statistic
        R_hat = np.sqrt(Var_hat / W)
    
        return R_hat


class SaveProgressCallback:
    """
    The Save Progress Callback class iteratively saves the collected samples and log-probability values to a HDF5 file.

    Args:
        filename (str): Name of the directory and file to save samples. Default is ``./chains.h5``.
        ncheck (int): The number of steps after which the samples are saved. Default is ``ncheck=100``.
    """

    def __init__(self, filename='./chains.h5', ncheck=100):

        if h5py is None:
            raise ImportError("You must install 'h5py' to use the SaveProgressCallback")

        self.directory = filename
        self.initialised = False
        self.ncheck = ncheck


    def __call__(self, i, x, y):
        """
        Method that calls the callback function.

        Args:
            i (int): Current iteration of the run.
            x (array): Numpy array containing the chain elements up to iteration i for every walker.
            y (array): Numpy array containing the log-probability values of all chain elements up to
                iteration i for every walker.
        Returns:
            True if the criteria are satisfied and sampling terminates or False if the criteria are
                not satisfied and sampling continues.
        
        """
        if i % self.ncheck == 0:
            if self.initialised:
                self.__save(x[i-self.ncheck:], y[i-self.ncheck:])
            else:
                self.__initialize_and_save(x[i-self.ncheck:], y[i-self.ncheck:])

        return None


    def __save(self, x, y):
        with h5py.File(self.directory, 'a') as hf:
            hf['samples'].resize((hf['samples'].shape[0] + x.shape[0]), axis = 0)
            hf['samples'][-x.shape[0]:] = x
            hf['logprob'].resize((hf['logprob'].shape[0] + y.shape[0]), axis = 0)
            hf['logprob'][-y.shape[0]:] = y


    def __initialize_and_save(self, x, y):
        with h5py.File(self.directory, 'w') as hf:
            hf.create_dataset('samples', data=x, compression="gzip", chunks=True, maxshape=(None,)+x.shape[1:])
            hf.create_dataset('logprob', data=y, compression="gzip", chunks=True, maxshape=(None,)+y.shape[1:]) 
        self.initialised  = True