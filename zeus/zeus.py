import sys
import numpy as np
from itertools import permutations, starmap
import random
from multiprocessing import Pool
from mpi4py.futures import MPIPoolExecutor
from psutil import cpu_count
from tqdm import tqdm
import logging

from .samples import samples
from .fwrapper import _FunctionWrapper
from .autocorr import _autocorr_time


class sampler:
    """
    An ensemble slice MCMC sampler.

    Args:
        logp (callable): A python function that takes a vector in the
            parameter space as input and returns the natural logarithm of the
            unnormalised posterior probability at that position.
        nwalkers (int): The number of walkers in the ensemble.
        ndim (int): The number of dimensions/parameters.
        args (list): Extra arguments to be passed into the logp.
        kwargs (list): Extra arguments to be passed into the logp.
        jump (float): Probability of random jump (Default is 0.1). It has to be <1 and >0.
        mu (float): This is the mu coefficient (default value is 3.7). Numerical tests verify this as the optimal choice.
        parallel (bool): If True (default is False) distribute workload to multiple CPUs.
        ncores (bool): The maximum number of cores to use if parallel=True (default is None, meaning all of them).
        verbose (bool): If True (default) print log statements.
    """
    def __init__(self,
                 logp,
                 nwalkers,
                 ndim,
                 args=None,
                 kwargs=None,
                 jump=0.1,
                 mu=3.7,
                 pool=None,
                 verbose=True):

        # Set up logger
        self.logger = logging.getLogger()
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        handler = logging.StreamHandler()
        self.logger.addHandler(handler)
        if verbose:
            self.logger.setLevel(logging.INFO)
        else:
            self.logger.setLevel(logging.WARNING)

        self.logp = _FunctionWrapper(logp, args, kwargs)
        self.nwalkers = int(nwalkers)
        self.ndim = int(ndim)
        self.jump = jump
        self.mu = mu * np.sqrt(2) / np.sqrt(self.ndim)
        self.pool = pool
        self.neval = 0
        self.samples = samples(self.ndim, self.nwalkers)
        self.X = None
        self.Z = None


    def run(self,
            start,
            nsteps=1000,
            thin=1,
            progress=True):
        '''
        Calling this method runs the mcmc sampler.

        Args:
            start (float) : Starting point for the walkers.
            nsteps (int): Number of steps/generations (default is 1000).
            thin (float): Thin the chain by this number (default is 1 no thinning).
            progress (bool): If True (default), show progress bar (requires tqdm).
        '''
        # Initialise ensemble of walkers
        logging.info('Initialising ensemble of %d walkers...', self.nwalkers)

        self.X = np.copy(start)

        if np.shape(self.X) != (self.nwalkers, self.ndim):
            raise ValueError("Incompatible input dimensions! Please provide array of shape (nwalkers, ndim) as the starting position.")

        self.Z = np.asarray(list(map(self.logp,self.X)))

        batch = list(np.arange(self.nwalkers))

        # Extend saving space
        self.nsteps = int(nsteps)
        self.thin = int(thin)

        self.samples.extend(self.nsteps//self.thin)

        if self.pool is None:
            distribute = map
        else:
            distribute = self.pool.map


        if progress:
            t = tqdm(total=nsteps, desc='Sampling progress : ')

        for i in range(self.nsteps):

            # Random jump
            gamma = 1.0
            if np.random.uniform(0.0,1.0) > 1.0 - self.jump:
                gamma = 2.0 / self.mu

            # Shuffle ensemble
            np.random.shuffle(batch)
            batch0 = batch[:int(self.nwalkers/2)]
            batch1 = batch[int(self.nwalkers/2):]
            sets = [[batch0,batch1],[batch1,batch0]]

            for ensembles in sets:
                active, inactive = ensembles
                perms = list(permutations(inactive,2))
                pairs = np.asarray(random.sample(perms,int(self.nwalkers/2))).T
                directions = self.mu * (self.X[pairs[0]]-self.X[pairs[1]]) * gamma

                mask = np.full(int(self.nwalkers/2),True)

                Zs = self.Z[active] - np.random.exponential(size=int(self.nwalkers/2))
                Ls = - np.random.uniform(0.0,1.0,size=int(self.nwalkers/2))
                Rs = Ls + 1.0

                x1s = np.empty(int(self.nwalkers/2))
                logp1s = np.empty(int(self.nwalkers/2))
                new_vectors = np.empty((int(self.nwalkers/2),self.ndim))

                indeces = np.arange(int(self.nwalkers/2))

                while len(mask[mask])>0:

                    x1s[mask] = Ls[mask] + np.random.uniform(0.0,1.0,size=len(mask[mask])) * (Rs[mask] - Ls[mask])

                    new_vectors[mask] = directions[mask] * x1s[mask][:,np.newaxis] + self.X[active][mask]

                    logp1s[mask] = np.asarray(list(distribute(self.logp,new_vectors[mask])))

                    self.neval += len(mask[mask])

                    for j in indeces[mask]:
                        if Zs[j] < logp1s[j]:
                            mask[j] = False
                        if x1s[j] < 0.0:
                            Ls[j] = x1s[j]
                        elif x1s[j] > 0.0:
                            Rs[j] = x1s[j]

                self.X[active] = new_vectors
                self.Z[active] = logp1s

            if (i+1) % self.thin == 0:
                self.samples.save(self.X)
            if progress:
                t.update()
        if progress:
            t.close()
        logging.info('Sampling Complete!')


    def reset(self):
        """
        Reset the state of the sampler. Delete any samples stored in memory.
        """
        self.samples = samples(self.ndim, self.nwalkers)


    @property
    def chain(self):
        """
        Returns the chains.

        Returns:
            Returns the chains of shape (nwalkers, nsteps, ndim).
        """
        return self.samples.chain


    def flatten(self, burn=0, thin=1):
        """
        Flatten the chain.

        Args:
            burn (int): The number of burn-in steps to remove from each walker (default is 0).
            thin (int): The ammount to thin the chain (default is 1, no thinning).

        Returns:
            2D Flattened chain.
        """
        return self.samples.flatten(burn, thin)


    @property
    def last(self):
        """
        Last position of the walkers.

        Returns:
            (array) : The last position of the walkers.
        """
        return self.X


    @property
    def autocorr_time(self):
        """
        Integrated Autocorrelation Time (IAT) of the Markov Chain.

        Returns:
            Array with the IAT of each parameter.
        """
        return _autocorr_time(self.chain[:,int(self.nsteps/(self.thin*2.0)):,:])


    @property
    def ess(self):
        """
        Effective Sampling Size (ESS) of the Markov Chain.

        Returns:
            ESS
        """
        return self.nwalkers * self.samples.length / np.mean(self.autocorr_time)


    @property
    def efficiency(self):
        """
        Effective Samples per Log Probability Evaluation.

        Returns:
            efficiency
        """
        return self.ess / self.neval


    @property
    def summary(self):
        """
        Summary of the MCMC run.
        """
        logging.info('Summary')
        logging.info('-------')
        logging.info('Number of Generations: ' + str(self.samples.length))
        logging.info('Number of Parameters: ' + str(self.ndim))
        logging.info('Number of Walkers: ' + str(self.nwalkers))
        logging.info('Mean Integrated Autocorrelation Time: ' + str(round(np.mean(self.autocorr_time),2)))
        logging.info('Effective Sample Size: ' + str(round(self.ess,2)))
        logging.info('Number of Log Probability Evaluations: ' + str(self.neval))
        logging.info('Effective Samples per Log Probability Evaluation: ' + str(round(self.efficiency,6)))
        if self.thin > 1:
            logging.info('Thinning rate: ' + str(self.thin))


    @property
    def one_sigma(self):
        fits = np.percentile(self.flatten(burn=int(self.nsteps/2.0)), [16, 50, 84], axis=0)
        return list(map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]), zip(*fits)))
