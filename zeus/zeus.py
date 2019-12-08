import sys
import numpy as np
from itertools import permutations, starmap
import random
from multiprocessing import Pool
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
                 parallel=False,
                 ncores=None,
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
        self.parallel = parallel
        self.ncores = ncores
        self.neval = 0
        self.samples = samples(self.ndim, self.nwalkers)
        self.X = None
        self.Z = None

        if self.ncores is None:
            self.ncores = min(int(self.nwalkers/2.0),cpu_count(logical=False))


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
        self.X = np.copy(start)

        if np.shape(self.X) != (self.nwalkers, self.ndim):
            raise ValueError("Incompatible input dimensions! Please provide array of shape (nwalkers, ndim) as the starting position.")

        if self.parallel:
            logging.info('Parallelizing ensemble of walkers using %d CPUs...', self.ncores)
        else:
            logging.info('Initialising ensemble of %d walkers...', self.nwalkers)

        self.Z = np.asarray(list(map(self.logp,self.X)))

        batch = list(np.arange(self.nwalkers))

        # Extend saving space
        self.nsteps = int(nsteps)
        self.thin = int(thin)

        self.samples.extend(self.nsteps//self.thin)


        def vec_diff(i, j):
            '''
            Returns the difference between two vectors.

            Args:
                i (int): Index of first vector.
                j (int): Index of second vector.

            Returns:
                The difference between the vector positions of the walkers i and j.
            '''
            return self.X[i] - self.X[j]


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
                pairs = random.sample(perms,int(self.nwalkers/2))
                self.directions = self.mu * np.array(list(starmap(vec_diff,pairs))) * gamma
                active_i = np.vstack((np.arange(int(self.nwalkers/2)),active)).T

                if not self.parallel:
                    results = list(map(self._slice1d, active_i))
                else:
                    with Pool(self.ncores) as pool:
                        results = list(pool.map(self._slice1d, active_i))

                Xinit = np.copy(self.X)
                for result in results:
                    k, w_k, x1, logp1, n = result
                    self.X[w_k] = x1 * self.directions[k] + Xinit[w_k]
                    self.Z[w_k] = logp1
                    self.neval += n

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



    def _slice1d(self, k_w):
        '''
        Samples the next point along the chosen direction.

        Args:
            k_w (int,int): index and label of walker.

        Returns:
            (list) : [k, w_k, x1, logp1, n]
        '''

        k, w_k = k_w
        x_init = np.copy(self.X[w_k])
        direction = self.directions[k]

        z = self.Z[w_k] - np.random.exponential()

        L = - np.random.uniform(0.0,1.0)
        R = L + 1.0

        n = 0
        while True:
            n += 1
            x1 = L + np.random.uniform(0.0,1.0) * (R - L)
            logp1 = self.logp(direction * x1 + x_init)
            if (z < logp1):
                break
            if (x1 < 0.0):
                L = x1
            elif (x1 > 0.0):
                R = x1

        return [k, w_k, x1, logp1, n]


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
        if self.parallel:
            logging.info('Number of CPUs: ' + str(self.ncores))
        if self.thin > 1:
            logging.info('Thinning rate: ' + str(self.thin))


    @property
    def one_sigma(self):
        fits = np.percentile(self.flatten(burn=int(self.nsteps/2.0)), [16, 50, 84], axis=0)
        return list(map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]), zip(*fits)))
