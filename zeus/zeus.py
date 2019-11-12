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
from .start import jitter


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
        mu (float): This is the mu coefficient (default value is 3.44). Numerical tests verify this as the optimal choice.
        parallel (bool): If True (default is False), use only 1 CPU, otherwise distribute to multiple.
        ncores (bool): The maximum number of cores to use if parallel=True (default is None, meaning all of them).
        verbose (bool): If True (default) print log statements.
    """
    def __init__(self,
                 logp,
                 nwalkers,
                 ndim,
                 args=None,
                 kwargs=None,
                 mu=3.44,
                 parallel=False,
                 ncores=None,
                 verbose=True):

        # Set up logger
        self.logger = logging.getLogger()
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s %(name)-2s %(levelname)-6s %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        if verbose:
            self.logger.setLevel(logging.INFO)
        else:
            self.logger.setLevel(logging.WARNING)

        self.logp = _FunctionWrapper(logp, args, kwargs)
        self.nwalkers = int(nwalkers)
        self.ndim = int(ndim)
        self.mu = mu * np.sqrt(2) / np.sqrt(self.ndim)
        self.parallel = parallel
        self.ncores = ncores
        self.nlogp = 0
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

        if self.parallel:
            logging.info('Parallelizing ensemble of walkers using %d CPUs...', self.ncores)

        self.samples.extend(nsteps)
        if self.X is None:
            self.start = np.copy(start)
            self.X = jitter(self.start, self.nwalkers, self.ndim)
            self.Z = np.asarray(list(map(self.logp,self.X)))
            logging.info('Starting sampling...')
        else:
            logging.info('Continuing sampling...')

        self.nsteps = int(nsteps)

        batch = list(np.arange(self.nwalkers))

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
            t = tqdm(total=nsteps)

        for i in range(self.nsteps):
            np.random.shuffle(batch)
            batch0 = batch[:int(self.nwalkers/2)]
            batch1 = batch[int(self.nwalkers/2):]
            sets = [[batch0,batch1],[batch1,batch0]]
            for ensembles in sets:
                active, inactive = ensembles
                perms = list(permutations(inactive,2))
                pairs = random.sample(perms,int(self.nwalkers/2))
                self.directions = self.mu * np.array(list(starmap(vec_diff,pairs)))
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
                    self.nlogp += n

            if i % thin == 0:
                self.samples.save(self.X)
            if progress:
                t.update()
        if progress:
            t.close()
        logging.info('Sampling Complete!')


    def _slice1d(self, k_w):
        '''
        Samples the next point along the chosen direction.

        Args:
            k_w (int,int): index and label of walker.

        Returns:
            (list) : [k, w_k, x1, n]
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
            logp1 = self._slicelogp(x1, x_init, direction)
            if (z < logp1):
                break
            if (x1 < 0.0):
                L = x1
            elif (x1 > 0.0):
                R = x1

        return [k, w_k, x1, logp1, n]


    def _slicelogp(self, x, x_init, direction):
        """
        Evaluate the log probability in a point along a specific direction.

        Args:
            x (ndarray): magnitude of new point along the chosen direction.
            x_init (ndarray): vector of initial point.
            direction (ndarray): vector of chosen direction.

        Returns:
            The logp at direction * x + x_init
        """
        return self.logp(direction * x + x_init)


    @property
    def chain(self):
        """
        Returns the chains.

        Returns:
            Returns the chains of shape (nwalkers, nsteps, ndim).
        """
        return self.samples.chain


    def flatten(self, burn=None, thin=1):
        """
        Flatten the chain.

        Args:
            burn (int): The number of burn-in steps to remove from each walker (default is None, which results to Nsteps/2).
            thin (int): The ammount to thin the chain (default is 1, no thinning).

        Returns:
            2D Flattened chain.
        """
        return self.samples.flatten(burn, thin)
