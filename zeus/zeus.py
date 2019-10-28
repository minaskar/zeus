import numpy as np
from itertools import permutations, starmap
import random
from multiprocessing import Pool

from .samples import samples
from .fwrapper import _FunctionWrapper
from .start import jitter

from tqdm import tqdm


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
        width (float): Width of initial slice (default is 1.0, do not change that).
        maxsteps (float): Maximum number of steps for stepping-out procedure of Slice Sampler (default is 1, do not change that).
        mu (float): This is the mu coefficient (default value is 2.5). Numerical tests verify this as the optimal choice.
        normalise (bool): If True (default is False) then normalise the direction vector (no reason to do that, unless you also change the width and the maxsteps).

    """
    def __init__(self,
                 logp,
                 nwalkers,
                 ndim,
                 args=None,
                 kwargs=None,
                 mu=2.5):
        self.logp = _FunctionWrapper(logp, args, kwargs)
        self.nwalkers = int(nwalkers)
        self.ndim = int(ndim)
        self.mu = mu
        self.nlogp = 0


    def run(self,
            start,
            nsteps=1000,
            thin=1,
            progress=True,
            parallel=False):
        '''
        Calling this method runs the mcmc sampler.

        Args:
            start (float) : Starting point for the walkers.
            nsteps (int): Number of steps/generations (default is 1000).
            thin (float): Thin the chain by this number (default is 1 no thinning).
            progress (bool): If True (default), show progress bar (requires tqdm).
            parallel (bool): If True run the walkers in parallel (default is False).
        '''

        self.start = np.copy(start)
        self.X = jitter(self.start, self.nwalkers, self.ndim)
        self.nsteps = int(nsteps)
        self.samples = samples(self.nsteps, self.nwalkers, self.ndim)

        walkers = np.arange(self.nwalkers)
        batches = np.array(list(map(np.random.permutation,np.broadcast_to(walkers, (nsteps,self.nwalkers)))))

        def vec_diff(i, j):
            ''' Returns the difference between two vectors'''
            return self.X[i] - self.X[j]

        if progress:
            t = tqdm(total=nsteps)

        for i in range(self.nsteps):
            batch = batches[i]
            batch0 = list(batch[:int(self.nwalkers/2)])
            batch1 = list(batch[int(self.nwalkers/2):])
            sets = [[batch0,batch1],[batch1,batch0]]

            for ensembles in sets:
                active, inactive = ensembles
                perms = list(permutations(inactive,2))
                pairs = random.sample(perms,int(self.nwalkers/2))
                self.directions = self.mu * np.asarray(list(starmap(vec_diff,pairs)))
                active_i = np.vstack((np.arange(int(self.nwalkers/2)),active)).T
                if not parallel:
                    loop = list(map(self.slice1d, active_i))
                else:
                    with Pool() as pool:
                        loop = list(pool.map(self.slice1d, active_i))

            if i % thin == 0:
                self.samples.append(self.X)
            if progress:
                t.update()
        if progress:
            t.close()


    def slice1d(self, k_w):
        '''
        Samples the next point along the chosen direction.

        Args:
            k_w (int,int): index and label of walker.
        '''

        k, w_k = k_w

        x_init = np.copy(self.X[w_k])
        x0 = np.linalg.norm(x_init)
        direction = self.directions[k]
        if self.normalise:
            direction /= np.linalg.norm(direction)

        # Sample z=log(y)
        z = self.slicelogp(0.0, x_init, direction) - np.random.exponential()

        # Stepping Out procedure
        L = - np.random.uniform(0.0,1.0)
        R = L + 1.0

        # Shrinkage procedure
        while True:
            x1 = L + np.random.uniform(0.0,1.0) * (R - L)

            if (z < self.slicelogp(x1, x_init, direction)):
                break

            if (x1 < 0.0):
                L = x1
            elif (x1 > 0.0):
                R = x1

        self.X[w_k] = x1 * direction + x_init

        return 1.0


    def slicelogp(self, x, x_init, direction):
        """
        Evaluate the log probability in a point along a specific direction.

        Args:
            x (ndarray): magnitude of new point along the chosen direction.
            x_init (ndarray): vector of initial point.
            direction (ndarray): vector of chosen direction.

        Returns:
            The logp at direction * x + x_init
        """
        self.nlogp += 1
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
