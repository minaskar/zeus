import numpy as np
from itertools import permutations, starmap
import random
from multiprocessing import Pool

from .samples import samples
from .fwrapper import _FunctionWrapper
from .start import jitter

from tqdm import tqdm


class sampler:
    '''
    An ensemble slice MCMC sampler.

    Args:
        logp (callable): A python function that takes a vector in the
            parameter space as input and returns the natural logarithm of the
            unnormalised posterior probability at that position.
        nwalkers (int): The number of walkers in the ensemble.
    '''

    def __init__(self,
                 logp,
                 nwalkers,
                 ndim,
                 args=None,
                 kwargs=None,
                 width=1.0,
                 maxsteps=1,
                 mu=2.5,
                 normalise=False):

        self.logp = _FunctionWrapper(logp, args, kwargs)
        self.nwalkers = int(nwalkers)
        self.ndim = int(ndim)
        self.width = width
        self.maxsteps = int(maxsteps)
        self.mu = mu
        self.normalise = normalise
        self.nlogp = 0


    def run(self,
            start,
            nsteps=1000,
            thin=1,
            progress=True,
            parallel=False):

        self.start = np.copy(start)
        self.X = jitter(self.start, self.nwalkers, self.ndim)
        self.nsteps = int(nsteps)
        self.samples = samples(self.nsteps, self.nwalkers, self.ndim)

        walkers = np.arange(self.nwalkers)
        batches = np.array(list(map(np.random.permutation,np.broadcast_to(walkers, (nsteps,self.nwalkers)))))

        def vec_diff(i, j):
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

        k, w_k = k_w

        x_init = np.copy(self.X[w_k])
        x0 = np.linalg.norm(x_init)
        direction = self.directions[k]
        if self.normalise:
            direction /= np.linalg.norm(direction)

        # Sample z=log(y)
        z = self.slicelogp(0.0, x_init, direction) - np.random.exponential()

        # Stepping Out procedure
        L = - self.width * np.random.uniform(0.0,1.0)
        R = L + self.width
        J = int(self.maxsteps * np.random.uniform(0.0,1.0))
        K = (self.maxsteps - 1) - J

        while (J > 0) and (z < self.slicelogp(L, x_init, direction)):
            L = L - self.width
            J = J - 1

        while (K > 0) and (z < self.slicelogp(R, x_init, direction)):
            R = R + self.width
            K = K - 1

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
        self.nlogp += 1
        return self.logp(direction * x + x_init)


    @property
    def chain(self):
        return self.samples.chain


    def flatten(self, burn=None, thin=1):
        return self.samples.flatten(burn, thin)
