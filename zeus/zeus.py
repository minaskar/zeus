import numpy as np
from itertools import permutations
from .samples import samples
from tqdm import tqdm

class sampler:

    def __init__(self,
                 logp,
                 nwalkers,
                 ndim,
                 width=1.0,
                 maxsteps=1):

        self.logp = logp
        self.nwalkers = nwalkers
        self.ndim = ndim
        self.width = width
        self.maxsteps = maxsteps
        self.nlogp = 0

    def run(self,
            start,
            nsteps=1000,
            thin=1,
            progress=True,
            parallel=False):

        X = np.copy(start)
        self.nsteps = nsteps
        self.samples = samples(self.nsteps, self.nwalkers, self.ndim)

        walkers = np.arange(self.nwalkers)
        batches = np.array(list(map(np.random.permutation,np.broadcast_to(walkers, (nsteps,self.nwalkers)))))

        for i in tqdm(range(nsteps)):
            batch = batches[i]
            batch0 = list(batch[:int(self.nwalkers/2)])
            batch1 = list(batch[int(self.nwalkers/2):])
            sets = [[batch0,batch1],[batch1,batch0]]

            for ensembles in sets:
                active, inactive = ensembles
                J_pairs = list(permutations(inactive, 2))
                for k, w_k in enumerate(active):
                    direction = 2.5 * (X[J_pairs[k][0]] - X[J_pairs[k][1]])
                    X[w_k] = self.slice1d(X[w_k], direction)

            if i % thin == 0:
                self.samples.append(X)


    def slice1d(self,
               x,
               direction):

        x_init = np.copy(x)
        x0 = np.linalg.norm(x)

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

        return x1 * direction + x_init


    def slicelogp(self, x, x_init, direction):
        self.nlogp += 1
        return self.logp(direction * x + x_init)


    @property
    def chain(self):
        return self.samples.chain


    def flatten(self, burn=None):
        return self.samples.flatten(burn)
