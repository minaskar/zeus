import numpy as np
from itertools import permutations
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
            thin=1):

        X = np.copy(start)
        self.nsteps = nsteps
        chain = []

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
                    direction = X[J_pairs[k][0]] - X[J_pairs[k][1]]
                    direction *= 2.5
                    #direction /= np.linalg.norm(direction) # This is not part of the original
                    X[w_k] = self.slice1d(X[w_k], direction)

            if i % thin == 0:
                chain.append(X.tolist())

        return np.swapaxes(np.array(chain), 0, 1)


    def slice1d(self,
               x,
               direction):

        x_init = np.copy(x)
        x0 = np.linalg.norm(x)

        # Sample y
        z = self.slicelogp(0.0, x_init, direction) - np.random.exponential()

        # Stepping Out procedure
        U = np.random.uniform(0.0,1.0)
        L = - self.width * U
        R = L + self.width
        V = np.random.uniform(0.0,1.0)
        J = int(self.maxsteps * V)
        K = (self.maxsteps - 1) - J

        while (J > 0) and (z < self.slicelogp(L, x_init, direction)):
            L = L - self.width
            J = J - 1

        while (K > 0) and (z < self.slicelogp(R, x_init, direction)):
            R = R + self.width
            K = K - 1

        # Shrinkage procedure
        while True:
            U = np.random.uniform(0.0,1.0)
            x1 = L + U * (R - L)

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


    #def flatten(self, burn=0):
    #    return 
