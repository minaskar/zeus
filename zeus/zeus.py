import numpy as np
from itertools import permutations
import random
from tqdm import tqdm
import logging

from .samples import samples
from .fwrapper import _FunctionWrapper
from .autocorr import _autocorr_time


class sampler:
    """
    A Differential Slice MCMC sampler.

    Args:
        logprob (callable): A python function that takes a vector in the
            parameter space as input and returns the natural logarithm of the
            unnormalised posterior probability at that position.
        nwalkers (int): The number of walkers in the ensemble.
        ndim (int): The number of dimensions/parameters.
        args (list): Extra arguments to be passed into the logp.
        kwargs (list): Extra arguments to be passed into the logp.
        maxsteps (int): Number of maximum stepping-out steps (Default is 10^4).
        jump (float): Probability of random jump (Default is 0.1). It has to be <1 and >0.
        mu (float): Scale factor (Default value is 10), this will be tuned if tune=True.
        tune (bool): Tune the scale factor to optimize performance (Default is True.)
        pool (bool): External pool of workers to distribute workload to multiple CPUs (default is None).
        verbose (bool): If True (default) print log statements.
    """
    def __init__(self,
                 logprob,
                 nwalkers,
                 ndim,
                 args=None,
                 kwargs=None,
                 maxsteps=10000,
                 jump=0.1,
                 mu=1.0,
                 tune=True,
                 walk=False,
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

        self.nwalkers = int(nwalkers)
        self.ndim = int(ndim)
        if self.nwalkers < 2 * self.ndim:
            raise ValueError("Please provide at least (2 * ndim) walkers.")
        elif self.nwalkers % 2 == 1:
            raise ValueError("Please provide an even number of walkers.")

        self.logprob = _FunctionWrapper(logprob, args, kwargs)

        self.jump = jump
        if self.jump < 0.0 or self.jump > 1.0:
            raise ValueError("Please provide jump probability in the range [0,1].")

        self.mu = mu #* np.sqrt(2) / np.sqrt(self.ndim)
        self.tune = tune
        self.maxsteps = maxsteps
        self.pool = pool
        self.samples = samples(self.ndim, self.nwalkers)

        self.walk = walk

        self.mus = []
        self.nconverge = 0


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
        if np.shape(start) != (self.nwalkers, self.ndim):
            raise ValueError("Incompatible input dimensions! Please provide array of shape (nwalkers, ndim) as the starting position.")
        X = np.copy(start)
        Z = np.asarray(list(map(self.logprob,X)))
        batch = list(np.arange(self.nwalkers))

        # Extend saving space
        self.nsteps = int(nsteps)
        self.thin = int(thin)
        self.samples.extend(self.nsteps//self.thin)

        # Devine Number of Log Prob Evaluations vector
        self.neval = np.zeros(self.nsteps)

        # Define task distributer
        if self.pool is None:
            distribute = map
        else:
            distribute = self.pool.map

        # Initialise progress bar
        if progress:
            t = tqdm(total=nsteps, desc='Sampling progress : ')

        # Main Loop
        for i in range(self.nsteps):

            # Initialise number of expansions & contractions
            nexp = 0
            ncon = 0

            # Random jump
            gamma = 1.0
            if np.random.uniform(0.0,1.0) > 1.0 - self.jump and (self.tune==False):
                gamma = 2.0 / self.mu

            # Shuffle ensemble
            np.random.shuffle(batch)
            batch0 = batch[:int(self.nwalkers/2)]
            batch1 = batch[int(self.nwalkers/2):]
            sets = [[batch0,batch1],[batch1,batch0]]

            # Loop over two sets
            for ensembles in sets:
                indeces = np.arange(int(self.nwalkers/2))
                # Define active-inactive ensembles
                active, inactive = ensembles

                if not self.walk:
                    # Compute Random Pair direction vectors
                    perms = list(permutations(inactive,2))
                    pairs = np.asarray(random.sample(perms,int(self.nwalkers/2))).T
                    directions = self.mu * (X[pairs[0]]-X[pairs[1]]) * gamma
                else:
                    mean = np.mean(X[inactive], axis=0)
                    cov = np.cov(X[inactive], rowvar=False)
                    directions = self.mu * np.random.multivariate_normal(mean,cov,size=int(self.nwalkers/2))

                # Get Z0 = LogP(x0)
                Z0 = Z[active] - np.random.exponential(size=int(self.nwalkers/2))

                # Set Initial Interval Boundaries
                L = - np.random.uniform(0.0,1.0,size=int(self.nwalkers/2))
                R = L + 1.0

                # Parallel stepping-out
                J = np.floor(self.maxsteps * np.random.uniform(0.0,1.0,size=int(self.nwalkers/2)))
                K = (self.maxsteps - 1) - J

                # Initialise number of Log prob calls
                ncall = 0

                # Left stepping-out
                mask_J = np.full(int(self.nwalkers/2),True)
                Z_L = np.empty(int(self.nwalkers/2))
                X_L = np.empty((int(self.nwalkers/2),self.ndim))

                while len(mask_J[mask_J])>0:
                    for j in indeces[mask_J]:
                        if J[j] < 1:
                            mask_J[j] = False
                    X_L[mask_J] = directions[mask_J] * L[mask_J][:,np.newaxis] + X[active][mask_J]
                    Z_L[mask_J] = np.asarray(list(distribute(self.logprob,X_L[mask_J])))
                    for j in indeces[mask_J]:
                        ncall += 1
                        nexp += 1
                        if Z0[j] < Z_L[j]:
                            L[j] = L[j] - 1.0
                            J[j] = J[j] - 1
                        else:
                            mask_J[j] = False


                # Right stepping-out
                mask_K = np.full(int(self.nwalkers/2),True)
                Z_R = np.empty(int(self.nwalkers/2))
                X_R = np.empty((int(self.nwalkers/2),self.ndim))

                while len(mask_K[mask_K])>0:
                    for j in indeces[mask_K]:
                        if K[j] < 1:
                            mask_K[j] = False
                    X_R[mask_K] = directions[mask_K] * R[mask_K][:,np.newaxis] + X[active][mask_K]
                    Z_R[mask_K] = np.asarray(list(distribute(self.logprob,X_R[mask_K])))
                    for j in indeces[mask_K]:
                        ncall += 1
                        nexp += 1
                        if Z0[j] < Z_R[j]:
                            R[j] = R[j] + 1.0
                            K[j] = K[j] - 1
                        else:
                            mask_K[j] = False


                # Shrinking procedure
                Widths = np.empty(int(self.nwalkers/2))
                Z_prime = np.empty(int(self.nwalkers/2))
                X_prime = np.empty((int(self.nwalkers/2),self.ndim))
                mask = np.full(int(self.nwalkers/2),True)

                while len(mask[mask])>0:
                    # Update Widths of intervals
                    Widths[mask] = L[mask] + np.random.uniform(0.0,1.0,size=len(mask[mask])) * (R[mask] - L[mask])

                    # Compute New Positions
                    X_prime[mask] = directions[mask] * Widths[mask][:,np.newaxis] + X[active][mask]

                    # Calculate LogP of New Positions
                    Z_prime[mask] = np.asarray(list(distribute(self.logprob,X_prime[mask])))

                    # Count LogProb calls
                    ncall += len(mask[mask])
                    ncon += len(mask[mask])

                    # Shrink slices
                    for j in indeces[mask]:
                        if Z0[j] < Z_prime[j]:
                            mask[j] = False
                        if Widths[j] < 0.0:
                            L[j] = Widths[j]
                        elif Widths[j] > 0.0:
                            R[j] = Widths[j]

                # Update Positions
                X[active] = X_prime
                Z[active] = Z_prime
                self.neval[i] += ncall

            # Tune scale factor using Robbins-Monro optimization
            if self.tune:
                self.mu *= 2.0 * nexp / (nexp + ncon)
                #print(nexp/(nexp+ncon))
                self.mus.append(self.mu)
                if np.abs(nexp / (nexp + ncon) - 0.5) < 0.05:
                    self.nconverge += 1
                if self.nconverge > 5:
                    self.tune = False




            # Save samples
            if (i+1) % self.thin == 0:
                self.samples.save(X)

            # Update progress bar
            if progress:
                t.update()

        # Close progress bar
        if progress:
            t.close()


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
    def ncall(self):
        """
        Number of Log Prob calls.

        Returns:
            ncall
        """
        return np.sum(self.neval)


    @property
    def efficiency(self):
        """
        Effective Samples per Log Probability Evaluation.

        Returns:
            efficiency
        """
        return self.ess / self.ncall


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
        logging.info('Number of Log Probability Evaluations: ' + str(self.ncall))
        logging.info('Effective Samples per Log Probability Evaluation: ' + str(round(self.efficiency,6)))
        if self.thin > 1:
            logging.info('Thinning rate: ' + str(self.thin))


    @property
    def one_sigma(self):
        fits = np.percentile(self.flatten(burn=int(self.nsteps/2.0)), [16, 50, 84], axis=0)
        return list(map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]), zip(*fits)))
