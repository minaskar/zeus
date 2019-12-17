import sys
sys.path.append("./../")
import numpy as np
import zeus
from mpi4py.futures import MPIPoolExecutor

def logp(x):
    return -0.5*np.dot(x-1.0,x-1.0)

ndim = 500
nwalkers = 2*ndim
nsteps = 50

start = 0.1* np.random.randn(nwalkers,ndim) - 1.0


if __name__ == '__main__':

    with MPIPoolExecutor() as executor:
        sampler = zeus.sampler(logp,nwalkers,ndim,pool=executor)
        sampler.run(start,nsteps)
        print(sampler.one_sigma)
        sampler.summary
