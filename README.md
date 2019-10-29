
# zeus: Lightning Fast MCMC

![](zeus.gif)

zeus is an efficient parallel Python implementation of the ensemble slice MCMC method. The latter is a combination of _Slice Sampling_ \(Neal 2003\) and _Adaptive Direction Sampling_ \(Gilks, Roberts, George 1993\) in a parallel ensemble framework with a few extra tricks. **The result is a fast and easy-to-use general-purpose MCMC sampler that works out-of-the-box without the need to tune any hyperparameters or provide a proposal distribution.** zeus produces independent samples of extremely low autocorrelation and converges to the target distribution faster than any other general-purpose gradient-free MCMC sampler.

## Example

For instance, if you wanted to draw samples from a 10-dimensional Gaussian, you would do

```python
import numpy as np
import zeus

def logp(x, ivar):
    return - 0.5 * np.sum(ivar * x**2.0)

nsteps, nwalkers, ndim = 1000, 100, 10
ivar = 1.0 / np.random.rand(ndim)
start = np.random.rand(ndim)

sampler = zeus.sampler(logp, nwalkers, ndim, args=[ivar])
sampler.run(start, nsteps)
```

## Installation

To install zeus using pip run

```bash
pip install git+https://github.com/minaskar/zeus
```

## Licence

Copyright 2019-2019 Minas Karamanis and contributors.

zeus is free software made available under the GPL-3.0 License. For details see the `LICENSE`.
