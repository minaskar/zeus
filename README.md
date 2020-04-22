![logo](https://github.com/minaskar/zeus/blob/master/logo.png)

**zeus is a pure-Python implementation of the *Ensemble Slice Sampling* method.**

- Fast & Robust *Bayesian Inference*,
- Efficient Markov Chain Monte Carlo,
- No hand-tuning,
- Excellent performance in terms of autocorrelation time and convergence rate,
- Scale to multiple CPUs without any extra effort.

[![GitHub](https://img.shields.io/badge/GitHub-minaskar%2Fzeus-blue)](https://github.com/minaskar/zeus)
[![arXiv](https://img.shields.io/badge/arXiv-2002.06212-red)](https://arxiv.org/abs/2002.06212)
[![Build Status](https://travis-ci.com/minaskar/zeus.svg?token=xnVWRZ3TFg1zxQYQyLs4&branch=master)](https://travis-ci.com/minaskar/zeus)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://github.com/minaskar/zeus/blob/master/LICENSE)
[![Documentation Status](https://readthedocs.org/projects/zeus-mcmc/badge/?version=latest&token=4455dbf495c5a4eaba52de26ac56628aad85eb3eadc90badfd1703d0a819a0f9)](https://zeus-mcmc.readthedocs.io/en/latest/?badge=latest)


## Example

For instance, if you wanted to draw samples from a 10-dimensional Gaussian, you would do something like:

```python
import numpy as np
import zeus

def logp(x, ivar):
    return - 0.5 * np.sum(ivar * x**2.0)

nsteps, nwalkers, ndim = 1000, 100, 10
ivar = 1.0 / np.random.rand(ndim)
start = np.random.randn(nwalkers,ndim)

sampler = zeus.sampler(logp, nwalkers, ndim, args=[ivar])
sampler.run(start, nsteps)
```

## Documentation

Read the docs at [zeus-mcmc.readthedocs.io](https://zeus-mcmc.readthedocs.io)


## Installation

To install zeus using pip run

```bash
pip install zeus-mcmc
```

## Attribution

Please cite [Karamanis & Beutler (2020)](https://arxiv.org/abs/2002.06212) if you find this code useful in your
research. The BibTeX entry for the paper is:

```bash
@article{zeus,
      title={Ensemble Slice Sampling},
      author={Minas Karamanis and Florian Beutler},
      year={2020},
      eprint={2002.06212},
      archivePrefix={arXiv},
      primaryClass={stat.ML}
}
```

## Licence

Copyright 2019-2020 Minas Karamanis and contributors.

zeus is free software made available under the GPL-3.0 License. For details see the `LICENSE` file.
