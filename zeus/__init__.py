__bibtex__ = """
@article{karamanis2021zeus,
             title={zeus: A Python implementation of Ensemble Slice Sampling for efficient Bayesian parameter inference},
             author={Karamanis, Minas and Beutler, Florian and Peacock, John A},
             journal={arXiv preprint arXiv:2105.03468},
             year={2021}
            }

    @article{karamanis2020ensemble,
             title = {Ensemble slice sampling: Parallel, black-box and gradient-free inference for correlated & multimodal distributions},
             author = {Karamanis, Minas and Beutler, Florian},
             journal = {arXiv preprint arXiv: 2002.06212},
             year = {2020}
            }
"""

from ._version import version

__version__ = version
__url__ = "https://zeus-mcmc.readthedocs.io"
__author__ = "Minas Karamanis"
__email__ = "minaskar@gmail.com"
__license__ = "GPL-3.0"
__description__ = "Lightning Fast MCMC"


from .ensemble import *
from .parallel import ChainManager
from .autocorr import AutoCorrTime
from .plotting import cornerplot
from . import moves, callbacks