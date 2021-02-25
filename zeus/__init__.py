__bibtex__ = """
@article{zeus,
        title={zeus: A Python Implementation of the Ensemble Slice Sampling method},
        author={Minas Karamanis and Florian Beutler},
        year={2021},
        note={in prep}
    }

@article{ess,
      title={Ensemble Slice Sampling},
      author={Minas Karamanis and Florian Beutler},
      year={2020},
      eprint={2002.06212},
      archivePrefix={arXiv},
      primaryClass={stat.ML}
}
"""

__version__ = "2.3.0"
__url__ = "https://zeus-mcmc.readthedocs.io"
__author__ = "Minas Karamanis"
__email__ = "minaskar@gmail.com"
__license__ = "GPL-3.0"
__description__ = "Lightning Fast MCMC"


from .ensemble import *
from .parallel import ChainManager
from .autocorr import AutoCorrTime
from .plotting import cornerplot