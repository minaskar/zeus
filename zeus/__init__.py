__version__ = "2.2.0"
__url__ = "https://zeus-mcmc.readthedocs.io"
__author__ = "Minas Karamanis"
__email__ = "minaskar@gmail.com"
__license__ = "GPL-3.0"
__description__ = "Lightning Fast MCMC"


from .ensemble import *
from .parallel import ChainManager
from .autocorr import AutoCorrTime
from .plotting import cornerplot