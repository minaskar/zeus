__version__ = "1.2.1"
__url__ = "https://zeus-mcmc.readthedocs.io"
__author__ = "Minas Karamanis"
__email__ = "minaskar@gmail.com"
__license__ = "GPL-3.0"
__description__ = "Lightning Fast MCMC"


from .ensemble import *
from .diagnostics import GelmanRubin, Geweke
from .parallel import ChainManager
from .autocorr import AutoCorrTime