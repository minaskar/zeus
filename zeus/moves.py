import numpy as np 
from itertools import permutations
import random

try:
    from scipy.stats import gaussian_kde
except ImportError:
    gaussian_kde = None

try:
    from sklearn.mixture import BayesianGaussianMixture
except ImportError:
    BayesianGaussianMixture = None



class DifferentialMove:
    r"""
    The `Karamanis & Beutler (2020) <https://arxiv.org/abs/2002.06212>`_ "Differential Move" with parallelization.
    When this Move is used the walkers move along directions defined by random pairs of walkers sampled (with no
    replacement) from the complementary ensemble. This is the default choice and performs well along a wide range
    of target distributions.

    Parameters
    ----------
        tune : bool
            If True then tune this move. Default is True.
        mu0 : float
            Default value of ``mu`` if ``tune=False``.

    """

    def __init__(self, tune=True, mu0=1.0):
        self.tune = tune
        self.mu0 = mu0


    def get_direction(self, X, mu):
        r"""
        Generate direction vectors.

        Parameters
        ----------
            X : array
                Array of shape ``(nwalkers//2, ndim)`` with the walker positions of the complementary ensemble.
            mu : float
                The value of the scale factor ``mu``.
        
        Returns
        -------
            directions : array
                Array of direction vectors of shape ``(nwalkers//2, ndim)``.
        """

        nsamples = X.shape[0]

        perms = list(permutations(np.arange(nsamples), 2))
        pairs = np.asarray(random.sample(perms,nsamples)).T

        if not self.tune:
            mu = self.mu0
        
        return 2.0 * mu * (X[pairs[0]]-X[pairs[1]]), self.tune


class GaussianMove:
    r"""
    The `Karamanis & Beutler (2020) <https://arxiv.org/abs/2002.06212>`_ "Gaussian Move" with parallelization.
    When this Move is used the walkers move along directions defined by random vectors sampled from the Gaussian
    approximation of the walkers of the complementary ensemble.

    Parameters
    ----------
        tune : bool
            If True then tune this move. Default is True.
        mu0 : float
            Default value of ``mu`` if ``tune=False``.

    """

    def __init__(self, tune=False, mu0=1.0, cov=None):
        self.tune = tune 
        self.mu0 = mu0
        self.cov = cov


    def get_direction(self, X, mu):
        r"""
        Generate direction vectors.

        Parameters
        ----------
            X : array
                Array of shape ``(nwalkers//2, ndim)`` with the walker positions of the complementary ensemble.
            mu : float
                The value of the scale factor ``mu``.
        
        Returns
        -------
            directions : array
                Array of direction vectors of shape ``(nwalkers//2, ndim)``.
        """

        nsamples = X.shape[0]

        mean = np.mean(X, axis=0)
        if self.cov is None:
            cov = np.cov(X, rowvar=False)
        else:
            cov = self.cov

        if not self.tune:
            mu = self.mu0

        return 2.0 * mu * np.random.multivariate_normal(np.zeros_like(mean),cov,size=nsamples), self.tune


class GlobalMove:
    r"""
    The `Karamanis & Beutler (2020) <https://arxiv.org/abs/2002.06212>`_ "Global Move" with parallelization.
    When this Move is used a Bayesian Gaussian Mixture (BGM) is fitted to the walkers of complementary ensemble.
    The walkers move along random directions which connect different components of the BGM in an attempt to
    facilitate mode jumping. This Move should be used when the target distribution is multimodal. This move should
    be used after any burnin period.

    Parameters
    ----------
        tune : bool
            If True then tune this move. Default is True.
        mu0 : float
            Default value of ``mu`` if ``tune=False``.
        rescale_cov : float
            Rescale the covariance matrices of the BGM components by this factor. This promotes mode jumping.
            Default value is 0.001.
        n_components : int
            The number of mixture components. Depending on the distribution of the walkers the model can
            decide not to use all of them.
    """

    def __init__(self, tune=True, mu0=1.0, rescale_cov=0.001, n_components=5):
        
        if BayesianGaussianMixture is None:
            raise ImportError("you need sklearn.mixture.BayesianGaussianMixture to use the GlobalMove")

        self.tune = tune
        self.mu0 = mu0
        self.rescale_cov = rescale_cov
        self.n_components = n_components


    def get_direction(self, X, mu):
        r"""
        Generate direction vectors.

        Parameters
        ----------
            X : array
                Array of shape ``(nwalkers//2, ndim)`` with the walker positions of the complementary ensemble.
            mu : float
                The value of the scale factor ``mu``.
        
        Returns
        -------
            directions : array
                Array of direction vectors of shape ``(nwalkers//2, ndim)``.
        """
        
        if not self.tune:
            mu = self.mu0

        n = X.shape[0]

        mixture = BayesianGaussianMixture(n_components=self.n_components)
        labels = mixture.fit_predict(X)
        means = mixture.means_
        covariances = mixture.covariances_

        i, j = np.random.choice(labels, 2, replace=False)
        if i != j:
            directions = np.random.multivariate_normal(means[i], covariances[i]*self.rescale_cov, size=n) - np.random.multivariate_normal(means[j], covariances[j]*self.rescale_cov, size=n)
            tune_once = False
        else:
            directions = mu * np.random.multivariate_normal(np.zeros_like(means[i]), covariances[i], size=n)
            if self.tune:
                tune_once = True
            else:
                tune_once = False

        return 2.0*directions, tune_once


class KDEMove:
    r"""
    The `Karamanis & Beutler (2020) <https://arxiv.org/abs/2002.06212>`_ "KDE Move" with parallelization.
    When this Move is used the distribution of the walkers of the complementary ensemble is traced using
    a Gaussian Kernel Density Estimation methods. The walkers then move along random direction vectos
    sampled from this distribution.

    Parameters
    ----------
        tune : bool
            If True then tune this move. Default is True.
        mu0 : float
            Default value of ``mu`` if ``tune=False``.
        bw_method :
            The bandwidth estimation method. See the scipy docs for allowed values.

    """

    def __init__(self, tune=False, mu0=1.0, bw_method=None):

        if gaussian_kde is None:
            raise ImportError("you need scipy.stats.gaussian_kde to use the KDEMove")

        self.tune = tune
        self.mu0 = mu0
        self.bw_method = bw_method


    def get_direction(self, X, mu):
        r"""
        Generate direction vectors.

        Parameters
        ----------
            X : array
                Array of shape ``(nwalkers//2, ndim)`` with the walker positions of the complementary ensemble.
            mu : float
                The value of the scale factor ``mu``.
        
        Returns
        -------
            directions : array
                Array of direction vectors of shape ``(nwalkers//2, ndim)``.
        """

        n = X.shape[0]

        kde = gaussian_kde(X.T, bw_method=self.bw_method)

        vectors = kde.resample(2*n).T
        directions = vectors[:n] - vectors[n:]

        if not self.tune:
            mu = self.mu0

        return 2.0 * mu * directions, self.tune


class RandomMove:
    r"""
    The `Karamanis & Beutler (2020) <https://arxiv.org/abs/2002.06212>`_ "Random Move" with parallelization.
    When this move is used the walkers move along random directions. There is no communication between the
    walkers and this Move corresponds to the vanilla Slice Sampling method. This Move should be used for
    debugging purposes only.

    Parameters
    ----------
        tune : bool
            If True then tune this move. Default is True.
        mu0 : float
            Default value of ``mu`` if ``tune=False``.

    """

    def __init__(self, tune=True, mu0=1.0):
        self.tune = tune
        self.mu0 = mu0


    def get_direction(self, X, mu):
        r"""
        Generate direction vectors.

        Parameters
        ----------
            X : array
                Array of shape ``(nwalkers//2, ndim)`` with the walker positions of the complementary ensemble.
            mu : float
                The value of the scale factor ``mu``.
        
        Returns
        -------
            directions : array
                Array of direction vectors of shape ``(nwalkers//2, ndim)``.
        """

        directions = np.random.normal(0.0, 1.0, size=X.shape)
        directions /= np.linalg.norm(directions, axis=0)

        if not self.tune:
            mu = self.mu0

        return 2.0 * mu * directions, self.tune