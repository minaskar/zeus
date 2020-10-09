import numpy as np

def cornerplot(samples,
               labels=None,
               weights=None,
               levels=None,
               span=None,
               quantiles=[0.025, 0.5, 0.975],
               truth=None,
               color=None,
               alpha=0.5,
               linewidth=1.5,
               fill=True,
               fontsize=10,
               show_titles=True,
               title_fmt='.2f',
               title_fontsize=12,
               cut=3,
               fig=None,
               size=(10,10)):
    r"""
    Plot corner-plot of samples.

    Parameters
    ----------
    samples : array
        Array of shape (nsamples, ndim) containing the samples.
    labels : list
        List of names of for the parameters.
    weights : array
        Array with weights (useful if different samples have different weights e.g. as in Nested Sampling).
    levels : list
        The quantiles used for plotting the smoothed 2-D distributions. If not provided, these default to 0.5, 1, 1.5, and 2-sigma contours.
    quantiles : list
        A list of fractional quantiles to overplot on the 1-D marginalized posteriors as titles. Default is ``[0.025, 0.5, 0.975]`` (spanning the 95%/2-sigma credible interval).
    truth : array
        Array specifying a point to be highlighted in the plot. It can be the true values of the parameters, the mean, median etc. By default this is None.
    color : str
        Matplotlib color to be used in the plot.
    alpha : float
        Transparency value of figure (Default is 0.5).
    linewidth : float
        Linewidth of plot (Default is 1.5).
    fill : bool
        If True (Default) the fill the 1D and 2D contours with color.
    fontsize : float
        Fontsize of axes labels. Default is 10.
    show_titles : bool
        Whether to display a title above each 1-D marginalized posterior showing the quantiles. Default is True.
    title_fmt : str
        Format of the titles. Default is ``.2f``.
    title_fontsize : float
        Fontsize of titles. Default is 12.
    cut : float
        Factor, multiplied by the smoothing bandwidth, that determines how far the evaluation grid extends past the extreme datapoints.
        When set to 0, truncate the curve at the data limits. Default is ``cut=3``.
    fig : (figure, axes)
        Pre-existing Figure and Axes for the plot. Otherwise create new internally. Default is None.
    size : (int, int)
        Size of the plot. Default is (10, 10).
    
    Returns
    -------
    Figure, Axes
        The matplotlib figure and axes.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    ndim = samples.shape[1]

    if labels is None:
        labels = [r"$x_{"+str(i+1)+"}$" for i in range(ndim)]

    if levels is None:
        levels = list(1.0 - np.exp(-0.5 * np.arange(0.5, 2.1, 0.5) ** 2))
    levels.append(1.0)
    
    if color is None:
        color = "tab:blue"
    
    # Determine plotting bounds.
    if span is None:
        span = [0.999999426697 for i in range(ndim)]
    span = list(span)
    if len(span) != ndim:
        raise ValueError("Dimension mismatch between samples and span.")
    for i, _ in enumerate(span):
        try:
            xmin, xmax = span[i]
        except:
            q = [0.5 - 0.5 * span[i], 0.5 + 0.5 * span[i]]
            span[i] = _quantile(samples[:,i], q, weights=weights)

    idxs = np.arange(ndim**2).reshape(ndim, ndim)
    tril = np.tril_indices(ndim)
    triu = np.triu_indices(ndim)
    lower = list(set(idxs[tril])-set(idxs[triu]))
    upper = list(set(idxs[triu])-set(idxs[tril]))
    
    if fig is None:
        figure, axes = plt.subplots(ndim, ndim, figsize=size, sharex=False)
    else:
        figure = fig[0]
        axes = fig[1]

    for idx, ax in enumerate(axes.flat):

        i = idx // ndim
        j = idx % ndim        
        
        if idx in lower:
            
            ax.set_ylim(span[i])
            ax.yaxis.set_major_locator(plt.MaxNLocator(5))

            if fill:
                ax = sns.kdeplot(x=samples[:,j], y=samples[:,i], weights=weights,
                                fill=True, color=color,
                                clip=None, cut=cut,
                                thresh=levels[0], levels=levels,
                                ax=ax, alpha=alpha, linewidth=0.0,
                                )
            ax = sns.kdeplot(x=samples[:,j], y=samples[:,i], weights=weights,
                             fill=False, color=color,
                             clip=None, cut=cut,
                             thresh=levels[0], levels=levels,
                             ax=ax, alpha=alpha, linewidth=linewidth,
                             )

            if truth is not None:
                ax.axvline(truth[j], color='k', lw=1.0)
                ax.axhline(truth[i], color='k', lw=1.0)

            if j == 0:
                ax.set_ylabel(labels[i], fontsize=fontsize)
                [l.set_rotation(45) for l in ax.get_yticklabels()]
            else:
                ax.yaxis.set_ticklabels([])

            if i == ndim - 1:
                ax.xaxis.set_major_locator(plt.MaxNLocator(5))
                ax.set_xlabel(labels[j], fontsize=fontsize)
                [l.set_rotation(45) for l in ax.get_xticklabels()]
            else:
                ax.set_xticklabels([])

                        
        elif idx in upper:
            ax.set_axis_off()
        else:

            ax.yaxis.set_major_locator(plt.NullLocator())

            if fill:
                ax = sns.kdeplot(x=samples[:,j],
                                fill=True, color=color, weights=weights,
                                clip=None, cut=cut,
                                ax=ax, linewidth=0.0, alpha=alpha,
                                )
            ax = sns.kdeplot(x=samples[:,j],
                             fill=None, color=color, weights=weights,
                             clip=None, cut=cut,
                             ax=ax, linewidth=linewidth, alpha=alpha,
                             )

            if truth is not None:
                ax.axvline(truth[j], color='k', lw=1.0)

            if i == ndim - 1:
                ax.set_xlabel(labels[j], fontsize=fontsize)
                [l.set_rotation(45) for l in ax.get_xticklabels()]
            else:
                ax.set_xticklabels([])
            
            if show_titles:
                ql, qm, qh = _quantile(samples[:,i], quantiles, weights=weights)
                q_minus, q_plus = qm - ql, qh - qm
                fmt = "{{0:{0}}}".format(title_fmt).format
                title = r"${{{0}}}_{{-{1}}}^{{+{2}}}$"
                title = title.format(fmt(qm), fmt(q_minus), fmt(q_plus))
                title = "{0} = {1}".format(labels[i], title)
                ax.set_title(title, fontsize=title_fontsize)
            

        ax.set_xlim(span[j])
        ax.xaxis.set_major_locator(plt.MaxNLocator(5))
    
    figure.subplots_adjust(top=0.95, right=0.95, wspace=.05, hspace=.05)

    return figure, axes


def _quantile(x, q, weights=None):
    """
    Compute (weighted) quantiles from an input set of samples.
    Parameters
    ----------
    x : `~numpy.ndarray` with shape (nsamps,)
        Input samples.
    q : `~numpy.ndarray` with shape (nquantiles,)
       The list of quantiles to compute from `[0., 1.]`.
    weights : `~numpy.ndarray` with shape (nsamps,), optional
        The associated weight from each sample.
    Returns
    -------
    quantiles : `~numpy.ndarray` with shape (nquantiles,)
        The weighted sample quantiles computed at `q`.
    """

    # Initial check.
    x = np.atleast_1d(x)
    q = np.atleast_1d(q)

    # Quantile check.
    if np.any(q < 0.0) or np.any(q > 1.0):
        raise ValueError("Quantiles must be between 0. and 1.")

    if weights is None:
        # If no weights provided, this simply calls `np.percentile`.
        return np.percentile(x, list(100.0 * q))
    else:
        # If weights are provided, compute the weighted quantiles.
        weights = np.atleast_1d(weights)
        if len(x) != len(weights):
            raise ValueError("Dimension mismatch: len(weights) != len(x).")
        idx = np.argsort(x)  # sort samples
        sw = weights[idx]  # sort weights
        cdf = np.cumsum(sw)[:-1]  # compute CDF
        cdf /= cdf[-1]  # normalize CDF
        cdf = np.append(0, cdf)  # ensure proper span
        quantiles = np.interp(q, cdf, x[idx]).tolist()
        return quantiles