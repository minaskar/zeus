import numpy as np 

def GelmanRubin(chains):
    '''
    Gelman-Rubin convergence statistic.
    '''

    chains = np.array(chains)

    # Number and length of chains
    M, N, _ = np.shape(chains)

    # Within chain variance
    W = 0.0
    for chain in chains:
        W += np.var(chain, axis=0)
    W /= M

    # Means of chains
    means = []
    for chain in chains:
        means.append(np.mean(chain, axis=0))
    means = np.array(means)
    
    # Between chain variance
    B = N * np.var(means, ddof=1, axis=0)

    # Weighted variance
    Var_hat = (1.0 - 1.0 / N) * W + B / N

    # Return R_hat statistic
    R_hat = np.sqrt(Var_hat / W)
    
    return R_hat


def Geweke(chain, first, last, intervals=20):
    '''
    Geweke convergence diagnostic.

    Parameters
    ----------
    chain : array
        1D array with samples corresponding to one parameter
    first : float
        Value in (0.0, 1.0)
    last : float
        Value in (0.1, 1.0)

    Returns
    -------
    zscores : array
        Array of z-scores for all intervals.
    '''


    for interval in (first, last):
        if interval <= 0 or interval >= 1:
            raise ValueError("Invalid intervals for Geweke convergence analysis", (first, last))
    if first + last >= 1:
        raise ValueError("Invalid intervals for Geweke convergence analysis", (first, last))

    # Initialize list of z-scores
    zscores = []

    # Last index value
    end = len(chain) - 1

    # Start intervals going up to the <last>% of the chain
    last_start_idx = (1 - last) * end

    # Calculate starting indices
    start_indices = np.linspace(0, last_start_idx, num=intervals, endpoint=True, dtype=int)

    # Loop over start indices
    for start in start_indices:
        # Calculate slices
        first_slice = chain[start : start + int(first * (end - start))]
        last_slice = chain[int(end - last * (end - start)) :]

        z_score = (first_slice.mean() - last_slice.mean())/np.sqrt(first_slice.var() + last_slice.var())

        zscores.append([start, z_score])

    return np.array(zscores)