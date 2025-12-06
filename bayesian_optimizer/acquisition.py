import numpy as np
import scipy.stats as sps


def weighted_expected_improvement(
    f_best: np.ndarray[np.float64] | float,
    mus: np.ndarray[np.float64] | float,
    sigmas: np.ndarray[np.float64] | float,
    weight=0.5,
) -> np.ndarray[np.float64]:
    """_summary_

    Args:
        f_best (np.ndarray[np.float64] | float): minimum observed f(x) so far
        mus (np.ndarray[np.float64] | float): mean from gaussian process
        sigmas (np.ndarray[np.float64] | float): sigma / uncertainty from gaussian process
        weight (float, optional): Weight for exploration versus exploitation,
                                    higher = more exploitation. Defaults to 0.5.

    Returns:
        np.ndarray[np.float64]: weighted expected improvement
    """

    mus = np.asarray(mus)
    sigmas = np.asarray(sigmas)

    # Output array
    wei = np.zeros_like(mus, dtype=float)

    # Returns a boolean array of same shape as sigmas
    # where values are True for sigmas > 0.0 and False
    # otherwise
    mask = sigmas > 0.0
    # if all sigmas are zero, return expected improvement as all zeroes
    if not np.any(mask):
        return wei

    # filter out standard deviations that are zeroes
    mu_masked = mus[mask]
    sigma_masked = sigmas[mask]

    # returns standardized improvement of same shape
    # as mu_masked and sigma_masked
    # equation from this paper: https://arxiv.org/pdf/2306.04262
    z = (f_best - mu_masked) / sigma_masked

    pdf = sps.norm.pdf(z)
    cdf = sps.norm.cdf(z)

    ei_exploit = (f_best - mu_masked) * cdf
    ei_explore = sigma_masked * pdf

    wei_masked = weight * ei_exploit + (1.0 - weight) * ei_explore

    # Clip improvements, negative improvements will be zero
    wei_masked = np.maximum(wei_masked, 0.0)

    # Fill in only where sigma > 0; this assigns the value of wei_masked
    # only to those indices that were not masked out; i.e. if the mask is
    # np.array([True, False, True]) and wei_masked = np.array([1.0, 0.5])
    # wei = np.array([1.0, 0.0, 0.5])
    wei[mask] = wei_masked
    return wei
