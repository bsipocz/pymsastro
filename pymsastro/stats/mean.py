# Licensed under a 3-clause BSD style license - see LICENSE.rst


from numpy import square, sqrt
from numpy.ma import mean
from scipy.stats import hmean, gmean

__all__ = ['root_mean_square', 'rms',
           'harmonic_mean', 'geometric_mean', 'arithmetic_mean',
           'quadratic_mean']


def root_mean_square(value, **kwargs):
    """
    Calculates the root mean square (quadratic mean).

    Parameters
    ----------
    value : ``Number`` or `numpy.ndarray`
        The measured values

    kwargs :
        ``kwargs`` for `numpy.mean`.

    Returns
    -------
    rms : ``Number`` or `numpy.ndarray`
        The quadratic mean.

    Notes
    -----
    The value is calculated by ``rms = sqrt(mean(square(value)))``
    """
    return sqrt(mean(square(value), **kwargs))


def harmonic_mean(value, *args, **kwargs):
    """
    Wrapper for :func:`scipy.stats.hmean`
    """
    return hmean(value, *args, **kwargs)


def geometric_mean(value, *args, **kwargs):
    """
    Wrapper for :func:`scipy.stats.gmean`
    """
    return gmean(value, *args, **kwargs)


def arithmetic_mean(value, *args, **kwargs):
    """
    Wrapper for :func:`numpy.mean`
    """
    return mean(value, *args, **kwargs)

rms = root_mean_square
quadratic_mean = root_mean_square
