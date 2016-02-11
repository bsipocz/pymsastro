# Licensed under a 3-clause BSD style license - see LICENSE.rst

import matplotlib.pyplot as plt

__all__ = ['pltImshow']


def pltImshow(array):
    """
    Uses `matplotlib.pyplot.imshow` but with predefined settings.

    Parameters
    ----------
    array : `numpy.ndarray`
        The 2D image to plot.

    Notes
    -----
    The function sets ``origin='lower'``, ``interpolation='none'`` and
    ``cmap=plt.get_cmap('gray')``
    """
    plt.imshow(array,
               origin='lower',
               interpolation='none',
               cmap=plt.get_cmap('gray'))
    plt.show()
