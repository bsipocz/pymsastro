# Licensed under a 3-clause BSD style license - see LICENSE.rst

__all__ = ['weights_calc_spectrum']


def weights_calc_spectrum(signal1, signal2, noise1, noise2,
                          method='optimal'):
    """
    Calculates the weighting factors for two spectra.

    Parameters
    ----------
    signal1, signal2 : `numpy.ndarray` or ``Number``
        The signal of spectrum 1 and spectrum 2.
    noise1, noise2 : `numpy.ndarray` or ``Number``
        The noise of spectrum 1 and spectrum 2.
    method : `str`, optional
        The method how to calculate the weighting factors. See Notes for
        further details.
        Default is ``"optimal"``.

    Returns
    -------
    weight1 : `numpy.ndarray` or ``Number``
        The weighting factor for spectrum 1.
    weight2 : `numpy.ndarray` or ``Number``
        The weighting factor for spectrum 2.

    Notes
    -----
    How the weighting factors are calculated depends on the method. In the
    following signal1, 2 is abbreviated by s1, s2 and noise1, 2 by n1, n2:

    - ``"optimal"`` :

      * Weight1 = 1 / (s2 * n1 * n1)
      * Weight2 = 1 / (s1 * n2 * n2)

    - ``"snr_squared"`` :

      * Weight1 = (s1 / n1)**2,
      * Weight2 = (s2 / n2)**2

    - ``"ivar"`` :

      * Weight1 = (1 / n1)**2,
      * Weight2 = (1 / n2)**2

    - ``"ierr"`` :

      * Weight1 = 1 / n1,
      * Weight2 = 1 / n2

    - ``"none"`` :

      * Weight1 = 1,
      * Weight2 = 1


    But all weighting factors are normalized afterwards so that
    ``weight1 + weight2 = 1`` before they are returned.
    """
    if method == 'optimal':
        method_func = _weighting_optimal
    elif method == 'snr_squared':
        method_func = _weighting_snr_squared
    elif method == 'ivar':
        method_func = _weighting_ivar
    elif method == 'ierr':
        method_func = _weighting_ierr
    elif method == 'none' or method is None:
        method_func = _weighting_none
    else:
        raise ValueError('Unknown weighting calculation method.')

    # Calculate the weighting factors
    weight1, weight2 = method_func(signal1, signal2, noise1, noise2)

    # Return the normalized weighting factors
    return _normalize_weights(weight1, weight2)


def _normalize_weights(weight1, weight2):
    """
    Normalize the weighting factors so that they add to one.
    """
    weightsum = weight1 + weight2
    return (weight1 / weightsum, weight2 / weightsum)


def _weighting_optimal(signal1, signal2, noise1, noise2):
    """See ``weights_spectrum()``"""
    weight1 = 1 / (signal2 * noise1**2)
    weight2 = 1 / (signal1 * noise2**2)
    return (weight1, weight2)


def _weighting_snr_squared(signal1, signal2, noise1, noise2):
    """See ``weights_spectrum()``"""
    weight1 = (signal1 ** 2 / noise1 ** 2)
    weight2 = (signal2 ** 2 / noise2 ** 2)
    return (weight1, weight2)


def _weighting_ivar(signal1, signal2, noise1, noise2):
    """See ``weights_spectrum()``"""
    weight1 = (1 / noise1 ** 2)
    weight2 = (1 / noise2 ** 2)
    return (weight1, weight2)


def _weighting_ierr(signal1, signal2, noise1, noise2):
    """See ``weights_spectrum()``"""
    weight1 = 1 / noise1
    weight2 = 1 / noise2
    return (weight1, weight2)


def _weighting_none(signal1, signal2, noise1, noise2):
    """See ``weights_spectrum()``"""
    weight1 = 1
    weight2 = 1
    return (weight1, weight2)
