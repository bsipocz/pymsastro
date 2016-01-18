# Licensed under a 3-clause BSD style license - see LICENSE.rst

from .weighting import weights_calc_spectrum, _normalize_weights
from . import SynSpectrum

__all__ = ['addSynSpectra']


def addSynSpectra(first, second, weighting=None, s_n_calculator=None,
                  returnreference=True):
    """
    Adds two instances of `SynSpec` with a given weighting

    Parameters
    ----------
    first, second : `SynSpectrum`
        The first and second spectrum.

    weighting : ``None``, `tuple` or `str`, optional
        Depeding on the input the weighting changes:

        - ``None`` : no weighting
            Neither spectrum will be weighted.
        - `tuple` : Custom weighting
            Needs to have exactly two elements, the first is the
            weighting factor for the first spectrum, the second is the
            weighting for the second.
        - ``'optimal'`` :
          Weight by inverse of (other signal times own variance)
        - ``'snr_squared'`` : Weight with squared signal to noise ratio
        - ``'ierr'`` : Weight by inverse error
        - ``'ivar'`` : Weight by inverse variance

        The weighting factors will be normalized during this function and need
        not to be in the input.
        Default is ``None``.

    s_n_calculator : ``None`` or `~pymsastro.stats.SNR`-like, optional
        A class that takes the signal with noise and has an attribute
        ``signal`` and ``noise`` which return the calculated signal and noise.
        If ``None`` then the theoretical signal and noise are taken from the
        synthetic spectrum class. If ``weighting`` is a tuple then this
        parameter must be ``None``.
        Default is ``None``.

    returnreference : `bool`, optional
        Also return the result of the weighted addition for the spectra without
        noise.
        Default is ``True``.

    Returns
    -------
    result : `numpy.ndarray`
        The result of the weighted addition of the spectra with noise.
    reference : `numpy.ndarray`
        The result of the weighted addition of the spectra without noise. Is
        only returned if ``returnreference`` is ``True``.

    Notes
    -----
    Since the exact signals and spectra are known the reference is calculated
    too if one wants to calculate the goodness of the weighted addition.
    """
    # Verify the input spectra are SynSpecs
    if (not isinstance(first, SynSpectrum) or
            not isinstance(second, SynSpectrum)):
        raise TypeError('Inputs must be SynSpecs.')

    # Evaluate the weighting

    if isinstance(weighting, tuple):
        # There must not be a calculator function if there are custom weights.
        if s_n_calculator is not None:
            raise TypeError('If weighting is a tuple the signal and noise'
                            'calculator function must be None.')
        # Custom weighting factors need to be normalized.
        weight1, weight2 = weighting
        weight1, weight2 = _normalize_weights(weight1, weight2)

    else:
        # Weights have to be calculated
        if s_n_calculator is None:
            # Using the ideal values from the synthetic spectrum class.
            weight1, weight2 = weights_calc_spectrum(first.signalsum,
                                                     second.signalsum,
                                                     first.noisequadsum,
                                                     second.noisequadsum,
                                                     weighting)
        else:
            # TODO: This could be put in a more general function since
            # this kind of weighting is always possible and does not
            # require the precise values from a synthetic spectrum!

            # Use the class that calculates the signal and noise
            s_n_spec1 = s_n_calculator(first.signalsum_with_noise)
            signal1 = s_n_spec1.signal
            noise1 = s_n_spec1.noise
            s_n_spec2 = s_n_calculator(second.signalsum_with_noise)
            signal2 = s_n_spec2.signal
            noise2 = s_n_spec2.noise
            # And then calculate the weights
            weight1, weight2 = weights_calc_spectrum(signal1, signal2,
                                                     noise1, noise2,
                                                     weighting)

    # Calculate the result with noise
    result = (weight1 * first.signalsum_with_noise +
              weight2 * second.signalsum_with_noise)

    if returnreference:
        reference = weight1 * first.signalsum + weight2 * second.signalsum
        return result, reference

    else:
        return result
