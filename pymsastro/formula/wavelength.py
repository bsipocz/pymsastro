# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy.constants import c as speed_of_light
import astropy.units as u

__all__ = ['wavelength_frequency']


def wavelength_frequency(wavelength=None, frequency=None):
    """
    Implements the wavelength (:math:`\\lambda`) - frequency (:math:`f`)
    relationship:

    .. math ::
        \\lambda = \\frac{c}{f}

    Parameters
    ----------
    wavelength : `~astropy.units.Quantity` or ``None``
        The wavelength.

    frequency : `~astropy.units.Quantity` or ``None``
        The frequency.

    Returns
    -------
    Parameter that was not None.

    Raises
    ------
    ValueError
        If more than one or no Parameter was ``None``.
    """
    if wavelength is None:
        return (speed_of_light / frequency).to(u.Angstrom)
    elif frequency is None:
        return (speed_of_light / wavelength).to(u.Hz)
    else:
        raise ValueError('One parameter must be None.')
