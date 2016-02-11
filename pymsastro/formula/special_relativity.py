# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy.constants import c as speed_of_light
import numpy as np

__all__ = ['lorentzfactor_velocity', 'betafactor_velocity']


def lorentzfactor_velocity(velocity=None, lorentzfactor=None):
    """
    Implements the lorentz factor (:math:`\\gamma`) - velocity (:math:`v`)
    relationship:

    .. math ::
        \\gamma = \\frac{1}{\\sqrt{1-(\\frac{v}{c})^2}}

    With :math:`c` being the speed of light.

    Parameters
    ----------
    velocity : `~astropy.units.Quantity` or None
        The velocity.

    lorentzfactor : ``Number``, `numpy.ndarray` or None
        The Lorentz-Factor.

    Returns
    -------
    Parameter that was not None.

    Raises
    ------
    ValueError
        If more than one or no Parameter was ``None``.
    """
    if lorentzfactor is None:
        return 1 / np.sqrt(1 - betafactor_velocity(velocity=velocity)**2)
    elif velocity is None:
        return speed_of_light * np.sqrt(1-1/lorentzfactor**2)
    else:
        raise ValueError('One parameter must be None.')


def betafactor_velocity(velocity=None, betafactor=None):
    """
    Implements the beta-factor (:math:`\\beta`) - velocity (:math:`v`)
    relationship:

    .. math ::
        \\beta = \\frac{v}{c}

    With :math:`c` being the speed of light.

    Parameters
    ----------
    velocity : `~astropy.units.Quantity` or None
        The velocity.

    betafactor : ``Number``, `numpy.ndarray` or None
        The Beta-Factor.

    Returns
    -------
    Parameter that was not None.

    Raises
    ------
    ValueError
        If more than one or no Parameter was ``None``.
    """
    if betafactor is None:
        return (velocity / speed_of_light).decompose()
    elif velocity is None:
        return speed_of_light * betafactor
    else:
        raise ValueError('One parameter must be None.')
