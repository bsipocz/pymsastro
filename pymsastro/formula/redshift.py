# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy.constants import c as speed_of_light
import astropy.units as u
import numpy as np

from .math import pq_formula
from .special_relativity import lorentzfactor_velocity, betafactor_velocity

__all__ = ['redshift_wavelength', 'redshift_frequency',
           'redshift_velocity_normal', 'redshift_velocity_relativistic',
           'redshift_velocity_relativistic_angle']


def redshift_wavelength(wavelength_observed=None, wavelength_emitted=None,
                        redshift=None):
    """
    Implements the redshift (:math:`z`) - wavelength (:math:`\\lambda`)
    relationship:

    .. math ::
        1 + z = \\frac{\\lambda_{\\text{observed}}}{ \\lambda_{\\text{emitted}}}

    Parameters
    ----------
    wavelength_observed : ``Number``, `numpy.ndarray` or None
        The observed wavelength.

    wavelength_emitted : ``Number``, `numpy.ndarray` or None
        The wavelength at emission (theoretical wavelength).

    redshift : ``Number``, `numpy.ndarray` or None
        The redshift (z).

    Returns
    -------
    Parameter that was not None.

    Raises
    ------
    ValueError
        If more than one or no Parameter was ``None``.
    """
    if redshift is None:
        return wavelength_observed / wavelength_emitted - 1
    elif wavelength_observed is None:
        return (1 + redshift) * wavelength_emitted
    elif wavelength_emitted is None:
        return wavelength_observed / (1 + redshift)
    else:
        raise ValueError('One parameter must be None.')


def redshift_frequency(frequency_observed=None, frequency_emitted=None,
                       redshift=None):
    """
    Implements the redshift (:math:`z`) - frequency (:math:`f`)
    relationship:

    .. math ::
        1 + z = \\frac{f_{\\text{emitted}}}{f_{\\text{observed}}}

    Parameters
    ----------
    frequency_observed : ``Number``, `numpy.ndarray` or None
        The observed frequency.

    frequency_emitted : ``Number``, `numpy.ndarray` or None
        The frequency at emission (theoretical frequency).

    redshift : ``Number``, `numpy.ndarray` or None
        The redshift (z).

    Returns
    -------
    Parameter that was not None.

    Raises
    ------
    ValueError
        If more than one or no Parameter was ``None``.
    """
    if redshift is None:
        return frequency_emitted / frequency_observed - 1
    elif frequency_emitted is None:
        return (1 + redshift) * frequency_observed
    elif frequency_observed is None:
        return frequency_emitted / (1 + redshift)
    else:
        raise ValueError('One parameter must be None.')


def redshift_velocity_nonrelativistic(velocity=None, redshift=None):
    """
    Implements the non-relativistic redshift (:math:`z`) - velocity (:math:`v`)
    relationship (Doppler effect for :math:`v \\ll c`):

    .. math ::
        z = \\frac{v}{c}

    With :math:`c` being the speed of light.

    Parameters
    ----------
    velocity : `~astropy.units.Quantity` or None
        The velocity in the line of sight. Must be much smaller than the
        lightspeed.

    redshift : ``Number``, `numpy.ndarray` or None
        The redshift (z).

    Returns
    -------
    Parameter that was not None.

    Raises
    ------
    ValueError
        If more than one or no Parameter was ``None``.
    """
    if redshift is None:
        return betafactor_velocity(velocity=velocity)
    elif velocity is None:
        return (redshift * speed_of_light).to(u.km / u.s)
    else:
        raise ValueError('One parameter must be None.')


def redshift_velocity_relativistic(velocity=None, redshift=None):
    """
    Implements the relativistic redshift (:math:`z`) - velocity (:math:`v`)
    relationship (Doppler effect for :math:`v \\gg 0` *but* only valid for
    special relativistic treatement not general relativistic one):

    .. math ::
        1 + z = \\frac{(1 + \\frac{v}{c})}{\\sqrt{1-(\\frac{v}{c})^2}}

    With :math:`c` being the speed of light.

    .. warning::
        This function should *not* be used for recession velocities due to
        cosmological redshift.

    Parameters
    ----------
    velocity : `~astropy.units.Quantity` or None
        The velocity in the line of sight. Must be much smaller than the
        lightspeed.

    redshift : ``Number``, `numpy.ndarray` or None
        The redshift (z).

    Returns
    -------
    Parameter that was not None.

    Raises
    ------
    ValueError
        If more than one or no Parameter was ``None``.
    """
    if redshift is None:
        left = 1 + betafactor_velocity(velocity=velocity)
        gamma = lorentzfactor_velocity(velocity=velocity)
        return left * gamma - 1
    elif velocity is None:
        # (1+z^2) = (1+beta)^2/(1-beta^2)
        # => beta^2 (1+(1+z^2)) + 2 beta - 2z - z^2
        # then pq formula.
        divisor = 1 + (1+redshift)**2
        p = 2 / divisor
        q = (- 2 * redshift - redshift**2) / divisor
        return (pq_formula(p, q) * speed_of_light).to(u.km / u.s)
    else:
        raise ValueError('One parameter must be None.')


def redshift_velocity_relativistic_angle(velocity=None, redshift=None,
                                         angle=None):
    """
    Implements the relativistic redshift (:math:`z`) - velocity (:math:`v`)
    with angle (:math:`\\theta`) relationship (Doppler effect for
    :math:`v \\gg 0` *but* only valid for special relativistic treatement
    not general relativistic one):

    .. math ::
        1+z=\\frac{(1+\\cos(\\theta)\\frac{v}{c})}{\\sqrt{1-(\\frac{v}{c})^2}}

    With :math:`c` being the speed of light.

    .. warning::
        This function should *not* be used for recession velocities due to
        cosmological redshift.

    Parameters
    ----------
    velocity : `~astropy.units.Quantity` or None
        The velocity in the line of sight. Must be much smaller than the
        lightspeed.

    redshift : ``Number``, `numpy.ndarray` or None
        The redshift (z).

    angle : `~astropy.units.Quantity` or None
        If angle is the angle between the direction of relative motion and the
        direction of emission in the observer's frame (zero angle is directly
        away from the observer) (Wikipedia).

    Returns
    -------
    Parameter that was not None.

    Raises
    ------
    ValueError
        If more than one or no Parameter was ``None``.

    Notes
    -----
    If ``angle`` is ``None`` there could be situations where the result is
    ``NaN``. This happens if the angle is very close to zero or multiples of
    180 degree.
    """
    if redshift is None:
        left = 1 + np.cos(angle) * betafactor_velocity(velocity=velocity)
        gamma = lorentzfactor_velocity(velocity=velocity)
        return left * gamma - 1
    elif velocity is None:
        # (1+z^2) = (1+cos(angle)*beta)^2/(1-beta^2)
        # => beta^2 (1+(1+z^2)) + 2*cos(angle)*beta - 2z - z^2
        # then pq formula.
        divisor = np.cos(angle)**2 + (1+redshift)**2
        p = 2 * np.cos(angle) / divisor
        q = (- 2 * redshift - redshift**2) / divisor
        return (pq_formula(p, q) * speed_of_light).to(u.km / u.s)
    elif angle is None:
        # Formula:
        # ((1+z)/gamma - 1) / (v/c) = cos(angle)
        beta = betafactor_velocity(velocity=velocity)
        other = (1 + redshift) / lorentzfactor_velocity(velocity=velocity)
        return np.arccos((other - 1) / beta)
    else:
        raise ValueError('One parameter must be None.')
