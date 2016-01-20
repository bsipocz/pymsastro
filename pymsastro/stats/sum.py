# Licensed under a 3-clause BSD style license - see LICENSE.rst

from numpy import sqrt, square
from numpy.ma import sum

__all__ = ['sum_square', 'ss', 'sum_of_squares', 'root_sum_square']


def sum_square(value,  **kwargs):
    """
    Calculates the sum of squares.

    Parameters
    ----------
    value : ``Number`` or `numpy.ndarray`
        The measured values

    kwargs :
        ``kwargs`` for `numpy.sum`.

    Returns
    -------
    ss : ``Number`` or `numpy.ndarray`
        The sum of squares.

    Notes
    -----
    The value is calculated by ``ss = sum(square(value))``
    """
    return sum(square(value), **kwargs)


def root_sum_square(value, **kwargs):
    """
    Calculates the root of the sum of squares.

    Parameters
    ----------
    value : ``Number`` or `numpy.ndarray`
        The measured values

    kwargs :
        ``kwargs`` for `numpy.sum`.

    Returns
    -------
    root_sum_square : ``Number`` or `numpy.ndarray`
        The root of the sum of squares.

    Notes
    -----
    The value is calculated by ``root_sum_square = sqrt(sum(square(value)))``
    """
    return sqrt(sum(square(value), **kwargs))

ss = sum_square
sum_of_squares = sum_square
