# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np

__all__ = ['pq_formula', 'abc_formula']


def pq_formula(p, q):
    """
    Implements the pq formula for solutions of quadratic equations:

    .. math ::
        x^2 + px + q = 0

        \\Rightarrow x_{1, 2} = -\\frac{p}{2} \\pm \\sqrt{(\\frac{p}{2})^2 - q}

    Parameters
    ----------
    p : ``Number`` or `numpy.ndarray`-like
        The linear factor.

    q : ``Number`` or `numpy.ndarray`-like
        The constant factor.

    Returns
    -------
    x1, x2 : ``Number`` or `numpy.ndarray`-like
        The solutions for the pq formula
    """
    p_half = p / 2
    root_term = np.sqrt(p_half**2 - q)
    solution1 = - p_half + root_term
    solution2 = - p_half - root_term
    return solution1, solution2


def abc_formula(a, b, c):
    """
    Implements the abc formula for solutions of quadratic equations:

    .. math ::
        ax^2 + bx + c = 0

        \\Rightarrow x_{1, 2} = \\frac{-b \\pm \\sqrt{b^2 - 4ac}}{2a}

    Parameters
    ----------
    a : ``Number`` or `numpy.ndarray`-like
        The quadratic factor.

    b : ``Number`` or `numpy.ndarray`-like
        The linear factor.

    c : ``Number`` or `numpy.ndarray`-like
        The constant factor.

    Returns
    -------
    x1, x2 : ``Number`` or `numpy.ndarray`-like
        The solutions for the pq formula
    """
    divisor = 2 * a
    root_term = np.sqrt(b*b - 4*a*c)
    solution1 = (- b + root_term) / divisor
    solution2 = (- b - root_term) / divisor
    return solution1, solution2
