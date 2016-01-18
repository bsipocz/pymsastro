# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np
from numpy.testing import (assert_almost_equal, assert_array_almost_equal,
                           assert_array_equal)

from .. import *


def test_error_int():
    a = 10
    b = 12
    assert signed_error(a, b) == -2
    assert absolute_error(a, b) == 2

    assert_almost_equal(relative_signed_error(a, b), -1/6)
    assert_almost_equal(relative_absolute_error(a, b), 1/6)

    assert_almost_equal(percentage_signed_error(a, b), -100/6)
    assert_almost_equal(percentage_absolute_error(a, b), 100/6)

    assert square_error(a, b) == 4
    assert square_relative_error(a, b) == 1/36

    assert mean_signed_error(a, b) == -2
    assert mean_absolute_error(a, b) == 2

    assert_almost_equal(mean_relative_signed_error(a, b), -1/6)
    assert_almost_equal(mean_relative_absolute_error(a, b), 1/6)

    assert_almost_equal(mean_percentage_signed_error(a, b), -100/6)
    assert_almost_equal(mean_percentage_absolute_error(a, b), 100/6)

    assert_almost_equal(mean_square_error(a, b), 4)
    assert_almost_equal(mean_square_relative_error(a, b), 1/36)

    assert_almost_equal(root_mean_square_error(a, b), 2)
    assert_almost_equal(root_mean_square_relative_error(a, b), 1/6)

    assert median_signed_error(a, b) == -2
    assert median_absolute_error(a, b) == 2

    assert_almost_equal(median_relative_signed_error(a, b), -1/6)
    assert_almost_equal(median_relative_absolute_error(a, b), 1/6)

    assert_almost_equal(median_percentage_signed_error(a, b), -100/6)
    assert_almost_equal(median_percentage_absolute_error(a, b), 100/6)

    assert median_square_error(a, b) == 4
    assert median_square_relative_error(a, b) == 1/36

    assert sum_square_error(a, b) == 4
    assert sum_square_relative_error(a, b) == 1/36


def test_error_float():
    a = 10.5
    b = 12.2
    assert_almost_equal(signed_error(a, b), -1.7)
    assert_almost_equal(absolute_error(a, b), 1.7)

    assert_almost_equal(relative_signed_error(a, b), -1.7/12.2)
    assert_almost_equal(relative_absolute_error(a, b), 1.7/12.2)

    assert_almost_equal(percentage_signed_error(a, b), -170/12.2)
    assert_almost_equal(percentage_absolute_error(a, b), 170/12.2)

    assert_almost_equal(square_error(a, b), 2.89)
    assert_almost_equal(square_relative_error(a, b), 2.89/148.84)

    assert_almost_equal(mean_signed_error(a, b), -1.7)
    assert_almost_equal(mean_absolute_error(a, b), 1.7)

    assert_almost_equal(mean_relative_signed_error(a, b), -1.7/12.2)
    assert_almost_equal(mean_relative_absolute_error(a, b), 1.7/12.2)

    assert_almost_equal(mean_percentage_signed_error(a, b), -170/12.2)
    assert_almost_equal(mean_percentage_absolute_error(a, b), 170/12.2)

    assert_almost_equal(mean_square_error(a, b), 2.89)
    assert_almost_equal(mean_square_relative_error(a, b), 2.89/148.84)

    assert_almost_equal(root_mean_square_error(a, b), 1.7)
    assert_almost_equal(root_mean_square_relative_error(a, b), 1.7/12.2)

    assert_almost_equal(median_signed_error(a, b), -1.7)
    assert_almost_equal(median_absolute_error(a, b), 1.7)

    assert_almost_equal(median_relative_signed_error(a, b), -1.7/12.2)
    assert_almost_equal(median_relative_absolute_error(a, b), 1.7/12.2)

    assert_almost_equal(median_percentage_signed_error(a, b), -170/12.2)
    assert_almost_equal(median_percentage_absolute_error(a, b), 170/12.2)

    assert_almost_equal(median_square_error(a, b), 2.89)
    assert_almost_equal(median_square_relative_error(a, b), 2.89/148.84)

    assert_almost_equal(sum_square_error(a, b), 2.89)
    assert_almost_equal(sum_square_relative_error(a, b), 2.89/148.84)


def test_error_one_ndarray():
    a = np.array([1, 2, 3])
    b = 2
    assert_array_equal(signed_error(a, b), np.array([-1, 0, 1]))
    assert_array_equal(absolute_error(a, b), np.array([1, 0, 1]))

    assert_array_equal(relative_signed_error(a, b), np.array([-0.5, 0, 0.5]))
    assert_array_equal(relative_absolute_error(a, b), np.array([0.5, 0, 0.5]))

    assert_array_equal(percentage_signed_error(a, b), np.array([-50, 0, 50]))
    assert_array_equal(percentage_absolute_error(a, b), np.array([50, 0, 50]))

    assert_array_equal(square_error(a, b), np.array([1, 0, 1]))
    assert_array_equal(square_relative_error(a, b), np.array([0.25, 0, 0.25]))

    assert_almost_equal(mean_signed_error(a, b), 0)
    assert_almost_equal(mean_absolute_error(a, b), 2/3)

    assert_almost_equal(mean_relative_signed_error(a, b), 0)
    assert_almost_equal(mean_relative_absolute_error(a, b), 1/3)

    assert_almost_equal(mean_percentage_signed_error(a, b), 0)
    assert_almost_equal(mean_percentage_absolute_error(a, b), 100/3)

    assert_almost_equal(mean_square_error(a, b), 2/3)
    assert_almost_equal(mean_square_relative_error(a, b), 0.5/3)

    assert_almost_equal(root_mean_square_error(a, b), np.sqrt(2/3))
    assert_almost_equal(root_mean_square_relative_error(a, b), np.sqrt(0.5/3))

    assert_almost_equal(median_signed_error(a, b), 0)
    assert_almost_equal(median_absolute_error(a, b), 1)

    assert_almost_equal(median_relative_signed_error(a, b), 0)
    assert_almost_equal(median_relative_absolute_error(a, b), 0.5)

    assert_almost_equal(median_percentage_signed_error(a, b), 0)
    assert_almost_equal(median_percentage_absolute_error(a, b), 50)

    assert_almost_equal(median_square_error(a, b), 1)
    assert_almost_equal(median_square_relative_error(a, b), 0.25)

    assert_almost_equal(sum_square_error(a, b), 2)
    assert_almost_equal(sum_square_relative_error(a, b), 0.5)


def test_error_two_ndarray():
    a = np.array([1, 2, 3])
    b = np.array([2, 2, 2])
    assert_array_equal(signed_error(a, b), np.array([-1, 0, 1]))
    assert_array_equal(absolute_error(a, b), np.array([1, 0, 1]))

    assert_array_equal(relative_signed_error(a, b), np.array([-0.5, 0, 0.5]))
    assert_array_equal(relative_absolute_error(a, b), np.array([0.5, 0, 0.5]))

    assert_array_equal(percentage_signed_error(a, b), np.array([-50, 0, 50]))
    assert_array_equal(percentage_absolute_error(a, b), np.array([50, 0, 50]))

    assert_almost_equal(square_error(a, b), np.array([1, 0, 1]))
    assert_almost_equal(square_relative_error(a, b), np.array([0.25, 0, 0.25]))

    assert_almost_equal(mean_signed_error(a, b), 0)
    assert_almost_equal(mean_absolute_error(a, b), 2/3)

    assert_almost_equal(mean_relative_signed_error(a, b), 0)
    assert_almost_equal(mean_relative_absolute_error(a, b), 1/3)

    assert_almost_equal(mean_percentage_signed_error(a, b), 0)
    assert_almost_equal(mean_percentage_absolute_error(a, b), 100/3)

    assert_almost_equal(mean_square_error(a, b), 2/3)
    assert_almost_equal(mean_square_relative_error(a, b), 0.5/3)

    assert_almost_equal(root_mean_square_error(a, b), np.sqrt(2/3))
    assert_almost_equal(root_mean_square_relative_error(a, b), np.sqrt(0.5/3))

    assert_almost_equal(median_signed_error(a, b), 0)
    assert_almost_equal(median_absolute_error(a, b), 1)

    assert_almost_equal(median_relative_signed_error(a, b), 0)
    assert_almost_equal(median_relative_absolute_error(a, b), 0.5)

    assert_almost_equal(median_percentage_signed_error(a, b), 0)
    assert_almost_equal(median_percentage_absolute_error(a, b), 50)

    assert_almost_equal(median_square_error(a, b), 1)
    assert_almost_equal(median_square_relative_error(a, b), 0.25)

    assert_almost_equal(sum_square_error(a, b), 2)
    assert_almost_equal(sum_square_relative_error(a, b), 0.5)


def test_error_one_ndarray_float():
    a = np.array([1, 2, 3])
    b = 2.3
    assert_array_almost_equal(signed_error(a, b),
                              np.array([-1.3, -0.3, 0.7]))
    assert_array_almost_equal(absolute_error(a, b),
                              np.array([1.3, 0.3, 0.7]))

    assert_array_almost_equal(relative_signed_error(a, b),
                              np.array([-1.3/2.3, -0.3/2.3, 0.7/2.3]))
    assert_array_almost_equal(relative_absolute_error(a, b),
                              np.array([1.3/2.3, 0.3/2.3, 0.7/2.3]))

    assert_array_almost_equal(percentage_signed_error(a, b),
                              np.array([-130/2.3, -30/2.3, 70/2.3]))
    assert_array_almost_equal(percentage_absolute_error(a, b),
                              np.array([130/2.3, 30/2.3, 70/2.3]))

    assert_array_almost_equal(square_error(a, b),
                              np.array([1.69, 0.09, 0.49]))
    assert_array_almost_equal(square_relative_error(a, b),
                              np.array([1.69, 0.09, 0.49])/5.29)

    assert_almost_equal(mean_signed_error(a, b), -0.3)
    assert_almost_equal(mean_absolute_error(a, b), 2.3/3)

    assert_almost_equal(mean_relative_signed_error(a, b), -0.3/2.3)
    assert_almost_equal(mean_relative_absolute_error(a, b), 1/3)

    assert_almost_equal(mean_percentage_signed_error(a, b), -30/2.3)
    assert_almost_equal(mean_percentage_absolute_error(a, b), 100/3)

    assert_almost_equal(mean_square_error(a, b), 2.27/3)
    assert_almost_equal(mean_square_relative_error(a, b), 2.27/5.29/3)

    assert_almost_equal(root_mean_square_error(a, b), np.sqrt(2.27/3))
    assert_almost_equal(root_mean_square_relative_error(a, b),
                        np.sqrt(2.27/5.29/3))

    assert_almost_equal(median_signed_error(a, b), -0.3)
    assert_almost_equal(median_absolute_error(a, b), 0.7)

    assert_almost_equal(median_relative_signed_error(a, b), -0.3/2.3)
    assert_almost_equal(median_relative_absolute_error(a, b), 0.7/2.3)

    assert_almost_equal(median_percentage_signed_error(a, b), -30/2.3)
    assert_almost_equal(median_percentage_absolute_error(a, b), 70/2.3)

    assert_almost_equal(median_square_error(a, b), 0.49)
    assert_almost_equal(median_square_relative_error(a, b), 0.49/5.29)

    assert_almost_equal(sum_square_error(a, b), 2.27)
    assert_almost_equal(sum_square_relative_error(a, b), 2.27/5.29)


def test_rms():
    a = 5
    assert root_mean_square(a) == 5

    a = np.array([1, 1, 3])
    assert_almost_equal(root_mean_square(a), np.sqrt(11/3))

    a = np.array([1, -1, -3])
    assert_almost_equal(root_mean_square(a), np.sqrt(11/3))

    a = np.array([1, 2, -3])
    assert_almost_equal(root_mean_square(a), np.sqrt(14/3))


def test_sum_square():
    a = 5
    assert sum_square(a) == 25

    a = np.array([1, 1, 3])
    assert_almost_equal(sum_square(a), 11)

    a = np.array([1, -1, -3])
    assert_almost_equal(sum_square(a), 11)

    a = np.array([1, 2, -3])
    assert_almost_equal(sum_square(a), 14)


def test_root_sum_square():
    a = 5
    assert root_sum_square(a) == 5

    a = np.array([1, 1, 3])
    assert_almost_equal(root_sum_square(a), np.sqrt(11))

    a = np.array([1, -1, -3])
    assert_almost_equal(root_sum_square(a), np.sqrt(11))

    a = np.array([1, 2, -3])
    assert_almost_equal(root_sum_square(a), np.sqrt(14))
