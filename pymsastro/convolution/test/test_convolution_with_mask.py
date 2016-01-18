# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np
from numpy.testing import (assert_almost_equal, assert_array_almost_equal)

from astropy.convolution import convolve as astropy_convolve
from scipy.ndimage.filters import convolve as scipy_convolve
from numpy import convolve as numpy_convolve

from .. import convolve_with_mask

from pytest import raises


def test_exception_explicit_and_implicit_mask():
    array = np.ma.array([1, 2, 3], mask=False)
    mask = np.ones(3, dtype=np.bool_)
    kernel = np.ones(1)
    with raises(TypeError):
        convolve_with_mask(array, kernel, mask)


def test_exception_mask_shape_not_equal_to_array_1():
    array = np.array([1, 2, 3])
    mask = np.ones(2, dtype=np.bool_)
    kernel = np.ones(1)
    with raises(ValueError):
        convolve_with_mask(array, kernel, mask)


def test_exception_mask_shape_not_equal_to_array_2():
    array = np.array([1, 2, 3, 4])
    mask = np.ones(3, dtype=np.bool_)
    kernel = np.ones(1)
    with raises(ValueError):
        convolve_with_mask(array, kernel, mask)


def test_exception_mask_ndim_not_equal_to_array_1():
    array = np.array([1, 2, 3])
    mask = np.ones((3, 1), dtype=np.bool_)
    kernel = np.ones(1)
    with raises(ValueError):
        convolve_with_mask(array, kernel, mask)


def test_exception_mask_ndim_not_equal_to_array_2():
    array = np.array([1, 2, 3]).reshape(3, 1)
    mask = np.ones(3, dtype=np.bool_)
    kernel = np.ones(1)
    with raises(ValueError):
        convolve_with_mask(array, kernel, mask)


def test_exception_array_ndim_wrong_1():
    array = np.array(1)
    kernel = np.array(1)
    with raises(ValueError):
        convolve_with_mask(array, kernel)


def test_exception_array_ndim_wrong_2():
    array = np.ones((16)).reshape(2, 2, 2, 2)
    kernel = np.ones(1).reshape(1, 1, 1, 1)
    with raises(ValueError):
        convolve_with_mask(array, kernel)


def test_exception_kernel_even_shape():
    array = np.ones(8)
    kernel = np.ones(2)
    with raises(ValueError):
        convolve_with_mask(array, kernel)


def test_no_mask_equal_to_mask_containing_zeros_1():
    array = np.random.normal(5, 0.5, 8)
    kernel = np.ones(3)
    mask = np.zeros(array.shape, dtype=np.bool_)
    conv1 = convolve_with_mask(array, kernel)
    conv2 = convolve_with_mask(array, kernel, mask)
    assert_array_almost_equal(conv1, conv2)


def test_no_mask_equal_to_mask_containing_zeros_2():
    array = np.random.normal(5, 0.5, 8).reshape(2, 2, 2)
    kernel = np.ones((1, 1, 1))
    mask = np.zeros(array.shape, dtype=np.bool_)
    conv1 = convolve_with_mask(array, kernel)
    conv2 = convolve_with_mask(array, kernel, mask)
    assert_array_almost_equal(conv1, conv2)


def test_convolve_identical_to_scipy():
    array = np.random.normal(5, 0.5, 100).reshape(10, 10)
    kernel = np.ones((3, 3))
    conv1 = convolve_with_mask(array, kernel)
    conv2 = scipy_convolve(array, kernel)
    # Because of different boundary handling the first and last element in each
    # dimension must be ignored.
    assert_array_almost_equal(conv1[1:-1, 1:-1], conv2[1:-1, 1:-1])


def test_convolve_identical_to_numpy():
    array = np.random.normal(5, 0.5, 100)
    kernel = np.ones(3)
    conv1 = convolve_with_mask(array, kernel)
    conv2 = numpy_convolve(array, kernel, mode='same')
    # Because of different boundary handling the first and last element in each
    # dimension must be ignored.
    assert_array_almost_equal(conv1[1:-1], conv2[1:-1])


def test_convolve_identical_to_astropy():
    array = np.random.normal(5, 0.5, 1000).reshape(10, 10, 10)
    kernel = np.ones((3, 3, 3))
    conv1 = convolve_with_mask(array, kernel)
    conv2 = astropy_convolve(array, kernel)
    # Because of different boundary handling the first and last element in each
    # dimension must be ignored.
    assert_array_almost_equal(conv1[1:-1, 1:-1, 1:-1], conv2[1:-1, 1:-1, 1:-1])


def test_convolve_rescale():
    array = np.random.normal(5, 0.5, 100).reshape(10, 10)
    kernel = np.ones((3, 3))
    conv1 = convolve_with_mask(array, kernel)
    conv2 = convolve_with_mask(array, kernel, rescale_kernel=False)
    assert_array_almost_equal(conv1, conv2*9)


def test_convolve_test_boundary_1():
    array = np.random.normal(5, 0.5, 10)
    kernel = np.ones(3)
    conv1 = convolve_with_mask(array, kernel)
    assert_almost_equal(conv1[0], (array[0]+array[1])/2*3)
    assert_almost_equal(conv1[-1], (array[-1]+array[-2])/2*3)


def test_convolve_test_boundary_2():
    array = np.random.normal(5, 0.5, 10)
    kernel = np.ones(5)
    conv1 = convolve_with_mask(array, kernel)
    assert_almost_equal(conv1[0], (array[0]+array[1]+array[2])/3*5)
    assert_almost_equal(conv1[-1], (array[-1]+array[-2]+array[-3])/3*5)
    assert_almost_equal(conv1[1], (array[0]+array[1]+array[2]+array[3])/4*5)
    assert_almost_equal(conv1[-2],
                        (array[-1]+array[-2]+array[-3]+array[-4])/4*5)


def test_convolve_test_masked_1():
    array = np.random.normal(5, 0.5, 10)
    kernel = np.ones(3)
    mask = np.zeros(array.shape)
    mask[4] = 1
    conv1 = convolve_with_mask(array, kernel, mask)
    assert_almost_equal(conv1[4], (array[3]+array[5])/2*3)


def test_convolve_test_masked_2():
    array = np.random.normal(5, 0.5, 10)
    kernel = np.ones(3)
    mask = np.zeros(array.shape)
    mask[4] = 1
    mask[5] = 1
    conv1 = convolve_with_mask(array, kernel, mask)
    assert_almost_equal(conv1[3], (array[3]+array[2])/2*3)
    assert_almost_equal(conv1[4], array[3]*3)
    assert_almost_equal(conv1[5], array[6]*3)
    assert_almost_equal(conv1[6], (array[6]+array[7])/2*3)


def test_convolve_test_masked_3():
    array = np.random.normal(5, 0.5, 10)
    kernel = np.ones(3)
    mask = np.zeros(array.shape)
    mask[4] = 1
    mask[5] = 1
    mask[6] = 1
    conv1 = convolve_with_mask(array, kernel, mask)
    assert_almost_equal(conv1[3], (array[3]+array[2])/2*3)
    assert_almost_equal(conv1[4], array[3]*3)
    assert_almost_equal(conv1[5], np.nan)
    assert_almost_equal(conv1[6], array[7]*3)
    assert_almost_equal(conv1[7], (array[7]+array[8])/2*3)


def test_convolve_test_masked_4():
    array = np.random.normal(5, 0.5, 10*3).reshape(10, 3)
    kernel = np.ones((3, 3))
    mask = np.zeros(array.shape)
    mask[4, 1] = 1
    conv1 = convolve_with_mask(array, kernel, mask)
    assert_almost_equal(conv1[4, 1], ((array[3, 1] + array[5, 1] +
                                      array[3, 0] + array[4, 0] +
                                      array[5, 0] +
                                      array[3, 2] + array[4, 2] +
                                      array[5, 2]) / 8*9))
