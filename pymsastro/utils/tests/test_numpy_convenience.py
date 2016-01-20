# Licensed under a 3-clause BSD style license - see LICENSE.rst

from ..numpy_convenience import (numpyAxisStringToNumber, numpyIsBroadcastable,
                                 numpyOffset, numpySliceWithReference,
                                 numpyOffsetWithReference, numpyIsDtype)

from pytest import raises
import numpy as np


def test_nc_axis():
    # String input is converted to number
    assert numpyAxisStringToNumber('h') == 1
    assert numpyAxisStringToNumber('v') == 0
    # int input is simply returned
    assert numpyAxisStringToNumber(1) == 1
    assert numpyAxisStringToNumber(0) == 0
    # unknown string input
    with raises(ValueError):
        numpyAxisStringToNumber('d')
    # unknown type input (not string, not int)
    with raises(TypeError):
        numpyAxisStringToNumber([1, 2])
    with raises(TypeError):
        numpyAxisStringToNumber(1.5)


def test_nc_isdtype():
    a = np.ones((3, 3), dtype=np.float64)
    assert numpyIsDtype(a, 'numerical')
    assert not numpyIsDtype(a, 'boolean')
    with raises(TypeError):
        numpyIsDtype(a, 'numeric')

    a = np.ones((3, 3), dtype=np.uint64)
    assert numpyIsDtype(a, 'numerical')
    assert not numpyIsDtype(a, 'boolean')
    with raises(TypeError):
        numpyIsDtype(a, 'numeric')

    a = np.ones((3, 3), dtype=np.bool_)
    assert not numpyIsDtype(a, 'numerical')
    assert numpyIsDtype(a, 'boolean')
    with raises(TypeError):
        numpyIsDtype(a, 'numeric')


def test_nc_broadcastable_not():
    # Completly different shape
    b = np.ones(3)
    a = np.ones(2)
    with raises(AssertionError):
        assert not numpyIsBroadcastable(a.shape, b.shape)


def test_nc_broadcastable_not_2():
    # One dimension not broadcastable
    b = np.ones((3, 3))
    a = np.ones((3, 2))
    with raises(AssertionError):
        assert not numpyIsBroadcastable(a.shape, b.shape)


def test_nc_broadcastable_not_3():
    # One dimension missing
    b = np.ones((3, 3))
    a = np.ones(3)
    with raises(AssertionError):
        assert not numpyIsBroadcastable(a.shape, b.shape)


def test_nc_broadcastable_not_4():
    # One dimension not broadcastable one only broadcastable
    b = np.ones((3, 3))
    a = np.ones((2, 1))
    with raises(AssertionError):
        assert not numpyIsBroadcastable(a.shape, b.shape)


def test_nc_broadcastable_not_5():
    # One dimension broadcastable but reference dimensions is 1
    b = np.ones((3, 1))
    a = np.ones((3, 2))
    with raises(AssertionError):
        assert not numpyIsBroadcastable(a.shape, b.shape)


def test_nc_broadcastable_1():
    # One dimension and is the same
    b = np.ones(3)
    a = np.ones(3)
    assert len(numpyIsBroadcastable(a.shape, b.shape)) == 0


def test_nc_broadcastable_2():
    # One dimension and is broadcastable
    b = np.ones(3)
    a = np.ones(1)
    assert len(numpyIsBroadcastable(a.shape, b.shape)) == 1
    assert numpyIsBroadcastable(a.shape, b.shape)[0] == 0


def test_nc_broadcastable_3():
    # Multiple dimensions and the same
    b = np.ones((3, 3))
    a = np.ones((3, 3))
    assert len(numpyIsBroadcastable(a.shape, b.shape)) == 0


def test_nc_broadcastable_4():
    # Multiple dimensions and broadcastable
    b = np.ones((3, 3))
    a = np.ones((3, 1))
    assert len(numpyIsBroadcastable(a.shape, b.shape)) == 1
    assert numpyIsBroadcastable(a.shape, b.shape)[0] == 1


def test_nc_broadcastable_5():
    # Multiple dimensions and broadcastable
    b = np.ones((3, 3))
    a = np.ones((1, 3))
    assert len(numpyIsBroadcastable(a.shape, b.shape)) == 1
    assert numpyIsBroadcastable(a.shape, b.shape)[0] == 0


def test_nc_broadcastable_6():
    # Multiple dimensions and broadcastable
    b = np.ones((3, 3))
    a = np.ones((1, 1))
    assert len(numpyIsBroadcastable(a.shape, b.shape)) == 2
    assert numpyIsBroadcastable(a.shape, b.shape)[0] == 0
    assert numpyIsBroadcastable(a.shape, b.shape)[1] == 1


def test_nc_slice_ref_1d():
    # Same shape - 1D
    array = np.ones(3)
    ref = np.ones(3)
    assert numpySliceWithReference(array, slice(1, 3), ref.shape).shape == (2,)

    # Broadcastable shape - 1D
    array = np.ones(1)
    ref = np.ones(3)
    assert numpySliceWithReference(array, slice(1, 3), ref.shape).shape == (1,)

    # Not Broadcastable shape - 1D
    array = np.ones(2)
    ref = np.ones(3)
    with raises(AssertionError):
        numpySliceWithReference(array, slice(1, 3), ref.shape)


def test_nc_slice_ref_2d_1dslice():
    # Same shape - 2D
    array = np.ones((3, 3))
    ref = np.ones((3, 3))
    assert numpySliceWithReference(array, slice(1, 3),
                                   ref.shape).shape == (2, 3)

    # Broadcastable shape - 2D
    array = np.ones((3, 1))
    ref = np.ones((3, 3))
    assert numpySliceWithReference(array, slice(1, 3),
                                   ref.shape).shape == (2, 1)

    # Broadcastable shape - 2D
    array = np.ones((1, 3))
    ref = np.ones((3, 3))
    assert numpySliceWithReference(array, slice(1, 3),
                                   ref.shape).shape == (1, 3)

    # Broadcastable shape - 2D
    array = np.ones((1, 1))
    ref = np.ones((3, 3))
    assert numpySliceWithReference(array, slice(1, 3),
                                   ref.shape).shape == (1, 1)

    # Not Broadcastable shape - 2D
    array = np.ones((1, 2))
    ref = np.ones((3, 3))
    with raises(AssertionError):
        numpySliceWithReference(array, slice(1, 3), ref.shape)


def test_nc_slice_ref_2d_2dslice():
    # Same shape - 2D
    array = np.ones((3, 3))
    ref = np.ones((3, 3))
    assert numpySliceWithReference(array, (slice(1, 3), slice(1, 3)),
                                   ref.shape).shape == (2, 2)

    # Broadcastable shape - 2D
    array = np.ones((3, 1))
    ref = np.ones((3, 3))
    assert numpySliceWithReference(array, (slice(1, 3), slice(1, 3)),
                                   ref.shape).shape == (2, 1)

    # Broadcastable shape - 2D
    array = np.ones((1, 3))
    ref = np.ones((3, 3))
    assert numpySliceWithReference(array, (slice(1, 3), slice(1, 3)),
                                   ref.shape).shape == (1, 2)

    # Broadcastable shape - 2D
    array = np.ones((1, 1))
    ref = np.ones((3, 3))
    assert numpySliceWithReference(array, (slice(1, 3), slice(1, 3)),
                                   ref.shape).shape == (1, 1)

    # Not Broadcastable shape - 2D
    array = np.ones((1, 2))
    ref = np.ones((3, 3))
    with raises(AssertionError):
        numpySliceWithReference(array, (slice(1, 3), slice(1, 3)), ref.shape)


def test_nc_slice_ref_3d_1dslice():
    # Same shape - 3D
    array = np.ones((3, 3, 3))
    ref = np.ones((3, 3, 3))
    assert numpySliceWithReference(array, slice(1, 3),
                                   ref.shape).shape == (2, 3, 3)

    # Broadcastable shape - 3D
    array = np.ones((3, 1, 3))
    assert numpySliceWithReference(array, slice(1, 3),
                                   ref.shape).shape == (2, 1, 3)

    # Broadcastable shape - 3D
    array = np.ones((1, 3, 3))
    assert numpySliceWithReference(array, slice(1, 3),
                                   ref.shape).shape == (1, 3, 3)

    # Broadcastable shape - 3D
    array = np.ones((3, 3, 1))
    assert numpySliceWithReference(array, slice(1, 3),
                                   ref.shape).shape == (2, 3, 1)

    # Broadcastable shape - 3D
    array = np.ones((1, 1, 3))
    assert numpySliceWithReference(array, slice(1, 3),
                                   ref.shape).shape == (1, 1, 3)

    # Broadcastable shape - 3D
    array = np.ones((3, 1, 1))
    assert numpySliceWithReference(array, slice(1, 3),
                                   ref.shape).shape == (2, 1, 1)

    # Broadcastable shape - 3D
    array = np.ones((1, 3, 1))
    assert numpySliceWithReference(array, slice(1, 3),
                                   ref.shape).shape == (1, 3, 1)

    # Broadcastable shape - 3D
    array = np.ones((1, 1, 1))
    assert numpySliceWithReference(array, slice(1, 3),
                                   ref.shape).shape == (1, 1, 1)

    # Not Broadcastable shape - 3D
    array = np.ones((1, 2, 3))
    with raises(AssertionError):
        numpySliceWithReference(array, slice(1, 3), ref.shape)


def test_nc_slice_ref_3d_2dslice():
    slicing = (slice(1, 3), slice(1, 3))
    # Same shape - 3D
    array = np.ones((3, 3, 3))
    ref = np.ones((3, 3, 3))
    assert numpySliceWithReference(array, slicing,
                                   ref.shape).shape == (2, 2, 3)

    # Broadcastable shape - 3D
    array = np.ones((3, 1, 3))
    assert numpySliceWithReference(array, slicing,
                                   ref.shape).shape == (2, 1, 3)

    # Broadcastable shape - 3D
    array = np.ones((1, 3, 3))
    assert numpySliceWithReference(array, slicing,
                                   ref.shape).shape == (1, 2, 3)

    # Broadcastable shape - 3D
    array = np.ones((3, 3, 1))
    assert numpySliceWithReference(array, slicing,
                                   ref.shape).shape == (2, 2, 1)

    # Broadcastable shape - 3D
    array = np.ones((1, 1, 3))
    assert numpySliceWithReference(array, slicing,
                                   ref.shape).shape == (1, 1, 3)

    # Broadcastable shape - 3D
    array = np.ones((3, 1, 1))
    assert numpySliceWithReference(array, slicing,
                                   ref.shape).shape == (2, 1, 1)

    # Broadcastable shape - 3D
    array = np.ones((1, 3, 1))
    assert numpySliceWithReference(array, slicing,
                                   ref.shape).shape == (1, 2, 1)

    # Broadcastable shape - 3D
    array = np.ones((1, 1, 1))
    assert numpySliceWithReference(array, slicing,
                                   ref.shape).shape == (1, 1, 1)

    # Not Broadcastable shape - 3D
    array = np.ones((1, 2, 3))
    with raises(AssertionError):
        numpySliceWithReference(array, slicing, ref.shape)


def test_nc_slice_ref_3d_3dslice():
    slicing = (slice(1, 3), slice(1, 3), slice(0, 2))
    # Same shape - 3D
    array = np.ones((3, 3, 3))
    ref = np.ones((3, 3, 3))
    assert numpySliceWithReference(array, slicing,
                                   ref.shape).shape == (2, 2, 2)

    # Broadcastable shape - 3D
    array = np.ones((3, 1, 3))
    assert numpySliceWithReference(array, slicing,
                                   ref.shape).shape == (2, 1, 2)

    # Broadcastable shape - 3D
    array = np.ones((1, 3, 3))
    assert numpySliceWithReference(array, slicing,
                                   ref.shape).shape == (1, 2, 2)

    # Broadcastable shape - 3D
    array = np.ones((3, 3, 1))
    assert numpySliceWithReference(array, slicing,
                                   ref.shape).shape == (2, 2, 1)

    # Broadcastable shape - 3D
    array = np.ones((1, 1, 3))
    assert numpySliceWithReference(array, slicing,
                                   ref.shape).shape == (1, 1, 2)

    # Broadcastable shape - 3D
    array = np.ones((3, 1, 1))
    assert numpySliceWithReference(array, slicing,
                                   ref.shape).shape == (2, 1, 1)

    # Broadcastable shape - 3D
    array = np.ones((1, 3, 1))
    assert numpySliceWithReference(array, slicing,
                                   ref.shape).shape == (1, 2, 1)

    # Broadcastable shape - 3D
    array = np.ones((1, 1, 1))
    assert numpySliceWithReference(array, slicing,
                                   ref.shape).shape == (1, 1, 1)

    # Not Broadcastable shape - 3D
    array = np.ones((1, 2, 3))
    with raises(AssertionError):
        numpySliceWithReference(array, slicing, ref.shape)


def test_nc_offset_failures():
    # More offset dimensions as image dimensions - 1D
    array = np.ones(5)
    offset = (1, 3)
    newshape = (6, 3)
    with raises(ValueError):
        numpyOffset(array, offset, newshape)

    # More offset dimensions as image dimensions - 2D
    array = np.ones((3, 3))
    offset = (1, 3, 2)
    newshape = (4, 5, 2)
    with raises(ValueError):
        numpyOffset(array, offset, newshape)

    # More offset dimensions as image dimensions - 3D
    array = np.ones((3, 3, 3))
    offset = (1, 3, 2, 4)
    newshape = (4, 5, 5, 4)
    with raises(ValueError):
        numpyOffset(array, offset, newshape)

    # Less offset dimensions as image dimensions - 2D
    array = np.ones((3, 3))
    offset = (1,)
    newshape = (4,)
    with raises(ValueError):
        numpyOffset(array, offset, newshape)

    # Less offset dimensions as image dimensions - 3D
    array = np.ones((3, 3, 3))
    offset = (1, 3)
    newshape = (4, 6)
    with raises(ValueError):
        numpyOffset(array, offset, newshape)

    # More newshape dimensions as image dimensions - 1D
    array = np.ones(5)
    offset = (1, )
    newshape = (6, 3)
    with raises(ValueError):
        numpyOffset(array, offset, newshape)

    # More newshape dimensions as image dimensions - 2D
    array = np.ones((3, 3))
    offset = (1, 3)
    newshape = (4, 6, 2)
    with raises(ValueError):
        numpyOffset(array, offset, newshape)

    # More newshape dimensions as image dimensions - 3D
    array = np.ones((3, 3, 3))
    offset = (1, 3, 2)
    newshape = (4, 6, 5, 4)
    with raises(ValueError):
        numpyOffset(array, offset, newshape)

    # Less newshape dimensions as image dimensions - 2D
    array = np.ones((3, 3))
    offset = (1, 2)
    newshape = (4, )
    with raises(ValueError):
        numpyOffset(array, offset, newshape)

    # Less newshape dimensions as image dimensions - 3D
    array = np.ones((3, 3, 3))
    offset = (1, 3, 2)
    newshape = (4, 6)
    with raises(ValueError):
        numpyOffset(array, offset, newshape)

    # Negative value inside offset
    array = np.ones((3, 3, 3))
    offset = (1, -3, 2)
    newshape = (4, 6, 5)
    with raises(ValueError):
        numpyOffset(array, offset, newshape)

    # Negative value inside newshape
    array = np.ones((3, 3, 3))
    offset = (1, 3, 2)
    newshape = (4, 6, -2)
    with raises(ValueError):
        numpyOffset(array, offset, newshape)

    # float value inside offset
    array = np.ones((3, 3, 3))
    offset = (1, 2.9, 2)
    newshape = (4, 6, 5)
    with raises(ValueError):
        numpyOffset(array, offset, newshape)

    # float value inside newshape
    array = np.ones((3, 3, 3))
    offset = (1, 3, 2)
    newshape = (4, 6, 5.1)
    with raises(ValueError):
        numpyOffset(array, offset, newshape)

    # newshape too small
    array = np.ones((3, 3, 3))
    offset = (1, 3, 2)
    newshape = (4, 5, 5)
    with raises(ValueError):
        numpyOffset(array, offset, newshape)

    # newshape differs from fill.shape
    array = np.ones(4)
    offset = (1,)
    newshape = (6,)
    fill = np.arange(10).astype(np.float_)
    with raises(ValueError):
        numpyOffset(array, offset, newshape, fill)


def test_nc_offset_1d():
    array = np.ones(3)
    offset = (1,)
    newshape = (4,)
    offsetted_array = numpyOffset(array, offset, newshape)
    assert offsetted_array.shape == (4, )
    reference = np.array([0, 1, 1, 1])
    np.testing.assert_array_equal(offsetted_array, reference)

    array = np.ones(3)
    offset = (2,)
    newshape = (6,)
    offsetted_array = numpyOffset(array, offset, newshape)
    assert offsetted_array.shape == (6, )
    reference = np.array([0, 0, 1, 1, 1, 0])
    np.testing.assert_array_equal(offsetted_array, reference)

    array = np.ones(3)
    offset = (1,)
    newshape = (6,)
    offsetted_array = numpyOffset(array, offset, newshape, fill=2)
    assert offsetted_array.shape == (6, )
    reference = np.array([2, 1, 1, 1, 2, 2])
    np.testing.assert_array_equal(offsetted_array, reference)

    array = np.ones(3)
    offset = (1,)
    newshape = (6,)
    fill = np.arange(6).astype(np.float_)
    offsetted_array = numpyOffset(array, offset, fill.shape, fill=fill)
    assert offsetted_array.shape == (6, )
    reference = np.array([0, 1, 1, 1, 4, 5])
    np.testing.assert_array_equal(offsetted_array, reference)


def test_nc_offset_2d():
    array = np.ones((3, 3))
    offset = (1, 2)
    newshape = (4, 5)
    offsetted_array = numpyOffset(array, offset, newshape)
    assert offsetted_array.shape == (4, 5)
    reference = np.array([[0, 0, 0, 0, 0],
                          [0, 0, 1, 1, 1],
                          [0, 0, 1, 1, 1],
                          [0, 0, 1, 1, 1]])
    np.testing.assert_array_equal(offsetted_array, reference)

    array = np.ones((3, 3))
    offset = (1, 2)
    newshape = (4, 6)
    offsetted_array = numpyOffset(array, offset, newshape)
    assert offsetted_array.shape == (4, 6)
    reference = np.array([[0, 0, 0, 0, 0, 0],
                          [0, 0, 1, 1, 1, 0],
                          [0, 0, 1, 1, 1, 0],
                          [0, 0, 1, 1, 1, 0]])
    np.testing.assert_array_equal(offsetted_array, reference)

    array = np.ones((3, 3))
    offset = (1, 2)
    newshape = (4, 5)
    offsetted_array = numpyOffset(array, offset, newshape, fill=2)
    assert offsetted_array.shape == (4, 5)
    reference = np.array([[2, 2, 2, 2, 2],
                          [2, 2, 1, 1, 1],
                          [2, 2, 1, 1, 1],
                          [2, 2, 1, 1, 1]])
    np.testing.assert_array_equal(offsetted_array, reference)


def test_nc_offset_reference_failures():
    # Not broadcastable
    array = np.ones((3, 2))
    refshape = (3, 3)
    offset = (1, 2)
    newshape = (4, 5)
    with raises(AssertionError):
        numpyOffsetWithReference(array, offset, newshape, refshape)

    # Fill not same shape as newshape
    array = np.ones(3)
    refshape = (3,)
    offset = (1,)
    newshape = (5,)
    fill = np.arange(6).astype(np.float_)
    with raises(ValueError):
        numpyOffsetWithReference(array, offset, newshape, refshape, fill=fill)


def test_nc_offset_reference():
    array = np.ones((3, 1))
    refshape = (3, 3)
    offset = (1, 2)
    newshape = (4, 5)
    offsetted_array = numpyOffsetWithReference(array, offset, newshape,
                                               refshape)
    assert offsetted_array.shape == (4, 1)
    reference = np.array([[0], [1], [1], [1]])
    np.testing.assert_array_equal(offsetted_array, reference)

    array = np.ones((1, 3))
    refshape = (3, 3)
    offset = (1, 2)
    newshape = (4, 5)
    offsetted_array = numpyOffsetWithReference(array, offset, newshape,
                                               refshape)
    assert offsetted_array.shape == (1, 5)
    reference = np.array([[0, 0, 1, 1, 1]])
    np.testing.assert_array_equal(offsetted_array, reference)
