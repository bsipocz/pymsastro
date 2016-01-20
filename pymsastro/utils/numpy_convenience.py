# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np

__all__ = ['numpyAxisStringToNumber', 'numpyIsBroadcastable', 'numpyOffset',
           'numpySliceWithReference', 'numpyOffsetWithReference',
           'numpyIsDtype']


def numpyAxisStringToNumber(axis):
    """
    Converts an axis string to the corresponding axis number for `numpy`.

    Parameters
    ----------
    axis : `str`, `int`
        The wanted axis as `str`. If it is an `int` the value is simply
        returned, assuming that the function call was unnecessary.

    Returns
    -------
    axis : `int`
        The axis as number

    Notes
    -----
    Abbreviations such as ``"h"`` for horizontal and ``"v"`` for vertical
    are supported.

    The complete list of possible values:

        - ``"h"``, ``"horiz"``, ``"horizontal"``, ``"bottomtop"`` (axis=1)
        - ``"v"``, ``"vert"``, ``"vertical"``, ``"leftright"`` (axis=0)
    """
    if isinstance(axis, int):
        return axis
    elif isinstance(axis, str):
        if axis in ('h', 'horiz', 'horizontal', 'bottomtop'):
            return 1
        elif axis in ('v', 'vert', 'vertical', 'leftright'):
            return 0
        else:
            raise ValueError('Unknown string "{0}" as axis '
                             'argument.'.format(axis))
    else:
        raise TypeError('Unknown type "{0}" as axis argument. Use ints or '
                        'strings.'.format(axis.__class__.__name__))


def numpyIsDtype(array, dtype):
    """
    This function takes a ``dtype`` and checks if the ``array`` is of this
    kind.

    Parameters
    ----------
    array : `numpy.ndarray`
        The array we want to investigate.

    dtype : `str`
        The dtype we want to check for. Can be:

        - ``"numerical"`` : `float`, `int`, etc.
        - ``"boolean"`` : `bool`

    Returns
    -------
    is_type : `bool`
        ``True`` if the array has this dtype, ``False`` if not.

    Notes
    -----
    An important difference to check the `numpy.ndarray.dtype` directly is
    that there are several different options for the different kinds. This
    function compares the ``numpy.ndarray.dtype.kind`` with known values.
    """
    if dtype == 'numerical':
        return array.dtype.kind in ['i', 'f', 'u']
    elif dtype == 'boolean':
        return array.dtype.kind in ['b']
    else:
        raise TypeError('Unknown dtype.')


def numpyIsBroadcastable(shape, reference_shape):
    """
    Checks if two `numpy.ndarray` shapes are broadcastable (only if they have
    the same number of dimensions).

    Parameters
    ----------
    shape, reference_shape : `tuple`
        The shapes of the two `numpy.ndarray` as are returned by
        `numpy.ndarray.shape`.

    Returns
    -------
    broadcast_dims : `list`
        A list of all dimensions that are not the same shape as the reference
        but are broadcastable.

    Raises
    ------
    AssertionError
        If the arrays are not broadcastable.
    """
    if len(shape) != len(reference_shape):
        raise AssertionError('Shapes do not have the same number of '
                             'dimensions')

    if shape == reference_shape:
        # identical shapes
        return []

    broadcast_dims = []
    for dim in range(len(shape)):
        if shape[dim] == reference_shape[dim]:
            # Identical to this axis
            pass
        elif shape[dim] == 1:
            # At least Broadcastable to axis
            broadcast_dims.append(dim)
        else:
            raise AssertionError('Dimension {0} is not broadcastable from {1}'
                                 ' to {2}'.format(dim, shape[dim],
                                                  reference_shape[dim]))

    return broadcast_dims


def numpySliceWithReference(array, item, refshape):
    '''
    Slice something but given a reference shape so broadcastable dimensions
    are not sliced.

    Sometimes we want to slice something that
    has the same number of dimensions but
    it is only broadcastable to the reference
    array but has not the same shape. This function
    tests if the ``array`` can or should be sliced
    with ``item`` given a ``refshape``.

    Parameters
    ----------
    array : `numpy.ndarray`
        The thing that we want to slice. Assumes that it has the same number of
        dimensions as the reference has.

    item : `slice` or `list`/`tuple` of `slice`
        The passed slicing item. As interpreted by ``__getitem__(item)``.

    refshape : `tuple`
        The shapes of the reference `numpy.ndarray` as is returned by
        `numpy.ndarray.shape`.

    Returns
    -------
    sliceded_array : `numpy.ndarray`
        The sliced array with regard to the reference shape.

    Raises
    ------
    AssertionError
        see :func:`numpyIsBroadcastable`.

    Notes
    -----
    Dimensions that are only broadcastable are not sliced. All the other
    dimensions are sliced like the reference would be.

    Does not allow for complex slicing.

    This method is designed to return a reference *not* a copy.
    '''
    # Check if it is broadcastable and get the dimensions where the array
    # is only broadcastable.
    b_dims = numpyIsBroadcastable(array.shape, refshape)

    # Easy case: Flag has the same shape as data
    if array.shape == refshape:
        # Simply slice
        return array[item]

    # Well now the not so easy cases:

    # Case 1 - Slice is 1D
    if isinstance(item, slice):
        # Case 1: Subcase 1: First dimension has the same shape as data:
        # Slice it
        if 0 in b_dims:
            return array
        # Case 1: Subcase 2: First dimension has only 1 element
        # (only broadcastable). Leave it alone!
        else:
            return array[item]

    # Case 2 - Array/List/Tuple of slices
    if isinstance(item, (list, tuple)):
        item_copy = [i for i in item]
        # Go through each entry of the b_dims (these are the dimensions that
        # are only broadcastable)
        for i in b_dims:
            # Replace the slice a "None" slice.
            if i < len(item_copy):
                item_copy[i] = slice(None, None, None)
            else:
                break
        return array[tuple(item_copy)]


def numpyOffset(array, offset, newShape=0, fill=0):
    '''
    Offsets an `numpy.ndarray` by inserting it into an filled array with a
    new shape.

    Parameters
    ----------
    array : `numpy.ndarray`
        The array that should be offset.

    offset : `tuple` of `int`
        The n-th element corresponds to the *positiv* offset along
        the n-th axis.

    newShape : `tuple` of `int`
        The n-th element corresonds to the number of elements after
        offsetting along the n-th axis.

    fill : ``Number``, `numpy.ndarray`, optional
        Fill the output before inserting the offsetted image with this value.
        If it is a `numpy.ndarray` it's shape must be the same as newShape.
        Default is ``0``.

    Returns
    -------
    array : `numpy.ndarray`
        The offsetted array.

    Raises
    ------
    ValueError

        - ``len(offset)`` is not equal to ``array.ndim``.
        - ``len(newShape)`` is not equal to ``array.ndim``.
        - ``offset`` or ``newShape`` contains negative values.
        - ``offset`` or ``newShape`` contains non-integers.
        - ``newShape`` is to small to contain the ``array`` with ``offset``.

    Notes
    -----
    ``offset`` and ``shape`` must have the same number of elements like the
    ``array`` has dimensions. So a 2D `numpy.ndarray` requires the ``offset``
    and ``newShape`` to be tuples with two elements each. For a 3D image it
    would be 3 elements each.

    The ``offset`` and ``newShape`` must *all* be positive integers.

    The returned image is a copy (probably), because the kind of `numpy`
    slicing (currently) forces it to be a copy.

    The empty elements of the offsetted image are set to zero.
    '''
    # Verfify dimensions
    if len(offset) != array.ndim:
        raise ValueError('Offset must be contain one element for '
                         'each dimension of the data.')
    if len(newShape) != array.ndim:
        raise ValueError('newShape must be contain one element '
                         'for each dimension of the data.')
    # Verify each set contains positive integers
    for i in offset:
        if int(i) != i:
            raise ValueError('Offset elements must be integers.')
        if i < 0:
            raise ValueError('Offset elements must be positive.')
    for i in range(array.ndim):
        if int(newShape[i]) != newShape[i]:
            raise ValueError('newShape elements must be integers.')
        if newShape[i] < 0:
            raise ValueError('newShape elements must be positive.')
        if newShape[i] < offset[i] + array.shape[i]:
            raise ValueError('newShape elements must be bigger or '
                             'the same than original shape in this'
                             ' dimension + offset.')

    # Determine the original shape as mutable object (a list)
    shape = [i for i in array.shape]
    # Placeholder for final shape and the part of the final shape
    # that is filled by the original
    finalshape = newShape
    insertHere = []

    # Determine the slice for finding where to insert the original
    for dim in range(array.ndim):
        insertHere.append(slice(offset[dim], offset[dim] + shape[dim]))

    # Offset
    if isinstance(fill, np.ndarray):
        if fill.shape != newShape:
            raise ValueError('Fill must have the shape as newShape if it is '
                             'a numpy array.')
        else:
            new_array = np.array(fill)
    elif fill == 0:
        new_array = np.zeros(finalshape, dtype=array.dtype)
    else:
        new_array = np.ones(finalshape, dtype=array.dtype) * fill

    new_array[insertHere] = array.data

    return new_array


def numpyOffsetWithReference(array, offset, newShape, refshape, fill=0):
    '''
    Offsets an `numpy.ndarray` but given a reference shape so broadcastable
    dimensions are not offsetted.

    Parameters
    ----------
    array, offset, newShape, fill :
        see :func:`numpyOffset`

    refshape : `tuple`
        The shapes of the reference `numpy.ndarray` as is returned by
        `numpy.ndarray.shape`.

    Returns
    -------
    array : `numpy.ndarray`
        The offsetted array.

    Raises
    ------
    AssertionError
        see :func:`numpyIsBroadcastable`

    ValueError
        see :func:`numpyOffset`

    See also
    --------
    numpyOffset
    '''
    # Check if it is broadcastable and return the dimensions that are only
    # broadcastable and not the same.
    b_dims = numpyIsBroadcastable(array.shape, refshape)

    offset_tmp = [i for i in offset]
    newShape_tmp = [i for i in newShape]

    # Go through the dimensions that are only broadcastable and set the
    # corresponding offset to zero and newshape to 1.
    for dim in b_dims:
        offset_tmp[dim] = 0
        newShape_tmp[dim] = 1

    return numpyOffset(array, offset_tmp, newShape_tmp, fill)
