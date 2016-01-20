# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np
import warnings
from astropy.utils.exceptions import AstropyUserWarning

from ..error import median_absolute_standard_deviation

__all__ = ['reject_minmax', 'reject_sigma_clip', 'reject_range']


def reject_minmax(array, nlow, nhigh, axis=0):
    '''
    Reject the ``nlow`` lowest values and ``nhigh`` highest values along the
    ``axis``.

    Parameters
    ----------
    array : `numpy.ndarray`-like
        The array in which to reject. Can also be a `numpy.ma.MaskedArray`.

    nlow : `int`
        The number of lowest values to reject.

    nhigh : `int`
        The number of highest values to reject.

    axis : `int`, optional
        The axis along which to reject. Cannot be ``None``!
        Default is ``0``.

    Returns
    -------
    masked_array : `numpy.ma.MaskedArray`
        The array where the ``nlow`` lowest and ``nhigh`` highest values along
        the ``axis`` are masked.

    Notes
    -----
    If the ``array`` already had a ``mask`` then this mask is taken into
    account and the returned array is masked where the ``array`` was masked and
    where the rejection masked the values.
    '''
    # Verify parameter are senseable
    if not isinstance(nlow, int) or not isinstance(nhigh, int):
        raise ValueError('nlow and nhigh must be integer.')
    if nlow < 0 or nhigh < 0:
        raise ValueError('nlow and nhigh must not be negative.')

    # If we dont need to reject simply return the stack.
    if nlow == 0 and nhigh == 0:
        raise ValueError('nlow and nhigh are both zero.')

    # No need to copy since we only alter the mask and not the data values.
    masked_array = np.ma.array(array, copy=True, keep_mask=True,
                               hard_mask=False)

    cmp = np.arange(masked_array.shape[axis])
    if axis != masked_array.ndim - 1:
        # Last axis doesn't need any special setup because it simply works with
        # numpy broadcasting
        expanddims = [i for i in range(masked_array.ndim)]
        for i in expanddims:
            if i != axis:
                cmp = np.expand_dims(cmp, axis=i)

    # Start with the lowest values
    for i in range(nlow):
        # Find the coordinate in axis direction where the minimum value is
        minCoord = np.expand_dims(np.ma.argmin(masked_array, axis=axis),
                                  axis=axis)
        # Mask those values
        newmask = cmp == minCoord
        # Combine the original mask with the newmask with the bitwise
        # or operator
        masked_array.mask |= newmask

    # Same for the highest values for each pixel
    for i in range(nhigh):
        maxCoord = np.expand_dims(np.ma.argmax(masked_array, axis=axis),
                                  axis=axis)
        newmask = cmp == maxCoord
        masked_array.mask |= newmask

    return masked_array


def reject_sigma_clip(data, sigma=3, sigma_lower=None, sigma_upper=None,
                      iters=5, cenfunc=np.ma.median,
                      stdfunc=median_absolute_standard_deviation, axis=None,
                      copy=True):
    """
    Very much like `astropy.stats.sigma_clip` except that the ``stdfunc``
    should be from the ``pymsastro.stats.error``.

    Parameters
    ----------
    stdfunc : ``callable``, optional
        The function that calculates the standard deviation from the calculated
        center (from ``cenfunc``).
        Default is `~pymsastro.stats.median_absolute_standard_deviation`

    Returns
    -------
    masked_array : `numpy.ma.MaskedArray`
        The array where the clipped values are masked.

    See also
    --------
    astropy.stats.sigma_clip

    Notes
    -----
    Only ``pymsastro.stats.error`` functions are allowed that take an ``axis``
    argument.

    Recommended is:

        - `~pymsastro.stats.median_absolute_standard_deviation` if the
          ``cenfunc`` is `numpy.ma.median`
        - `~pymsastro.stats.root_mean_square_error` if the ``cenfunc`` is
          `numpy.ma.mean`
    """
    if sigma_lower is None:
        sigma_lower = sigma
    if sigma_upper is None:
        sigma_upper = sigma

    if axis is not None:
        cenfunc_in = cenfunc
        stdfunc_in = stdfunc
        # These differ from astropy function
        cenfunc = lambda d: np.expand_dims(cenfunc_in(d, axis=axis),
                                           axis=axis)
        stdfunc = lambda d, ref: np.expand_dims(stdfunc_in(d, ref, axis=axis),
                                                axis=axis)

    if np.any(~np.isfinite(data)):
        data = np.ma.masked_invalid(data)
        warnings.warn("Input data contains invalid values (NaNs or infs), "
                      "which were automatically masked.", AstropyUserWarning)

    filtered_data = np.ma.array(data, copy=copy)

    if iters is None:
        i = -1
        lastrej = filtered_data.count() + 1
        while filtered_data.count() != lastrej:
            i += 1
            lastrej = filtered_data.count()
            # The following 3 lines differ from astropy
            center = cenfunc(filtered_data)
            deviation = filtered_data - center
            std = stdfunc(filtered_data, center)
            filtered_data.mask |= np.ma.masked_less(deviation,
                                                    -std * sigma_lower).mask
            filtered_data.mask |= np.ma.masked_greater(deviation,
                                                       std * sigma_upper).mask
    else:
        for i in range(iters):
            # The following 3 lines differ from astropy
            center = cenfunc(filtered_data)
            deviation = filtered_data - center
            std = stdfunc(filtered_data, center)
            filtered_data.mask |= np.ma.masked_less(deviation,
                                                    -std * sigma_lower).mask
            filtered_data.mask |= np.ma.masked_greater(deviation,
                                                       std * sigma_upper).mask

    # prevent filtered_data.mask = False (scalar) if no values are clipped
    if filtered_data.mask.shape == ():
        filtered_data.mask = False   # .mask shape will now match .data shape

    return filtered_data


def reject_range(array, low=None, high=None):
    '''
    Reject the pixel which are below ``low`` or above ``high``.

    Parameters
    ----------
    array : `numpy.ndarray`-like
        The array in which to reject. Can also be a `numpy.ma.MaskedArray`.

    low : ``Number``
        All pixel with values below this value will be rejected. ``None``
        means no such check. Default is ``None``.

    high : ``Number``
        All pixel with values above this value will be rejected. ``None``
        means no such check. Default is ``None``.

    Returns
    -------
    masked_array : `numpy.ma.MaskedArray`
        The array where the rejected values are masked.
    '''
    # Verify parameter are senseable
    if low is None and high is None:
        raise ValueError('Either Parameter low or high must be set.')

    if low is not None:
        if not isinstance(low, (int, float)):
            raise ValueError('Parameter low must be a number.')

    if high is not None:
        if not isinstance(high, (int, float)):
            raise ValueError('Parameter high must be a number.')

    if low is not None and high is not None:
        if low >= high:
            raise ValueError('Parameter low must be smaller than high.')

    # No need to copy since we only alter the mask and not the data values.
    masked_array = np.ma.array(array, copy=True, keep_mask=True,
                               hard_mask=False)

    if low is not None:
        masked_array = np.ma.masked_less(masked_array, low)

    if high is not None:
        masked_array = np.ma.masked_greater(masked_array, high)

    return masked_array
