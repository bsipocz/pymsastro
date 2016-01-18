# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np

from numba import jit

__all__ = ['convolve_with_mask']


def convolve_with_mask(array, kernel, mask=None, rescale_kernel=True):
    """
    Convolution of an array with mask.

    Parameters
    ----------
    array : `numpy.ndarray` or `numpy.ma.MaskedArray`
        The array which shall be convolved.

    kernel : `numpy.ndarray`
        The kernel (or footprint) for the convolution. The sum of the
        ``kernel`` must not be 0 (or very close to it).

    mask : `numpy.ndarray` or ``None``, optional
        Masked values in the ``array``. Elements where the mask is 1 or
        ``True`` are intrepreted as masked. If ``None`` then the array is
        assumed to contain no masked values or provide it's own mask.
        Default is ``None``.

    rescale_kernel : `bool`, optional
        In order to allow for ignoring masked values the kernel must be
        normalized (divided by it's sum). If this is ``True`` the result is
        multiplied with the sum afterwards, if ``False`` the result of the
        convolution will be as if each kernel element was divided by the sum
        of the kernel.
        Default is ``True``.

    Returns
    -------
    conv : `numpy.ndarray`
        The convolved array.

    Notes
    -----
    1. If the ``array`` parameter has a ``mask`` attribute then ``array.data``
       is interpreted as ``array`` and ``array.mask`` as ``mask`` parameter (
       and the explicit ``mask`` parameter must be ``None``). This allows using
       `~numpy.ma.MaskedArray` objects as ``array`` parameter.

    2. No border handling is possible, if the kernel extends beyond the
       image these ``outside`` values are treated as if they were masked.

    3. Since the kernel is internally normalized (in order to allow ignoring
       the masked values) the sum of the kernel must not be 0 because each
       element in the convolved image would be ``Inf`` since it is divided by
       the sum of the kernel. Even a kernel sum close to zero should be avoided
       because otherwise floating value precision might play a significant
       role.

    4. If no mask is given this function might be slower than
       :func:`numpy.convolve` or :func:`scipy.ndimage.filters.convolve` (which
       also allow for more options concerning boundary handling.)
    """
    # Check if an implicit mask is given and if so use the mask as mask and
    # the data as array.
    if hasattr(array, 'mask'):
        if mask is not None:
            raise TypeError('Array contains a mask and an explicit mask was'
                            'given. Cannot evaluate which to use.')
        mask = array.mask
        array = array.data

    # If no mask is given then create an empty one.
    if mask is None:
        mask = np.zeros(array.shape, dtype=np.bool_)
    else:
        # Check if the shape is the same. There might be cases where the
        # array contained a mask attribute but the mask has a different shape
        # than the data!
        if array.shape != mask.shape:
            raise ValueError('Array and Mask must have the same shape.')

    # Evaluate how many dimensions the array has.
    ndim = array.ndim

    # Kernel must have the same number of dimensions
    if kernel.ndim != ndim:
        raise ValueError('Array and Kernel must have the same number of '
                         'dimensions.')

    # Kernel must have odd shape in each dimension
    for i in kernel.shape:
        if i % 2 == 0:
            raise ValueError('Kernel must have an odd shape')

    kernel_sum = np.sum(kernel)

    if ndim == 1:
        conv = _convolve_with_mask_1d(array, kernel, mask)
    elif ndim == 2:
        conv = _convolve_with_mask_2d(array, kernel, mask)
    elif ndim == 3:
        conv = _convolve_with_mask_3d(array, kernel, mask)
    else:
        raise ValueError('Array must have 1, 2 or 3 dimensions.')

    if rescale_kernel:
        return kernel_sum * conv
    else:
        return conv


@jit(nopython=True, nogil=True, cache=True)
def _convolve_with_mask_1d(image, kernel, mask):
    nx = image.shape[0]
    nkx = kernel.shape[0]
    wkx = nkx // 2

    result = np.zeros_like(image)

    for i in range(0, nx, 1):
        iimin = max(i - wkx, 0)
        iimax = min(i + wkx + 1, nx)
        num = 0.
        div = 0.
        for ii in range(iimin, iimax, 1):
            if mask[ii] == 0:
                iii2 = wkx + ii - i
                num += kernel[iii2] * image[ii]
                div += kernel[iii2]
        if div == 0.0:
            result[i] = np.nan
        else:
            result[i] = num / div
    return result


@jit(nopython=True, nogil=True, cache=True)
def _convolve_with_mask_2d(image, kernel, mask):
    nx = image.shape[0]
    ny = image.shape[1]
    nkx = kernel.shape[0]
    nky = kernel.shape[1]
    wkx = nkx // 2
    wky = nky // 2

    result = np.zeros_like(image)

    for i in range(0, nx, 1):
        iimin = max(i - wkx, 0)
        iimax = min(i + wkx + 1, nx)
        for j in range(0, ny, 1):
            jjmin = max(j - wky, 0)
            jjmax = min(j + wky + 1, ny)
            num = 0.
            div = 0.
            for ii in range(iimin, iimax, 1):
                iii2 = wkx + ii - i
                for jj in range(jjmin, jjmax, 1):
                    if mask[ii, jj] == 0:
                        jjj2 = wky + jj - j
                        num += kernel[iii2, jjj2] * image[ii, jj]
                        div += kernel[iii2, jjj2]
            if div == 0.0:
                result[i, j] = np.nan
            else:
                result[i, j] = num / div
    return result


@jit(nopython=True, nogil=True, cache=True)
def _convolve_with_mask_3d(image, kernel, mask):
    nx = image.shape[0]
    ny = image.shape[1]
    nz = image.shape[2]
    nkx = kernel.shape[0]
    nky = kernel.shape[1]
    nkz = kernel.shape[2]
    wkx = nkx // 2
    wky = nky // 2
    wkz = nkz // 2

    result = np.zeros_like(image)

    for i in range(0, nx, 1):
        iimin = max(i - wkx, 0)
        iimax = min(i + wkx + 1, nx)
        for j in range(0, ny, 1):
            jjmin = max(j - wky, 0)
            jjmax = min(j + wky + 1, ny)
            for k in range(0, nz, 1):
                kkmin = max(k - wkz, 0)
                kkmax = min(k + wkz + 1, nz)
                num = 0.
                div = 0.
                for ii in range(iimin, iimax, 1):
                    iii2 = wkx + ii - i
                    for jj in range(jjmin, jjmax, 1):
                        jjj2 = wky + jj - j
                        for kk in range(kkmin, kkmax, 1):
                            if mask[ii, jj, kk] == 0:
                                kkk2 = wkz + kk - k
                                num += (kernel[iii2, jjj2, kkk2] *
                                        image[ii, jj, kk])
                                div += kernel[iii2, jjj2, kkk2]
                if div == 0.0:
                    result[i, j, k] = np.nan
                else:
                    result[i, j, k] = num / div
    return result
