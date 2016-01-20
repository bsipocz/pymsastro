# Licensed under a 3-clause BSD style license - see LICENSE.rst

__all__ = ['NDImage']

# Numpy / Scipy / Matplotlib
import numpy as np

# import matplotlib.cm as cm

# Astropy
from astropy import log
from astropy.io import fits
from astropy.wcs import WCS
import astropy.units as u

# Future astropy
from .compat.nddata import NDData
from .compat.ndslicing import NDSlicingMixin
from .compat.ndarithmetic import NDArithmeticMixin
from .compat.nduncertainty import NDUncertainty, StdDevUncertainty
# from nddata_base import NDDataBase
# from ndio import NDIOMixin
# from nduncertainty import UnknownUncertainty

# Python modules
from copy import deepcopy


class FitsExtensions(object):
    '''
    Name of Fits extensions that have special meaning.

    - bpm (bad pixel mask):
        Will be used while reading as mask for the data
    - uncertainty:
        Will be used as uncertainty on the data
    - flag:
        Prefix for a flag extension.
    '''
    bpm = ('BPM', )
    uncertainty = ('UNCERT', )
    flag = ('FLAG', )


class FitsHeaderCards(object):
    '''
    Header Keywords that will be used for control flow:

    - uncertainty_type:
        the type of uncertainty (only sensable if extension uncertainty is
        present). Like ``std`` for standard deviation error
    - unit:
        the unit of the data points. Like ``adu`` if it's an analog digital
        unit. Must be a string that can be converted to an
        `~astropy.units.Unit`
    '''
    flag_name = ('FLAGNAME', )
    uncertainty_type = ('UNCERTYP', )
    uncertainty_unit = ('UNCERUNT', )
    unit = ('BUNIT', )
    crPix = ('CRPIX', )
    saturation = ('SATURATE', )


class Verbose(object):
    '''
    Each class and function may have a verbose output. If you want to enable
    it, change it the corresponding value to ``True``.
    '''
    readFits = False
    writeFits = False
    mask_saturated = False
    ndimage_offset = False
    ndimage_slice = False
    ndimage_chgRefPix = False
    ndimage_normAxis = False

    ndimagecollec_offset = False

    ndstack_reject = False

    verified_already = False


class NDImage(NDArithmeticMixin, NDSlicingMixin, NDData):
    '''
    Just NDData with Mixins with:
    - `verify` function to check properties satisfy my requirements
    - `filename` property to follow the useage of the image.
    '''

    def __init__(self, *args, **kwargs):
        # Pop additional keywords
        if 'flags' in kwargs:
            if 'copy' in kwargs and kwargs['copy']:
                self._flags = deepcopy(kwargs.pop('flags'))
            else:
                self._flags = kwargs.pop('flags')
        else:
            self._flags = {}

        super(NDImage, self).__init__(*args, **kwargs)
        # Replace an empty meta with an empty fits header
        if len(self._meta) <= 0:
            self._meta = fits.Header()
        self._verified = []

    @property
    def flags(self):
        '''
        Flags of the data.

        Not yet handled:
        - init !DONE!
        - slicing !DONE!
        - arithmetics !IGNORE FLAGS
        - read/write !DONE!
        - stack !IGNORE FLAGS!
        - offset !DONE!
        - verify !DONE!
        - plot/show !DONE!
        TODO: Maybe introduce offset-flags :-D
        '''
        return self._flags

    def add_flag(self, kw, value, replace=False):
        '''
        Adds a flag to the flag collection.

        Parameters
        ----------
        kw: `str`
            The keyword in the flags collection.
            Should be short (4-5 characters)!
        value: `np.ndarray`
            The value for the flags. Should have the
            same shape as the data or be broadcastable.
        replace: `bool`, optional
            If ``True`` the flag (if already present)
            will be replaced with a warning. If ``False`` it
            will raise an Exception. Default is ``False``
        '''
        if kw in self._flags:
            if replace:
                log.info('Replacing flag {0}'.format(kw))
            else:
                raise ValueError('Cannot replace flag because explicitly '
                                 'forbidden')
        self._flags[kw] = value

    @property
    def size(self):
        '''
        Size of the data
        '''
        return self.data.size

    @property
    def verified(self):
        '''
        Some operations need the image to be verified

        Returns
        -------
        ``List``:
            Key: What has been verified
            - ``general``: See method ``verify``

        Notes
        -----
        Each time a property is updated this should be emptied.
        THIS IS NOT DONE YET.
        '''
        return self._verified

    def _slice(self, item):
        '''
        Overloaded to change reference pixel in meta and to handle flags.

        TODO: Use decorator to get in the documentation.
        '''
        # Call super slicing
        kwargs = super(NDImage, self)._slice(item)
        # Slice meta
        kwargs['meta'] = self._slice_meta(item)
        # Slice flags
        kwargs['flags'] = self._slice_flags(item)
        return kwargs

    def _slice_flags(self, item):
        '''
        TODO: Use decorator to get in the documentation.
        '''
        # The important thing is that I allowed flags to
        # be only broadcastable, so
        # slicing isn't straightforward here.
        flags = self.flags
        # Empty dict?
        if not flags:
            # Return the empty dict
            return flags

        return_flags = {}
        # Every flag needs to be sliced:
        for flag in flags:
            return_flags[flag] = self._slice_maybe_broadcastable(flags[flag],
                                                                 item)

        return return_flags

    def _slice_maybe_broadcastable(self, thing, item):
        '''
        Sometimes we want to slice something that
        has the same number of dimensions but
        it is only broadcastable to the original
        data not really the same shape. Then
        this method should be used. It tests if the
        ``thing`` can or should be sliced with
        ``item``.

        Parameters
        ----------
        thing: `np.ndarray`
            The thing that we want to slice.
        item: `slice` or `list`/`tuple` of slices
            The passed slicing item.

        Returns
        -------
        sliced thing: `np.ndarray`

        Notes
        -----
        Takes the data property as reference for the shape.

        Does not allow for complex slicing.

        This method is designed to return a reference NOT a copy.
        '''
        # Placeholder for each dimension that is only broadcastable:
        slice_not_dim = slice(None, None, None)

        # Easy case: Flag has the same shape as data
        if thing.shape == self.data.shape:
            # Simply slice
            return thing[item]

        # Well now the not so easy case:
        # Case 1 - Slice is 1D
        if isinstance(item, slice):
            # Case 1: Subcase 1: First dimension has the same shape as data:
            # Slice it
            if thing.shape[0] == self.data.shape[0]:
                return thing[item]
            # Case 1: Subcase 2: First dimension has only 1 element
            # (only broadcastable). Leave it alone!
            elif thing.shape[0] == 1:
                return thing
            else:
                raise ValueError('The first dimension is neither the same nor '
                                 'broadcastable to data.')
        # Case 2 - Array/List/Tuple of slices
        if isinstance(item, (list, tuple)):
            item_copy = [i for i in item]
            # Go through each dimension of the slice-item
            for i in range(len(item)):
                # Case 2: Subcase 1: This dimension has the same shape
                # as data: Slice it
                if thing.shape[i] == self.data.shape[i]:
                    pass
                # Case 2: Subcase 2: This dimension has only 1 element
                # (only broadcastable). Leave it alone!
                elif thing.shape[i] == 1:
                    # Replace it by the "do not slice this dimension" slice
                    item_copy[i] = slice_not_dim
                else:
                    raise ValueError('The {0} dimension is neither '
                                     'the same nor broadcastable to '
                                     'data.'.format(i))
            return thing[tuple(item_copy)]

    def _slice_meta(self, item):
        '''
        TODO: Use decorator to get in the documentation.
        '''
        meta = self.meta

        # Check if reference pixels are given
        editMeta = False
        # No meta means no change
        if len(meta) > 0:
            for kw in FitsHeaderCards.crPix:
                kw_num = kw + '1'
                if kw_num in meta:
                    editMeta = True

        if editMeta:
            shifts = []
            # If the item is a slice the user tried to slice 1D
            if isinstance(item, slice):
                if self.data.ndim > 1:
                    raise ValueError('Cannot slice {0}D image with a 1D '
                                     'slice.'.format(self.data.ndim))
                else:
                    if item.start is not None and item.start != 0:
                        shifts.append(-item.start)

            elif isinstance(item, tuple):
                if len(item) != self.data.ndim:
                    raise ValueError('Cannot slice {0}D image with a {1}D '
                                     'slice.'.format(self.data.ndim,
                                                     len(item)))
                else:
                    for i in range(len(item)):
                        if item[i].start is not None and item[i].start != 0:
                            shifts.append(-item[i].start)
                        else:
                            shifts.append(0)

            else:
                raise ValueError('Got too complicated slice (not tuple of '
                                 'slices).')

            return self._changeReferencePixel(shifts, deepcopy(meta))

        return meta

    def _changeReferencePixel(self, values, header=None):
        '''
        Slicing and offsetting can render the header
        useless because the reference pixel value is
        not correct anymore. Because FITS axis and
        numpy axes are reversed and one is zero based and
        the other 1-based I thought it safest to
        have a seperate method for this.

        Parameters
        ----------
        value: Tuple of numbers
            The change of the reference pixel. Be careful that offset has
            a positive shift and slicing
            should be negative.
        header: astropy.io.fits.Header object, optional
            The header which should be changed. If not given or None it
            will update the own meta.
        '''
        # Use the own meta if no header was given
        if header is None:
            header = self.meta

        # Verify that each dimension has a shift
        ndim = self.data.ndim
        if len(values) != ndim:
            raise ValueError('Number of shifts does not match the number '
                             'of dimensions')

        # Check each possible header card for reference pixel extremly slow
        # if many elements in header
        # and list
        for kw in FitsHeaderCards.crPix:
            # Check each dimension
            for i in range(ndim):
                # Build the right keyword the right number is (ndim - i)
                # because python is zero-based
                # while fits is not and the order is reversed.
                kw_full = kw + str(ndim - i)
                # Check if it is in the header
                if kw_full in header:
                    # Change it
                    if Verbose.ndimage_chgRefPix:
                        log.info('Updated Header Keyword: "{0}" Value "{1}" =>'
                                 ' "{2}"'.format(kw_full, header[kw_full],
                                                 header[kw_full] + values[i]))
                    header[kw_full] += values[i]

        return header

    def _verify(self):
        if 'general' in self._verified:
            if Verbose.verified_already:
                log.info('NDImage is already generally verified')
        else:
            # Data must be a numpy array
            if not isinstance(self._data, np.ndarray):
                raise ValueError('data must be a numpy array')
            # Check for numerical dtype ...
            if self._data.dtype.kind not in ['i', 'f', 'u']:
                raise ValueError('data must be numerical')

            # Check meta is a fits header (meta is always set!)
            if not isinstance(self._meta, fits.Header):
                raise ValueError('meta must be a astropy.io.fits header '
                                 'object')

            # Check unit is a unit
            if self._unit is not None:
                # Must be something resembling a BaseUnit
                if not isinstance(self._unit, u.UnitBase):
                    raise ValueError('unit must be a astropy.units Unit '
                                     'object')
            else:
                # If it is not set assume dimensionless unscaled
                self._unit = u.dimensionless_unscaled

            # Check wcs if set.
            if self._wcs is not None:
                if not isinstance(self._wcs, WCS):
                    raise ValueError('wcs must be a astropy.wcs WCS object')

            # Check mask if set
            if self._mask is not None:
                # Mask must be a numpy array
                if not isinstance(self._mask, np.ndarray):
                    raise ValueError('mask must be a numpy array')
                # Check for boolean dtype
                if self._mask.dtype.kind not in ['b']:
                    raise ValueError('mask must be boolean')
                # Check shape matches
                if self._mask.shape != self._data.shape:
                    raise ValueError('mask shape does not match data')

            # Check uncertainty if set
            if self._uncertainty is not None:
                # Must be a StdDevUncertainty
                if not isinstance(self._uncertainty, StdDevUncertainty):
                    raise ValueError('uncertainty must be a StdDevUncertainty')
                # Must contain a numpy array
                if self._uncertainty.array is not None:
                    if not isinstance(self._uncertainty.array, np.ndarray):
                        raise ValueError('uncertainty must be a numpy array')
                    # Check uncertainty has numerical dtype
                    if self._uncertainty.array.dtype.kind not in ['i',
                                                                  'f', 'u']:
                        raise ValueError('uncertainty must be numerical')
                    # Check shape matches
                    if self._uncertainty.array.shape != self._data.shape:
                        raise ValueError('uncertainty shape does '
                                         'not match data')
                    # Check unit matches (no need for None handling since
                    # we assign
                    # dimensionless_unscaled if any unit is None)
                    if self._uncertainty.unit != self._unit:
                        raise ValueError('uncertainty unit does '
                                         'not match data')

            # Check flags if set
            if len(self._flags) > 0:
                for flag_key in self.flags:
                    if not isinstance(self.flags[flag_key], np.ndarray):
                        raise ValueError('flag "{0}" must be a numpy array not'
                                         ' {1}' .format(flag_key,
                                      self.flags[flag_key].__class__.__name__))
                    if not verifyShapeBroadcastable(self.flags[flag_key].shape,
                                                    self._data.shape):
                        raise ValueError('flag "{0}" must have the same shape '
                                         'or be broadcastable to the shape of '
                                         'the data {1} {2}'.format(flag_key,
                                                                   self.data.shape,
                                                                   self.flags[flag_key].shape))
            # Add to verified
            self._verified.append('general')

    def offset(self, offset, newShape):
        '''
        Offsets the image.

        Parameters
        ----------
        offset: Tuple of Integer
            The n-th element corresponds to the positiv offset along
            the nth axis.
        newShape: Tuple of Integer
            The n-th element corresonds to the number of elements after
            offsetting along the nth axis.

        Notes
        -----
        All the parameters must have the same number of elements like the
        saved image has
        dimensions. So a 2D image requires the ``offset`` and ``newShape``
        to be tuples
        with two elements each. For a 3D image it would be 3 elements each.

        The offset elements must *all* be positive integers.

        For FITS files the first dimension is the vertical and the second
        one is the horizontal axis.

        The returned image is a copy (probably), because the kind of numpy
        slicing (currently) forces
        it to be a copy.
        '''
        # Verfify dimensions
        if len(offset) != self.data.ndim:
            raise ValueError('Offset must be contain one element for '
                             'each dimension of the data.')
        if len(newShape) != self.data.ndim:
            raise ValueError('newShape must be contain one element '
                             'for each dimension of the data.')
        # Verify each set contains positive integers
        for i in offset:
            if int(i) != i:
                raise ValueError('Offset elements must be integers.')
            if i < 0:
                raise ValueError('Offset elements must be positive.')
        for i in range(self.data.ndim):
            if int(newShape[i]) != newShape[i]:
                raise ValueError('newShape elements must be integers.')
            if newShape[i] < 0:
                raise ValueError('newShape elements must be positive.')
            if newShape[i] < offset[i] + self.data.shape[i]:
                raise ValueError('newShape elements must be bigger or '
                                 'the same than original shape in this'
                                 ' dimension + offset.')

        # Verify the integrity of the image (unfortunatly this has some
        # components that may be too strict
        # here:
        # unit of data and uncertainty must be the same, wcs must be a WCS
        #  object, ... - only the shape
        # check and the type checks for np.ndarrays are really mandatory)
        self._verify()

        # Determine the original shape
        shape = [i for i in self.data.shape]
        # Placeholder for final shape and the part of the final shape
        # that is filled by the original
        finalshape = newShape
        insertHere = []

        # Determine the slice for finding where to insert the original
        for dim in range(self.data.ndim):
            insertHere.append(slice(offset[dim], offset[dim] + shape[dim]))
        if Verbose.ndimage_offset:
            log.info('Offsetted image with shape "{0}" to shape "{1}" with '
                     'offset "{2}"'.format(shape, finalshape, offset))

        # Start offsetting the ...
        # ... Data
        new_data = np.zeros(finalshape, dtype=self.data.dtype)
        new_data[insertHere] = self.data
        if Verbose.ndimage_offset:
            log.info('Offsetted data')

        # ... Mask
        if self._mask is not None:
            new_mask = np.zeros(finalshape, dtype=np.bool_)
            new_mask[insertHere] = self.mask
            if Verbose.ndimage_offset:
                log.info('Offsetted mask')
        else:
            new_mask = None

        # ... Uncertainty
        if self._uncertainty is not None:
            # Only possible if uncertainty holds a numpy array (in principle
            # this is checked by verify.
            # But maybe someone forgot to call it, it's only mandatory
            # for NDImageCollection.
            if isinstance(self._uncertainty.array, np.ndarray):
                new_uncert = self._uncertainty.__class__(
                                np.zeros(finalshape,
                                         dtype=self._uncertainty.array.dtype),
                                copy=False)
                new_uncert.array[insertHere] = self._uncertainty.array
            else:
                raise ValueError('Uncertainty must hold a numpy.ndarray not'
                      ' a "{0}"!'.format(self._uncertainty.__class__.__name__))
            if Verbose.ndimage_offset:
                log.info('Offsetted uncertainty')
        else:
            new_uncert = None

        # ... Flags
        new_flags = {}
        if self.flags:
            for flag in self.flags:
                new_flags[flag] = self._offsetBroadcastable(self.flags[flag],
                                                            offset, finalshape)

        # ... Meta
        new_meta = self._changeReferencePixel(offset, deepcopy(self.meta))

        # ... WCS
        if self.wcs is not None:
            # Reparse the WCS from the new meta
            new_wcs = WCS(new_meta)
            if Verbose.ndimage_offset:
                log.info('Reparsed WCS information')
        else:
            new_wcs = None

        # ... Unit
        new_unit = self.unit

        # Create a new instance (not copied because only meta needs a copy
        #  and is copied before)
        return self.__class__(new_data, uncertainty=new_uncert,
                              mask=new_mask, wcs=new_wcs,
                              meta=new_meta, unit=new_unit,
                              flags=new_flags, copy=False)

    def _offsetBroadcastable(self, thing, offset, newShape):
        '''
        Creates an empty placeholder with the new shape and inserts the old
        one in it with a given offset.
        This method is adapted to also manage "just" broadcastable elements.

        Parameters
        ----------
        thing: `np.ndarray`
            The original that will be offset and inserted
        offset: `list` or `tuple` of integer
            Should come from the ``offset`` method because then it is checked
            for validity
        newShape: `tuple` of Integer
            Should come from the ``offset`` method because then it is checked
            for validity
        '''
        # Copy them as lists so they can be edited
        newShape_copy = [i for i in newShape]
        insert_slice = [0 for i in offset]

        # Check each dimension if it is the same as the datas or only
        # broadcastable
        for i in range(self.data.ndim):
            if thing.shape[i] == self.data.shape[i]:
                insert_slice[i] = slice(offset[i], offset[i]+thing.shape[i],
                                        None)
                pass
            elif thing.shape[i] == 1:
                # Broadcastable
                newShape_copy[i] = 1
                insert_slice[i] = slice(None, None, None)
            else:
                raise ValueError('Trying to offset something that is not '
                                 'broadcastable to data.')

        # Now create an empty array with the new shape and the old dtype
        offset_thing = np.zeros(tuple(newShape_copy), dtype=thing.dtype)
        offset_thing[tuple(insert_slice)] = thing
        return offset_thing

    def _normalizeAlongAxis(self, axis, func,
                            iters=0, iters_bound_min=0.5, iters_bound_max=2,
                            iters_smart=True,
                            keepNorm=True):
        '''
        Normalizes (divides by a number) the image along one axis.

        Parameters
        ----------
        axis: `int`
            Along which axis to normalize.
        func: `callable`
            This will determine the scaling factor for normalization. Must
            take an axis argument.
        iters: `int`, optional
            Number of masking iterations. After each masking iteration the
            image is normalized again. 0
            means the image is normalized without masking. A value of 1 means,
            normalization, masking
            and normalized again.
        iters_bound_min: ``None``, ``Number``, optional
            In each masking iteration values below this value are masked or
            none if ``None``. Default is
            ``None``.
        iters_bound_max: ``None``, ``Number``, optional
            In each masking iteration values above this value are masked or
            none if ``None``. Default is
            ``None``.
        iters_smart: `bool`, optional
            If ``True`` the iterations will stop if no additional pixel are
            rejected in this iteration.
            If ``False`` it does not check this and simply runs the number of
            iterations. Default is
            ``True``.
        keepNorm: `bool`, optional
            If ``True`` the method will save the normalization in the flags1d
            of the return.
        '''
        # Check params
        if not isinstance(axis, int):
            raise ValueError('Axis parameter must be an integer')
        elif axis != 0 and axis != 1:
            raise ValueError('Axis parameter must be 0 or 1, more is not '
                             'implemented yet')
        if not isinstance(iters, int):
            raise ValueError('iters parameter must be an integer')
        if iters < 0:
            raise ValueError('iters parameter must be positive')
        if iters > 0:
            # Check that something should be masked when iterating:
            if iters_bound_min is None and iters_bound_max is None:
                raise ValueError('iters were given but no boundaries.')

            if iters_bound_min is not None:
                if not isinstance(iters_bound_min, (int, float)):
                    raise ValueError('iters_bound_min parameter must a number')
                elif iters_bound_min < 0:
                    iters_bound_min = abs(iters_bound_min)
                    log.info('iters_bound_min was smaller than zero -> used '
                             'the absolute value.')
                if iters_bound_min >= 1:
                    raise ValueError('iters_bound_min parameter must be '
                                     'smaller than 1.')

            if iters_bound_max is not None:
                if not isinstance(iters_bound_max, (int, float)):
                    raise ValueError('iters_bound_max parameter must a number')
                elif iters_bound_max < 0:
                    iters_bound_max = abs(iters_bound_max)
                    log.info('iters_bound_max was smaller than zero -> used '
                             'the absolute value.')
                if iters_bound_max <= 1:
                    raise ValueError('iters_bound_min parameter must be '
                                     'bigger than 1.')

        data = self.data
        if self.mask is None:
            mask = False
        else:
            mask = self.mask
        if self.uncertainty is not None:
            uncert = self.uncertainty.array
        else:
            uncert = 0
        if Verbose.ndimage_normAxis:
            print('=============================Original'
                  '=============================')
            print('Data')
            print(data)
            print('Mask')
            print(mask)
            print('Uncertainty')
            print(uncert)

        # Make it a masked array for the function call
        data_masked = np.ma.array(data, mask=mask)

        # Call the function and store the result. Then invert the result and
        #  multiply it with data and
        # uncertainty
        res = func(data_masked, axis=axis)
        if keepNorm:
            normKeep = np.ma.expand_dims(res, axis=axis)
        norm = 1 / res
        # norm is a masked array since the input was a masked array. A mask
        # of True means that a complete
        # row/col was masked.
        if np.any(norm.mask):
            log.info('One row/column is completly masked')
        # To allow broadcasting with arbitary axis we need to expand
        # the dimension again
        norm = np.ma.expand_dims(norm, axis=axis)
        # Alter the data and uncertainty
        data_new = norm.data * data
        # TODO: Uncertainty propagation?
        uncert_new = norm.data * uncert
        mask_new = mask
        if Verbose.ndimage_normAxis:
            print('=============================Iteration 0'
                  '=============================')
            print('Normalize with')
            print(norm.data)
            print('Data')
            print(data_new)
            print('Mask')
            print(mask_new)
            print('Uncertainty')
            print(uncert_new)

        # Now do the masking iteration
        for iteration in range(iters):
            # Mask
            if iters_bound_min is not None:
                mask_bounds_min = data_new < iters_bound_min
            else:
                mask_bounds_min = False

            if iters_bound_max is not None:
                mask_bounds_max = data_new > iters_bound_max
            else:
                mask_bounds_max = False

            mask_bounds = mask_bounds_min | mask_bounds_max

            if iters_smart:
                # Create a temporary placeholder to compare against previous
                # result
                mask_new_tmp = mask_new | mask_bounds
                if np.all(mask_new_tmp == mask_new):
                    log.info('Stopping iteration because no pixel where '
                             'rejected in iteration {0}'.format(iteration + 1))
                    break
                # Use the placeholder
                mask_new = mask_new_tmp
            else:
                # The same as above but without placeholder since we don't
                # need to compare.
                mask_new = mask_new | mask_bounds

            # Renormalize ...

            # Make it a masked array for the function call
            data_masked = np.ma.array(data_new, mask=mask_new)

            # Call the function and store the result. Then invert the result
            # and multiply it with data and
            # uncert
            res = func(data_masked, axis=axis)
            if keepNorm:
                normKeep *= np.ma.expand_dims(res, axis=axis)
            norm = 1 / res
            # norm is a masked array since the input was a masked array. A mask
            # of True means that a
            # complete row/col was masked.
            if np.any(norm.mask):
                log.info('One row/column is completly masked')
            # To allow broadcasting with arbitary axis we need to expand
            # the dimension again
            norm = np.ma.expand_dims(norm, axis=axis)
            # Alter data and uncertainty
            data_new = norm.data * data_new
            # TODO: Uncertainty propagation?
            uncert_new = norm.data * uncert_new

            if Verbose.ndimage_normAxis:
                print('=============================Iteration {0}'
                      '============================='.format(iteration + 1))
                print('Masked below {0}'.format(iters_bound_min))
                print(mask_bounds_min)
                print('Masked above {0}'.format(iters_bound_max))
                print(mask_bounds_max)
                print('Combined mask (this iteration)')
                print(mask_bounds)
                print('Combined mask with previous mask')
                print(mask_new)
                print('Normalize with')
                print(norm.data)
                print('Data')
                print(data_new)
                print('Mask')
                print(mask_new)
                print('Uncertainty')
                print(uncert_new)

        # Finished iterating, return a new instance
        # ... but first make the uncertainty a StdDevUncertainty again.
        if np.all(uncert_new == 0):
            uncert_new = None
        else:
            uncert_new = self.uncertainty.__class__(uncert_new, copy=False)

        flags_new = self.flags
        if keepNorm:
            flags_new['NORM'] = normKeep

        return self.__class__(data_new, uncertainty=uncert_new,
                              mask=mask_new, wcs=self.wcs,
                              meta=self.meta, unit=self.unit,
                              flags=flags_new, copy=False)
