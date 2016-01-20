# Licensed under a 3-clause BSD style license - see LICENSE.rst

from abc import ABCMeta, abstractproperty

import six
from numpy import array, median, abs, isfinite, ndarray
# No need to import median from numpy.ma since we either discard the mask or
# delete the masked values.

from ...utils.decorator_collection import lazyproperty_readonly

from ..error import rmse, rmse_rel

__all__ = ['SNR', 'DER_SNR', 'RMSE_SNR']


@six.add_metaclass(ABCMeta)
class SNR(object):
    """
    Metaclass for signal to noise calculations.

    Parameters
    ----------
    flux : `numpy.ndarray`-like
        The flux of the spectrum.

    ignore_zeros : `bool`, optional
        If ``True`` all values that are exactly zero are discarded before any
        calculation is perfomed.
        Default is ``True``.

    ignore_nan_inf : `bool`, optional
        If ``True`` all values that are ``NaN`` (not a number) or ``inf``
        (infinite) are discarded before any calculation is perfomed.
        Default is ``True``.

    ignore_masked : `bool`, optional
        If ``True`` all values that are masked are discarded before any
        calculation is perfomed. This may not be crucial if the calculation
        depends on `numpy` UFUNCs because they default for
        `~numpy.ma.MaskedArray` to
        the appropriate masked ndarray func but for others this might be
        very important. Assumes that the ``flux`` has it's values saved in
        a data attribute and the mask in a mask attribute. The mask must follow
        the ``numpy`` convention that the masked values have a value of 1 and
        unmasked values a 0. (`numpy.ma.MaskedArray` works this way so this is
        used as reference).
        Default is ``True``.

    verbose : `bool`, optional
        If ``True`` print verbose output which values are deleted.
        Default is ``False``.

    Attributes
    ----------
    `signal` : ``Number`` or `numpy.ndarray`

    `noise` : ``Number`` or `numpy.ndarray`

    `signal_to_noise_ratio` : ``Number`` or `numpy.ndarray`

    `snr` : ``Number`` or `numpy.ndarray`

    """

    def __init__(self, flux, ignore_zeros=True, ignore_nan_inf=True,
                 ignore_masked=True, verbose=False):
        self.verbose = verbose
        self._flux = array(flux, copy=True, subok=True)
        self._ignore_zeros = ignore_zeros
        self._ignore_nan_inf = ignore_nan_inf
        self._ignore_masked = ignore_masked
        self._delete()

    def _delete(self):
        """
        Deletes zeros and/or NaNs/Infs in the spectrum.

        Prior to any calculation some values shoul be excluded to allow
        signal and noise calculation.
        """
        # Only delete if it wasn't deleted before.
        if not hasattr(self, '_deleted'):
            # Delete masked pixel if it is a masked array
            if self._ignore_masked and hasattr(self._flux, 'mask'):
                self._delete_masked()
            elif hasattr(self._flux, 'mask'):
                self._flux = self._flux.data

            # Delete pixel where the flux is zero.
            if self._ignore_zeros:
                self._delete_zeros()

            # Delete all NaNs and infs.
            if self._ignore_nan_inf:
                self._delete_nan_inf()

            # Set deleted flag so it will not try to delete again.
            self._deleted = True

    def _delete_masked(self):
        # Delete masked pixel if it is a masked array
        if self.verbose:
            print('Deleting masked values.')
        self._flux = self._flux.data[~self._flux.mask]

    def _delete_zeros(self):
        # Delete pixel where the flux is zero.
        if self.verbose:
            print('Deleting values of 0.')
        self._flux = self._flux[self._flux != 0]

    def _delete_nan_inf(self):
        # Delete all NaNs and infs.
        if self.verbose:
            print('Deleting NaN and Inf values.')
        self._flux = self._flux[isfinite(self._flux)]

    @abstractproperty
    def signal(self):
        """
        The signal of the spectrum.
        """
        return None

    @abstractproperty
    def noise(self):
        """
        The noise of the spectrum.
        """
        return None

    @abstractproperty
    def signal_to_noise_ratio(self):
        """
        The signal divided by the noise of the spectrum.
        """
        return None

    @property
    def snr(self):
        """
        Equivalent to `signal_to_noise_ratio`.
        """
        return self.signal_to_noise_ratio


class DER_SNR(SNR):
    """
    http://www.stecf.org/software/ASTROsoft/DER_SNR/ with minor modifications
    and more comments.

    Calculates the signal and noise with median and median absolute deviation,
    skipping every second pixel in case of the noise.

    Raises
    ------
    ValueError
        Needs at least 4 elements after deletion of invalid values.

    See also
    --------
    SNR
    """
    def _delete(self):
        super(DER_SNR, self)._delete()
        # Check that there are at least 4 pixel left otherwise terminate
        # with a ValueError.
        n = len(self._flux)
        if n <= 4:
            raise ValueError('Not enough values to calculate signal and '
                             'noise. At least 4 are needed but only {0} left'
                             '.'.format(n))

    @lazyproperty_readonly
    def signal(self):
        return median(self._flux)

    @lazyproperty_readonly
    def noise(self):
        n = len(self._flux)

        # The noise is the median absolute deviation of 2 times the central
        # pixel minus the pixel two on the left and two on the right.
        # It's 1 ignored pixel since most CCDs have correlated noise in
        # neighboring pixels.
        noise = median(abs(2.0 * self._flux[2:n-2] -
                           self._flux[0:n-4] - self._flux[4:n]))
        # Median deviation needs an additional factor (see median absolute
        # deviation) and we also need to compensate that we take 3
        # different values (error propagation).
        return 0.6052697 * noise

    @lazyproperty_readonly
    def signal_to_noise_ratio(self):
        return self.signal / self.noise


class RMSE_SNR(SNR):
    """
    Computes the noise and signal to noise ratio with the help of a reference
    spectrum based on RMSE (the root-mean-square-error).

    Parameters
    ----------
    reference : ``Number`` or `numpy.ndarray`
        The reference spectrum to which to compare the flux
    args, kwargs :
        Parameters needed for `SNR`.

    See also
    --------
    SNR
    """
    def __init__(self, reference, *args, **kwargs):
        # Take the reference spectrum and pass all the other stuff to SNR
        self._reference = reference
        super(RMSE_SNR, self).__init__(*args, **kwargs)

    def _delete_masked(self):
        # Delete masked pixel in the reference before deleting them in flux
        if isinstance(self._reference, ndarray):
            self._reference = self._reference[~self._flux.mask]
        super(RMSE_SNR, self)._delete_masked()

    def _delete_zeros(self):
        # Delete zeros in the reference before deleting them in flux
        if isinstance(self._reference, ndarray):
            self._reference = self._reference[self._flux != 0]
        super(RMSE_SNR, self)._delete_zeros()

    def _delete_nan_inf(self):
        # Delete NaN and Infs in the reference before deleting them in flux
        if isinstance(self._reference, ndarray):
            self._reference = self._reference[isfinite(self._flux)]
        super(RMSE_SNR, self)._delete_nan_inf()

    @property
    def signal(self):
        # This is just the reference spectrum.
        return self._reference

    @lazyproperty_readonly
    def noise(self):
        # Use the RMS of the absolute error between flux and reference.
        return rmse(self._flux, self._reference)

    @lazyproperty_readonly
    def signal_to_noise_ratio(self):
        # Use the inverse of the RMS of the relative error between flux
        # and reference.
        return 1 / rmse_rel(self._flux, self._reference)
