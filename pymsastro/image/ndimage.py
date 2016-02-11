# Licensed under a 3-clause BSD style license - see LICENSE.rst

# Future astropy
from .compat.nddata import NDData
from .compat.ndslicing import NDSlicingMixin
from .compat.ndarithmetic import NDArithmeticMixin
from .compat.nduncertainty import StdDevUncertainty
from astropy.io import fits
from astropy.wcs import WCS
from copy import deepcopy
import numpy as np

__all__ = ['NDImage']


class NDImage(NDArithmeticMixin, NDSlicingMixin, NDData):

    @classmethod
    def fromfits(cls, name, folder=''):
        """
        Reads a FITS file and returns the image as `NDImage` instance.

        Parameters
        ----------
        name : `str`
            The name of the FITS Image.

        folder : `str`
            The path to the folder of the image.

        Notes
        -----
        It assumes that the ``uncertainty`` and ``mask`` are saved as
        HDU extensions called ``UNCERT`` and ``MASK``.

        It doesn't try to parse the ``unit`` even though it should be
        in the header with the keyword ``BUNIT``.
        """
        hdus = fits.open(folder + name)
        data = hdus[0].data.astype(np.float64)
        meta = hdus[0].header
        wcs = WCS(meta)
        mask = None
        uncert = None
        if 'mask' in hdus:
            mask = hdus['mask'].data.astype(np.bool_)
        if 'uncert' in hdus:
            uncert = StdDevUncertainty(hdus['uncert'].data)
        hdus.close()
        return cls(data, mask=mask, uncertainty=uncert, wcs=wcs, meta=meta)

    def tofits(self, name, folder=''):
        """
        Writes a FITS file based on ``data`` (`NDImage`).

        Parameters
        ----------
        name : `str`
            The name of the FITS Image.

        folder : `str`
            The path to the folder of the image.

        Notes
        -----
        It saves the ``uncertainty`` and ``mask`` as HDU extensions called
        ``UNCERT`` and ``MASK``.

        It does *not* save the ``unit`` of the `NDImage`.

        It overwrites existing images when the path (``folder + name``) is the
        same.
        """
        header = deepcopy(self.meta)

        if self.wcs is not None:
            header.extend(self.wcs.to_header(), useblanks=False, update=True)

        hdus = [fits.PrimaryHDU(self.data, header)]

        if self.mask is not None:
            hduMask = fits.ImageHDU(self.mask.astype(np.uint8), name='mask')
            hdus.append(hduMask)

        if self.uncertainty is not None:
            hduUncert = fits.ImageHDU(self.uncertainty.array, name='uncert')
            hdus.append(hduUncert)

        hdulist = fits.HDUList(hdus)
        hdulist.writeto(folder + name, clobber=True, output_verify='fix+warn')
