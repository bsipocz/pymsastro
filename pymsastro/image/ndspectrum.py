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
import matplotlib.pyplot as plt
from astropy.modeling import models, fitting

__all__ = ['NDSpectrum']


class NDSpectrum(NDData):

    def __init__(self, wavelengths, data, mask=None):
        self._wave = wavelengths
        self._data = data
        self._mask = mask

    def plot(self, markers=None):
        if self._mask is not None:
            wave = self._wave[~self._mask]
            data = self._data[~self._mask]
        else:
            wave = self._wave
            data = self._data

        plt.scatter(wave, data)
        if markers is not None:
            ar_max = np.max(data)
            for i in markers:
                if markers[i] > wave[0] and markers[i] < wave[-1]:
                    plt.axvline(markers[i], ymax=ar_max, color='r')
                    plt.annotate(i, xy=(markers[i], 0), xytext=(markers[i], 0),
                                 arrowprops=dict(facecolor='black',
                                                 shrink=0.05),
                                 )
        plt.show()

    def slice_waverange(self, wavemin, wavemax):
        range_wave = (self._wave > wavemin) & (self._wave < wavemax)
        newdata = self._data[range_wave]
        newwave = self._wave[range_wave]
        if self._mask is not None:
            newmask = self._mask[range_wave]
        else:
            newmask = None
        return self.__class__(newwave, newdata, newmask)

    def fit(self, model=models.Gaussian1D, plot=True, continuum=None,
            **kwargs):
        if self._mask is not None:
            wave = self._wave[~self._mask]
            data = self._data[~self._mask]
        else:
            wave = self._wave
            data = self._data

        if continuum is None:
            median = np.ma.median(data)
            data = data-median
        else:
            data = data-continuum
        g_init = model(**kwargs)

        fit_g = fitting.LevMarLSQFitter()
        g = fit_g(g_init, wave, data)
        if plot:
            finer_wave = np.linspace(wave[0], wave[-1], 10000)
            plt.plot(finer_wave, g(finer_wave), label='Fit')
            plt.scatter(wave, data, label='data')
            plt.legend()
            plt.show()
        return g

    def fitcont(self, leftrange, rightrange, plot=True):
        if self._mask is not None:
            wave = self._wave[~self._mask]
            data = self._data[~self._mask]
        else:
            wave = self._wave
            data = self._data

        originalwave = deepcopy(wave)
        valid = ((wave > leftrange[0]) & (wave < leftrange[1]) |
                 (wave > rightrange[0]) & (wave < rightrange[1]))

        wave = wave[valid]
        data = data[valid]
        g_init = models.Linear1D()
        fit_g = fitting.LinearLSQFitter()
        g = fit_g(g_init, wave, data)
        if plot:
            plt.plot(originalwave, g(originalwave), label='Fit')
            plt.scatter(wave, data, label='data')
            plt.legend()
            plt.show()
        return g(originalwave)
