# Licensed under a 3-clause BSD style license - see LICENSE.rst

from abc import ABCMeta, abstractmethod, abstractproperty

import six

from .. import addSynSpectra
from ....stats import DER_SNR, RMSE_SNR

__all__ = ['AddSynSpectrumAnalyse',
           'AddSynSpectrumAnalyseSpectrum1',
           'AddSynSpectrumAnalyseSpectrum2',
           'AddSynSpectrumAnalyseDERIErr',
           'AddSynSpectrumAnalyseDERIVar',
           'AddSynSpectrumAnalyseDEROptimal',
           'AddSynSpectrumAnalyseDERSNRSquared',
           'AddSynSpectrumAnalyseIdealIErr',
           'AddSynSpectrumAnalyseIdealIVar',
           'AddSynSpectrumAnalyseIdealOptimal',
           'AddSynSpectrumAnalyseIdealSNRSquared',
           'AddSynSpectrumAnalyseNone']


@six.add_metaclass(ABCMeta)
class AddSynSpectrumAnalyse(object):
    """
    Metaclass container for holding the results of the addition of synthetic
    spectra for
    different weightings. Analysation is done by `~pymsastro.stats.DER_SNR`
    and `~pymsastro.stats.RMSE_SNR`.

    The method `add` (which adds the two spectra and returns the result
    and the reference) and the property `identification` (which name should
    this kind of addition have to be easily recognizeable.) are abstract and
    must be overridden.
    """
    def __init__(self):
        # Initialize the result-lists
        self._der_signal = []
        self._der_noise = []
        self._der_snr = []
        self._rmse_noise = []
        self._rmse_snr = []

    @abstractproperty
    def identification(self):
        """
        How is the kind of addition called. Must return a `str`.
        """
        return None

    @property
    def result(self):
        """
        A `dict` holding the results.
        """
        return {'der_signal': self._der_signal,
                'der_noise':  self._der_noise,
                'der_snr':    self._der_snr,
                'rmse_noise': self._rmse_noise,
                'rmse_snr':   self._rmse_snr}

    @abstractmethod
    def add(self, spectrum1, spectrum2):
        """
        This method should do the addition.

        Parameters
        ----------
        spectrum1, spectrum2 : `~pymsastro.spectrum.synthetic.SynSpectrum`
            see :meth:`analyze`

        Returns
        -------
        addedSpectrumWithNoise : `numpy.ndarray`
            The added spectrum with noise

        addedSpectrumRef : `numpy.ndarray`
            The added spectrum without noise as reference.

        Notes
        -----
        Must be overridden to yield the correct results.
        """
        return (None, None)

    def analyze(self, spectrum1, spectrum2):
        """
        Takes two spectra and adds them and then analyzes the result. Uses the
        abstract method :meth:`add` which controls how the spectra are added.

        Parameters
        ----------
        spectrum1, spectrum2 : `~pymsastro.spectrum.synthetic.SynSpectrum`
            The first and second spectrum that should be added.
        """
        # Calculate the result
        result, reference = self.add(spectrum1, spectrum2)

        # Calculate resulting snr, noise with DER and RMSE class
        der = DER_SNR(result)
        rmse = RMSE_SNR(reference, result)

        # Save the results
        self._der_signal.append(der.signal)
        self._der_noise.append(der.noise)
        self._der_snr.append(der.snr)
        self._rmse_noise.append(rmse.noise)
        self._rmse_snr.append(rmse.snr)


class AddSynSpectrumAnalyseSpectrum1(AddSynSpectrumAnalyse):
    """
    Just calculates the results for Spectrum 1 without addition.
    """
    @property
    def identification(self):
        """'Spec1          '"""
        return 'Spec1          '

    def add(self, spectrum1, spectrum2):
        return (spectrum1.signalsum_with_noise, spectrum1.signalsum)


class AddSynSpectrumAnalyseSpectrum2(AddSynSpectrumAnalyse):
    """
    Just calculates the results for Spectrum 1 without addition.
    """
    @property
    def identification(self):
        """'Spec2          '"""
        return 'Spec2          '

    def add(self, spectrum1, spectrum2):
        return (spectrum2.signalsum_with_noise, spectrum2.signalsum)


class AddSynSpectrumAnalyseNone(AddSynSpectrumAnalyse):
    """
    Just calculates the results for no weighting.
    """
    @property
    def identification(self):
        """'None           '"""
        return 'None           '

    def add(self, spectrum1, spectrum2):
        return addSynSpectra(spectrum1, spectrum2,
                             weighting=None,
                             s_n_calculator=None)


class AddSynSpectrumAnalyseIdealOptimal(AddSynSpectrumAnalyse):
    """
    Just calculates the results for optimal weighting with the theoretical
    signal and noise.
    """
    @property
    def identification(self):
        """'Ideal - Optimal'"""
        return 'Ideal - Optimal'

    def add(self, spectrum1, spectrum2):
        return addSynSpectra(spectrum1, spectrum2,
                             weighting='optimal',
                             s_n_calculator=None)


class AddSynSpectrumAnalyseIdealSNRSquared(AddSynSpectrumAnalyse):
    """
    Just calculates the results for snr squared weighting with the theoretical
    signal and noise.
    """
    @property
    def identification(self):
        """'Ideal - SNR^2  '"""
        return 'Ideal - SNR^2  '

    def add(self, spectrum1, spectrum2):
        return addSynSpectra(spectrum1, spectrum2,
                             weighting='snr_squared',
                             s_n_calculator=None)


class AddSynSpectrumAnalyseIdealIVar(AddSynSpectrumAnalyse):
    """
    Just calculates the results for inverse variance weighting with the
    theoretical signal and noise.
    """
    @property
    def identification(self):
        """'Ideal - 1/N^2  '"""
        return 'Ideal - 1/N^2  '

    def add(self, spectrum1, spectrum2):
        return addSynSpectra(spectrum1, spectrum2,
                             weighting='ivar',
                             s_n_calculator=None)


class AddSynSpectrumAnalyseIdealIErr(AddSynSpectrumAnalyse):
    """
    Just calculates the results for inverse error (standard deviation)
    weighting with the theoretical signal and noise.
    """
    @property
    def identification(self):
        """'Ideal - 1/N    '"""
        return 'Ideal - 1/N    '

    def add(self, spectrum1, spectrum2):
        return addSynSpectra(spectrum1, spectrum2,
                             weighting='ierr',
                             s_n_calculator=None)


class AddSynSpectrumAnalyseDEROptimal(AddSynSpectrumAnalyse):
    """
    Just calculates the results for optimal weighting with the
    signal and noise determined by `~pymsastro.stats.DER_SNR`.
    """
    @property
    def identification(self):
        """'DER   - Optimal'"""
        return 'DER   - Optimal'

    def add(self, spectrum1, spectrum2):
        return addSynSpectra(spectrum1, spectrum2,
                             weighting='optimal',
                             s_n_calculator=DER_SNR)


class AddSynSpectrumAnalyseDERSNRSquared(AddSynSpectrumAnalyse):
    """
    Just calculates the results for snr squared weighting with the
    signal and noise determined by `~pymsastro.stats.DER_SNR`.
    """
    @property
    def identification(self):
        """'DER   - SNR^2  '"""
        return 'DER   - SNR^2  '

    def add(self, spectrum1, spectrum2):
        return addSynSpectra(spectrum1, spectrum2,
                             weighting='snr_squared',
                             s_n_calculator=DER_SNR)


class AddSynSpectrumAnalyseDERIVar(AddSynSpectrumAnalyse):
    """
    Just calculates the results for inverse variance weighting with the
    signal and noise determined by `~pymsastro.stats.DER_SNR`.
    """
    @property
    def identification(self):
        """'DER   - 1/N^2  '"""
        return 'DER   - 1/N^2  '

    def add(self, spectrum1, spectrum2):
        return addSynSpectra(spectrum1, spectrum2,
                             weighting='ivar',
                             s_n_calculator=DER_SNR)


class AddSynSpectrumAnalyseDERIErr(AddSynSpectrumAnalyse):
    """
    Just calculates the results for inverse error (standard deviation)
    weighting with the signal and noise determined by
    `~pymsastro.stats.DER_SNR`.
    """
    @property
    def identification(self):
        """'DER   - 1/N    '"""
        return 'DER   - 1/N    '

    def add(self, spectrum1, spectrum2):
        return addSynSpectra(spectrum1, spectrum2,
                             weighting='ierr',
                             s_n_calculator=DER_SNR)
