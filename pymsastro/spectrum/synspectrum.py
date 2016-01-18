# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np
import matplotlib.pyplot as plt

from . import SynSignal

from ..utils import lazyproperty_readonly

__all__ = ['SynSpectrum']


class SynSpectrum(object):
    """
    A container for multiple `SynSignal` instances.

    Parameters
    ----------
    signals : `numpy.ndarray` or a `list`/`tuple` containing `numpy.ndarray`
        The signal or the signals

    wavelengths : `numpy.ndarray`, ``None``, optional
        The wavelengths associated with the signal(s). If ``None`` then pixel
        space is assumed (creating wavelengths with :func:`numpy.arange`)
        Default is ``None``

    shotnoise : `bool`, optional
        Apply shot noise to the signal. If ``True`` then shotnoise is used, if
        ``False`` then shotnoise is ignored.
        Default is ``True``.

    constnoise : ``False`` or ``Number``
        If ``False`` no constant noise is added. If a number is given the
        number is interpreted as the standard deviation of the constant
        noise.
        Default is ``False``

    Raises
    ------
    TypeError

        - The signals could not be wrapped in a `SynSignal`.
        - The wavelengths is not None and not a `numpy.ndarray`.
        - The shotnoise is not a `bool`.
        - The constnoise is not False or a ``Number`` or ``False``.

    ValueError

        - If the signals do not have the same shape.
        - If the wavelengths has not the same shape as the signals.

    Attributes
    ----------
    `signals` : `list` of `SynSignal`

    `wavelengths` : `numpy.ndarray`

    `shotnoise` : `bool`

    `constnoise` : ``Number``

    `constnoise_array` : `numpy.ndarray`

    `signalsum` : `numpy.ndarray`

    `signalsum_with_noise` : `numpy.ndarray`

    `signalsum_norm_with_noise` : `numpy.ndarray`

    `noisequadsum` : `numpy.ndarray`

    Notes
    -----
    All attributes are cached so these must not be altered since the
    quantities are not recalculated.
    """

    def __init__(self, signals, wavelengths=None, shotnoise=True,
                 constnoise=False):
        # If only a single signal is given wrap it inside a list.
        if isinstance(signals, np.ndarray):
            signals = [signals]

        # Convert each signal to a SynSignal and check that all shapes are
        # equal
        if isinstance(signals, (list, tuple)):
            _signals = [None] * len(signals)
            for i in range(len(signals)):
                # This checks if it was a numpy ndarray and 1D
                _signals[i] = SynSignal(signals[i])
                # Check that the shape is the same
                if _signals[i].signal.shape != _signals[0].signal.shape:
                    thisshape = _signals[i].signal.shape
                    refshape = _signals[0].signal.shape
                    raise ValueError('Signals must have the same shape'
                                     'Got {0} and {1}.'
                                     ''.format(thisshape[0], refshape[0]))
        else:
            raise TypeError('Signals must be a numpy array or a list/tuple. '
                            'Must not be a {0}'
                            ''.format(signals.__class__.__name__))

        # Check wavelengths
        if wavelengths is None:
            wavelengths = np.arange(_signals[0].signal.size)
        elif not isinstance(wavelengths, np.ndarray):
            raise TypeError('Wavelengths must be a numpy ndarray or None.')
        elif _signals[0].signal.shape != wavelengths.shape:
            raise ValueError('Wavelength must have the same shape as the '
                             'signal. Got {0} and {1}'
                             ''.format(wavelengths.shape,
                                       _signals[0].signal.shape[0]))

        # Shotnoise must be a boolean.
        if not isinstance(shotnoise, bool):
            raise TypeError('Shotnoise must be of boolean type.')

        # Check constnoise for correct type
        if (not isinstance(constnoise, (int, float)) and
                constnoise is not False):
            raise TypeError('Constnoise must be an integer/float or False.')

        self._signals = _signals
        self._wavelengths = wavelengths
        self._shotnoise = shotnoise
        self._constnoise = constnoise

    @lazyproperty_readonly
    def signals(self):
        """
        The different signals in the spectrum.
        """
        return self._signals

    @lazyproperty_readonly
    def wavelengths(self):
        """
        Containing the wavelengths for the signal(s).
        """
        return self._wavelengths

    @lazyproperty_readonly
    def shotnoise(self):
        """
        If shotnoise is included in the synthetic spectrum.
        """
        return self._shotnoise

    @lazyproperty_readonly
    def constnoise(self):
        """
        The standard deviation of the simulated constant noise.
        """
        return self._constnoise

    @lazyproperty_readonly
    def constnoise_array(self):
        """
        The array that is created to simulate constant noise. If
        ``constnoise`` is 0 or ``False`` this will only containts zeroes.

        Computed with :func:`numpy.random.normal` with mean value 0.
        """
        elements = self.signals[0].signal.size
        if self.constnoise:
            return np.random.normal(0, self.constnoise, elements)
        else:
            return np.zeros(elements)

    @lazyproperty_readonly
    def signalsum(self):
        """
        The sum of the signals without noise.
        """
        signalsum = 0

        # Each single signal added
        for i in self.signals:
            signalsum = signalsum + i.signal

        return signalsum

    @lazyproperty_readonly
    def signalsum_with_noise(self):
        """
        The sum of the signals with noise.
        """
        signalsum_noise = 0

        # Shot noise
        if self.shotnoise:
            for i in self.signals:
                signalsum_noise = signalsum_noise + i.signal_with_noise
        else:
            signalsum_noise = self.signalsum

        # Constant noise
        signalsum_noise = signalsum_noise + self.constnoise_array

        return signalsum_noise

    @lazyproperty_readonly
    def signalsum_norm_with_noise(self):
        """
        The normalized signalsum with noise.

        The sum of the signals with noise divided by the signalsum to yield
        the normalized spectrum with noise.
        """
        return self.signalsum_with_noise / self.signalsum

    @lazyproperty_readonly
    def noisequadsum(self):
        """
        The square root of the sum of the theoretical noise standard
        deviations squared.
        """
        noisequadsum = 0

        # Shot noise
        if self.shotnoise:
            for i in self.signals:
                noisequadsum = noisequadsum + i.noise ** 2

        # Constant noise
        noisequadsum = noisequadsum + self.constnoise ** 2

        return np.sqrt(noisequadsum)

    def plot(self):
        """
        Visualize the different signals and resulting spectrum.

        Notes
        -----
        1. With more than 5 different signals the colors in the plot will
           be repeated so for visual comprehension having 5 or less
           signals is most effective.

        2. If any signal or the spectrum has any noise component the
           theoretical signal is also plotted with the same color but as
           dashed line.
        """
        colors = ('b', 'g', 'r', 'c', 'm')
        # Plot each signal and if shot noise is enabled also the signal with
        # noise.
        for i in range(len(self.signals)):
            plt.plot(self.wavelengths,
                     self.signals[i].signal,
                     '--', color=colors[i % 5],
                     label='Signal {0}'.format(i))
            if self.shotnoise:
                plt.plot(self.wavelengths,
                         self.signals[i].signal_with_noise,
                         '-', color=colors[i % 5],
                         label='Signal {0} with noise'.format(i))

        # Plot the constant noise if there is one
        if self.constnoise:
            plt.plot(self.wavelengths,
                     self.constnoise_array,
                     '-', color='y',
                     label='Constant noise')
        # Plot the theoretical sum of all the signals (without noise) but only
        # if there is more than one signal
        if (len(self.signals) > 1 or
                (len(self.signals) == 1 and self.constnoise)):
            plt.plot(self.wavelengths, self.signalsum,
                     '--', color='k',
                     label='Sum of Signals')
            # Plot the sum of the signals with noise if any noise source is
            # present.
            if self.shotnoise is not False or self.constnoise is not False:
                plt.plot(self.wavelengths,
                         self.signalsum_with_noise,
                         '-', color='k',
                         label='Sum of Signals with Noise')

        plt.xlim(np.min(self.wavelengths), 1.5*np.max(self.wavelengths))
        plt.xlabel('Pixel / Wavelength')
        plt.ylabel('Counts')
        plt.title('Synthetic Spectrum and individual signals')

        plt.legend()
        plt.show()

    def plot_signalsum(self):
        """
        Plots the signalsum and the signalsum with noise and the region where
        the one sigma and three sigma regions are.
        """
        plt.fill_between(self.wavelengths,
                         self.signalsum-self.noisequadsum,
                         self.signalsum+self.noisequadsum,
                         facecolor='green', alpha=0.3,
                         label='1 sigma deviation')
        plt.fill_between(self.wavelengths,
                         self.signalsum-3*self.noisequadsum,
                         self.signalsum+3*self.noisequadsum,
                         facecolor='red', alpha=0.1,
                         label='3 sigma deviation')
        plt.scatter(self.wavelengths, self.signalsum_with_noise,
                    label="measured signal")
        plt.plot(self.wavelengths, self.signalsum,
                 label="theoretical signal")
        plt.title('Signalsum with Noise')
        plt.ylabel('Counts')
        plt.xlabel('Wavelength / Pixel')
        plt.legend()
        plt.show()
