# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np
import matplotlib.pyplot as plt

from ..utils import lazyproperty_readonly

__all__ = ['SynSignal']


class SynSignal(object):
    """
    Container for a synthetic signal.

    Parameters
    ----------
    signal : `numpy.ndarray`
        The synthetic signal

    Raises
    ------
    TypeError
        - If signal is not a `numpy.ndarray`
        - If signal shape is not 1D.

    Attributes
    ----------
    `signal` : `numpy.ndarray`

    `signal_with_noise` : `numpy.ndarray`

    `noise` : `numpy.ndarray`

    Notes
    -----
    The attributes must not be altered since they are cached and the quantities
    will not be recalculated!
    """
    def __init__(self, signal):
        # Signal must be a numpy ndarray
        if isinstance(signal, np.ndarray):
            if signal.ndim != 1:
                raise TypeError('Signal must be 1D not {0}D.'
                                ''.format(signal.ndim))
            self._signal = signal
        else:
            raise TypeError('Signal must be a numpy array not {0}.'
                            ''.format(signal.__class__.__name__))

    @lazyproperty_readonly
    def signal(self):
        """
        The synthetic signal without noise.
        """
        return self._signal

    @lazyproperty_readonly
    def signal_with_noise(self):
        """
        The synthetic signal with shotnoise.

        The synthetic signal passed through :func:`numpy.random.poisson` to
        simulate shot noise (photon noise). The result will only
        contain integers.
        """
        return np.random.poisson(self.signal)

    @lazyproperty_readonly
    def noise(self):
        """
        The computed theoretical noise.

        The theoretical noise associated with the signal. For
        shot noise this is simply the square root of the signal
        without noise.
        """
        return np.sqrt(self.signal)

    def plot(self):
        """
        Plots the signal and the signal with noise and the region where the
        one sigma and three sigma regions are.
        """
        elements = self.signal.size
        plt.fill_between(np.arange(elements),
                         self.signal-self.noise,
                         self.signal+self.noise,
                         facecolor='green', alpha=0.3,
                         label='1 sigma deviation')
        plt.fill_between(np.arange(elements),
                         self.signal-3*self.noise,
                         self.signal+3*self.noise,
                         facecolor='red', alpha=0.1,
                         label='3 sigma deviation')
        plt.scatter(np.arange(elements), self.signal_with_noise,
                    label="measured signal")
        plt.plot(np.arange(elements), self.signal, label="theoretical signal")
        plt.title('Signal with Noise')
        plt.ylabel('Counts')
        plt.xlabel('Wavelength / Pixel')
        plt.legend()
        plt.show()
