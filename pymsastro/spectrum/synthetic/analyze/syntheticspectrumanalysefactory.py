# Licensed under a 3-clause BSD style license - see LICENSE.rst

from .syntheticspectrumanalyse import (AddSynSpectrumAnalyseNone,
    AddSynSpectrumAnalyseSpectrum1, AddSynSpectrumAnalyseSpectrum2,
    AddSynSpectrumAnalyseDERIErr, AddSynSpectrumAnalyseDERIVar,
    AddSynSpectrumAnalyseDEROptimal, AddSynSpectrumAnalyseDERSNRSquared,
    AddSynSpectrumAnalyseIdealIErr, AddSynSpectrumAnalyseIdealIVar,
    AddSynSpectrumAnalyseIdealOptimal, AddSynSpectrumAnalyseIdealSNRSquared)

from .. import SynSpectrum
from ....utils import json_write, json_read

import numpy as np
import matplotlib.pyplot as plt

__all__ = ['AnalyseFactory']


class AnalyseFactory(object):
    """
    Uses all the different Analysation classes to compute the goodness for
    different spectra.

    Parameters
    ----------
    signals1, signals2 : `numpy.ndarray` or a `list`, `tuple` of these
        The signals that are converted to a
        `~pymsastro.spectrum.synthetic.SynSpectrum`.
    """

    def __init__(self, signals1, signals2):
        # Create instances of all the different analysation classes.
        self._analyse = [AddSynSpectrumAnalyseNone(),
                         AddSynSpectrumAnalyseSpectrum1(),
                         AddSynSpectrumAnalyseSpectrum2(),
                         AddSynSpectrumAnalyseDERIErr(),
                         AddSynSpectrumAnalyseDERIVar(),
                         AddSynSpectrumAnalyseDEROptimal(),
                         AddSynSpectrumAnalyseDERSNRSquared(),
                         AddSynSpectrumAnalyseIdealIErr(),
                         AddSynSpectrumAnalyseIdealIVar(),
                         AddSynSpectrumAnalyseIdealOptimal(),
                         AddSynSpectrumAnalyseIdealSNRSquared()]

        if isinstance(signals1, np.ndarray):
            self._signals1 = [signals1]
        elif isinstance(signals1, (list, tuple)):
            for i in signals1:
                if not isinstance(i, np.ndarray):
                    raise TypeError('each signal must be a np.array not a {0}.'
                                    ''.format(i.__class__.__name__))
            self._signals1 = signals1
        else:
            raise TypeError('signal1 must be a np.ndarray or a list/tuple '
                            'of those not a {0}.'
                            ''.format(signals1.__class__.__name__))

        if isinstance(signals2, np.ndarray):
            self._signals2 = [signals2]
        elif isinstance(signals2, (list, tuple)):
            for i in signals2:
                if not isinstance(i, np.ndarray):
                    raise TypeError('each signal must be a np.array not a {0}.'
                                    ''.format(i.__class__.__name__))
            self._signals2 = signals2
        else:
            raise TypeError('signal2 must be a np.ndarray or a list/tuple '
                            'of those not a {0}.'
                            ''.format(signals2.__class__.__name__))

    @property
    def result(self):
        """
        Returns all the results of the `AddSynSpectrumAnalyse` classes used in
        this factory.

        Currently all avaiable factories are used.
        """
        if hasattr(self, '_result'):
            return self._result
        tmp = {}
        for i in self._analyse:
            tmp[i.identification] = i.result
        return tmp

    def run_analysation(self, factor1, factor2, readnoise1, readnoise2):
        """
        Runs the analysation.

        Parameters
        ----------
        factor1, factor2 : ``Number`` or `numpy.ndarray`
            Each signal of spectrum 1 will be multiplied by factor1, and
            the same for spectrum 2 with factor2.

        readnoise1, readnoise2 : ``Number`` or `numpy.ndarray`
            Spectrum 1 will have a readnoise standard deviation given by
            readnoise1 and the same for readnoise2 and spectrum 2.

        Raises
        ------
        ValueError

            - Class contains a loaded result (class was created with
              `fromfile`)
            - Parameter were not the same size (or Numbers)

        Notes
        -----
        If no parameter is a `~numpy.ndarray` then the analysation will only
        run once with the numbers given.

        If one parameter is a `~numpy.ndarray` then the analysation will run as
        many times as there are elements in the array. The other parameters
        will be the same for each run.

        If more than one parameteris a `~numpy.ndarray` then the analysation
        will run as many times as there are elements in the array. So each
        array must contain the same number of elements. And in the ``i-th``
        run the ``i-th`` element of each array is used. Parameters given as
        ``Number`` will stay the same for each run.
        """
        if hasattr(self, '_result'):
            raise ValueError('Class already contains a loaded result.')
        # One run is the minimum.
        runs = 1

        self._static = {}
        self._variable = {}

        # Check how many runs have to be done.
        if isinstance(factor1, np.ndarray):
            runs = factor1.size
            self._variable['factor1'] = factor1.tolist()
        else:
            self._static['factor1'] = factor1

        if isinstance(factor2, np.ndarray):
            if runs != 1 and runs != factor2.size:
                raise ValueError('Arrays have different sizes.')
            elif runs == 1:
                runs = factor2.size
            self._variable['factor2'] = factor2.tolist()
        else:
            self._static['factor2'] = factor2

        if isinstance(readnoise1, np.ndarray):
            if runs != 1 and runs != readnoise1.size:
                raise ValueError('Arrays have different sizes.')
            elif runs == 1:
                runs = readnoise1.size
            self._variable['readnoise1'] = readnoise1.tolist()
        else:
            self._static['readnoise1'] = readnoise1

        if isinstance(readnoise2, np.ndarray):
            if runs != 1 and runs != readnoise2.size:
                raise ValueError('Arrays have different sizes.')
            elif runs == 1:
                runs = readnoise2.size
            self._variable['readnoise2'] = readnoise2.tolist()
        else:
            self._static['readnoise2'] = readnoise2

        self._static['runs'] = runs

        # Make the numbers to ndarrays so that we can iterate easily
        if not isinstance(factor1, np.ndarray):
            factor1 = np.ones(runs) * factor1
        if not isinstance(factor2, np.ndarray):
            factor2 = np.ones(runs) * factor2
        if not isinstance(readnoise1, np.ndarray):
            readnoise1 = np.ones(runs) * readnoise1
        if not isinstance(readnoise2, np.ndarray):
            readnoise2 = np.ones(runs) * readnoise2

        # Make the runs
        for run in range(runs):

            signal1_tmp = []
            for i in self._signals1:
                if 'elements' not in self._static:
                    self._static['elements'] = i.size
                signal1_tmp.append(i * factor1[run])

            signal2_tmp = []
            for i in self._signals2:
                signal2_tmp.append(i * factor2[run])

            spec1 = SynSpectrum(signal1_tmp, None, True, readnoise1[run])
            spec2 = SynSpectrum(signal2_tmp, None, True, readnoise2[run])

            for i in self._analyse:
                i.analyze(spec1, spec2)

    def save(self, filename):
        """
        Saves the results in a JSON file.

        Parameters
        ----------
        filename : `str`
            The filename of the saved results.
        """
        res = {'result': self.result,
               'variable': self._variable,
               'static': self._static,
               'signal1': [],
               'signal2': []}
        for i in self._signals1:
            res['signal1'].append(i.tolist())
        for i in self._signals2:
            res['signal2'].append(i.tolist())
        json_write(filename, res)

    @classmethod
    def fromfile(cls, filename):
        """
        Loads a previously saved result and return an instance with the
        results.

        Parameters
        ----------
        filename : `str`
            The filename of the saved results.

        Returns
        -------
        instance: `AddSynSpectrumAnalyseFactory`
            An instance with the previously saved results from the JSON file.

        Notes
        -----
        The returned instance is not useable for another analysation.
        """
        prev = json_read(filename)
        signal1 = []
        for i in prev['signal1']:
            signal1.append(np.array(i))
        signal2 = []
        for i in prev['signal2']:
            signal2.append(np.array(i))
        new = cls(signal1, signal2)
        new._result = prev['result']
        new._static = prev['static']
        new._variable = prev['variable']
        return new

    def plot(self, measure='rmse_snr', include_ideal=True, include_der=False):
        """
        Plot the results. Requires an analysation run before!

        Parameters
        ----------
        measure : `str`, optional
            The measured quantity to plot. Can be:

            - ``"der_signal"``: signal measured with `~pymsastro.stats.DER_SNR`
            - ``"der_noise"``: noise measured with `~pymsastro.stats.DER_SNR`
            - ``"der_snr"``: signal-to-noise-ratio measured with
              `~pymsastro.stats.DER_SNR`
            - ``"rmse_noise"``: noise measured with `~pymsastro.stats.RMSE_SNR`
            - ``"rmse_snr"``: signal-to-noise-ratio measured with
              `~pymsastro.stats.RMSE_SNR`

            Default is ``"rmse_snr"``.

        include_ideal : `bool`, optional
            Plot the added spectra where weighting was determined using
            the theoretical values.
            Default is ``True``.

        include_der : `bool`, optional
            Plot the added spectra where weighting was determined using
            `~pymsastro.stats.DER_SNR`.
            Default is ``False``.

        Raises
        ------
        ValueError

            - ``measure`` was a not recognized type
            - `run_analysation` was not invoked or had only one run.
        """
        # Check measure is an allowed string
        if measure not in ['der_signal', 'der_noise', 'der_snr',
                           'rmse_noise', 'rmse_snr']:
            raise ValueError('Unknown measure.')

        # Always plot Spec1 and Spec2 and None
        what_to_plot = ['Spec1          ', 'Spec2          ',
                        'None           ']

        # Include ideal weighting
        if include_ideal:
            tmp = []
            for i in self.result:
                if i.startswith('Ideal'):
                    tmp.append(i)
            what_to_plot += sorted(tmp)

        # Include DER weighting
        if include_der:
            tmp = []
            for i in self.result:
                if i.startswith('DER'):
                    tmp.append(i)
            what_to_plot += sorted(tmp)

        # Setup visual enhancing through colors and style
        counter = 0
        colors = ('k', 'b', 'g', 'r', 'c', 'm', 'y')
        style= ('-', '--')

        # Uncomment the following line and comment the one later to use xkcd
        # stype plots :-)
        # with plt.xkcd():
        if True:
            # We have stored the variable parameters inside it and can use them
            # here. Make sure it is not empty!
            if self._variable:
                x_label = ''
                for i in self._variable:
                    if x_label != '':
                        x_label = ('{0} & ({1} Range: {2} - {3})'
                                   ''.format(x_label, i, self._variable[i][0],
                                             self._variable[i][-1]))
                    else:
                        x_axis = np.array(self._variable[i])
                        x_label = str(i)
            else:
                raise ValueError('Please do an analysation run with more than'
                                 'one iteration before plotting.')
            for i in what_to_plot:

                data = np.array(self.result[i][measure])

                if i.startswith('Spec'):
                    plt.fill_between(x_axis, 0, data,
                                     facecolor='black', alpha=0.3)
                else:
                    plt.plot(x_axis, data,
                             style[int(counter/7)],
                             color=colors[counter % 7],
                             label=i)
                    counter += 1
            # plt.yscale('log')
            plt.xlabel(x_label)
            plt.ylabel(measure)
            plt.legend(loc=4)
            plt.title(str(self._static))
            plt.show()
