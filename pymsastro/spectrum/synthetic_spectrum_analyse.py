# Licensed under a 3-clause BSD style license - see LICENSE.rst

from operator import itemgetter

from .synthetic_spectrum import addSynSpec
from ..stats import der_snr, snr_rmse

__all__ = ['SynSpecAddAnalyse']


class SynSpecAddAnalyse(object):
    """
    Adds two synthetic spectra with every combination of weights.

    Parameters
    ----------
    spectrum1, spectrum2: ``SynSpec`` instances
        The first and second synthetic spectrum
    """
    def __init__(self, spectrum1, spectrum2):
        self._spectrum1 = spectrum1
        self._spectrum2 = spectrum2
        self._signal = []
        self._noise = []
        self._snr = []
        self._rms_rel = []
        self._rms_abs = []

    def _add(self):
        spectrum1 = self._spectrum1
        spectrum2 = self._spectrum2

        added = {}

        added['Spec1          '] = (spectrum1.signalsum_with_noise,
                                    spectrum1.signalsum)
        added['Spec2          '] = (spectrum2.signalsum_with_noise,
                                    spectrum2.signalsum)
        added['None           '] = addSynSpec(spectrum1, spectrum2,
                                              weighting=None,
                                              s_n_calc_func=None)
        added['Ideal - Optimal'] = addSynSpec(spectrum1, spectrum2,
                                              weighting='optimal',
                                              s_n_calc_func=None)
        added['Ideal - SNR^2  '] = addSynSpec(spectrum1, spectrum2,
                                              weighting='snr_squared',
                                              s_n_calc_func=None)
        added['Ideal - 1/N^2  '] = addSynSpec(spectrum1, spectrum2,
                                              weighting='ivar',
                                              s_n_calc_func=None)
        added['Ideal - 1/N    '] = addSynSpec(spectrum1, spectrum2,
                                              weighting='ierr',
                                              s_n_calc_func=None)
        added['DER   - Optimal'] = addSynSpec(spectrum1, spectrum2,
                                              weighting='optimal',
                                              s_n_calc_func=der_snr)
        added['DER   - SNR^2  '] = addSynSpec(spectrum1, spectrum2,
                                              weighting='snr_squared',
                                              s_n_calc_func=der_snr)
        added['DER   - 1/N^2  '] = addSynSpec(spectrum1, spectrum2,
                                              weighting='ivar',
                                              s_n_calc_func=der_snr)
        added['DER   - 1/N    '] = addSynSpec(spectrum1, spectrum2,
                                              weighting='ierr',
                                              s_n_calc_func=der_snr)

        self._added = added

    def _analyze(self, name, result_and_reference):
        result = result_and_reference[0]
        reference = result_and_reference[1]

        # Calculate S and N with DER ###################
        signal_der, noise_der = der_snr(result)

        # Calculate SNR with DER ###################
        snr_der = signal_der / noise_der

        # Calculate rel and abs RMSE ###################
        rms_abs, rms_rel = snr_rmse(result, reference)

        self._signal[name].append(signal_der)
        self._noise[name].append(noise_der)
        self._snr[name].append(snr_der)
        self._rms_rel[name].append(rms_rel)
        self._rms_abs[name].append(rms_abs)

    def _print_results(self):

        # Print dicts in sorted order ###################
        headline = '-'*20 + '{0}' + '-'*20

        print(headline.format('SNR'))
        sorted_snr = sorted(snr_der.items(), key=itemgetter(1),
                            reverse=True)
        for i in range(len(sorted_snr)):
            print('{0} SNR: {1}'.format(sorted_snr[i][0],
                                        sorted_snr[i][1]))

        print(headline.format('relative RMSE'))
        sorted_rmsrel = sorted(rms_rel.items(), key=itemgetter(1))
        for i in range(len(sorted_rmsrel)):
            print('{0} rel RMSE: {1}'.format(sorted_rmsrel[i][0],
                                             sorted_rmsrel[i][1]))

        print(headline.format('Signal'))
        sorted_signal = sorted(signal_der.items(), key=itemgetter(1),
                               reverse=True)
        for i in range(len(sorted_signal)):
            print('{0} Signal: {1}'.format(sorted_signal[i][0],
                                           sorted_signal[i][1]))

        print(headline.format('Noise'))
        sorted_noise = sorted(noise_der.items(), key=itemgetter(1))
        for i in range(len(sorted_noise)):
            print('{0} Noise: {1}'.format(sorted_noise[i][0],
                                          sorted_noise[i][1]))

        print(headline.format('absolute RMSE'))
        sorted_rmsabs = sorted(rms_abs.items(), key=itemgetter(1))
        for i in range(len(sorted_rmsabs)):
            print('{0} abs RMSE: {1}'.format(sorted_rmsabs[i][0],
                                             sorted_rmsabs[i][1]))

        return signal_der, noise_der, snr_der, rms_rel, rms_abs
