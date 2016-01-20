# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np
from numpy.testing import assert_equal, assert_almost_equal, assert_array_equal
from pytest import raises

from .. import DER_SNR, RMSE_SNR


def test_der_snr_exception_to_few_values():
    signal_with_noise = np.array([5, 1, 5, 2])
    with raises(ValueError):
        DER_SNR(signal_with_noise)


def test_der_snr():
    signal_with_noise = np.array([5, 1, 2, 3, 1, 4, 6])
    snr_calc = DER_SNR(signal_with_noise)
    assert_almost_equal(snr_calc.signal, 3)
    assert_almost_equal(snr_calc.noise, 1.2105394)
    assert_almost_equal(snr_calc.snr, 3/1.2105394)

    signal_with_noise2 = np.array([0, 0, 0, 5, 1, 2, 3, 1, 4, 6])
    snr_calc2 = DER_SNR(signal_with_noise2)
    assert_array_equal(snr_calc2._flux, signal_with_noise)
    assert_almost_equal(snr_calc2.signal, snr_calc.signal)
    assert_almost_equal(snr_calc2.noise, snr_calc.noise)
    assert_almost_equal(snr_calc2.snr, snr_calc.snr)

    signal_with_noise3 = np.array([np.inf, np.nan, -np.inf,
                                   5, 1, 2, 3, 1, 4, 6])
    snr_calc3 = DER_SNR(signal_with_noise3)
    assert_array_equal(snr_calc3._flux, signal_with_noise)
    assert_almost_equal(snr_calc3.signal, snr_calc.signal)
    assert_almost_equal(snr_calc3.noise, snr_calc.noise)
    assert_almost_equal(snr_calc3.snr, snr_calc.snr)

    signal_with_noise4 = np.array([214, 18, 14, 5, 1, 2, 3, 1, 4, 6])
    mask4 = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0])
    signal_with_noise4 = np.ma.array(signal_with_noise4, mask=mask4)
    snr_calc4 = DER_SNR(signal_with_noise4)
    assert_array_equal(snr_calc4._flux, signal_with_noise)
    assert_almost_equal(snr_calc4.signal, snr_calc.signal)
    assert_almost_equal(snr_calc4.noise, snr_calc.noise)
    assert_almost_equal(snr_calc4.snr, snr_calc.snr)


def test_der_snr_delete_zeros():
    signal = np.array([5, 1, 5, 2, 1, 0, 5])
    snr1 = DER_SNR(signal)
    assert len(signal) == 7
    assert signal[5] == 0
    assert len(snr1._flux) == 6
    assert snr1._flux[5] == 5
    snr2 = DER_SNR(signal, ignore_zeros=False)
    assert len(signal) == 7
    assert len(snr2._flux) == 7
    assert_equal(snr2._flux[5], 0)
    assert_array_equal(signal, snr2._flux)


def test_der_snr_delete_nan():
    signal = np.array([5, 1, 5, 2, 1, np.nan, 5])
    snr1 = DER_SNR(signal)
    assert len(signal) == 7
    assert_equal(signal[5], np.nan)
    assert len(snr1._flux) == 6
    assert snr1._flux[5] == 5
    snr2 = DER_SNR(signal, ignore_nan_inf=False)
    assert len(signal) == 7
    assert len(snr2._flux) == 7
    assert_equal(snr2._flux[5], np.nan)
    assert_array_equal(signal, snr2._flux)


def test_der_snr_delete_inf():
    signal = np.array([5, 1, 5, 2, 1, np.inf, 5])
    snr1 = DER_SNR(signal)
    assert len(signal) == 7
    assert_equal(signal[5], np.inf)
    assert len(snr1._flux) == 6
    assert snr1._flux[5] == 5
    snr2 = DER_SNR(signal, ignore_nan_inf=False)
    assert len(signal) == 7
    assert len(snr2._flux) == 7
    assert_equal(snr2._flux[5], np.inf)
    assert_array_equal(signal, snr2._flux)


def test_der_snr_delete_masked():
    signal = np.ma.array([5, 1, 5, 2, 1, 100, 5], mask=[0, 0, 0, 0, 0, 1, 0])
    snr1 = DER_SNR(signal)
    assert len(signal) == 7
    assert_equal(signal.data[5], 100)
    assert len(snr1._flux) == 6
    assert snr1._flux[5] == 5
    snr2 = DER_SNR(signal, ignore_masked=False)
    assert len(signal) == 7
    assert len(snr2._flux) == 7
    assert_equal(snr2._flux[5], 100)
    assert_array_equal(signal.data, snr2._flux)


def test_rmse_snr():
    signal_with_noise = np.array([5, 1, 2, 3, 1, 4, 6])
    snr_calc = RMSE_SNR(3, signal_with_noise)
    assert_almost_equal(snr_calc.signal, 3)
    assert_almost_equal(snr_calc.noise, 1.8126539343499315)
    assert_almost_equal(snr_calc.snr, 3/1.8126539343499315)

    signal_with_noise2 = np.array([0, 0, 0, 5, 1, 2, 3, 1, 4, 6])
    snr_calc2 = RMSE_SNR(3, signal_with_noise2)
    assert_array_equal(snr_calc2._flux, signal_with_noise)
    assert_almost_equal(snr_calc2.signal, snr_calc.signal)
    assert_almost_equal(snr_calc2.noise, snr_calc.noise)
    assert_almost_equal(snr_calc2.snr, snr_calc.snr)

    signal_with_noise3 = np.array([np.inf, np.nan, -np.inf,
                                   5, 1, 2, 3, 1, 4, 6])
    snr_calc3 = RMSE_SNR(3, signal_with_noise3)
    assert_array_equal(snr_calc3._flux, signal_with_noise)
    assert_almost_equal(snr_calc3.signal, snr_calc.signal)
    assert_almost_equal(snr_calc3.noise, snr_calc.noise)
    assert_almost_equal(snr_calc3.snr, snr_calc.snr)

    signal_with_noise4 = np.array([214, 18, 14, 5, 1, 2, 3, 1, 4, 6])
    mask4 = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0])
    signal_with_noise4 = np.ma.array(signal_with_noise4, mask=mask4)
    snr_calc4 = RMSE_SNR(3, signal_with_noise4)
    assert_array_equal(snr_calc4._flux, signal_with_noise)
    assert_almost_equal(snr_calc4.signal, snr_calc.signal)
    assert_almost_equal(snr_calc4.noise, snr_calc.noise)
    assert_almost_equal(snr_calc4.snr, snr_calc.snr)


def test_rmse_snr_delete_zeros():
    signal = np.array([5, 1, 5, 2, 1, 0, 5])
    snr1 = RMSE_SNR(1, signal)
    assert len(signal) == 7
    assert signal[5] == 0
    assert len(snr1._flux) == 6
    assert snr1._flux[5] == 5
    snr2 = RMSE_SNR(1, signal, ignore_zeros=False)
    assert len(signal) == 7
    assert len(snr2._flux) == 7
    assert_equal(snr2._flux[5], 0)
    assert_array_equal(signal, snr2._flux)


def test_rmse_snr_delete_nan():
    signal = np.array([5, 1, 5, 2, 1, np.nan, 5])
    snr1 = RMSE_SNR(1, signal)
    assert len(signal) == 7
    assert_equal(signal[5], np.nan)
    assert len(snr1._flux) == 6
    assert snr1._flux[5] == 5
    snr2 = RMSE_SNR(1, signal, ignore_nan_inf=False)
    assert len(signal) == 7
    assert len(snr2._flux) == 7
    assert_equal(snr2._flux[5], np.nan)
    assert_array_equal(signal, snr2._flux)


def test_rmse_snr_delete_inf():
    signal = np.array([5, 1, 5, 2, 1, np.inf, 5])
    snr1 = RMSE_SNR(1, signal)
    assert len(signal) == 7
    assert_equal(signal[5], np.inf)
    assert len(snr1._flux) == 6
    assert snr1._flux[5] == 5
    snr2 = RMSE_SNR(1, signal, ignore_nan_inf=False)
    assert len(signal) == 7
    assert len(snr2._flux) == 7
    assert_equal(snr2._flux[5], np.inf)
    assert_array_equal(signal, snr2._flux)


def test_rmse_snr_delete_masked():
    signal = np.ma.array([5, 1, 5, 2, 1, 100, 5], mask=[0, 0, 0, 0, 0, 1, 0])
    snr1 = RMSE_SNR(1, signal)
    assert len(signal) == 7
    assert_equal(signal.data[5], 100)
    assert len(snr1._flux) == 6
    assert snr1._flux[5] == 5
    snr2 = RMSE_SNR(1, signal, ignore_masked=False)
    assert len(signal) == 7
    assert len(snr2._flux) == 7
    assert_equal(snr2._flux[5], 100)
    assert_array_equal(signal.data, snr2._flux)
