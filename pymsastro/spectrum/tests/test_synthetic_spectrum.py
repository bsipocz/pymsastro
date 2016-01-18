# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal
from pytest import raises

from .. import SynSignal, SynSpectrum


def test_synsignal_not_ndarray():
    a = [1, 2, 3]  # list
    with raises(TypeError):
        SynSignal(a)

    a = (1, 2, 3)  # tuple
    with raises(TypeError):
        SynSignal(a)

    a = 5  # number
    with raises(TypeError):
        SynSignal(a)


def test_synsignal_multidim_ndarray():
    a = np.array([1, 2, 3]).reshape(1, 3)  # 2d
    with raises(TypeError):
        SynSignal(a)

    a = np.ones((2, 2, 2))  # 3d
    with raises(TypeError):
        SynSignal(a)

    a = np.ones((2, 2, 2, 2))  # 4d
    with raises(TypeError):
        SynSignal(a)

    a = np.array(2)  # Scalar
    with raises(TypeError):
        SynSignal(a)


def test_synsignal():
    a = np.array([1, 2, 3])
    sig = SynSignal(a)
    assert_array_equal(sig.signal, a)
    assert_array_equal(sig.noise, np.sqrt(a))


def test_synspec_wrong_signal():

    # Same test cases as with Signal but this time for spectrum

    a = [1, 2, 3]
    with raises(TypeError):
        SynSpectrum(a)

    a = (1, 2, 3)
    with raises(TypeError):
        SynSpectrum(a)

    a = 5
    with raises(TypeError):
        SynSpectrum(a)

    a = np.array([1, 2, 3]).reshape(1, 3)
    with raises(TypeError):
        SynSpectrum(a)

    a = np.ones((2, 2, 2))
    with raises(TypeError):
        SynSpectrum(a)

    a = np.ones((2, 2, 2, 2))
    with raises(TypeError):
        SynSpectrum(a)

    a = np.array(2)
    with raises(TypeError):
        SynSpectrum(a)


def test_synspec_different_signal_size():

    a = np.array([1, 2, 3])  # 3 size
    b = np.array([1, 2, 3, 4])  # 4 size
    with raises(ValueError):
        SynSpectrum([a, b])


def test_synspec_signals_not_correct_type():

    a = np.array([1, 2, 3])
    b = np.array([1, 2, 3, 4])
    with raises(TypeError):
        SynSpectrum({1: a, 2: b})  # dict


def test_synspec_wavelength_not_correct_size():

    a = np.array([1, 2, 3])  # size 3
    b = np.array([1, 2, 3, 4])  # size 4
    with raises(ValueError):
        SynSpectrum(a, b)


def test_synspec_wavelength_not_correct_type():

    a = np.array([1, 2, 3])
    b = [1, 2, 3, 4]  # list
    with raises(TypeError):
        SynSpectrum(a, b)


def test_synspec_shotnoise_not_correct_type():

    a = np.array([1, 2, 3])
    with raises(TypeError):
        SynSpectrum(a, shotnoise=5)  # int

    a = np.array([1, 2, 3])
    with raises(TypeError):
        SynSpectrum(a, shotnoise=[True, True])  # list of bool


def test_synspec_constnoise_not_correct_type():

    a = np.array([1, 2, 3])
    with raises(TypeError):
        SynSpectrum(a, constnoise=[1, 2])  # list


def test_synspec():
    # Test all combinations I can think of right now (ndarray, list, tuple)
    possible_signals = (np.array([1, 2, 3]),
                        [np.array([1, 2, 3]), np.array([1, 2, 3])],
                        (np.array([1, 2, 3]), np.array([1, 2, 3]),
                         np.array([1, 2, 3])))
    for signal in possible_signals:
        for wave in [np.array([1, 2, 3]), None]:
            for shotnoise in [True, False]:
                for constnoise in [False, 1, 2.5, 5]:
                    SynSpectrum(signal, wave, shotnoise, constnoise)


def test_synspec_sums_0():
    # Test all combinations I can think of right now (ndarray, list, tuple)
    signals = ((np.array([1, 2, 3]), np.array([1, 2, 3]), np.array([1, 2, 3])))
    spec = SynSpectrum(signals, shotnoise=False, constnoise=False)
    added_signals = signals[0]+signals[1]+signals[2]
    assert_array_equal(added_signals, spec.signalsum)
    assert_array_equal(added_signals, spec.signalsum_with_noise)
    assert_array_equal(np.zeros(3), spec.noisequadsum)
    assert_array_equal(np.ones(3), spec.signalsum_norm_with_noise)


def test_synspec_sums_1():
    # Test all combinations I can think of right now (ndarray, list, tuple)
    signals = ((np.array([1, 2, 3]), np.array([1, 2, 3]), np.array([1, 2, 3])))
    spec = SynSpectrum(signals, shotnoise=False, constnoise=2)
    added_signals = signals[0]+signals[1]+signals[2]
    assert_array_equal(added_signals, spec.signalsum)
    assert_array_equal(added_signals + spec.constnoise_array,
                       spec.signalsum_with_noise)
    assert_array_equal(np.ones(3)*2, spec.noisequadsum)
    assert_array_equal((added_signals + spec.constnoise_array) / added_signals,
                       spec.signalsum_norm_with_noise)


def test_synspec_sums_2():
    # Test all combinations I can think of right now (ndarray, list, tuple)
    signals = ((np.array([1, 2, 3]), np.array([1, 2, 3]), np.array([1, 2, 3])))
    spec = SynSpectrum(signals, shotnoise=True, constnoise=False)
    added_signals = signals[0]+signals[1]+signals[2]
    added_signals_noisy = (spec.signals[0].signal_with_noise +
                           spec.signals[1].signal_with_noise +
                           spec.signals[2].signal_with_noise)
    assert_array_equal(added_signals, spec.signalsum)
    assert_array_equal(added_signals_noisy, spec.signalsum_with_noise)
    assert_array_almost_equal(np.sqrt([3, 6, 9]), spec.noisequadsum)
    assert_array_almost_equal((added_signals_noisy) / added_signals,
                              spec.signalsum_norm_with_noise)


def test_synspec_sums_3():
    # Test all combinations I can think of right now (ndarray, list, tuple)
    signals = ((np.array([1, 2, 3]), np.array([1, 2, 3]), np.array([1, 2, 3])))
    spec = SynSpectrum(signals, shotnoise=True, constnoise=1)
    added_signals = signals[0]+signals[1]+signals[2]
    added_signals_noisy = (spec.signals[0].signal_with_noise +
                           spec.signals[1].signal_with_noise +
                           spec.signals[2].signal_with_noise +
                           spec.constnoise_array)
    assert_array_equal(added_signals, spec.signalsum)
    assert_array_equal(added_signals_noisy, spec.signalsum_with_noise)
    assert_array_almost_equal(np.sqrt([4, 7, 10]), spec.noisequadsum)
    assert_array_almost_equal((added_signals_noisy) / added_signals,
                              spec.signalsum_norm_with_noise)
