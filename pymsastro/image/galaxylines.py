# Licensed under a 3-clause BSD style license - see LICENSE.rst

from ..formula.redshift import redshift_wavelength
from copy import deepcopy

__all__ = ['GalaxyAbsorptionLines', 'GalaxyEmissionLines', 'GalaxyLinesAKoch',
           'NightSkyLines']


class GalaxyAbsorptionLines(object):
    lines = {'Ca K 3934': 3934.777,
             'Ca H 3969': 3969.588,
             'Ca G 4305': 4305.61,
             'Mg 5176': 5176.7,
             'Na 5895': 5895.6,
             'CaII 8500': 8500.36,
             'CaII 8544': 8544.44,
             'CaII 8664': 8664.52}

    @staticmethod
    def get(redshift=0):
        if redshift != 0:
            ret = deepcopy(GalaxyAbsorptionLines.lines)
            for i in ret:
                ret[i] = redshift_wavelength(redshift=redshift,
                                             wavelength_emitted=ret[i])
            return ret
        else:
            return GalaxyAbsorptionLines.lines


class GalaxyEmissionLines(object):
    lines = {'O VI 1033 (Quasar)': 1033.82,
             'Ly alpha 1215 (Quasar)': 1215.24,
             'N V 1240 (Quasar)': 1240.81,
             'O I 1305 (None)': 1305.53,
             'C II 1335 (None)': 1335.31,
             'Si IV 1397 (None)': 1397.61,
             'Si IV + O IV 1399 (Quasar)': 1399.8,
             'C IV 1549 (Quasar)': 1549.48,
             'He II 1640 (None)': 1640.4,
             'O III 1665 (None)': 1665.85,
             'Al III 1857 (None)': 1857.4,
             'C III 1908 (Quasar)': 1908.734,
             'C II 2326 (Quasar)': 2326.0,
             'Ne IV 2439 (None)': 2439.5,
             'Mg II 2799 (both)': 2799.117,
             'Ne V 3346 (None)': 3346.79,
             'Ne VI 3426 (None)': 3426.85,
             'O II 3727 (both)': 3727.092,
             'O II 3729 (None)': 3729.875,
             'He I 3889 (None)': 3889.0,
             'S II 4072 (None)': 4072.3,
             'H delta 4102 (both)': 4102.89,
             'H gamma 4341 (both)': 4341.68,
             'O III 4364 (None)': 4364.436,
             'H beta 4862 (both)': 4862.68,
             'O III 4932 (None)': 4932.603,
             'O III 4960 (both)': 4960.295,
             'O III 5008 (both)': 5008.240,
             'O I 6302 (None)': 6302.046,
             'O I 6365 (None)': 6365.536,
             'N I 6529 (None)': 6529.03,
             'N II 6549 (Galaxy)': 6549.86,
             'H alpha 6564 (both)': 6564.61,
             'N II 6585 (Galaxy)': 6585.27,
             'S II 6718 (Galaxy)': 6718.29,
             'S II 6732 (Galaxy)': 6732.67}

    @staticmethod
    def get(redshift=0):
        if redshift != 0:
            ret = deepcopy(GalaxyEmissionLines.lines)
            for i in ret:
                ret[i] = redshift_wavelength(redshift=redshift,
                                             wavelength_emitted=ret[i])
            return ret
        else:
            return GalaxyEmissionLines.lines


class GalaxyLinesAKoch(object):
    lines = {'[Si I] 6717': 6717,
             '[Si I] 6731': 6731,
             'He I 6678': 6678,
             '[N II] 6583': 6583,
             'H alpha 6562': 6562.78,
             '[N II] 6548': 6548,
             '[OI] 6300': 6300.23,
             'He I 5876': 5876,
             '[O III] 5007': 5007,
             '[O III] 4959': 4959,
             'H beta 4861': 4861.34,
             '[O III] 4363': 4363,
             'H gamma 4341': 4340.48,
             'H delta 4102': 4101.75,
             'Ca K 3968': 3968.49,
             'Ca H 3933': 3933.68,
             'H zeta 3889': 3889.05,
             'Ne 3868': 3868,
             'H eta 3835': 3835.39,
             'H theta 3797': 3797.90,
             'H iota 3770': 3770.63,
             'H kappa 3750': 3750.15,
             '[O II] 3727': 3727,
             '[O II] 3729': 3729,
             'Mg II 2796': 2796.352,
             'Mg II 2803': 2803.531,
             'Fe II 2586': 2586.650,
             'Fe II 2600': 2600.1729,
             'C IV 1548': 1548.195,
             'C IV 1550': 1550.770,
             'NaD 5889': 5889.951,
             'NaD 5895': 5895.924}

    @staticmethod
    def get(redshift=0):
        if redshift != 0:
            ret = deepcopy(GalaxyLinesAKoch.lines)
            for i in ret:
                ret[i] = redshift_wavelength(redshift=redshift,
                                             wavelength_emitted=ret[i])
            return ret
        else:
            return GalaxyLinesAKoch.lines


class NightSkyLines(object):
    lines = {'5197': 5197.928223,
             '5200': 5200.285645,
             '5577': 5577.346680,
             '5888': 5888.192383,
             '5889': 5889.958984,
             '5895': 5895.932129,
             '5915': 5915.307617,
             '5932': 5932.864258,
             '5953': 5953.358887,
             '5953 2': 5953.489746,
             '6234': 6234.304199,
             '6235': 6235.965332,
             '6257': 6257.970215,
             '6287': 6287.442871,
             '6300': 6300.308594,
             '6306': 6306.933594,
             '6321': 6321.408691,
             '6329': 6329.784180,
             '6329 2': 6329.927734,
             '6363': 6363.782715,
             '6498': 6498.736816,
             '6863': 6863.970703,
             '6923': 6923.192383,
             '7238': 7238.791504,
             '7240': 7240.194336,
             '7244': 7244.943848,
             '7276': 7276.424316,
             '7284': 7284.455566,
             '7316': 7316.289551,
             '7340': 7340.900879,
             '7358': 7358.680176,
             '7369': 7369.272461,
             '7369 2': 7369.495117,
             '7571': 7571.750977,
             '7712': 7712.229004,
             '7712 2': 7712.635742,
             '7716': 7716.922363,
             '7750': 7750.656250,
             '7794': 7794.120605,
             '7808': 7808.479980,
             '7821': 7821.519043,
             '7841': 7841.284668,
             '7853': 7853.258789,
             '7853 2': 7853.523926,
             '7870': 7870.738281,
             '7913': 7913.717773,
             '7964': 7964.654297,
             '7993': 7993.333984,
             '8288': 8288.608398,
             '8298': 8298.906250,
             '8344': 8344.613281}

    @staticmethod
    def get(redshift=0):
        return NightSkyLines.lines
