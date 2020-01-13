import numpy as np
import numpy.testing as npt

import spharpy

def test_dolph_cheby_mainlobe():
    N = 7
    theta0 = np.pi/6
    d_nm = spharpy.beamforming.dolph_chebyshev_weights(
        N, theta0, design_criterion='mainlobe')

    truth = np.loadtxt('tests/data/dolph_cheby_mainlobe.csv', delimiter=',')
    npt.assert_allclose(d_nm, truth)


def test_dolph_cheby_sidelobe():
    N = 7
    R_dB = 50
    R = 10**(R_dB/20)
    d_nm = spharpy.beamforming.dolph_chebyshev_weights(
        N, R, design_criterion='sidelobe')

    truth = np.loadtxt('tests/data/dolph_cheby_sidelobe.csv', delimiter=',')
    npt.assert_allclose(d_nm, truth)
