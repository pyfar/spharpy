import numpy as np
import numpy.testing as npt
import pytest

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


def test_re_max():
    N = 7
    g_nm = spharpy.beamforming.rE_max_weights(N, normalize=False)

    truth = np.loadtxt('tests/data/re_max_weights.csv', delimiter=',')
    npt.assert_allclose(g_nm, truth)

    g_nm_norm = spharpy.beamforming.rE_max_weights(N, normalize=True)

    Y = spharpy.spherical.spherical_harmonic_basis_real(
        N, spharpy.samplings.Coordinates(1, 0, 0))

    npt.assert_allclose(Y @ np.diag(g_nm_norm) @ Y.T, 1)


def test_max_front_back():
    N = 7
    f_nm_norm = spharpy.beamforming.maximum_front_back_ratio_weights(
        N, normalize=True)

    Y = spharpy.spherical.spherical_harmonic_basis_real(
        N, spharpy.samplings.Coordinates(1, 0, 0))

    npt.assert_allclose(Y @ np.diag(f_nm_norm) @ Y.T, 1)

    with pytest.raises(RuntimeError, match='did not converge'):
        spharpy.beamforming.maximum_front_back_ratio_weights(30)
