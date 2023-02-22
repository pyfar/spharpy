"""
Tests for modal strength function
"""
from scipy.io import loadmat
import spharpy.spherical as sh
import numpy as np


def test_modal_strength_open():
    n_max = 5
    n_bins = 128
    k = np.linspace(0.5, 15, n_bins)
    fname = './tests/data/modal_strength_open.mat'
    reference = loadmat(fname)['matrix']
    bn = sh.modal_strength(n_max, k, arraytype='open')
    np.testing.assert_allclose(bn, reference)


def test_modal_strength_rigid():
    n_max = 5
    n_bins = 128
    k = np.linspace(0.5, 15, n_bins)
    # ref data has wrong sign (calculated as i^(n-1) instead of i^(n+1)

    fname = './tests/data/modal_strength_rigid.mat'
    reference = -loadmat(fname)['matrix']
    bn = sh.modal_strength(n_max, k, arraytype='rigid')
    np.testing.assert_allclose(bn, reference)
