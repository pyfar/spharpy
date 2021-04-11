"""
Tests for modal strength function
"""

import sys
from filehandling import read_matrix_from_mat

sys.path.append('./')

import pytest
import spharpy.spherical as sh
import numpy as np
from spharpy import special

def test_modal_strength_open():
    n_max = 5
    n_sh = (n_max+1)**2
    n_bins = 128
    k = np.linspace(0.5, 15, n_bins)
    reference = read_matrix_from_mat('./tests/data/modal_strength_open.mat')
    bn = sh.modal_strength(n_max, k, arraytype='open')
    np.testing.assert_allclose(bn, reference)


def test_modal_strength_rigid():
    n_max = 5
    n_bins = 128
    k = np.linspace(0.5, 15, n_bins)
    # ref data has wrong sign (calculated as i^(n-1) instead of i^(n+1)
    reference = -read_matrix_from_mat('./tests/data/modal_strength_rigid.mat')
    bn = sh.modal_strength(n_max, k, arraytype='rigid')
    np.testing.assert_allclose(bn, reference)
