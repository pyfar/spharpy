"""
Tests for acn and order, degree indexing functions
"""

import sys
from filehandling import read_matrix_from_mat

sys.path.append('./')

import pytest
import spharpy.spherical as sh
import numpy as np

def test_acn2nm_single_val():
    n,m = sh.acn2nm(0)
    assert n == 0
    assert m == 0

    n, m = sh.acn2nm(2)
    assert n == 1
    assert m == 0

    n, m = sh.acn2nm(1)
    assert n == 1
    assert m == -1


def test_nm2acn_single_val():
    acn = sh.nm2acn(0, 0)
    assert acn == 0

    acn = sh.nm2acn(1, 0)
    assert acn == 2

    acn = sh.nm2acn(1, -1)
    assert acn == 1

def test_acn2nm_array():
    n_ref = np.array([0, 1, 1, 1])
    m_ref = np.array([0, -1, 0, 1])

    n_max = 1
    n_sh = (n_max + 1)**2
    acn = np.arange(0, n_sh)
    n, m = sh.acn2nm(acn)

    np.testing.assert_equal(n, n_ref)
    np.testing.assert_equal(m, m_ref)

def test_nm2acn_array():
    n = np.array([0, 1, 1, 1])
    m = np.array([0, -1, 0, 1])

    n_max = 1
    n_sh = (n_max + 1)**2
    acn_ref = np.arange(0, n_sh)

    acn = sh.nm2acn(n, m)

    np.testing.assert_equal(acn, acn_ref)
