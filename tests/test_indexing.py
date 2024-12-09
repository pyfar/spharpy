""" Tests for sh channel indexing"""

import spharpy.spherical as sh
import numpy as np


def test_acn2nm_single_val():
    n, m = sh.acn2nm(0)
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


def test_identity_matrix_n_nm():
    reference = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 1, 1, 1, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 1, 1, 1, 1, 1]])

    identity = sh.sph_identity_matrix(2, type='n-nm')
    np.testing.assert_allclose(reference, identity)


def test_sid_indexing():
    n_max = 2
    reference_n = np.array([0, 1, 1, 1, 2, 2, 2, 2, 2])
    reference_m = np.array([0, 1, -1, 0, 2, -2, 1, -1, 0])

    sid_n, sid_m = sh.sid(n_max)

    np.testing.assert_equal(reference_n, sid_n)
    np.testing.assert_equal(reference_m, sid_m)


def test_sid2acn():
    n_max = 2
    # indexing starts at 0 here, reference was calculated
    # with indexing starting at 1.
    reference_acn = np.array([1, 3, 4, 2, 6, 8, 9, 7, 5]) - 1

    acn_indices = sh.sid2acn(n_max)
    np.testing.assert_equal(reference_acn, acn_indices)


def test_nm2fuma_single_val():
    fuma = sh.nm2fuma(0, 0)
    assert fuma == 0

    fuma = sh.nm2fuma(1, 0)
    assert fuma == 1

    fuma = sh.nm2fuma(1, -1)
    assert fuma == 3


def test_nm2fuma_array():
    n = np.array([0, 1, 1])
    m = np.array([0, 0, -1])

    fuma = sh.nm2fuma(n, m)
    assert fuma[0] == 0
    assert fuma[1] == 1
    assert fuma[2] == 3
