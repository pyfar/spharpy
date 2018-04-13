""" Tests for sh channel indexing"""

import pytest
import numpy as np
import spharpy.indexing

def test_identity_matrix_n_nm():
    reference = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 1, 1, 1, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 1, 1, 1, 1, 1]])

    identity = spharpy.indexing.sph_identity_matrix(2, type='n-nm')
    np.testing.assert_allclose(reference, identity)

def test_sid_indexing():
    n_max = 2
    reference_n = np.array([0, 1, 1, 1, 2, 2, 2, 2, 2])
    reference_m = np.array([0, 1, -1, 0, 2, -2, 1, -1, 0])


    sid_n, sid_m = spharpy.indexing.sid(n_max)

    np.testing.assert_equal(reference_n, sid_n)
    np.testing.assert_equal(reference_m, sid_m)

def test_sid2acn():
    n_max = 2
    # indexing starts at 0 here, reference was calculated
    # with indexing starting at 1.
    reference_acn = np.array([1, 3, 4, 2, 6, 8, 9, 7, 5]) - 1

    acn_indeces = spharpy.indexing.sid2acn(n_max)
    np.testing.assert_equal(reference_acn, acn_indeces)
