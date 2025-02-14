import spharpy.spherical as sh
import numpy as np
import pytest


@pytest.mark.parametrize(('index', 'expected'), [
    (0, np.sqrt(1/2)), (1, np.sqrt(1/3)), (2, np.sqrt(1/3))
])
def test_n3d_to_maxn_single_val(index, expected):
    assert sh.n3d_to_maxn(index) == expected


def test_n3d_to_maxn_array():
    acn = np.array([0, 1, 2])
    maxN_ref = np.array([np.sqrt(1 / 2),
                         np.sqrt(1 / 3),
                         np.sqrt(1 / 3)])

    maxN_norm = sh.n3d_to_maxn(acn)
    np.testing.assert_equal(maxN_norm, maxN_ref)


@pytest.mark.parametrize(('index', 'expected'), [
    (0, 1), (1, 1/np.sqrt(3)), (2, 1/np.sqrt(5))
])
def test_n3d_to_sn3d_norm_single_val(index, expected):
    assert sh.n3d_to_sn3d_norm(index) == expected


def test_n3d_to_sn3d_norm_array():
    n = np.array([0, 1, 2])
    sn3d_norm_ref = np.array([1, 1/np.sqrt(3), 1/np.sqrt(5)])
    sn3d_norm = sh.n3d_to_sn3d_norm(n)

    np.testing.assert_equal(sn3d_norm, sn3d_norm_ref)
