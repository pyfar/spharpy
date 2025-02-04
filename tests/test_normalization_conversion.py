import spharpy.spherical as sh
import numpy as np


def test_n3d_to_maxn():
    maxN_norm = sh.n3d_to_maxn(0)
    assert maxN_norm == np.sqrt(1 / 2)

    maxN_norm = sh.n3d_to_maxn(1)
    assert maxN_norm == np.sqrt(1 / 3)

    maxN_norm = sh.n3d_to_maxn(2)
    assert maxN_norm == np.sqrt(1 / 3)


def test_n3d_to_sn3d_norm():
    sn3d_norm = sh.n3d_to_sn3d_norm(0)
    assert sn3d_norm == 1

    sn3d_norm = sh.n3d_to_sn3d_norm(1)
    assert sn3d_norm == 1/np.sqrt(3)

    sn3d_norm = sh.n3d_to_sn3d_norm(2)
    assert sn3d_norm == 1/np.sqrt(5)
