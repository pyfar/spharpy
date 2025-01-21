import spharpy.spherical as sh
import numpy as np


def test_n3d_to_maxn():
    maxN_norm = sh.n3d2_to_maxn(0)
    assert maxN_norm == np.sqrt(1 / 2)

    maxN_norm = sh.n3d2_to_maxn(1)
    assert maxN_norm == np.sqrt(1 / 3)

    maxN_norm = sh.n3d2_to_maxn(2)
    assert maxN_norm == np.sqrt(1 / 3)


def test_n3d_to_sn3d_norm():
    raise NotImplementedError()
