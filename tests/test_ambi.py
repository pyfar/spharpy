import numpy.testing as npt
from pytest import raises, warns, mark
from spharpy.ambi import AmbisonicsSignal, sht, isht
import numpy as np
import pyfar as pf


def test_sht_assert_num_channels():
    "test assert match of number of channels and number of sampling positions"
    n_max = 3
    signal = pf.Signal(data=np.zeros((7, 512)), sampling_rate=48000)
    coords = pf.Coordinates.from_spherical_elevation(np.zeros((8)),
                                                     np.zeros((8)),
                                                     np.ones((8)))

    with raises(ValueError, match="Signal shape does not match number of "
                                  "coordinates."):
        _ = sht(signal, coords, n_max)


def test_sht_wrong_axis():
    "test warning wrong axis"
    n_max = 3
    signal = pf.Signal(data=np.zeros((8, 1, 512)), sampling_rate=48000)
    coords = pf.Coordinates.from_spherical_elevation(np.zeros((8)),
                                                     np.zeros((8)),
                                                     np.ones((8)))

    with warns(UserWarning, match="Compute SHT along axis=0."):
        _ = sht(signal, coords, n_max, axis=1)


def test_sht_invalid_domain():
    "test warning invalid domain"
    n_max = 3
    signal = pf.Signal(data=np.zeros((8, 1, 512)), sampling_rate=48000)
    coords = pf.Coordinates.from_spherical_elevation(np.zeros((8)),
                                                     np.zeros((8)),
                                                     np.ones((8)))

    with raises(ValueError, match="Domain should be ``'time'`` or ``'freq'`` "
                                  "but is XXX."):
        _ = sht(signal, coords, n_max, domain='XXX')


@mark.parametrize("n_max", [3, 12, 44])
@mark.parametrize("sh_kind", ["real"])
@mark.parametrize("tkh_eps", [0, 1e-12])
@mark.parametrize("use_weights", [False, True])
def test_back_and_forth(n_max, sh_kind, tkh_eps, use_weights):
    coords = sph.samplings.equiangular(44)
    # generate unit amplitude ambisonics signal
    a_nm = AmbisonicsSignal(np.ones(((n_max+1) ** 2, 16)),
                            sh_kind=sh_kind, sampling_rate=48000)
    a = isht(a_nm, coords)
    a_eval_nm = sht(a, coords, n_max=n_max, sh_kind=sh_kind,
                    tikhonov_eps=tkh_eps, use_weights=use_weights)

    npt.assert_allclose(a_nm.time, a_eval_nm.time, rtol=1e-8)


test_sht_assert_num_channels()
