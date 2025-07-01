import numpy as np
import numpy.testing as npt
import pyfar as pf
from pytest import raises, warns, mark
from spharpy.sht import sht, isht
from spharpy import SphericalHarmonicSignal
from spharpy import samplings


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

    with warns(UserWarning, match="Compute spherical harmonics transform "
                                  "along axis = 0."):
        _ = sht(signal, coords, n_max, axis=1)


@mark.parametrize("n_max", [3, 12, 44])
@mark.parametrize("basis_type", ["real", "complex"])
def test_back_and_forth(n_max, basis_type):

    sampling = samplings.equiangular(50)

    if basis_type == 'real':
        data = np.ones((1, (n_max+1) ** 2, 16))
        is_complex = False
    else:
        data = np.ones((1, (n_max+1) ** 2, 16), dtype=complex)
        is_complex = True

    # generate unit amplitude sh signal
    a_nm = SphericalHarmonicSignal(data, basis_type='real',
                                   channel_convention='acn',
                                   condon_shortley=True,
                                   normalization='n3d',
                                   sampling_rate=48000,
                                   is_complex=is_complex)

    a = isht(a_nm, sampling)
    a_eval_nm = sht(a, sampling, n_max=n_max, basis_type=basis_type)

    npt.assert_allclose(a_nm.time, a_eval_nm.time, rtol=1e-8)
