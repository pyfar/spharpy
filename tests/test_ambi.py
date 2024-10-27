import numpy.testing as npt
from pytest import raises, warns, mark
from spharpy.ambi import AmbisonicsSignal, sht, isht
import numpy as np
import pyfar as pf
from spharpy import samplings
from spharpy.spherical import SphericalHarmonics


def test_ambisonics_signal_init():
    """Test to init Signal without optional parameters."""

    signal = AmbisonicsSignal(np.array([1., 2., 3.]), 44100,
                              n_max=1, basis_type='real',
                              channel_convention='acn',
                              condon_shortley=True,
                              normalization='n3d')
    assert isinstance(signal, AmbisonicsSignal)


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
@mark.parametrize("basis_type", ["real", "complex"])
def test_back_and_forth(n_max, basis_type):

    tmp = samplings.equiangular(50)
    coords = pf.Coordinates.from_spherical_colatitude(
        azimuth=tmp.azimuth, colatitude=tmp.colatitude,
        radius=np.ones_like(tmp.azimuth))

    if basis_type == 'real':
        data = np.ones(((n_max+1) ** 2, 16))
        is_complex = False
    else:
        data = np.ones(((n_max+1) ** 2, 16), dtype=complex)
        is_complex = True

    # generate unit amplitude ambisonics signal
    a_nm = AmbisonicsSignal(data,
                            n_max=1, basis_type='real',
                            channel_convention='acn',
                            condon_shortley=True,
                            normalization='n3d',
                            sampling_rate=48000,
                            is_complex=is_complex)

    a = isht(a_nm, coords)
    a_eval_nm = sht(a, coords, n_max=n_max, basis_type=basis_type)

    npt.assert_allclose(a_nm.time, a_eval_nm.time, rtol=1e-8)
