import numpy as np
import numpy.testing as npt
import pyfar as pf
from pytest import raises, warns, mark
from spharpy.sht import sht, isht
from spharpy.classes.sh import SphericalHarmonicDefinition, SphericalHarmonics
from spharpy import SphericalHarmonicSignal
from spharpy import SphericalHarmonicTimeData
from spharpy import SphericalHarmonicFrequencyData
from spharpy import samplings


def test_sht_assert_num_channels():
    "test assert match of number of channels and number of sampling positions"
    n_max = 3
    signal = pf.Signal(data=np.zeros((7, 512)), sampling_rate=48000)
    coords = pf.Coordinates.from_spherical_elevation(np.zeros((8)),
                                                     np.zeros((8)),
                                                     np.ones((8)))

    sh = SphericalHarmonics(n_max=n_max, coordinates=coords)
    with raises(ValueError, match="Signal shape does not match number of "
                                  "coordinates."):
        _ = sht(signal, sh)


def test_sht_wrong_axis():
    "test warning wrong axis"
    n_max = 3
    signal = pf.Signal(data=np.zeros((8, 1, 512)), sampling_rate=48000)
    coords = pf.Coordinates.from_spherical_elevation(np.zeros((8)),
                                                     np.zeros((8)),
                                                     np.ones((8)))
    sh = SphericalHarmonics(n_max=n_max, coordinates=coords)
    with warns(UserWarning, match="Compute spherical harmonics transform "
                                  "along axis = 0."):
        _ = sht(signal, sh, axis=1)


@mark.parametrize("n_max", [3, 12, 20])
@mark.parametrize("basis_type", ["real", "complex"])
@mark.parametrize("normalization", ["N3D", "SN3D"])
@mark.parametrize("condon_shortley", [True, False])
def test_back_and_forth(n_max, basis_type, normalization, condon_shortley):

    sampling = samplings.equiangular(n_max=n_max)

    data = np.ones((1, (n_max+1) ** 2, 16), dtype=complex)
    is_complex = True

    if basis_type == 'real':
        data = np.real(data)
        is_complex = False

    # generate unit amplitude sh signal
    a_nm = SphericalHarmonicSignal(data,
                                   basis_type=basis_type,
                                   channel_convention='ACN',
                                   condon_shortley=condon_shortley,
                                   normalization=normalization,
                                   sampling_rate=48000,
                                   is_complex=is_complex)

    a = isht(a_nm, sampling)
    sh = SphericalHarmonics(n_max=n_max,
                            ncoordinates=sampling,
                            basis_type=basis_type,
                            normalization=normalization,
                            condon_shortley=condon_shortley)
    a_eval_nm = sht(a, sh)

    npt.assert_allclose(a_nm.time, a_eval_nm.time, rtol=1e-8)
