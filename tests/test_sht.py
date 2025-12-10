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


def test_sht_input_parameter():
    input_signal = np.zeros((3, 12, 2))
    n_max = 2
    sampling = samplings.equiangular(n_max=n_max)                                            
    sh = SphericalHarmonics(n_max=n_max, coordinates=sampling)
    with raises(ValueError, match="Input signal must be a Signal, TimeData, "
                f"or FrequencyData but is {type(input_signal)}"):
        _ = sht(signal=input_signal, spherical_harmonics=sh)

    # test if SH in SphericalHarmonics object


def test_sht_output_parameter():
    n_max = 1
    sampling = samplings.equiangular(n_max=n_max)                                            
    sh = SphericalHarmonics(n_max=n_max, coordinates=sampling)

    # test signal
    signal = pf.Signal(data=np.zeros((4, 512)), sampling_rate=48000)
    test = sht(signal=signal, spherical_harmonics=sh)
    assert isinstance(test, SphericalHarmonicSignal)

    # test TimeData
    signal = pf.TimeData(data=np.zeros(1, 4, 8),
                         times=[1, 2, 3, 4, 5, 6, 7, 8])
    test = sht(signal=signal, spherical_harmonics=sh)
    
    assert isinstance(test, SphericalHarmonicTimeData)

    # test FrequencyData
    signal = pf.FrequencyData(data=np.zeros(4, 512), frequencies=np.range(512))
    test = sht(signal=signal, spherical_harmonics=sh)
    assert isinstance(test, SphericalHarmonicFrequencyData)


def test_sht_assert_num_channels():
    "test assert match of number of channels and number of sampling positions"
    n_max = 3
    signal = pf.Signal(data=np.zeros((7, 512)), sampling_rate=48000)
    sampling = samplings.equiangular(n_max=n_max)
    sh = SphericalHarmonics(n_max=n_max, coordinates=sampling)

    with raises(ValueError, match="Spherical samples of provided axis does "
                                  "not match the number of spherical "
                                  "harmonics basis functions."):
        _ = sht(signal, sh, axis=0)


def test_sht_auto_axis():
    "test warning wrong axis"
    n_max = 3
    signal = pf.Signal(data=np.zeros((7, 1, 32)), sampling_rate=48000)
    sampling = samplings.equiangular(n_max=n_max)
    sh = SphericalHarmonics(n_max=n_max, coordinates=sampling)

    with raises(ValueError, match="No axes matches the number of spherical "
                                  "harmonics basis functions"):
        _ = sht(signal, sh, axis='auto')

    signal = pf.Signal(data=np.zeros((8, 8, 32)), sampling_rate=48000)
    with raises(ValueError, match="To many axis match the number of spherical "
                                  "harmonics basis functions"):
        _ = sht(signal, sh, axis='auto')


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
                            coordinates=sampling,
                            basis_type=basis_type,
                            normalization=normalization,
                            condon_shortley=condon_shortley)
    a_eval_nm = sht(a, sh)

    npt.assert_allclose(a_nm.time, a_eval_nm.time, rtol=1e-8)
