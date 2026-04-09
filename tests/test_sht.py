import numpy as np
import numpy.testing as npt
import pyfar as pf
import pytest
from spharpy.sht import sht, isht
from spharpy.classes.sh import SphericalHarmonics
from spharpy import SphericalHarmonicSignal
from spharpy import SphericalHarmonicTimeData
from spharpy import SphericalHarmonicFrequencyData
from spharpy import samplings


def test_sht_input_parameter():
    input_signal = np.zeros((3, 12, 2))
    n_max = 2
    sampling = samplings.equiangular(n_max=n_max)                                            
    sh = SphericalHarmonics(n_max=n_max, coordinates=sampling)
    with pytest.raises(ValueError, match="Input signal must be a Signal, "
                                         "TimeData, or FrequencyData but"
                                         f"is {type(input_signal)}"):
        _ = sht(signal=input_signal, spherical_harmonics=sh)

    # test if SH in SphericalHarmonics object


def test_sht_output_parameter():
    n_max = 1
    sampling = samplings.equiangular(n_max=n_max)             
    sh = SphericalHarmonics(n_max=n_max, coordinates=sampling)

    # test Signal
    signal = pf.Signal(data=np.zeros((1, 16, 4)), sampling_rate=48000)
    test = sht(signal=signal, spherical_harmonics=sh)
    assert isinstance(test, SphericalHarmonicSignal)

    # test TimeData
    signal = pf.TimeData(data=np.zeros((1, 16, 4)),
                         times=[1, 2, 3, 4])
    test = sht(signal=signal, spherical_harmonics=sh)
    assert isinstance(test, SphericalHarmonicTimeData)

    # test FrequencyData
    signal = pf.FrequencyData(data=np.zeros((1, 16, 4)),
                              frequencies=[1, 2, 3, 4])
    test = sht(signal=signal, spherical_harmonics=sh)
    assert isinstance(test, SphericalHarmonicFrequencyData)


def test_sht_assert_num_channels():
    """test assert match of number of channels and number of sampling positions"""
    n_max = 3
    signal = pf.Signal(data=np.zeros((7, 512)), sampling_rate=48000)
    sampling = samplings.equiangular(n_max=n_max)
    sh = SphericalHarmonics(n_max=n_max, coordinates=sampling)

    with pytest.raises(ValueError,
                       match="Spherical samples of provided axis does "
                             "not match the number of spherical "
                             "harmonics basis functions."):
        _ = sht(signal, sh, axis=0)


def test_isht_input_parameter():
    n_max = 1
    data = np.zeros((1, (n_max+1) ** 2, 16))
    sampling = samplings.gaussian(n_max=n_max)
    with pytest.raises(ValueError, match="Input signal has to be "
                                         "SphericalHarmonicSignal, "
                                         "SphericalHarmonicTimeData, or "
                                         "SphericalHarmonicFrequencyData "
                                         f"but is {type(data)}"):
        _ = isht(sh_signal=data, coordinates=sampling)


def test_isht_output_parameter():
    n_max = 1
    data = np.zeros((1, (n_max+1) ** 2, 5))
    sampling = samplings.gaussian(n_max=n_max)

    # test Signal
    a_nm = SphericalHarmonicSignal(data,
                                   basis_type='real',
                                   channel_convention='ACN',
                                   condon_shortley=True,
                                   normalization='N3D',
                                   sampling_rate=48000)
    test = isht(sh_signal=a_nm, coordinates=sampling)
    assert isinstance(test, pf.Signal)

    # test TimeData
    a_nm = SphericalHarmonicTimeData(data,
                                     times=[1, 2, 3, 4, 5],
                                     basis_type='real',
                                     channel_convention='ACN',
                                     condon_shortley=True,
                                     normalization='N3D')
    test = isht(sh_signal=a_nm, coordinates=sampling)
    assert isinstance(test, pf.TimeData)

    # test FrequencyData
    a_nm = SphericalHarmonicFrequencyData(
        data,
        frequencies=[1, 2, 3, 4, 5],
        basis_type='real',
        channel_convention='ACN',
        condon_shortley=True,
        normalization='N3D')

    test = isht(sh_signal=a_nm, coordinates=sampling)
    assert isinstance(test, pf.FrequencyData)


def test_sht_auto_axis():
    """test warning wrong axis"""
    n_max = 3
    signal = pf.Signal(data=np.zeros((7, 1, 32)), sampling_rate=48000)
    sampling = samplings.equiangular(n_max=n_max)
    sh = SphericalHarmonics(n_max=n_max, coordinates=sampling)

    with pytest.raises(ValueError,
                       match="No axes matches the number of spherical "
                             "harmonics basis functions"):
        _ = sht(signal, sh, axis='auto')

    signal = pf.Signal(data=np.zeros((64, 64, 64)), sampling_rate=48000)
    with pytest.raises(ValueError,
                       match="Too many axis match the number of "
                             "spherical harmonics basis functions"):
        _ = sht(signal, sh, axis='auto')


def test_in_out_dimensions():
    n_max = 3
    n_samples = 128
    sampling = samplings.equiangular(n_max=n_max)
    signal = pf.Signal(data=np.zeros((sampling.csize, n_samples)),
                       sampling_rate=48000)
    sh = SphericalHarmonics(n_max=n_max, coordinates=sampling)

    sh_signal = sht(signal, sh, axis=0)
    assert sh_signal.n_samples == n_samples
    assert sh_signal.cshape[-1] == int(np.power(n_max+1, 2))
    assert sh_signal.cshape[0] == 1

    signal = pf.Signal(data=np.zeros((1, sampling.csize, n_samples)),
                       sampling_rate=48000)
    sh = SphericalHarmonics(n_max=n_max, coordinates=sampling)

    sh_signal = sht(signal, sh, axis=1)
    assert sh_signal.n_samples == n_samples
    assert sh_signal.cshape[-1] == int(np.power(n_max+1, 2))
    assert sh_signal.cshape[0] == 1

    signal = pf.Signal(data=np.zeros((sampling.csize, 1, n_samples)),
                       sampling_rate=48000)
    sh = SphericalHarmonics(n_max=n_max, coordinates=sampling)

    sh_signal = sht(signal, sh, axis=0)
    assert sh_signal.n_samples == n_samples
    assert sh_signal.cshape[-1] == int(np.power(n_max+1, 2))
    assert sh_signal.cshape[0] == 1

    signal = pf.Signal(data=np.zeros((sampling.csize, 1, 2, 3, n_samples)),
                       sampling_rate=48000)
    sh = SphericalHarmonics(n_max=n_max, coordinates=sampling)

    sh_signal = sht(signal, sh, axis=0)
    assert sh_signal.n_samples == n_samples
    assert sh_signal.cshape[-1] == int(np.power(n_max+1, 2))
    assert sh_signal.cshape[0] == 1
    assert sh_signal.cshape[1] == 2
    assert sh_signal.cshape[2] == 3

    signal = pf.Signal(data=np.zeros((1, 2, sampling.csize, 3, n_samples)),
                       sampling_rate=48000)
    sh = SphericalHarmonics(n_max=n_max, coordinates=sampling)

    sh_signal = sht(signal, sh, axis=2)
    assert sh_signal.n_samples == n_samples
    assert sh_signal.cshape[-1] == int(np.power(n_max+1, 2))
    assert sh_signal.cshape[0] == 1
    assert sh_signal.cshape[1] == 2
    assert sh_signal.cshape[2] == 3


@mark.parametrize("n_max", [1, 3, 12, 20])
@mark.parametrize("basis_type", ["real", "complex"])
@mark.parametrize("normalization", ["N3D", "SN3D"])
@mark.parametrize("condon_shortley", [True, False])
def test_back_and_forth(n_max, basis_type, normalization, condon_shortley):

    sampling = samplings.gaussian(n_max=n_max)
    # create unit amplitude SH coefficients
    data = np.zeros((1, (n_max+1) ** 2, 16), dtype=complex)
    if normalization == 'N3D':
        data[0, 0, :] = np.sqrt(4 * np.pi)
    else:
        data[0, 0, :] = 1.0

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
    assert a_nm.n_samples == a.n_samples
    sh = SphericalHarmonics(n_max=n_max,
                            coordinates=sampling,
                            basis_type=basis_type,
                            normalization=normalization,
                            condon_shortley=condon_shortley)
    a_eval_nm = sht(a, sh)
    assert a_eval_nm.n_samples == a.n_samples
    npt.assert_allclose(a_nm.time, a_eval_nm.time, rtol=1e-14, atol=1e-14)
