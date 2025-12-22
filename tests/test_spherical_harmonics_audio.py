from spharpy.classes.audio import (
    SphericalHarmonicTimeData,
    SphericalHarmonicFrequencyData,
    _atleast_3d_first_dimension)
from spharpy import SphericalHarmonicDefinition
import numpy as np
import deepdiff


def test_atleast_3d_data():
    data = np.ones((4, 4))
    data_3d = _atleast_3d_first_dimension(data)
    assert data_3d.ndim == 3
    assert data_3d.shape == (1, 4, 4)

    data_3d = _atleast_3d_first_dimension(np.ones((2, 4, 4)))

    assert data_3d.ndim == 3
    assert data_3d.shape == (2, 4, 4)

    data_3d = _atleast_3d_first_dimension(np.array(1))
    assert data_3d.ndim == 3
    assert data_3d.shape == (1, 1, 1)


def test_init_sh_time_data():
    data = np.ones((1, 4, 4))
    times = [1, 2, 3, 4]
    sh_time_data = SphericalHarmonicTimeData(
        data, times,  basis_type='real', normalization='SN3D',
        channel_convention="ACN", condon_shortley=False,
        comment="")
    assert isinstance(sh_time_data, SphericalHarmonicTimeData)
    np.testing.assert_allclose(sh_time_data.time, data)


def test_sh_time_data_from_sh_definition():
    """Test init SphericalHarmonicsSignal."""

    shd = SphericalHarmonicDefinition(
        n_max=0, basis_type="real",
        channel_convention="ACN",
        normalization="N3D",
        condon_shortley=False)

    data = np.ones((1, 4, 4))
    times = [1, 2, 3, 4]

    time_data_def = SphericalHarmonicTimeData.from_definition(
        sh_definition=shd, data=data, times=times)

    assert isinstance(time_data_def, SphericalHarmonicTimeData)

    time_data = SphericalHarmonicTimeData(
            data,
            times, basis_type='real',
            channel_convention='ACN',
            normalization='N3D',
            condon_shortley=False)

    assert not deepdiff.DeepDiff(
        time_data_def.__dict__, time_data.__dict__)


def test_init_sh_frequency_data():
    data = np.ones((1, 4, 4), dtype=complex)
    frequencies = [1, 2, 3, 4]
    sh_freq_data = SphericalHarmonicFrequencyData(
        data, frequencies, basis_type='real', normalization='SN3D',
        channel_convention="ACN", condon_shortley=False,
        comment="")
    assert isinstance(sh_freq_data, SphericalHarmonicFrequencyData)
    np.testing.assert_allclose(sh_freq_data.freq, data)


def test_sh_freq_data_from_sh_definition():
    """Test init SphericalHarmonicsSignal."""

    shd = SphericalHarmonicDefinition(
        n_max=0, basis_type="real",
        channel_convention="ACN",
        normalization="N3D",
        condon_shortley=False)

    data = np.ones((1, 4, 4))
    freqs = [1, 2, 3, 4]

    freq_data_def = SphericalHarmonicFrequencyData.from_definition(
        sh_definition=shd, data=data, frequencies=freqs)

    assert isinstance(freq_data_def, SphericalHarmonicFrequencyData)

    freq_data = SphericalHarmonicFrequencyData(
            data,
            freqs, basis_type='real',
            channel_convention='ACN',
            normalization='N3D',
            condon_shortley=False)

    assert not deepdiff.DeepDiff(
        freq_data_def.__dict__, freq_data.__dict__)