from pytest import raises
from spharpy.classes.audio import (
    _SphericalHarmonicAudio,
    SphericalHarmonicTimeData,
    SphericalHarmonicFrequencyData,
    _atleast_3d_first_dimension)
import numpy as np


def test_atleast_3d_data():
    data = np.ones((4, 4))
    data_3d = _atleast_3d_first_dimension(data)
    assert data_3d.ndim == 3
    assert data_3d.shape == (1, 4, 4)

    data_3d = _atleast_3d_first_dimension(np.ones((2, 4, 4)))

    assert data_3d.ndim == 3
    assert data_3d.shape == (2, 4, 4)


def test_init_sh_time_data():
    data = np.ones((1, 4, 4))
    times = [1, 2, 3, 4]
    sh_time_data = SphericalHarmonicTimeData(
        data, times,  basis_type='real', normalization='SN3D',
        channel_convention="ACN", condon_shortley=False,
        comment="")
    assert isinstance(sh_time_data, SphericalHarmonicTimeData)
    np.testing.assert_allclose(sh_time_data.time, data)


def test_init_sh_frequency_data():
    data = np.ones((1, 4, 4), dtype=complex)
    frequencies = [1, 2, 3, 4]
    sh_freq_data = SphericalHarmonicFrequencyData(
        data, frequencies, basis_type='real', normalization='SN3D',
        channel_convention="ACN", condon_shortley=False,
        comment="")
    assert isinstance(sh_freq_data, SphericalHarmonicFrequencyData)
    np.testing.assert_allclose(sh_freq_data.freq, data)
