from pytest import raises
from spharpy.classes.audio import SphericalHarmonicAudio
from spharpy.classes.audio import SphericalHarmonicTimeData
from spharpy.classes.audio import SphericalHarmonicFrequencyData
import numpy as np
import re


def test_init_sh_audio():
    sh_audio = SphericalHarmonicAudio(
        basis_type='real', normalization='sn3d', channel_convention="acn",
        condon_shortley=False, domain="time", comment="")
    assert isinstance(sh_audio, SphericalHarmonicAudio)


def test_init_sh_time_data():
    data = np.ones((1, 4, 4))
    times = [1, 2, 3, 4]
    sh_time_data = SphericalHarmonicTimeData(
        data, times,  basis_type='real', normalization='sn3d',
        channel_convention="acn", condon_shortley=False,
        comment="")
    assert isinstance(sh_time_data, SphericalHarmonicTimeData)


def test_init_sh_frequency_data():
    data = np.ones((1, 4, 4))
    frequencies = [1, 2, 3, 4]
    sh_freq_data = SphericalHarmonicFrequencyData(
        data, frequencies, basis_type='real', normalization='sn3d',
        channel_convention="acn", condon_shortley=False,
        comment="")
    assert isinstance(sh_freq_data, SphericalHarmonicFrequencyData)