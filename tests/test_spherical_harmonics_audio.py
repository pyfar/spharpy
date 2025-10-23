from pytest import raises
from spharpy.classes.sh_audio import _SphericalHarmonicAudio
from spharpy.classes.sh_audio import SphericalHarmonicTimeData
from spharpy.classes.sh_audio import SphericalHarmonicFrequencyData
import numpy as np


def test_init_sh_time_data():
    data = np.ones((1, 4, 4))
    times = [1, 2, 3, 4]
    sh_time_data = SphericalHarmonicTimeData(
        data, times,  basis_type='real', normalization='SN3D',
        channel_convention="acn", condon_shortley=False,
        comment="")
    assert type(sh_time_data) == SphericalHarmonicTimeData


def test_init_sh_frequency_data():
    data = np.ones((1, 4, 4))
    frequencies = [1, 2, 3, 4]
    sh_freq_data = SphericalHarmonicFrequencyData(
        data, frequencies, basis_type='real', normalization='SN3D',
        channel_convention="acn", condon_shortley=False,
        comment="")
    assert type(sh_freq_data) == SphericalHarmonicFrequencyData