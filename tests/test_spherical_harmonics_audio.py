from spharpy.classes.audio import (
    SphericalHarmonicTimeData,
    SphericalHarmonicFrequencyData,
    _atleast_3d_first_dimension)
from spharpy import SphericalHarmonicDefinition
import numpy as np
import pyfar as pf
import numpy.testing as npt
import pytest
import re


def test_atleast_3d_data():
    """Test extending the shape of 1D and 2D array likes."""

    # Prepend dimension for 2D array
    data = np.ones((4, 4))
    data_3d = _atleast_3d_first_dimension(data)
    assert data_3d.ndim == 3
    assert data_3d.shape == (1, 4, 4)

    # Do not change 3D array
    data_3d = _atleast_3d_first_dimension(np.ones((2, 4, 4)))

    assert data_3d.ndim == 3
    assert data_3d.shape == (2, 4, 4)

    # Prepend two dimensions for 1D array
    data_3d = _atleast_3d_first_dimension(np.array([1, 2, 3, 4]))
    assert data_3d.ndim == 3
    assert data_3d.shape == (1, 1, 4)

    # Convert lists to numpy array
    data = [1, 2, 3, 4]
    data_3d_list = _atleast_3d_first_dimension(data)
    npt.assert_equal(data_3d_list, data_3d)


def test_init_sh_time_data():
    data = np.ones((1, 4, 4))
    times = [1, 2, 3, 4]
    sh_time_data = SphericalHarmonicTimeData(
        data, times,  basis_type='real', normalization='SN3D',
        channel_convention="ACN", condon_shortley=False,
        comment="", caxis_spherical_harmonics=-1)
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
        sh_definition=shd, data=data, times=times,
        caxis_spherical_harmonics=-1)

    assert isinstance(time_data_def, SphericalHarmonicTimeData)

    time_data = SphericalHarmonicTimeData(
            data,
            times, basis_type='real',
            channel_convention='ACN',
            normalization='N3D',
            condon_shortley=False,
            caxis_spherical_harmonics=-1)

    assert time_data == time_data_def


def test_init_sh_frequency_data():
    data = np.ones((1, 4, 4), dtype=complex)
    frequencies = [1, 2, 3, 4]
    sh_freq_data = SphericalHarmonicFrequencyData(
        data, frequencies, basis_type='real', normalization='SN3D',
        channel_convention="ACN", condon_shortley=False,
        comment="", caxis_spherical_harmonics=-1)
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
            condon_shortley=False,
            caxis_spherical_harmonics=-1)

    assert freq_data == freq_data_def


def test_init_sh_time_data_invalid_sh_caxis():
    data = np.ones((1, 4, 4))
    times = [1, 2, 3, 4]

    caxis_sh = (1, 'b', 'c')
    with pytest.raises(ValueError,
                       match=re.escape("caxis_spherical_harmonics must "
                                       "contain only integers, but "
                                       f"got {type(caxis_sh[1]).__name__}.")):
        SphericalHarmonicTimeData(
            data, times,  basis_type='real', normalization='SN3D',
            channel_convention="ACN", condon_shortley=False,
            comment="", caxis_spherical_harmonics=caxis_sh)


def test_sh_freq_data_invalid_sh_caxis():
    data = np.ones((1, 4, 4), dtype=complex)
    frequencies = [1, 2, 3, 4]

    caxis_sh = (1, 'b', 'c')
    with pytest.raises(ValueError,
                       match=re.escape("caxis_spherical_harmonics must "
                                       "contain only integers, but "
                                       f"got {type(caxis_sh[1]).__name__}.")):
        SphericalHarmonicFrequencyData(
            data, frequencies,  basis_type='real', normalization='SN3D',
            channel_convention="ACN", condon_shortley=False,
            comment="", caxis_spherical_harmonics=caxis_sh)


def test_init_sh_time_data_wrong_sh_caxis():
    data = np.ones((1, 4, 4))
    times = [1, 2, 3, 4]

    with pytest.raises(ValueError,
                       match=re.escape("caxis_spherical_harmonics contains "
                                       "invalid axis. Axis must be in the "
                                       "range [-3, 3], but is -4.")):
        SphericalHarmonicTimeData(
            data, times,  basis_type='real', normalization='SN3D',
            channel_convention="ACN", condon_shortley=False,
            comment="", caxis_spherical_harmonics=-4)


def test_init_sh_frequency_data_wrong_sh_caxis():
    data = np.ones((1, 4, 4), dtype=complex)
    frequencies = [1, 2, 3, 4]

    with pytest.raises(ValueError,
                       match=re.escape("caxis_spherical_harmonics contains "
                                       "invalid axis. Axis must be in the "
                                       "range [-3, 3], but is -4.")):
        SphericalHarmonicFrequencyData(
            data, frequencies, basis_type='real', normalization='SN3D',
            channel_convention="ACN", condon_shortley=False,
            comment="", caxis_spherical_harmonics=-4)


def test_init_time_data_default_caxis_spherical_harmonics():
    """Test default caxis_sh and getter."""
    data = np.ones((1, 4, 4))
    times = [1, 2, 3, 4]

    time_data = SphericalHarmonicTimeData(
            data, times,  basis_type='real', normalization='SN3D',
            channel_convention="ACN", condon_shortley=False,
            comment="")
    assert time_data.caxis_spherical_harmonics == (-1,)


def test_init_freq_data_default_caxis_spherical_harmonics():
    """Test default caxis_sh and getter."""
    data = np.ones((1, 4, 4), dtype=complex)
    frequencies = [1, 2, 3, 4]

    freq_data = SphericalHarmonicFrequencyData(
            data, frequencies, basis_type='real', normalization='SN3D',
            channel_convention="ACN", condon_shortley=False,
            comment="")
    assert freq_data.caxis_spherical_harmonics == (-1,)


@pytest.mark.parametrize("caxis_sh", [-1, (-1,), (-2,)])
def test_init_time_data_caxis_spherical_harmonics(caxis_sh):
    """Test caxis_sh init and getter."""

    data = np.ones((1, 4, 4))
    times = [1, 2, 3, 4]

    time_data = SphericalHarmonicTimeData(
            data, times,  basis_type='real', normalization='SN3D',
            channel_convention="ACN", condon_shortley=False,
            comment="", caxis_spherical_harmonics=caxis_sh)

    expected = (caxis_sh,) if isinstance(caxis_sh, int) else caxis_sh
    assert time_data.caxis_spherical_harmonics == expected


@pytest.mark.parametrize("caxis_sh", [-1, (-1,), (-2,)])
def test_init_freq_data_caxis_spherical_harmonics(caxis_sh):
    """Test caxis_sh init and getter."""

    data = np.ones((1, 4, 4), dtype=complex)
    frequencies = [1, 2, 3, 4]

    freq_data = SphericalHarmonicFrequencyData(
            data, frequencies, basis_type='real', normalization='SN3D',
            channel_convention="ACN", condon_shortley=False,
            comment="", caxis_spherical_harmonics=caxis_sh)

    expected = (caxis_sh,) if isinstance(caxis_sh, int) else caxis_sh
    assert freq_data.caxis_spherical_harmonics == expected


def test_transpose_time_data():
    data = np.ones((3, 4, 256))
    times = range(256)
    caxis_sh = (-1,)

    time_data = SphericalHarmonicTimeData(
            data, times,  basis_type='real', normalization='SN3D',
            channel_convention="ACN", condon_shortley=False,
            comment="", caxis_spherical_harmonics=caxis_sh)

    time_data_transposed = time_data.transpose()
    data_transposed = np.moveaxis(time_data.time, 0, 1)

    assert np.allclose(data_transposed, time_data_transposed.time)
    assert time_data_transposed.caxis_spherical_harmonics == (-2, )


def _sh_time_data(cshape, caxis_sh, n_samples=8):
    """Create SphericalHarmonicTimeData with random data of given cshape."""
    rng = np.random.default_rng(42)
    data = rng.standard_normal(cshape + (n_samples, ))
    return SphericalHarmonicTimeData(
        data, range(n_samples), basis_type='real', normalization='SN3D',
        channel_convention="ACN", condon_shortley=False,
        comment="", caxis_spherical_harmonics=caxis_sh)


def test_init_spherical_harmonic_channels():
    """Test the private arrays tracking the spherical harmonic channels."""
    time_data = _sh_time_data((2, 4, 3), (-2, ))

    assert time_data._is_sh_channel.shape == (2, 4, 3)
    assert np.all(time_data._is_sh_channel)

    assert time_data._acn.shape == (2, 4, 3, 1)
    npt.assert_equal(time_data._acn[0, :, 0, 0], [0, 1, 2, 3])
    npt.assert_equal(time_data._acn[1, :, 2, 0], [0, 1, 2, 3])


def test_init_spherical_harmonic_channels_multiple_caxes():
    """Test the ACNs for multiple spherical harmonic channel axes."""
    time_data = _sh_time_data((4, 4, 2), (-3, -2))

    assert time_data._acn.shape == (4, 4, 2, 2)
    npt.assert_equal(time_data._acn[:, 1, 0, 0], [0, 1, 2, 3])
    npt.assert_equal(time_data._acn[1, :, 0, 0], [1, 1, 1, 1])
    npt.assert_equal(time_data._acn[1, :, 0, 1], [0, 1, 2, 3])
    npt.assert_equal(time_data._acn[:, 1, 0, 1], [1, 1, 1, 1])


def test_caxis_sh_non_sh_axis_of_same_size():
    """Test that a non SH axis with (N+1)^2 channels is not an SH axis."""
    time_data = _sh_time_data((4, 4), (-2, ))
    assert time_data.caxis_spherical_harmonics == (-2, )

    time_data = _sh_time_data((4, 4), (-1, ))
    assert time_data.caxis_spherical_harmonics == (-1, )


def test_caxis_sh_order_zero_singleton_axes():
    """Singleton channel axes are interchangeable for n_max=0."""
    time_data = _sh_time_data((1, 1), (-2, ))
    assert time_data.caxis_spherical_harmonics == (-1, )
    assert time_data.n_max == 0


def test_reshape_invalid_caxis_sh():
    """Reshaping is possible but caxis_spherical_harmonics raises an error if
    the SH channels are not contained in a channel axis.
    """
    time_data = _sh_time_data((4, 3), (-2, ))

    reshaped = time_data.reshape((3, 4))
    assert reshaped.cshape == (3, 4)

    with pytest.raises(ValueError, match=re.escape(
            "The spherical harmonic coefficients are not contained in a "
            "channel axis")):
        _ = reshaped.caxis_spherical_harmonics

    # data can not be converted without valid SH channel axis
    with pytest.raises(ValueError, match=re.escape(
            "The spherical harmonic coefficients are not contained in a "
            "channel axis")):
        _ = reshaped.time

    # n_max is still available
    assert reshaped.n_max == 1


def test_reshape_restores_caxis_sh():
    """Reshaping back to a valid cshape restores the SH channel axis."""
    time_data = _sh_time_data((4, 3), (-2, ))

    restored = time_data.reshape((3, 4)).reshape((4, 3))

    assert restored.caxis_spherical_harmonics == (-2, )
    npt.assert_equal(restored.time, time_data.time)


def test_reshape_valid_new_caxis_sh():
    data = np.ones((4, 2, 8, 256))
    times = range(256)
    caxis_sh = (-3,)

    time_data = SphericalHarmonicTimeData(
        data, times, basis_type='real', normalization='SN3D',
        channel_convention="ACN", condon_shortley=False,
        comment="", caxis_spherical_harmonics=caxis_sh)

    # reshape such that the non sh axis got reshaped
    reshaped = time_data.reshape((4, 16))

    # check if caxis_sh was updated properly
    assert reshaped.caxis_spherical_harmonics == (-2, )

    # check if new cshape is as desired
    assert reshaped.cshape == (4, 16)

    # original object must stay untouched (reshape returns a copy)
    assert time_data.cshape == (4, 2, 8)


def test_reshape_add_singleton_axis():
    """Test reshaping with additional singleton channel axes."""
    time_data = _sh_time_data((2, 4), (-1, ))

    reshaped = time_data.reshape((2, 1, 4, 1))
    assert reshaped.caxis_spherical_harmonics == (-2, )
    npt.assert_equal(reshaped.time, time_data.time[:, None, :, None])


@pytest.mark.parametrize(("cshape", "caxis_sh"), [
    ((4, ), (-1, )),
    ((1, 4), (-1, )),
    ((4, 1), (-2, ))])
def test_flatten_valid_caxis_sh(cshape, caxis_sh):
    """Flattening is valid if only the SH channel axis is left."""
    time_data = _sh_time_data(cshape, caxis_sh)

    flat = time_data.flatten()
    assert flat.cshape == (4, )
    assert flat.caxis_spherical_harmonics == (-1, )
    npt.assert_equal(flat.time, time_data.time.reshape((4, -1)))


@pytest.mark.parametrize(("cshape", "caxis_sh"), [
    ((2, 4), (-1, )),
    ((4, 2), (-2, )),
    ((4, 4), (-2, -1))])
def test_flatten_invalid_caxis_sh(cshape, caxis_sh):
    """Flattening is possible, but caxis_spherical_harmonics raises an error
    until the object is reshaped.
    """
    time_data = _sh_time_data(cshape, caxis_sh)

    flat = time_data.flatten()
    assert flat.cshape == (np.prod(cshape), )
    with pytest.raises(ValueError, match=re.escape(
            "The spherical harmonic coefficients are not contained in a "
            "channel axis")):
        _ = flat.caxis_spherical_harmonics

    # reshape to original cshape
    restored = flat.reshape(cshape)
    assert restored.caxis_spherical_harmonics == caxis_sh
    npt.assert_equal(restored.time, time_data.time)


def test_transpose_multiple_caxes_sh():
    """Test transposing an object with multiple SH channel axes."""
    time_data = _sh_time_data((4, 4, 2), (-3, -2))

    transposed = time_data.transpose(2, 0, 1)
    assert transposed.cshape == (2, 4, 4)
    assert transposed.caxis_spherical_harmonics == (-2, -1)
    npt.assert_equal(
        transposed.time, np.transpose(time_data.time, (2, 0, 1, 3)))

    # swap SH axes
    transposed = time_data.transpose(1, 0, 2)
    assert transposed.caxis_spherical_harmonics == (-2, -3)


def test_transpose_flattened():
    """Transposing a flattened object keeps the SH channels."""
    time_data = _sh_time_data((2, 4), (-1, ))

    restored = time_data.transpose().flatten().reshape((4, 2)).transpose()
    assert restored.caxis_spherical_harmonics == (-1, )
    npt.assert_equal(restored.time, time_data.time)


def test_set_time_with_new_cshape():
    """Setting data with a different cshape keeps the SH channel axis."""
    time_data = _sh_time_data((2, 4), (-1, ))

    time_data.time = np.ones((3, 2, 4, 8))
    assert time_data.cshape == (3, 2, 4)
    assert time_data.caxis_spherical_harmonics == (-1, )
    assert time_data._acn.shape == (3, 2, 4, 1)


def test_set_time_invalid_caxis_sh():
    """Setting data requires a valid SH channel axis."""
    time_data = _sh_time_data((2, 4), (-1, )).flatten()

    with pytest.raises(ValueError, match=re.escape(
            "The spherical harmonic coefficients are not contained in a "
            "channel axis")):
        time_data.time = np.ones((8, 8))


def test_broadcast_cshape():
    """Test the SH channels after broadcasting with pyfar."""
    time_data = _sh_time_data((4, ), (-1, ))

    broadcasted = pf.utils.broadcast_cshape(time_data, (3, 4))
    assert broadcasted.cshape == (3, 4)
    assert broadcasted.caxis_spherical_harmonics == (-1, )
