import numpy as np


def sht(signal, coordinates, n_max, basis_type="real", domain=None, axis=0,
        channel_convention='acn', normalization='n3d', condon_shortley=True,
        inverse_transform='pseudo_inverse'):
    """Compute the spherical harmonics transform at a certain order N

    Parameters
    ----------
    signal: Signal
        the signal for which the spherical harmonics transform is computed
    coordinates: :class:`spharpy.samplings.Coordinates`, :doc:`pf.Coordinates
        <pyfar:classes/pyfar.coordinates>`
        Coordinates on which the signal has been sampled
    n_max: integer
        Spherical harmonic order
    sh_kind: str
        Use real or complex valued SH bases ``'real'``, ``'complex'``
        default is ``'real'``
    axis: integer
        Axis along which the SH transform is computed

    Returns
    ----------
    AmbisonicsSignal

    References
    ----------

        [1] Rafaely, B. (2015). Fundamentals of Spherical Array Processing,
            (J. Benesty and W. Kellermann, Eds.) Springer Berlin Heidelberg,
            2nd ed., 196 pages. doi:10.1007/978-3-319-99561-8
        [2] Ramani Duraiswami, Dimitry N. Zotkin, and Nail A. Gumerov: "Inter-
            polation and range extrapolation of HRTFs." IEEE Int. Conf.
            Acoustics, Speech, and Signal Processing (ICASSP), Montreal,
            Canada, May 2004, p. 45-48, doi: 10.1109/ICASSP.2004.1326759.
    """

    if domain is None:
        domain = signal.domain

    #  coordinates = convert_coordinates(coordinates)

    if not signal.cshape[axis] == coordinates.csize:
        if coordinates.csize not in signal.cshape:
            raise ValueError("Signal shape does not match "
                             "number of coordinates.")
        else:
            axis = signal.cshape.index(coordinates.csize)
            warnings.warn(f"Compute SHT along axis={axis}.", UserWarning)

    spherical_harmonics = SphericalHarmonics(
        n_max=n_max, coordinates=coordinates, basis_type=basis_type,
        channel_convention=channel_convention,
        normalization=normalization, inverse_transform=inverse_transform,
        condon_shortley=condon_shortley)

    if domain == "time":
        data = signal.time
    elif domain == "freq":
        data = signal.freq
    else:
        raise ValueError("Domain should be ``'time'`` or ``'freq'`` but "
                         f"is {domain}.")

    Y_inv = spherical_harmonics.basis_inv  # [1] Eq. 3.34
    data_nm = np.tensordot(Y_inv, data, [1, axis])

    return AmbisonicsSignal(data=data_nm, domain=domain, n_max=n_max,
                            basis_type=basis_type, normalization=normalization,
                            channel_convention=channel_convention,
                            condon_shortley=condon_shortley,
                            sampling_rate=signal.sampling_rate,
                            fft_norm=signal.fft_norm,
                            comment=signal.comment)


def isht(ambisonics_signal, coordinates):
    """Compute the inverse spherical harmonics transform at a certain order N

    Parameters
    ----------
    ambisonics_signal: Signal
        The ambisonics signal for which the inverse spherical harmonics
        transform is computed
    coordinates: :class:`spharpy.samplings.Coordinates`, :doc:`pf.Coordinates
                 <pyfar:classes/pyfar.coordinates>`
        Coordinates for which the inverse SH transform is computed
    """
    # get spherical harmonics basis functions of same type as the the
    # ambisonics signals but for the passed coordinates
    spherical_harmonics = SphericalHarmonics(
        ambisonics_signal.n_max,
        coordinates=coordinates,
        basis_type=ambisonics_signal.basis_type,
        channel_convention=ambisonics_signal.channel_convention,
        normalization=ambisonics_signal.normalization,
        condon_shortley=ambisonics_signal.condon_shortley)

    if ambisonics_signal.domain == "time":
        data_nm = ambisonics_signal.time
    else:
        data_nm = ambisonics_signal.freq

    # perform inverse transform
    data = np.tensordot(spherical_harmonics.basis, data_nm, [1, 0])

    return Signal(data, ambisonics_signal.sampling_rate,
                  fft_norm=ambisonics_signal.fft_norm,
                  comment=ambisonics_signal.comment)
