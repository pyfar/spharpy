import numpy as np
import scipy.special as special
import spharpy.special as _special
import pyfar as pf
import warnings
from spharpy import SamplingSphere


def acn_to_nm(acn):
    r"""
    Calculate the order n and degree m from the linear coefficient index.

    The linear index corresponds to the Ambisonics Channel Convention [#]_.

    .. math::

        n = \lfloor \sqrt{\mathrm{acn} + 1} \rfloor - 1

        m = \mathrm{acn} - n^2 -n


    References
    ----------
    .. [#]  C. Nachbar, F. Zotter, E. Deleflie, and A. Sontacchi, “Ambix - A
            Suggested Ambisonics Format (revised by F. Zotter),” International
            Symposium on Ambisonics and Spherical Acoustics,
            vol. 3, pp. 1-11, 2011.


    Parameters
    ----------
    acn : ndarray, int
        Linear index

    Parameters
    ----------
    n : ndarray, int
        Spherical harmonic order
    m : ndarray, int
        Spherical harmonic degree

    """
    acn = np.asarray(acn, dtype=int)

    n = (np.ceil(np.sqrt(acn + 1)) - 1)
    m = acn - n**2 - n

    n = n.astype(int, copy=False)
    m = m.astype(int, copy=False)

    return n, m


def nm_to_acn(n, m):
    r"""
    Calculate the linear index coefficient for a order n and degree m,

    The linear index corresponds to the Ambisonics Channel Convention [#]_.

    .. math::

        \mathrm{acn} = n^2 + n + m

    References
    ----------
    .. [#]  C. Nachbar, F. Zotter, E. Deleflie, and A. Sontacchi, “Ambix - A
            Suggested Ambisonics Format (revised by F. Zotter),” International
            Symposium on Ambisonics and Spherical Acoustics,
            vol. 3, pp. 1-11, 2011.


    Parameters
    ----------
    n : ndarray, int
        Spherical harmonic order
    m : ndarray, int
        Spherical harmonic degree

    Returns
    -------
    acn : ndarray, int
        Linear index

    """
    n = np.asarray(n, dtype=int)
    m = np.asarray(m, dtype=int)

    if n.size != m.size:
        raise ValueError("n and m need to be of the same size")

    return n**2 + n + m


def nm_to_fuma(n, m):
    r"""
    Calculate the FuMa channel index for a given spherical harmonic order n
    and degree m, according to the FuMa (Furse-Malham)
    Channel Ordering Convention.

    Parameters
    ----------
    n : integer, ndarray
        Spherical harmonic order
    m : integer, ndarray
        Spherical harmonic degree

    Returns
    -------
    fuma : integer
        FuMa channel index

    References
    ----------
    .. [#]  D. Malham, "Higher order Ambisonic systems” Space in Music –
             Music in Space (Mphil thesis). University of York. pp. 2–3., 2003.
    """

    fuma_mapping = [0, 2, 3, 1, 8, 6, 4, 5, 7, 15, 13, 11, 9, 10, 12, 14]

    n = np.asarray([n], dtype=int)
    m = np.asarray([m], dtype=int)

    if n.shape != m.shape:
        raise ValueError("n and m need to be of the same size")

    # convert (n, m) to the ACN index
    acn = nm_to_acn(n, m)

    if np.any(acn < 0) or np.any(acn >= len(fuma_mapping)):
        raise ValueError(
            "nm2fuma only supports up to 3rd order"
        )

    acn = np.atleast_2d(acn).T
    fuma = np.array([], dtype=int)
    for a in acn:
        fuma = np.append(fuma, fuma_mapping.index(a))

    return fuma


def fuma_to_nm(fuma):
    r"""
    Calculate the spherical harmonic order n and degree m for a linear
    coefficient index, according to the FuMa (Furse-Malham)
    Channel Ordering Convention [#]_.

    FuMa = WXYZ | RSTUV | KLMNOPQ
    ACN = WYZX | VTRSU | QOMKLNP

    Parameters
    ----------
    fuma : integer, ndarray
        FuMa channel index

    Returns
    -------
    n : integer, ndarray
        Spherical harmonic order
    m : integer, ndarray
        Spherical harmonic degree

    References
    ----------
    .. [#]  D. Malham, "Higher order Ambisonic systems” Space in Music –
             Music in Space (Mphil thesis). University of York. pp. 2–3., 2003.
    """

    fuma_mapping = [0, 2, 3, 1, 8, 6, 4, 5, 7, 15, 13, 11, 9, 10, 12, 14]

    if not isinstance(fuma, np.ndarray):
        fuma = np.asarray([fuma], dtype=int)

    if np.any(fuma) < 0 or np.any(fuma >= len(fuma_mapping)):
        raise ValueError(
            "Invalid FuMa channel index, must be between 0 and 15 "
            "(supported up to 3rd order)"
        )

    acn = np.array([], dtype=int)
    for f in fuma:
        acn = np.append(acn, fuma_mapping[int(f)])

    n, m = acn_to_nm(acn)
    return n, m


def n3d_to_maxn(acn):
    """
    Calculate the scaling factor which converts from N3D (normalized 3D)
    normalization to max N normalization. ACN must be less or equal to 15.

    Parameters
    ----------
    acn : integer, ndarray
          linear index

    Returns
    -------
    maxN : float
        Scaling factor which converts from N3D to max N
    """

    if not isinstance(acn, np.ndarray):
        acn = np.asarray([acn], dtype=int)

    if np.any(acn) > 15:
        raise ValueError("acn must be less than or "
                         "equal to 15")
    valid_maxN = [
        np.sqrt(1 / 2),
        np.sqrt(1 / 3),
        np.sqrt(1 / 3),
        np.sqrt(1 / 3),
        2 / np.sqrt(15),
        2 / np.sqrt(15),
        np.sqrt(1 / 5),
        2 / np.sqrt(15),
        2 / np.sqrt(15),
        np.sqrt(8 / 35),
        3 / np.sqrt(35),
        np.sqrt(45 / 224),
        np.sqrt(1 / 7),
        np.sqrt(45 / 224),
        3 / np.sqrt(35),
        np.sqrt(8 / 35),
    ]

    maxN = np.array([], dtype=int)
    for a in acn:
        maxN = np.append(maxN, valid_maxN[int(a)])

    return maxN


def n3d_to_sn3d_norm(n):
    """
    Calculate the scaling factor which converts from N3D (normalized 3D)
    normalization to SN3D (Schmidt semi-normalized 3D) normalization.

    Parameters
    ----------
    n : integer, ndarray
        Spherical harmonic order

    Returns
    -------
    sn3d : float, ndarray
        normalization factor which converts from N3D to SN3D
    """
    return 1 / np.sqrt(2 * n + 1)


def renormalize(data, channel_convention, current_norm, target_norm, axis):
    """
    Renormalize spherical harmonics coefficients or basis functions.

    Parameters
    ----------
    data : ndarray
        Data which should be renormalized, either spherical harmonics
        coefficients or basis functions.
    channel_convention : str
        Channel convention of the data which should be renormalized. Valid
        conventions are `"acn"` or `"fuma"`.
    current_norm : str
        Current normalization. Valid normalizations are `"n3d"`, `"maxN"`, or
        `"sn3d"`.
    target_norm : str
        Desired normalization. Valid normalizations are `"n3d"`,
        `"maxN"`, or `"sn3d"`.
    axis : integer
        Axis along which the renormalization should be applied

    Returns
    -------
    data : ndarray
        Renormalized data
    """
    if channel_convention not in ["acn", "fuma"]:
        raise ValueError("Invalid channel convention. Has to be 'acn' "
                         f"or 'fuma', but is {channel_convention}")

    if current_norm not in ["n3d", "maxN", "sn3d"]:
        raise ValueError("Invalid normalization. Has to be 'sn3d', "
                         f"'n3d', or 'maxN', but is {current_norm}")

    if target_norm not in ["n3d", "maxN", "sn3d"]:
        raise ValueError("Invalid normalization. Has to be 'sn3d', "
                         f"'n3d', or 'maxN', but is {target_norm}")
    acn = np.arange(data.shape[axis])

    if channel_convention == "fuma":
        orders, _ = fuma_to_nm(acn)
    else:
        orders, _ = acn_to_nm(acn)

    # prepare helper for reshaping
    shape = [1] * data.ndim
    shape[axis] = -1

    data_renorm = data.copy()
    if current_norm == 'n3d':
        if target_norm == "sn3d":
            data_renorm *= n3d_to_sn3d_norm(orders).reshape(shape)
        elif target_norm == "maxN":
            data_renorm *= n3d_to_maxn(acn).reshape(shape)

    if current_norm == 'sn3d':
        # convert to n3d
        data_renorm /= n3d_to_sn3d_norm(orders).reshape(shape)
        if target_norm == "maxN":
            data_renorm *= n3d_to_maxn(acn).reshape(shape)

    if current_norm == 'maxN':
        # convert to n3d
        data_renorm /= n3d_to_maxn(acn).reshape(shape)
        if target_norm == "sn3d":
            data_renorm *= n3d_to_sn3d_norm(orders).reshape(shape)
    return data_renorm


def change_channel_convention(data, current, target, axis):
    """
    Change the channel convention of spherical harmonics coefficients
    or basis functions.

    Parameters
    ----------
    data : ndarray
        Data of which channel convention should be changed. Either spherical
        harmonics coefficients or bases functions.
    current : str
        Current channel convention. Valid conventions are `"acn"` or `"fuma"`.
    target : str
        Desired channel convention. Valid conventions are `"acn"` or `"fuma"`.
    axis : integer
        Axis along which the channel convention should be changed

    Returns
    -------
    data : ndarray
        Data with changed channel convention
    """
    if current not in ["acn", "fuma"]:
        raise ValueError("Invalid current channel convention. Has to be "
                         f"'acn' or 'fuma', but is {current}")

    if target not in ["acn", "fuma"]:
        raise ValueError("Invalid target channel convention. Has to be 'acn' "
                         f"or 'fuma', but is {target}")

    acn = np.arange(data.shape[axis])
    if current == 'acn':
        n, m = acn_to_nm(acn)
        idx = nm_to_fuma(n, m)
    elif current == 'fuma':
        n, m = fuma_to_nm(acn)
        idx = nm_to_acn(n, m)

    return np.take(data, idx, axis=axis)


def spherical_harmonic_basis(
        n_max, coordinates, normalization="n3d", channel_convention="acn",
        condon_shortley='auto'):
    r"""
    Calculates the complex valued spherical harmonic basis matrix.

    See [#]_ and [#]_ for details.
    See also :py:func:`spherical_harmonic_basis_real`.

    .. math::
        Y_n^m(\theta, \phi) =  CS_m N_{nm} P_{nm}(cos(\theta)) e^{im\phi}

    - :math:`n` is the degree
    where:

    - :math:`n` is the degree
    - :math:`m` is the order
    - :math:`P_{nm}` is the associated Legendre function
    - :math:`N_{nm}` is the normalization term
    - :math:`CS_m` is the Condon-Shortley phase term
    - :math:`\theta` is the colatitude (angle from the positive z-axis)
    - :math:`\phi` is the azimuth (angle from the positive x-axis in the
      xy-plane), see :doc:`pf.Coordinates <pyfar:classes/pyfar.coordinates>`

    References
    ----------
    .. [#]  E. G. Williams, Fourier Acoustics. Academic Press, 1999.
    .. [#]  B. Rafaely, Fundamentals of Spherical Array Processing, vol. 8.
            Springer, 2015.

    Parameters
    ----------
    n_max : integer
        Spherical harmonic order
    coordinates : :py:class:`pf.Coordinates <pyfar.coordinates.Coordinates>` \
        or :py:class:`sp.SamplingSphere \
        <spharpy.samplings.coordinates.SamplingSphere>`
        objects with sampling points for which the basis matrix is calculated
    normalization : str, optional
        Normalization convention, either ``'n3d'``, ``'maxN'`` or ``'sn3d'``.
        The default is ``'n3d'``.
        (maxN is only supported up to 3rd order)
    channel_convention : str, optional
        Channel ordering convention, either ``'acn'`` or ``'fuma'``.
        The default is ``'acn'``.
        (FuMa is only supported up to 3rd order)
    condon_shortley : bool or str, optional
        Whether to include the Condon-Shortley phase term. If ``True`` or
        ``'auto'``, Condon-Shortley is included, if ``False`` it is not
        included. The default is ``'auto'``.

    Returns
    -------
    Y : ndarray, complex
        Complex spherical harmonic basis matrix

    Examples
    --------
    >>> import spharpy
    >>> n_max = 2
    >>> coordinates = spharpy.samplings.icosahedron()
    >>> Y = spharpy.spherical.spherical_harmonic_basis(n_max, coordinates)

    """
    if channel_convention == "fuma" and n_max > 3:
        raise ValueError(
            "FuMa channel convention is only supported up to 3rd order.")

    if normalization == "maxN" and n_max > 3:
        raise ValueError(
            "MaxN normalization is only supported up to 3rd order.")

    if not isinstance(condon_shortley, bool) and condon_shortley != 'auto':
        raise ValueError(
            "Condon_shortley has to be a bool, or 'auto'.")

    if condon_shortley == 'auto':
        condon_shortley = True

    n_coeff = (n_max + 1) ** 2

    basis = np.zeros((coordinates.csize, n_coeff), dtype=complex)

    for acn in range(n_coeff):
        if channel_convention == "fuma":
            order, degree = fuma_to_nm(acn)
        else:
            order, degree = acn_to_nm(acn)
        basis[:, acn] = _special.spherical_harmonic(
            order, degree, coordinates.colatitude, coordinates.azimuth
        )
        if normalization == "sn3d":
            basis[:, acn] *= n3d_to_sn3d_norm(order)
        elif normalization == "maxN":
            basis[:, acn] *= n3d_to_maxn(acn)
        if not condon_shortley:
            # Condon-Shortley phase term is already included in
            # the special.spherical_harmonic function
            # so need to divide by (-1)^m
            basis[:, acn] /= (-1) ** float(degree)
    return basis


def spherical_harmonic_basis_gradient(n_max, coordinates, normalization="n3d",
                                      channel_convention="acn",
                                      condon_shortley='auto'):
    r"""
    Calculates the unit sphere gradients of the complex spherical harmonics.


    The angular parts of the gradient are defined as

    .. math::

        \nabla_{(\theta, \phi)} Y_n^m(\theta, \phi) =
        \frac{1}{\sin \theta} \frac{\partial Y_n^m(\theta, \phi)}
        {\partial \phi} \vec{e}_\phi +
        \frac{\partial Y_n^m(\theta, \theta)}
        {\partial \theta} \vec{e}_\theta .


    This implementation avoids singularities at the poles using identities
    derived in [#]_.


    References
    ----------
    .. [#]  E. G. Williams, Fourier Acoustics. Academic Press, 1999.
    .. [#]  J. Du, C. Chen, V. Lesur, and L. Wang, “Non-singular spherical
            harmonic expressions of geomagnetic vector and gradient tensor
            fields in the local north-oriented reference frame,” Geoscientific
            Model Development, vol. 8, no. 7, pp. 1979-1990, Jul. 2015.

    Parameters
    ----------
    n_max : int
        Spherical harmonic order
    coordinates : :py:class:`pf.Coordinates <pyfar.coordinates.Coordinates>` \
        or :py:class:`sp.SamplingSphere \
        <spharpy.samplings.coordinates.SamplingSphere>`
        objects with sampling points for which the basis matrix is
        calculated
    normalization : str, optional
        Normalization convention, either ``'n3d'``, ``'maxN'`` or ``'sn3d'``.
        The default is ``'n3d'``.
        (maxN is only supported up to 3rd order)
    channel_convention : str, optional
        Channel ordering convention, either ``'acn'`` or ``'fuma'``.
        The default is ``'acn'``.
        (FuMa is only supported up to 3rd order)
    condon_shortley : bool or str, optional
        Whether to include the Condon-Shortley phase term. If ``True`` or
        ``'auto'``, Condon-Shortley is included, if ``False`` it is not
        included. The default is ``'auto'``.

    Returns
    -------
    grad_theta : ndarray, complex
        Gradient with regard to the co-latitude angle.
    grad_azimuth : ndarray, complex
        Gradient with regard to the azimuth angle.

    Examples
    --------
    >>> import spharpy
    >>> n_max = 2
    >>> coordinates = spharpy.samplings.icosahedron()
    >>> grad_theta, grad_phi = /
        spharpy.spherical.spherical_harmonic_basis_gradient(n_max, coordinates)


    """
    if channel_convention == "fuma" and n_max > 3:
        raise ValueError(
            "FuMa channel convention is only supported up to 3rd order.")

    if normalization == "maxN" and n_max > 3:
        raise ValueError(
            "MaxN normalization is only supported up to 3rd order.")

    if not isinstance(condon_shortley, bool) and condon_shortley != 'auto':
        raise ValueError(
            "Condon_shortley has to be a bool, or 'auto'.")

    if condon_shortley == 'auto':
        condon_shortley = True

    n_points = coordinates.csize
    n_coeff = (n_max+1)**2
    theta = coordinates.colatitude
    phi = coordinates.azimuth
    grad_theta = np.zeros((n_points, n_coeff), dtype=complex)
    grad_phi = np.zeros((n_points, n_coeff), dtype=complex)

    for acn in range(n_coeff):
        if channel_convention == "fuma":
            n, m = fuma_to_nm(acn)
        else:
            n, m = acn_to_nm(acn)

        grad_theta[:, acn] = _special.spherical_harmonic_derivative_theta(
            n, m, theta, phi
        )
        grad_phi[:, acn] = _special.spherical_harmonic_gradient_phi(
            n, m, theta, phi)

        factor = 1.0
        if normalization == "sn3d":
            factor = n3d_to_sn3d_norm(n)
        elif normalization == "maxN":
            factor *= n3d_to_maxn(acn)

        if not condon_shortley:
            # Condon-Shortley phase term is already included in
            # the special.spherical_harmonic function
            # so need to divide by (-1)^m
            factor /= (-1) ** float(m)

        grad_theta[:, acn] *= factor
        grad_phi[:, acn] *= factor

    return grad_theta, grad_phi


def spherical_harmonic_basis_real(
        n_max, coordinates, normalization="n3d", channel_convention="acn",
        condon_shortley='auto'):
    r"""
    Calculates the real valued spherical harmonic basis matrix.
    See also :py:func:`spherical_harmonic_basis`.

    .. math::

        Y_n^m(\theta, \phi) = \sqrt{\frac{2n+1}{4\pi}
        \frac{(n-|m|)!}{(n+|m|)!}} P_n^{|m|}(\cos \theta)
        \begin{cases}
            \displaystyle \cos(|m|\phi),  & \text{if $m \ge 0$} \newline
            \displaystyle \sin(|m|\phi) ,  & \text{if $m < 0$}
        \end{cases}

    Parameters
    ----------
    n_max : int
        Spherical harmonic order
    coordinates : :py:class:`pf.Coordinates <pyfar.coordinates.Coordinates>` \
        or :py:class:`sp.SamplingSphere \
        <spharpy.samplings.coordinates.SamplingSphere>`
        objects with sampling points for which the basis matrix is
        calculated
    normalization : str, optional
        Normalization convention, either ``'n3d'``, ``'maxN'`` or ``'sn3d'``.
        The default is ``'n3d'``.
        (maxN is only supported up to 3rd order)
    channel_convention : str, optional
        Channel ordering convention, either ``'acn'`` or ``'fuma'``.
        The default is ``'acn'``.
        (FuMa is only supported up to 3rd order)
    condon_shortley : bool or str, optional
        Whether to include the Condon-Shortley phase term. If ``True``,
        Condon-Shortley is included, if ``False`` or ``'auto'``, it is not
        included. The default is ``'auto'``.

    Returns
    -------
    Y : ndarray, float
        Real valued spherical harmonic basis matrix.


    """
    if channel_convention == "fuma" and n_max > 3:
        raise ValueError(
            "FuMa channel convention is only supported up to 3rd order.")

    if normalization == "maxN" and n_max > 3:
        raise ValueError(
            "MaxN normalization is only supported up to 3rd order.")

    if not isinstance(condon_shortley, bool) and condon_shortley != 'auto':
        raise ValueError(
            "Condon_shortley has to be a bool, or 'auto'.")

    if condon_shortley == 'auto':
        condon_shortley = False

    n_coeff = (n_max + 1) ** 2

    basis = np.zeros((coordinates.csize, n_coeff), dtype=float)

    for acn in range(n_coeff):
        if channel_convention == "fuma":
            order, degree = fuma_to_nm(acn)
        else:
            order, degree = acn_to_nm(acn)
        basis[:, acn] = _special.spherical_harmonic_real(
            order, degree, coordinates.colatitude, coordinates.azimuth
        )
        if normalization == "sn3d":
            basis[:, acn] *= n3d_to_sn3d_norm(order)
        elif normalization == "maxN":
            basis[:, acn] *= n3d_to_maxn(acn)
        if condon_shortley:
            # Condon-Shortley phase term is not included in
            # the special.spherical_harmonic function
            basis[:, acn] *= (-1) ** float(degree)

    return basis


def spherical_harmonic_basis_gradient_real(n_max, coordinates,
                                           normalization="n3d",
                                           channel_convention="acn",
                                           condon_shortley='auto'):
    r"""
    Calculates the unit sphere gradients of the real valued spherical
    harmonics.

    The spherical harmonic functions are fully normalized (N3D) and follow
    the AmbiX phase convention [#]_.

    The angular parts of the gradient are defined as

    .. math::

        \nabla_{(\theta, \phi)} Y_n^m(\theta, \phi) =
        \frac{1}{\sin \theta} \frac{\partial Y_n^m(\theta, \phi)}
        {\partial \phi} \vec{e}_\phi +
        \frac{\partial Y_n^m(\theta, \theta)}
        {\partial \theta} \vec{e}_\theta .


    This implementation avoids singularities at the poles using identities
    derived in [#]_.


    References
    ----------
    .. [#]  C. Nachbar, F. Zotter, E. Deleflie, and A. Sontacchi, “Ambix - A
            Suggested Ambisonics Format (revised by F. Zotter),” International
            Symposium on Ambisonics and Spherical Acoustics,
            vol. 3, pp. 1-11, 2011.
    .. [#]  J. Du, C. Chen, V. Lesur, and L. Wang, “Non-singular spherical
            harmonic expressions of geomagnetic vector and gradient tensor
            fields in the local north-oriented reference frame,” Geoscientific
            Model Development, vol. 8, no. 7, pp. 1979-1990, Jul. 2015.

    Parameters
    ----------
    n_max : int
        Spherical harmonic order
    coordinates : :py:class:`pf.Coordinates <pyfar.coordinates.Coordinates>` \
        or :py:class:`sp.SamplingSphere \
        <spharpy.samplings.coordinates.SamplingSphere>`
        objects with sampling points for which the basis matrix is
        calculated
    normalization : str, optional
        Normalization convention, either ``'n3d'``, ``'maxN'`` or ``'sn3d'``.
        The default is ``'n3d'``.
        (maxN is only supported up to 3rd order)
    channel_convention : str, optional
        Channel ordering convention, either ``'acn'`` or ``'fuma'``.
        The default is ``'acn'``.
        (FuMa is only supported up to 3rd order)
    condon_shortley : bool or str, optional
        Whether to include the Condon-Shortley phase term. If ``True``,
        Condon-Shortley is included, if ``False`` or ``'auto'``, it is not
        included. The default is ``'auto'``.

    Returns
    -------
    grad_theta : ndarray, float
        Gradient with respect to the co-latitude angle.
    grad_phi : ndarray, float
        Gradient with respect to the azimuth angle.

    """
    if channel_convention == "fuma" and n_max > 3:
        raise ValueError(
            "FuMa channel convention is only supported up to 3rd order.")

    if normalization == "maxN" and n_max > 3:
        raise ValueError(
            "MaxN normalization is only supported up to 3rd order.")

    if not isinstance(condon_shortley, bool) and condon_shortley != 'auto':
        raise ValueError(
            "Condon_shortley has to be a bool, or 'auto'.")

    if condon_shortley == 'auto':
        condon_shortley = False

    n_points = coordinates.csize
    n_coeff = (n_max + 1) ** 2
    theta = coordinates.colatitude
    phi = coordinates.azimuth
    grad_theta = np.zeros((n_points, n_coeff), dtype=float)
    grad_phi = np.zeros((n_points, n_coeff), dtype=float)

    for acn in range(n_coeff):
        if channel_convention == "fuma":
            n, m = fuma_to_nm(acn)
        else:
            n, m = acn_to_nm(acn)

        grad_theta[:, acn] = \
            _special.spherical_harmonic_derivative_theta_real(
                n, m, theta, phi)
        grad_phi[:, acn] = \
            _special.spherical_harmonic_gradient_phi_real(
                n, m, theta, phi)

        factor = 1.0
        if normalization == "sn3d":
            factor = n3d_to_sn3d_norm(n)
        elif normalization == "maxN":
            factor *= n3d_to_maxn(acn)

        if condon_shortley:
            # Condon-Shortley phase term is not included in
            # the special.spherical_harmonic function
            factor *= (-1) ** float(m)

        grad_theta[:, acn] *= factor
        grad_phi[:, acn] *= factor

    return grad_theta, grad_phi


def modal_strength(n_max,
                   kr,
                   arraytype='rigid'):
    r"""
    Modal strength function for microphone arrays.

    .. math::

        b(kr) =
        \begin{cases}
            \displaystyle 4\pi i^n j_n(kr),  & \text{open} \newline
            \displaystyle  4\pi i^{(n-1)} \frac{1}{(kr)^2 h_n^\prime(kr)},
                & \text{rigid} \newline
            \displaystyle  4\pi i^n (j_n(kr) - i j_n^\prime(kr)),
                & \text{cardioid}
        \end{cases}


    This implementation uses the second order Hankel function, see [#]_ for an
    overview of the corresponding sign conventions.

    References
    ----------
    .. [#]  V. Tourbabin and B. Rafaely, “On the Consistent Use of Space and
            Time Conventions in Array Processing,” vol. 101, pp. 470–473, 2015.


    Parameters
    ----------
    n_max : int
        The spherical harmonic order
    kr : ndarray, float
        Wave number * radius
    arraytype : string
        Array configuration. Can be a microphones mounted on a rigid sphere,
        on a virtual open sphere or cardioid microphones on an open sphere.

    Returns
    -------
    B : ndarray, float
        Modal strength diagonal matrix

    """
    n_coeff = (n_max+1)**2
    n_bins = kr.shape[0]

    modal_strength_mat = np.zeros((n_bins, n_coeff, n_coeff), dtype=complex)

    for n in range(n_max+1):
        bn = _modal_strength(n, kr, arraytype)
        for m in range(-n, n+1):
            acn = n*n + n + m
            modal_strength_mat[:, acn, acn] = bn

    return np.squeeze(modal_strength_mat)


def _modal_strength(n, kr, config):
    """Helper function for the calculation of the modal strength for
    plane waves"""
    if config == 'open':
        ms = 4*np.pi*pow(1.0j, n) * _special.spherical_bessel(n, kr)
    elif config == 'rigid':
        ms = 4*np.pi*pow(1.0j, n+1) / \
            _special.spherical_hankel(n, kr, derivative=True) / (kr)**2
    elif config == 'cardioid':
        ms = 4*np.pi*pow(1.0j, n) * \
            (_special.spherical_bessel(n, kr) -
                1.0j * _special.spherical_bessel(n, kr, derivative=True))
    else:
        raise ValueError("Invalid configuration.")

    return ms


def aperture_vibrating_spherical_cap(
        n_max,
        rad_sphere,
        rad_cap):
    r"""
    Aperture function for a vibrating spherical cap.

    The cap has radius :math:`r_c` and is mounted in a rigid sphere with
    radius :math:`r_s` [#]_, [#]_

    .. math::

        a_n (r_{s}, \alpha) = 4 \pi
        \begin{cases}
            \displaystyle \left(2n+1\right)\left[
                P_{n-1} \left(\cos\alpha\right) -
                P_{n+1} \left(\cos\alpha\right) \right],
                & {n>0} \newline
            \displaystyle  (1 - \cos\alpha)/2,  & {n=0}
        \end{cases}

    where :math:`\alpha = \arcsin \left(\frac{r_c}{r_s} \right)` is the
    aperture angle.

    Parameters
    ----------
    n_max : integer, ndarray
        Maximal spherical harmonic order
    r_sphere : double, ndarray
        Radius of the sphere
    r_cap : double
        Radius of the vibrating cap

    Returns
    -------
    A : ndarray, float
        Aperture function in diagonal matrix form with shape
        :math:`[(n_{max}+1)^2~\times~(n_{max}+1)^2]`

    References
    ----------
    .. [#]  E. G. Williams, Fourier Acoustics. Academic Press, 1999.
    .. [#]  B. Rafaely and D. Khaykin, “Optimal Model-Based Beamforming and
             Independent Steering for Spherical Loudspeaker Arrays,” IEEE
             Transactions on Audio, Speech, and Language Processing, vol. 19,
             no. 7, pp. 2234-2238, 2011

    Notes
    -----
    Eq. (3) in the second Ref. contains an error, here, the power of 2 on pi is
    omitted on the normalization term.

    """
    angle_cap = np.arcsin(rad_cap / rad_sphere)
    arg = np.cos(angle_cap)
    n_sh = (n_max+1)**2

    aperture = np.zeros((n_sh, n_sh), dtype=float)

    aperture[0, 0] = (1-arg)*2*np.pi
    for n in range(1, n_max+1):
        legendre_minus = special.legendre(n-1)(arg)
        legendre_plus = special.legendre(n+1)(arg)
        legendre_term = legendre_minus - legendre_plus
        for m in range(-n, n+1):
            acn = nm_to_acn(n, m)
            aperture[acn, acn] = legendre_term * 4 * np.pi / (2*n+1)

    return aperture


def radiation_from_sphere(
        n_max,
        rad_sphere,
        k,
        distance,
        density_medium=1.2,
        speed_of_sound=343.0):
    r"""
    Radiation function in SH for a vibrating spherical cap.
    Includes the radiation impedance and the propagation to a arbitrary
    distance from the sphere.
    The sign and phase conventions result in a positive pressure response for
    a positive cap velocity with the intensity vector pointing away from the
    source. [#]_, [#]_

    TODO: This function does not have a test yet.

    References
    ----------
    .. [#]  E. G. Williams, Fourier Acoustics. Academic Press, 1999.
    .. [#]  F. Zotter, A. Sontacchi, and R. Höldrich, “Modeling a spherical
            loudspeaker system as multipole source,” in Proceedings of the 33rd
            DAGA German Annual Conference on Acoustics, 2007, pp. 221-222.


    Parameters
    ----------
    n_max : integer
        Maximal spherical harmonic order
    rad_sphere : float
        Radius of the sphere
    k : ndarray, float
        Wave number
    distance : float
        Radial distance from the center of the sphere
    density_medium : float
        Density of the medium surrounding the sphere. Default is 1.2 for air.
    speed_of_sound : float
        Speed of sound in m/s

    Returns
    -------
    R : ndarray, float
        Radiation function in diagonal matrix form with shape
        :math:`[K \times (n_{max}+1)^2~\times~(n_{max}+1)^2]`

    """
    n_sh = (n_max+1)**2

    k = np.atleast_1d(k)
    n_bins = k.shape[0]
    radiation = np.zeros((n_bins, n_sh, n_sh), dtype=complex)

    for n in range(n_max+1):
        hankel = _special.spherical_hankel(n, k*distance, kind=2)
        hankel_prime = _special.spherical_hankel(
            n, k*rad_sphere, kind=2, derivative=True)
        radiation_order = -1j * hankel/hankel_prime * \
            density_medium * speed_of_sound
        for m in range(-n, n+1):
            acn = nm_to_acn(n, m)
            radiation[:, acn, acn] = radiation_order

    return radiation


def sid(n_max):
    """Calculates the SID indices up to spherical harmonic order n_max.
    The SID indices were originally proposed by Daniel [#]_, more recently
    ACN indexing has been favored and is used in the AmbiX format [#]_.

    Parameters
    ----------
    n_max : int
        The maximum spherical harmonic order

    Returns
    -------
    sid_n : ndarray, int
        The SID indices for all orders
    sid_m : ndarray, int
        The SID indices for all degrees

    References
    ----------
    .. [#]  J. Daniel, “Représentation de champs acoustiques, application à la
            transmission et à la reproduction de scènes sonores complexes dans
            un contexte multimédia,” Dissertation, l’Université Paris 6, Paris,
            2001.

    .. [#]  C. Nachbar, F. Zotter, E. Deleflie, and A. Sontacchi, “Ambix - A
            Suggested Ambisonics Format (revised by F. Zotter),” International
            Symposium on Ambisonics and Spherical Acoustics,
            vol. 3, pp. 1-11, 2011.

    """
    n_sh = (n_max+1)**2
    sid_n = sph_identity_matrix(n_max, 'n-nm').T @ np.arange(0, n_max+1)
    sid_m = np.zeros(n_sh, dtype=int)
    idx_n = 0
    for n in range(1, n_max+1):
        for m in range(1, n+1):
            sid_m[idx_n + 2*m-1] = n-m+1
            sid_m[idx_n + 2*m] = -(n-m+1)
        sid_m[idx_n + 2*n + 1] = 0
        idx_n += 2*n+1

    return sid_n, sid_m


def sid_to_acn(n_max):
    """Convert from SID channel indexing to ACN indeces.
    Returns the indices to achieve a corresponding linear acn indexing.

    Parameters
    ----------
    n_max : int
        The maximum spherical harmonic order.

    Returns
    -------
    acn : ndarray, int
        The SID indices sorted according to a respective linear ACN indexing.
    """
    sid_n, sid_m = sid(n_max)
    linear_sid = nm_to_acn(sid_n, sid_m)
    return np.argsort(linear_sid)


def sph_identity_matrix(n_max, type='n-nm'):
    """Calculate a spherical harmonic identity matrix.

    Parameters
    ----------
    n_max : int
        The spherical harmonic order.
    type : str, optional
        The type of identity matrix. Currently only 'n-nm' is implemented.

    Returns
    -------
    identity_matrix : ndarray, int
        The spherical harmonic identity matrix.

    Examples
    --------

    The identity matrix can for example be used to decompress from order only
    vectors to a full order and degree representation.

    >>> import spharpy
    >>> import matplotlib.pyplot as plt
    >>> n_max = 2
    >>> E = spharpy.spherical.sph_identity_matrix(n_max, type='n-nm')
    >>> a_n = [1, 2, 3]
    >>> a_nm = E.T @ a_n
    >>> a_nm
    array([1, 2, 2, 2, 3, 3, 3, 3, 3])

    The matrix E in this case has the following form.

    .. plot::

        >>> import spharpy
        >>> import matplotlib.pyplot as plt
        >>> n_max = 2
        >>> E = spharpy.spherical.sph_identity_matrix(n_max, type='n-nm')
        >>> plt.matshow(E, cmap=plt.get_cmap('Greys'))
        >>> plt.gca().set_aspect('equal')

    """
    n_sh = (n_max+1)**2

    if type != 'n-nm':
        raise NotImplementedError

    identity_matrix = np.zeros((n_max+1, n_sh), dtype=int)

    for n in range(n_max+1):
        m = np.arange(-n, n+1)
        linear_nm = nm_to_acn(np.tile(n, m.shape), m)
        identity_matrix[n, linear_nm] = 1

    return identity_matrix



class SphericalHarmonics:
    r"""
    Compute spherical harmonic basis matrices, their inverses, and gradients.
    The spherical harmonic Ynm is given by:

    .. math::
        Y_{nm} = N_{nm} P_{nm}(cos(\theta)) T_{nm}(\phi)

    where:
    

    - :math:`n` is the degree
    - :math:`m` is the order
    - :math:`P_{nm}` is the associated Legendre function
    - :math:`N_{nm}` is the normalization term
    - :math:`T_{nm}` is a term that depends on whether the harmonics are real or complex  
    - :math:`\theta` is the colatitude (angle from the positive z-axis)
    - :math:`\phi` is the azimuth (angle in the x-y plane from the x-axis)

    The normalization term :math:`N_{nm}` is given by:

    .. math::
        N_{nm}^{\text{SN3D}} =
        \sqrt{\frac{2n+1}{4\pi} \frac{(n-|m|)!}{(n+|m|)!}}

        N_{nm}^{\text{N3D}} = N_{nm}^{\text{SN3D}} \sqrt{\frac{2n+1}{2}}

        N_{nm}^{\text{MaxN}} = ... (max of N3D)

    The associated Legendre function :math:`P_{nm}` is defined as:

    .. math::
        P_{nm}(x) = (1-x^2)^{|m|/2} (d/dx)^n (x^2-1)^n

    The term :math:`T_{nm}` is defined as:
    
    - For complex-valued harmonics:  
        .. math::  
            T_{nm} = e^{im\phi}  
    - For real-valued harmonics:  
        .. math::  
            T_{nm} = \begin{cases}  
                        \cos(m\phi) & \text{if } m \ge 0 \\  
                        \sin(m\phi) & \text{if } m < 0  
                    \end{cases}  

    The spherical harmonics are orthogonal on the unit sphere, i.e.,

    .. math::
        \int_{sphere} Y_{nm} Y_{n'm'}* d\omega = \delta_{nn'} \delta_{mm'}

    where:

    - :math:`*` denotes complex conjugation
    - :math:`\delta_{nn'}` is the Kronecker delta function
    - :math:`d\omega` is the solid angle element
    - The integral is over the entire sphere

    The class supports the following conventions:

    - normalization: Defines the normalization convention:

      - ``'n3d'``: Uses the 3D normalization
        (also known as Schmidt semi-normalized).
      - ``'maxN'``: Uses the maximum norm
        (also known as fully normalized).
      - ``'sn3d'``: Uses the SN3D normalization
        (also known as Schmidt normalized).

    - channel_convention: Defines the channel ordering convention.

        - ``'acn'``: Follows the Ambisonic Channel Number (ACN) convention.
        - ``'fuma'``: Follows the Furse-Malham (FuMa) convention.
        (FuMa is only supported up to 3rd order)

    - inverse_method: Defines the type of inverse transform.

        - ``'pseudo_inverse'``: Uses the Moore-Penrose pseudo-inverse
         for the inverse transform.
        - ``'quadrature'``: Uses quadrature for the inverse transform.
        - ``None``: No inverse transform is applied and basis is returned.

    Parameters
    ----------
    n_max : int
        Maximum spherical harmonic order
    coordinates : :py:class:`pyfar.Coordinates`, spharpy.SamplingSphere  
        objects with sampling points for which the basis matrix is
        calculated
    basis_type : str, optional
        Type of spherical harmonic basis, either ``'complex'`` or
        ``'real'``. The default is ``'real'``.
    normalization : str, optional
        Normalization convention, either ``'n3d'``, ``'maxN'`` or
        ``'sn3d'``. The default is ``'n3d'``.  
        (maxN is only supported up to 3rd order)
    channel_convention : str, optional
        Channel ordering convention, either ``'acn'`` or ``'fuma'``.
        The default is ``'acn'``.
        (FuMa is only supported up to 3rd order)
    inverse_method : str, optional
        Inverse transform type, either ``'pseudo_inverse'`` or
        ``'quadrature'``. The default is "pseudo_inverse".
    condon_shortley : bool or str, optional
        Whether to include the Condon-Shortley phase term. If ``True``,
        Condon-Shortley is included, if ``False`` it is not
        included. The default is ``'auto'``, which corresponds to
        ``True`` for complex basis and ``False`` for real basis.
        
    """

    def __init__(
        self,
        n_max,
        coordinates,
        basis_type="real",
        normalization="n3d",
        channel_convention="acn",
        inverse_method="pseudo_inverse",
        condon_shortley='auto',
    ):
        # initialize private attributes
        self._n_max = None
        self._coordinates = pf.Coordinates()
        self._basis_type = None
        self._inverse_method = None
        self._channel_convention = None
        self._condon_shortley = None
        self._normalization = None
        self._reset_compute_attributes()

        self.n_max = n_max
        self.coordinates = coordinates
        self.basis_type = basis_type
        self.inverse_method = inverse_method
        self.channel_convention = channel_convention
        self.normalization = normalization
        self.condon_shortley = condon_shortley

    # Properties
    @property
    def condon_shortley(self):
        """Get or set the Condon-Shortley phase term."""
        return self._condon_shortley

    @condon_shortley.setter
    def condon_shortley(self, value):
        """Get or set the Condon-Shortley phase term."""
        if isinstance(value, str):
            if value != 'auto':
                raise ValueError("condon_shortley must be a bool or the string 'auto'")
        elif not isinstance(value, bool):
            raise TypeError("condon_shortley must be a bool or the string 'auto'")
        if value != self._condon_shortley:
            self._reset_compute_attributes()
        self._condon_shortley = value

    @property
    def n_max(self):
        """Get or set the spherical harmonic order."""
        return self._n_max

    @n_max.setter
    def n_max(self, value):
        """Get or set the spherical harmonic order."""
        if value < 0:
            raise ValueError("n_max must be a positive integer")
        if value % 1 != 0:
            raise ValueError("n_max must be an integer value")
        if self.channel_convention == "fuma" and value > 3:
            raise ValueError("n_max > 3 is not allowed with 'fuma' "
                             "channel convention")
        if self.normalization == "maxN" and value > 3:
            raise ValueError("n_max > 3 is not allowed with 'maxN' "
                             "normalization")
        if int(value) != self._n_max:
            self._reset_compute_attributes()
            self._n_max = int(value)  # Cast to int for safety

    @property
    def coordinates(self):
        """Get or set the coordinates object."""
        return self._coordinates

    @coordinates.setter
    def coordinates(self, value):
        """Get or set the coordinates object."""
        if not isinstance(value, pf.Coordinates):
            raise TypeError("coordinates must be a pyfar.Coordinates "
                            "object or spharpy.SamplingSphere object")
        if value.cdim != 1:
            raise ValueError("Coordinates must be a 1D array")
        if value.csize == 0:
            raise ValueError("Coordinates cannot be empty")
        if value != self._coordinates:
            self._reset_compute_attributes()
            self._coordinates = value

    @property
    def basis_type(self):
        """Get or set the type of spherical harmonic basis."""
        return self._basis_type

    @basis_type.setter
    def basis_type(self, value):
        """Get or set the type of spherical harmonic basis."""
        if value not in ["complex", "real"]:
            raise ValueError("Invalid basis type, only "
                             "'complex' and 'real' are supported")
        if value != self._basis_type:
            self._reset_compute_attributes()
            self._basis_type = value

    @property
    def inverse_method(self):
        """Get or set the type of inverse transform."""
        return self._inverse_method

    @inverse_method.setter
    def inverse_method(self, value):
        """Get or set the inverse transform type."""
        allowed = ["pseudo_inverse", "quadrature"]
        if value not in allowed:
            raise ValueError("Invalid inverse method. Allowed values are 'inverse' or "
                             "'quadrature'")
        # If using simple pf.Coordinates and not SamplingSphere, 'auto' is not allowed
        if not isinstance(self.coordinates, SamplingSphere) and value == "quadrature":
            raise ValueError("'quadrature' requires coordinates in a SamplingSphere object")
        if value != self._inverse_method:
            self._reset_compute_attributes()
            self._inverse_method = value

    @property
    def channel_convention(self):
        """Get or set the channel ordering convention."""
        return self._channel_convention

    @channel_convention.setter
    def channel_convention(self, value):
        """Get or set the channel order convention."""
        if value not in ["acn", "fuma"]:
            raise ValueError("Invalid channel convention, "
                             "currently only 'acn' "
                             "and 'fuma' are supported"
                             )
        if value == "fuma" and self.n_max > 3:
            raise ValueError("n_max > 3 is not allowed with 'fuma' " \
                            "channel convention")
        if value != self._channel_convention:
            self._reset_compute_attributes()
            self._channel_convention = value

    @property
    def normalization(self):
        """Get or set the normalization convention."""
        return self._normalization

    @normalization.setter
    def normalization(self, value):
        """Get or set the normalization convention."""
        if value not in ["n3d", "maxN", "sn3d"]:
            raise ValueError(
                "Invalid normalization, "
                "currently only 'n3d', 'maxN', 'sn3d' are "
                "supported"
            )
        if value == "maxN" and self.n_max > 3:
            raise ValueError("n_max > 3 is not allowed with " \
                            "'maxN' normalization")
        if value != self._normalization:
            self._reset_compute_attributes()
            self._normalization = value

    @property
    def basis(self):
        """Get the spherical harmonic basis matrix."""
        if self._basis is None:
            self._compute_basis()
        return self._basis

    @property
    def basis_gradient_theta(self):
        """Get the gradient of the basis matrix with respect to theta."""
        if self._basis_gradient_theta is None:
            self._compute_basis_gradient()
        return self._basis_gradient_theta

    @property
    def basis_gradient_phi(self):
        """Get the gradient of the basis matrix with respect to phi."""  
        if self._basis_gradient_phi is None:
            self._compute_basis_gradient()
        return self._basis_gradient_phi

    def _compute_basis(self):
        """
        Compute the basis matrix for the SphericalHarmonics class.
        """
        if self.basis_type == "complex":
            function = spherical_harmonic_basis
        elif self.basis_type == "real":
            function = spherical_harmonic_basis_real
        else:
            raise ValueError(
                "Invalid basis type, should be either 'complex' or 'real'"
            )
        self._basis = function(
            self.n_max, self.coordinates,
            self.normalization, self.channel_convention)

    def _compute_basis_gradient(self):
        """
        Compute the gradient of the basis matrix for the SphericalHarmonics class
        """
        if any(
            (self.normalization in ["maxN", "sn3d"],
             self.channel_convention == "fuma")
        ):
            raise ValueError(
            f"Gradient computation not supported for normalization "
            f"'{self.normalization}' and "
            f"channel convention '{self.channel_convention}'."
            )
        else:
            if self.basis_type == "complex":
                function = spherical_harmonic_basis_gradient
            elif self.basis_type == "real":
                function = spherical_harmonic_basis_gradient_real
            else:
                raise ValueError(
                    "Invalid basis type, should be either 'complex' or 'real'"
                )
            self._basis_gradient_theta, self._basis_gradient_phi = function(
                self.n_max, self.coordinates)

    @property
    def basis_inv(self):
        """Get or set the inverse basis matrix."""
        if self._basis is None:
            self._compute_basis()
        if self._basis_inv is None:
            self._compute_inverse()
        return self._basis_inv

    def _compute_inverse(self):
        """
        Compute the inverse basis matrix for the SphericalHarmonics class
        """
        if self._basis is None:
            self._compute_basis()
        if not isinstance(self.coordinates, SamplingSphere):
            warnings.warn("Using a simple pf.Coordinates object for inverse_basis." \
            " Make sure this is intended.")
        if self.inverse_method == "pseudo_inverse":
            self._basis_inv = np.linalg.pinv(self._basis)
        elif self.inverse_method == "quadrature":
            self.coordinates.quadrature = True
            self._basis_inv = np.einsum('ij,i->ji', np.conj(self._basis), 
                                        self.coordinates.weights)
    def _reset_compute_attributes(self):
        """Reset the computed attributes for the SphericalHarmonics class in
        case of changes in the parameters."""
        self._basis = None
        self._basis_gradient_theta = None
        self._basis_gradient_phi = None
        self._basis_inv = None
        
