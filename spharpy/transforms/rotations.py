"""
Rotation/Translation operations for data in the spherical harmonic domains.
"""

import numpy as np
import spharpy
from scipy.special import eval_jacobi, factorial
from scipy.spatial.transform import Rotation as ScipyRotation
from spharpy import (
    SphericalHarmonicSignal,
    SphericalHarmonicTimeData,
    SphericalHarmonicFrequencyData,
    SphericalHarmonicDefinition,
)
from spharpy.classes.audio import _SphericalHarmonicAudio
from typing import Union


class SphericalHarmonicRotation(ScipyRotation):
    """Class for rotations of coordinates and data.

    Examples
    --------
    The following examples demonstrate how to create a rotation and apply it
    to spherical harmonic coefficients and corresponding spherical harmonic
    data containers using the implemented methods and operators.

    .. plot::
        :context: close-figs

        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> import spharpy

        Define the convention and order of the spherical harmonics and a set of
        coefficients. Note that the generated coefficients correspond to a
        dipole oriented along the y-axis.

        >>> definition = spharpy.SphericalHarmonicDefinition(n_max=1)
        >>> coefficients = np.array([0, 1, 0, 0])

        Define a rotation by 45 degrees around the z-axis based on the Euler
        angles.

        >>> angle = np.pi / 4
        >>> R = spharpy.transforms.SphericalHarmonicRotation.from_euler(
        >>>     'z', angle)

        The spherical harmonic rotation matrix can be obtained and applied
        to the using the following:

        >>> D = R.as_spherical_harmonic_matrix(definition)
        >>> print(f"Rotated coefficients: {np.round(D @ coefficients, 2)}")
        ... # Rotated coefficients: [ 0.   0.71 0.   -0.71]

        To visualize the effect of the rotation, we can expand the series
        expansion on the unit sphere and plot the original and rotated
        data:

        >>> sampling = spharpy.samplings.equal_area(0, n_points=250)
        >>> Y = spharpy.SphericalHarmonics.from_definition(
        >>>     definition, coordinates=sampling)
        ...
        >>> _, axs = plt.subplots(
        >>>     1, 2, subplot_kw={'projection': '3d'}, figsize=(5, 2.5))
        >>> spharpy.plot.balloon_wireframe(
        >>>     sampling, Y.basis @ coefficients, ax=axs[0], colorbar=False)
        >>> spharpy.plot.balloon_wireframe(
        >>>     sampling, Y.basis @ D @ coefficients, ax=axs[1],
        >>>     colorbar=False)

    .. plot::
        :context: close-figs

        The rotation can also be applied to spherical harmonic data objects
        using the overloaded multiplication operator. This includes the
        class object itself as well as
        :class:`~spharpy.SphericalHarmonicFrequencyData`,
        :class:`~spharpy.SphericalHarmonicTimeData`, and
        :class:`~spharpy.SphericalHarmonicSignal`.

        The following example demonstrates the application to an arbitrary
        :class:`~spharpy.SphericalHarmonicFrequencyData` object containign the
        same series expansion as above:

        >>> frequency_data = spharpy.SphericalHarmonicFrequencyData(
        >>>     np.atleast_2d(coefficients).T, frequencies=1e3,
        >>>     basis_type=definition.basis_type,
        >>>     normalization=definition.normalization,
        >>>     channel_convention=definition.channel_convention,
        >>>     condon_shortley=definition.condon_shortley)

        The rotation can now be applied using the `*` operator:
        Note that due to the multiplication with itself, the rotation
        is applied twice, which corresponds to a rotation of 90 degrees around
        the z-axis.

        >>> rotated_frequency_data = R * R * frequency_data

        The effect of the rotation can again be visualized by evaluating the
        series expansion on the unit sphere:

        >>> _, axs = plt.subplots(
        >>>     1, 2, subplot_kw={'projection': '3d'}, figsize=(5, 2.5))
        >>> spharpy.plot.balloon_wireframe(
        >>>     sampling, np.squeeze(Y.basis @ frequency_data.freq),
        >>>     ax=axs[0], colorbar=False)
        >>> spharpy.plot.balloon_wireframe(
        >>>     sampling, np.squeeze(Y.basis @ rotated_frequency_data.freq),
        >>>     ax=axs[1], colorbar=False)


    Note
    ----
    Initializing using the constructor is not advised. Always use the
    respective ``from_*`` method.

    """

    def __init__(
            self,
            quat,
            *args, **kwargs):
        """Initialize.

        Parameters
        ----------
        quat : array_like, shape (N, 4) or (4,)
            Each row is a (possibly non-unit norm) quaternion in scalar-last
            (x, y, z, w) format. Each quaternion will be normalized to unit
            norm.
        *args : optional
            Arguments are passed to
            :py:class:`~scipy.spatial.transform.Rotation`
        **kwargs : optional
            Keyword arguments are passed to
            :py:class:`~scipy.spatial.transform.Rotation`

        Returns
        -------
        SphericalHarmonicRotation
            The rotation object.

        Note
        ----
        Initializing using the constructor is not advised. Always use the
        respective ``from_quat`` method.

        """
        super().__init__(quat, *args, **kwargs)

    def as_spherical_harmonic_matrix(
            self,
            spherical_harmonic_definition : SphericalHarmonicDefinition,
        ) -> np.ndarray:
        """Express the rotation as spherical harmonic rotation matrices.

        Supports complex and real-valued spherical harmonics.

        Parameters
        ----------
        spherical_harmonic_definition : SphericalHarmonicDefinition
            Definition of the spherical harmonics.

        Returns
        -------
        np.ndarray, complex or float
            Stack of block-diagonal rotation matrices with individual shapes
            :math:`[(n_{max}+1)^2, (n_{max}+1)^2]`.

        Note
        ----
        At the moment, only spherical harmonic rotations matrices following
        the ``ACN`` indexing and ``N3D`` normalization are supported.

        """
        if spherical_harmonic_definition.normalization != 'N3D':
            raise NotImplementedError(
                "Only 'N3D' normalization is supported for spherical "
                "harmonic rotations.")
        if spherical_harmonic_definition.channel_convention != 'ACN':
            raise NotImplementedError(
                "Only 'ACN' channel convention is supported for spherical "
                "harmonic rotations.")

        euler_angles = np.atleast_2d(self.as_euler('zyz'))
        n_matrices = euler_angles.shape[0]

        n_sh = (spherical_harmonic_definition.n_max + 1)**2
        if spherical_harmonic_definition.basis_type == 'real':
            dtype = float
            rotation_function = wigner_d_rotation_real
        elif spherical_harmonic_definition.basis_type == 'complex':
            dtype = complex
            rotation_function = wigner_d_rotation

        D = np.zeros((n_matrices, n_sh, n_sh), dtype=dtype)

        for idx, angles in enumerate(euler_angles):
            D[idx, :, :] = rotation_function(
                spherical_harmonic_definition.n_max,
                angles[0], angles[1], angles[2])

        return np.squeeze(D)

    def apply(
            self,
            target : Union[
                SphericalHarmonicTimeData,
                SphericalHarmonicFrequencyData,
                SphericalHarmonicSignal],
        ) -> Union[
            SphericalHarmonicTimeData,
            SphericalHarmonicFrequencyData,
            SphericalHarmonicSignal]:
        """Apply the rotation to spherical harmonic data object.

        Note that the spherical harmonic definition of the target object is
        used to generate a fitting spherical harmonic rotation matrix.

        Parameters
        ----------
        target : SphericalHarmonicTimeData, SphericalHarmonicFrequencyData, SphericalHarmonicSignal
            Spherical harmonic series expansion to be rotated.

        Returns
        -------
        SphericalHarmonicTimeData, SphericalHarmonicFrequencyData, SphericalHarmonicSignal
            The rotated signal.
        """  # noqa: E501
        # Audio objects use ACN/N3D internally
        target_definition = SphericalHarmonicDefinition(
            n_max=target.n_max,
            basis_type=target.basis_type,
            normalization='N3D',
            channel_convention='ACN',
        )
        D = self.as_spherical_harmonic_matrix(target_definition)
        M = np.linalg.multi_dot(list(D)) if D.ndim > 2 else D

        rotated_data = np.einsum('ij, ljt -> lit', M, target._data)
        rotated = target.copy()
        rotated._data = rotated_data
        return rotated

    def __mul__(self, other):
        """Multiplication with rotations or application to signals."""
        if isinstance(other, ScipyRotation):
            result = super().__mul__(other)
            return SphericalHarmonicRotation.from_quat(result.as_quat())
        elif isinstance(other, _SphericalHarmonicAudio):
            return self.apply(other)
        else:
            raise ValueError(
                "Multiplication is only supported for "
                "SphericalHarmonicRotation and SphericalHarmonicSignal.")


def rotation_z_axis(n_max, angle):
    r"""Rotation matrix for complex spherical harmonics around the z-axis
    by a given angle. The rotation is performed such that positive angles
    result in a counter clockwise rotation of the data [#]_.

    .. math::

        c_{nm}(\theta, \phi + \xi) = e^{-im\xi} c_{nm}(\theta, \phi)

    Parameters
    ----------
    n_max : integer
        Spherical harmonic order
    angle : number
        Rotation angle in radians `[0, 2 \\pi]`

    Returns
    -------
    array_like, complex, shape :math:`((n_{max}+1)^2, (n_{max}+1)^2)`
        Diagonal rotation matrix evaluated for the specified angle

    References
    ----------
    .. [#]  N. A. Gumerov and R. Duraiswami, “Recursions for the computation
            of multipole translation and rotation coefficients for the 3-d
            helmholtz equation,” vol. 25, no. 4, pp. 1344–1381, 2003.


    Examples
    --------
    >>> import numpy as np
    >>> import spharpy
    >>> n_max = 1
    >>> sh_vec = np.array([0, 1, 0, 0])
    >>> rotMat = spharpy.transforms.rotation_z_axis(n_max, np.pi/2)
    >>> sh_vec_rotated = rotMat @ sh_vec
    """

    acn = np.arange(0, (n_max+1)**2)
    m = spharpy.spherical.acn_to_nm(acn)[1]
    rotation_phi = np.exp(-1j*angle*m)

    return np.diag(rotation_phi)


def rotation_z_axis_real(n_max, angle):
    r"""Rotation matrix for real-valued spherical harmonics around the z-axis
    by a given angle. The rotation is performed such that positive angles
    result in a counter clockwise rotation of the data [#]_.

    Parameters
    ----------
    n_max : integer
        Spherical harmonic order
    angle : number
        Rotation angle in radians `[0, 2 \pi]`

    Returns
    -------
    array_like, float, shape :math:`((n_max+1)^2, (n_max+1)^2)`
        Block-diagonal Rotation matrix evaluated for the specified angle.

    References
    ----------
    .. [#] M. Kronlachner, “Spatial Transformations for the Alteration of
           Ambisonic Recordings,” Master Thesis, University of Music and
           Performing Arts, Graz, 2014.


    Examples
    --------
    >>> import numpy as np
    >>> import spharpy
    >>> n_max = 1
    >>> sh_vec = np.array([0, 1, 0, 0])
    >>> rotMat = spharpy.transforms.rotation_z_axis_real(n_max, np.pi/2)
    >>> sh_vec_rotated = rotMat @ sh_vec

    """
    acn = np.arange(0, (n_max + 1) ** 2)
    n, m = spharpy.spherical.acn_to_nm(acn)
    acn_reverse_degree = n ** 2 + n - m

    rotation_phi = np.zeros(((n_max + 1) ** 2, (n_max + 1) ** 2))
    mask = m == 0
    rotation_phi[acn[mask], acn[mask]] = 1.0

    mask_pos = m > 0
    mask_neg = m < 0
    rotation_phi[acn[mask_pos], acn[mask_pos]] = np.cos(
        np.abs(m[mask_pos]) * angle)
    rotation_phi[acn[mask_neg], acn[mask_neg]] = np.cos(
        np.abs(m[mask_neg]) * angle)

    # non diagonal
    rotation_phi[acn[mask_pos], acn_reverse_degree[mask_pos]] = -np.sin(
        np.abs(m[mask_pos]) * angle)
    rotation_phi[acn[mask_neg], acn_reverse_degree[mask_neg]] = np.sin(
        np.abs(m[mask_neg]) * angle)

    return rotation_phi


def wigner_d_rotation(n_max, alpha, beta, gamma):
    r"""Wigner-D rotation matrix for Euler rotations by angles
    (\alpha, \beta, \gamma) around the (z,y,z)-axes.
    The implementation follows [#]_. and rotation is performed such that
    positive angles result in a counter clockwise rotation of the data.

    .. math::

        D_{m^\prime,m}^n(\alpha, \beta, \gamma) =
        e^{-im^\prime\alpha} d_{m^\prime,m}^n(\beta) e^{-im\gamma}

    Parameters
    ----------
    n_max : int
        Spherical harmonic order
    alpha : float
        First z-axis rotation angle
    beta : float
        Y-axis rotation angle
    gamma : float
        Second z-axis rotation angle

    Returns
    -------
    array_like, complex,:math:`((n_max+1)^2, (n_max+1)^2)`
        Block diagonal rotation matrix


    Examples
    --------
    >>> import numpy as np
    >>> import spharpy
    >>> n_max = 1
    >>> sh_vec = np.array([0, 0, 1, 0])
    >>> rotMat = spharpy.transforms.wigner_d_rotation(n_max, 0, np.pi/4, 0)
    >>> sh_vec_rotated = rotMat @ sh_vec

    References
    ----------
    .. [#] B. Rafaely, Fundamentals of Spherical Array Processing, 1st ed.,
           vol. 8. Springer-Verlag GmbH Berlin Heidelberg, 2015.

    """

    n_sh = (n_max+1)**2

    R = np.zeros((n_sh, n_sh), dtype=complex)

    for row in np.arange(0, (n_max+1)**2):
        n_dash, m_dash = spharpy.spherical.acn_to_nm(row)
        for column in np.arange(0, (n_max+1)**2):
            n, m = spharpy.spherical.acn_to_nm(column)
            if n == n_dash:
                rot_alpha = np.exp(-1j*m_dash*alpha)
                rot_beta = wigner_d_function(n, m_dash, m, beta)
                rot_gamma = np.exp(-1j*m*gamma)
                R[row, column] = rot_alpha * rot_beta * rot_gamma

    return R


def wigner_d_rotation_real(n_max, alpha, beta, gamma):
    r"""Wigner-D rotation matrix for Euler rotations for real-valued spherical
    harmonics by angles (\alpha, \beta, \gamma) around the (z,y,z)-axes.
    The implementation follows [#]_ and the rotation is performed such that
    positive angles result in a counter clockwise rotation of the data.

    Parameters
    ----------
    n_max : int
        Spherical harmonic order
    alpha : float
        First z-axis rotation angle
    beta : float
        Y-axis rotation angle
    gamma : float
        Second z-axis rotation angle

    Returns
    -------
    array_like, float, :math:`((n_max+1)^2, (n_max+1)^2)`
        Block diagonal rotation matrix

    Examples
    --------
    >>> import numpy as np
    >>> import spharpy
    >>> n_max = 1
    >>> sh_vec = np.array([0, 0, 1, 0])
    >>> rotMat = spharpy.transforms.wigner_d_rotation_real(
    >>>     n_max, 0, np.pi/4, 0)
    >>> sh_vec_rotated = rotMat @ sh_vec

    References
    ----------
    .. [#]  M. A. Blanco, M. Flórez, and M. Bermejo, “Evaluation of the
            rotation matrices in the basis of real spherical harmonics,”
            Journal of Molecular Structure: THEOCHEM,  vol. 419, no. 1–3,
            pp. 19–27, Dec. 1997, doi: 10.1016/S0166-1280(97)00185-1.

    """
    n_sh = (n_max+1)**2
    R = np.zeros((n_sh, n_sh), dtype=float)

    for row_acn in np.arange(0, (n_max+1)**2):
        for col_acn in np.arange(0, (n_max+1)**2):
            n, m = spharpy.spherical.acn_to_nm(col_acn)
            n_dash, m_dash = spharpy.spherical.acn_to_nm(row_acn)
            if n == n_dash:
                # minus beta opposite rotation direction
                d_l_1 = wigner_d_function(n, np.abs(m_dash), np.abs(m), -beta)
                d_l_2 = wigner_d_function(n, np.abs(m), -np.abs(m_dash), -beta)

                R[row_acn, col_acn] = \
                    _sign(m_dash) * _Phi(m, alpha) * _Phi(m_dash, gamma) * \
                    (d_l_1 + (-1)**int(m) * d_l_2)/2 \
                    - _sign(m) * _Phi(-m, alpha) * _Phi(-m_dash, gamma) * \
                    (d_l_1 - (-1)**int(m) * d_l_2)/2

    return R


def _sign(x):
    """
    Returns sign of x, differs from numpy definition for x=0.
    """
    if x < 0:
        sign = -1
    else:
        sign = 1

    return sign


def _Phi(m, angle):
    """
    Rotation Matrix around z-axis for real Spherical Harmonics as defined in
    Blanco et al., Evaluation of the rotation matrices in the basis of real
    spherical harmonics, eq.(8).
    """
    if m > 0:
        phi = np.sqrt(2)*np.cos(m*angle)
    elif m == 0:
        phi = 1
    elif m < 0:
        # minus due to differing phase convention
        phi = -np.sqrt(2)*np.sin(np.abs(m)*angle)

    return phi


def wigner_d_function(n, m_dash, m, beta):
    r"""Wigner-d function for rotations around the y-axis.
    Convention as defined in [#]_.

    Parameters
    ----------
    n : int
        order
    m_dash : int
        degree
    m : int
        degree
    beta : float
        Rotation angle

    Returns
    -------
    float
        Wigner-d symbol

    References
    ----------
    .. [#] B. Rafaely, Fundamentals of Spherical Array Processing, 1st ed.,
           vol. 8. Springer-Verlag GmbH Berlin Heidelberg, 2015.

    """

    if m >= m_dash:
        sign = 1
    else:
        sign = (-1)**int(m-m_dash)

    mu = np.abs(m_dash - m)
    nu = np.abs(m_dash + m)
    s = n - (mu + nu)/2

    norm = (factorial(s)*factorial(s+mu+nu))/(factorial(s+mu)*factorial(s+nu))
    P = eval_jacobi(s, mu, nu, np.cos(beta))
    d = sign * np.sqrt(norm) * np.sin(beta/2)**mu * np.cos(beta/2)**nu * P

    return d
