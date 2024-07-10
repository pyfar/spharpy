import numpy as np
import scipy.special as special
import spharpy.special as _special
from spharpy.samplings.helpers import calculate_sampling_weights
import pyfar as pf
# from functools import lru_cache
import logging as logger


class SphericalHarmonics:
    r"""
    Compute spherical harmonic basis matrices, their inverses, and gradients.
    The spherical harmonic Ynm is given by:
    .. math::
        Y_{nm} = N_{nm} P_{nm}(cos(\theta)) T_{nm}(\phi)

    where:
    - $n$ is the degree
    - $m$ is the order
    - $P_{nm}$ is the associated Legendre function
    - $N_{nm}$ is the normalization term
    - $T_{nm}$ is a term that depends on whether the
     harmonics are real or complex
    - $\theta$ is the colatitude (angle from the positive z-axis)
    - $\phi$ is the azimuth (angle from the positive x-axis in the xy-plane)

    The normalization term Nnm is given by:

    .. math::
        N_{nm}^{\text{SN3D}} =
        \sqrt{\frac{2n+1}{4\pi} \frac{(n-|m|)!}{(n+|m|)!}}

        N_{nm}^{\text{N3D}} = N_{nm}^{\text{SN3D}} \sqrt{\frac{2n+1}{2}}

        N_{nm}^{\text{MaxN}} = ... (max of N3D)

    The associated Legendre function Pnm is given by:

    .. math::
        P_{nm}(x) = (1-x^2)^{|m|/2} (d/dx)^n (x^2-1)^n

    The term Tnm is given by:
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
    - $*$ denotes the complex conjugate
    - $\delta$ is the Kronecker delta
    - $d\omega$ is the differential solid angle
    - The integral is over the entire sphere

    The class supports the following conventions:

    - basis_type: Defines the type of spherical harmonic basis.
    It can be either 'complex' or 'real'.
        - ``'complex'``: Uses complex-valued spherical harmonics.
        - ``'real'``: Uses real-valued spherical harmonics.

    - normalization: Defines the normalization convention.
     It can be 'n3d', 'maxN', or 'sn3d'.
        - ``'n3d'``: Uses the 3D normalization
        (also known as Schmidt semi-normalized).
        - ``'maxN'``: Uses the maximum norm
        (also known as fully normalized).
        - ``'sn3d'``: Uses the SN3D normalization
        (also known as Schmidt normalized).

    - channel_convention: Defines the channel ordering convention.
    It can be either 'acn' or 'fuma'.
        - ``'acn'``: Follows the Ambisonic Channel Number (ACN) convention.
        - ``'fuma'``: Follows the Furse-Malham (FuMa) convention.

    - inverse_transform: Defines the type of inverse transform.
     It can be 'pseudo_inverse', 'quadrature', or None.
        - ``'pseudo_inverse'``: Uses the Moore-Penrose pseudo-inverse
         for the inverse transform.
        - ``'quadrature'``: Uses quadrature for the inverse transform.
        - ``None``: No inverse transform is applied.
    """

    def __init__(
        self,
        n_max,
        coords,
        basis_type="real",
        normalization="n3d",
        channel_convention="acn",
        inverse_transform=None,
        weights=None,
        condon_shortley=True
    ):
        r"""
        Initialize the SphericalHarmonics class.

        Parameters
        ----------
        n_max : int
            Maximum spherical harmonic order
        coords : :doc:`pf.Coordinates <pyfar:classes/pyfar.coordinates>`
            Coordinate object with sampling points for
            which the basis matrix is calculated
        basis_type : str, optional
            Type of spherical harmonic basis, either ``'complex'`` or
            ``'real'``. The default is ``'complex'``.
        normalization : str, optional
            Normalization convention, either ``'n3d'``, ``'maxN'`` or
             ``'sn3d'``. The default is ``'n3d'``.
            (maxN is only supported up to 3rd order)
        channel_convention : str, optional
            Channel ordering convention, either ``'acn'`` or ``'fuma'``.
            The default is ``'acn'``.
            (FuMa is only supported up to 3rd order)
        inverse_transform : str, optional
            Inverse transform type, either ``'pseudo_inverse'`` or
            ``'quadrature'``. The default is None.
        weights : array-like, optional
            Sampling weights for the quadrature transform.
            Required if `quadrature` is chosen.
        condon_shortley : bool, optional
            Whether to include the Condon-Shortley phase term.
            The default is True.
        """
        if not isinstance(coords, pf.Coordinates):
            raise TypeError("coords must be a pyfar.Coordinates object")
        self.n_max = n_max
        self.weights = weights
        self.coords = coords
        self.basis_type = basis_type
        self.inverse_transform = inverse_transform
        self.channel_convention = channel_convention  # ch. ord. conv.
        self.normalization = normalization  # gain norm. conv.
        self.basis = None
        (self.basis_gradient_theta,
         self.basis_gradient_phi) = (None, None)
        self._basis_inv = None
        (self._basis_inv_gradient_theta,
         self._basis_inv_gradient_phi) = (None, None)
        self._condon_shortley = condon_shortley
        # Store previous values for comparison
        self._prev_n_max = self.n_max
        self._prev_coords = self.coords
        self._prev_basis_type = self.basis_type
        self._prev_inverse_transform = self.inverse_transform
        self._prev_channel_convention = self.channel_convention
        self._prev_normalization = self.normalization
        self._prev_condon_shortley = condon_shortley

    # Properties
    @property
    def n_max(self):
        return self._n_max

    @n_max.setter
    def n_max(self, value):
        if value < 0:
            raise ValueError("n_max must be a positive integer")
        if value % 1 != 0:
            raise ValueError("n_max must be an integer value")
        self._n_max = int(value)  # Cast to int for safety

    @property
    def weights(self):
        """Sampling weights for numeric integration."""
        return self._weights

    @weights.setter
    def weights(self, weights):
        if weights is None:
            self._weights = None
            return
        if len(weights) != len(self.coords.cartesian):
            raise ValueError(
                "The number of weights has to be equal to \
                    the number of sampling points."
            )

        weights = np.asarray(weights, dtype=float)
        norm = np.linalg.norm(weights, axis=-1)

        if not np.allclose(norm, 4 * np.pi):
            weights *= 4 * np.pi / norm

        self._weights = weights

    @property
    def coords(self):
        return self._coords

    @coords.setter
    def coords(self, value):
        if not isinstance(value, pf.Coordinates):
            raise TypeError("coords must be a pyfar.Coordinates object")
        if value.cdim != 1:
            raise ValueError("Coordinates must be 1D")
            # TODO: Allow for 2D coordinates
        self._coords = value

    @property
    def basis_type(self):
        return self._basis_type

    @basis_type.setter
    def basis_type(self, value):
        assert value in ["complex", "real"], (
            "Invalid basis type, currently only 'complex' and"
            " 'real' "
            "are supported"
        )
        self._basis_type = value

    @property
    def inverse_transform(self):
        return self._inverse_transform

    @inverse_transform.setter
    def inverse_transform(self, value):
        assert value in ["pseudo_inverse", "quadrature", None], (
            "Invalid inverse transform type, "
            "currently only 'pseudo_inverse' "
            "and 'quadrature' are supported"
        )
        self._inverse_transform = value

    @property
    def channel_convention(self):
        return self._channel_convention

    @channel_convention.setter
    def channel_convention(self, value):
        assert value in ["acn", "fuma"], (
            "Invalid channel convention, currently only 'acn' "
            "and 'fuma' are supported"
        )
        self._channel_convention = value

    @property
    def normalization(self):
        return self._normalization

    @normalization.setter
    def normalization(self, value):
        assert value in ["n3d", "maxN", "sn3d"], (
            "Invalid normalization, "
            "currently only 'n3d', 'maxN', 'sn3d' are "
            "supported"
        )
        self._normalization = value

    @property
    def basis(self):
        if self._basis is None:
            self._compute_basis()
        self._check_properties(compute='basis')
        return self._basis

    @property
    def basis_gradient_theta(self):
        if self._basis_gradient_theta is None:
            self._compute_basis_gradient()
        self._check_properties(compute='basis_gradient')
        return self._basis_gradient_theta

    @property
    def basis_gradient_phi(self):
        if self._basis_gradient_phi is None:
            self._compute_basis_gradient()
        self._check_properties(compute='basis_gradient')
        return self._basis_gradient_phi

    # @lru_cache(maxsize=128)
    def _compute_basis(self):
        """
        Compute the basis matrix for the SphericalHarmonics class.
        """
        logger.info("Computing basis matrix for n_max=%d", self.n_max)
        if self.basis_type == "complex":
            self.basis = spherical_harmonic_basis(
                self.n_max, self.coords,
                self.normalization, self.channel_convention
            )
        elif self.basis_type == "real":
            self.basis = spherical_harmonic_basis_real(
                self.n_max, self.coords,
                self.normalization, self.channel_convention
            )
        else:
            raise ValueError(
                "Invalid signal type, should be either 'complex' or 'real'"
            )

    def compute_basis(self):
        try:
            return self._compute_basis()
        except Exception as e:
            raise ValueError("Error computing basis:", e) from e

    # @lru_cache(maxsize=128)
    def _compute_basis_gradient(self):
        logger.info("Computing basis gradient for n_max=%d", self.n_max)
        if any(
            (self.normalization in ["maxN", "sn3d"],
             self.channel_convention == "fuma")
        ):
            raise ValueError(
                "Gradient computation not supported for MaxN, "
                "SN3D normalization or FuMa channel ordering"
            )
        else:
            if self.basis_type == "complex":
                grad_theta, grad_phi = spherical_harmonic_basis_gradient(
                    self.n_max, self.coords
                )
                self.basis_gradient_theta, self.basis_gradient_phi = (
                    grad_theta,
                    grad_phi,
                )
            elif self.basis_type == "real":
                grad_theta, grad_phi = spherical_harmonic_basis_gradient_real(
                    self.n_max, self.coords
                )
                self.basis_gradient_theta, self.basis_gradient_phi = (
                    grad_theta,
                    grad_phi,
                )
            else:
                raise ValueError(
                    "Invalid signal type, should be either 'complex' or 'real'"
                )

    def compute_basis_gradient(self):
        try:
            return self._compute_basis_gradient()
        except Exception as e:
            raise ValueError("Error computing basis gradient:", e) from e

    @property
    def basis_inv(self):
        if self._basis_inv is None:
            self.compute_inverse()
        self._check_properties(compute='basis_inv')
        return self._basis_inv

    def compute_inverse(self):
        """
        Compute the inverse transform matrix for the specified transform type

        The inverse transform matrix is calculated based on the specified
        `inverse_transform`. If ``'pseudo_inverse' is chosen,
        the Moore-Penrose pseudo-inverse is used.
        If ``'quadrature'`` is chosen, the inverse is computed
        using the conjugate transpose of the basis matrix
        multiplied by 4 * pi * weights.

        Returns
        -------
        None
        """
        if self.basis is None:
            self.compute_basis()
        if self.inverse_transform is not None:
            assert self.inverse_transform in ["pseudo_inverse",
                                              "quadrature"], (
                "Invalid inverse transform type, "
                "currently only 'pseudo_inverse' "
                "and 'quadrature' are supported"
            )
        # print("computing inverse basis using ", self.inverse_transform)
        elif self.inverse_transform is None:
            ValueError("Inverse transform type not specified")
        if self.inverse_transform == "pseudo_inverse":
            self._basis_inv = np.linalg.pinv(self.basis)
        elif self.inverse_transform == "quadrature":
            if self.weights is None:
                print(
                    "Warning: No weights specified for quadrature transform,"
                    "calculating weights using voronoi tessellation of sphere"
                )
                self.weights = calculate_sampling_weights(self.coords)
            self._basis_inv = np.conj(self.basis).T * self.weights

    @property
    def basis_inv_gradient_theta(self):
        if (
            self._basis_inv_gradient_theta is None
            or self._basis_inv_gradient_phi is None
        ):
            self.compute_inverse_gradient()
        self._check_properties(compute='basis_inv_gradient')
        return self._basis_inv_gradient_theta

    @property
    def basis_inv_gradient_phi(self):
        if (
            self._basis_inv_gradient_theta is None
            or self._basis_inv_gradient_phi is None
        ):
            self.compute_inverse_gradient()
        self._check_properties(compute='basis_inv_gradient')
        return self._basis_inv_gradient_phi

    def compute_inverse_gradient(self):
        if self.inverse_transform is None:
            ValueError("Inverse transform type not specified")
        elif self.inverse_transform == "pseudo_inverse":
            self._basis_inv_gradient_theta = np.linalg.pinv(
                self.basis_gradient_theta)
            self._basis_inv_gradient_phi = np.linalg.pinv(
                self.basis_gradient_phi)
        elif self.inverse_transform == "quadrature":
            if self.weights is None:
                print(
                    "Warning: No weights specified for quadrature transform,"
                    "calculating weights using voronoi tessellation of sphere"
                )
                self.weights = calculate_sampling_weights(self.coords)
            self._basis_inv_gradient_theta = np.conj(
                self.basis_gradient_theta).T * (4 * np.pi * self.weights)
            self._basis_inv_gradient_phi = np.conj(
                self.basis_gradient_phi).T * (4 * np.pi * self.weights)

    @basis.setter
    def basis(self, value):
        self._basis = value

    @basis_gradient_theta.setter
    def basis_gradient_theta(self, value):
        self._basis_gradient_theta = value

    @basis_gradient_phi.setter
    def basis_gradient_phi(self, value):
        self._basis_gradient_phi = value

    def _check_properties(self, compute=None):
        # Check if any crucial properties have changed
        if (self.n_max != self._prev_n_max or
                self.coords != self._prev_coords or
                self.basis_type != self._prev_basis_type or
                self.inverse_transform != self._prev_inverse_transform or
                self.channel_convention != self._prev_channel_convention or
                self.normalization != self._prev_normalization):
            if compute == 'basis':
                self._compute_basis()
            elif compute == 'basis_gradient':
                self._compute_basis_gradient()
            elif compute == 'basis_inv':
                self.compute_inverse()
            elif compute == 'basis_inv_gradient':
                self.compute_inverse_gradient()
            # And update the previous property values
            self._prev_n_max = self.n_max
            self._prev_coords = self.coords
            self._prev_basis_type = self.basis_type
            self._prev_inverse_transform = self.inverse_transform
            self._prev_channel_convention = self.channel_convention
            self._prev_normalization = self.normalization


def n3d_to_maxn(acn):
    """
    Converts N3D normalization to max N normalization
    Parameters
    ----------
    acn : int

    Returns
    -------
    maxN : float
        Maximum norm for spherical harmonics of order N
    """
    assert acn <= 15
    maxN = [
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
    return maxN[acn]


def n3d_to_sn3d_norm(m, n):
    """
    Converts N3D normalization to SN3D normalization
    Parameters
    ----------
    m : int
        Spherical harmonic degree
    n : int
        Spherical harmonic order (n >= |m|)

    Returns
    -------
    sn3d : float
        SN3D normalization factor
    """
    return 1 / np.sqrt(2 * n + 1)


def fuma_to_nm(fuma):
    r"""
    Calculate the spherical harmonic order n and degree m for a linear
    coefficient index, according to the FuMa (Furse-Malham)
    Channel Ordering Convention [2]_.

    FuMa = WXYZ | RSTUV | KLMNOPQ
    ACN = WYZX | VTRSU | QOMKLNP


    References
    ----------
    [2]  D. Malham, "Higher order Ambisonic systems” Space in Music –
    Music in Space (Mphil thesis).
         University of York. pp. 2–3. , 2003.

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
    """

    fuma_mapping = [0, 2, 3, 1, 8, 6, 4, 5, 7, 15, 13, 11, 9, 10, 12, 14]

    if fuma < 0 or fuma >= len(fuma_mapping):
        raise ValueError(
            "Invalid FuMa channel index, must be between 0 and 15 "
            "(supported up to 3rd order)"
        )

    acn = fuma_mapping[fuma]
    n, m = acn_to_nm(acn)  # Assuming you have the acn_to_nm function defined
    return n, m


def spherical_harmonic_basis(
    n_max, coords, normalization="n3d", channel_convention="acn",
    condon_shortley=True
):
    r"""
    Calculates the complex valued spherical harmonic basis matrix.
    See also :func:`spherical_harmonic_basis_real`.

    .. math::
        Y_n^m(\theta, \phi) =  CS_m N_{nm} P_{nm}(cos(\theta)) e^{im\phi}

    where:
    - $n$ is the degree
    - $m$ is the order
    - $P_{nm}$ is the associated Legendre function
    - $N_{nm}$ is the normalization term
    - $CS_m$ is the Condon-Shortley phase term
    - $\theta$ is the colatitude (angle from the positive z-axis)
    - $\phi$ is the azimuth (angle from the positive x-axis in the xy-plane)

    References
    ----------
    .. [#]  E. G. Williams, Fourier Acoustics. Academic Press, 1999.
    .. [#]  B. Rafaely, Fundamentals of Spherical Array Processing, vol. 8.
            Springer, 2015.


    Parameters
    ----------
    n_max : integer
        Spherical harmonic order
    coords : :doc:`pf.Coordinates <pyfar:classes/pyfar.coordinates>`
        Coordinate object with sampling points for which the basis matrix is
        calculated
    normalization : str, optional
        Normalization convention, either 'n3d', 'maxN' or 'sn3d'.
        The default is 'n3d'.
        (maxN is only supported up to 3rd order)
    channel_convention : str, optional
        Channel ordering convention, either 'acn' or 'fuma'.
        The default is 'acn'.
        (FuMa is only supported up to 3rd order)
    condon_shortley : bool, optional
        Whether to include the Condon-Shortley phase term.
        The default is True.

    Returns
    -------
    Y : ndarray, complex
        Complex spherical harmonic basis matrix


    >>> import spharpy
    >>> n_max = 2
    >>> coords = spharpy.samplings.icosahedron()
    >>> Y = spharpy.spherical.spherical_harmonic_basis(n_max, coords)

    """

    n_coeff = (n_max + 1) ** 2

    basis = np.zeros((coords.csize, n_coeff), dtype=complex)

    for acn in range(n_coeff):
        if channel_convention == "fuma":
            order, degree = fuma_to_nm(acn)
        else:
            order, degree = acn_to_nm(acn)
        basis[:, acn] = _special.spherical_harmonic(
            order, degree, coords.colatitude, coords.azimuth
        )
        if normalization == "sn3d":
            basis[:, acn] *= n3d_to_sn3d_norm(degree, order)
        elif normalization == "maxN":
            basis[:, acn] *= n3d_to_maxn(acn)
        if not condon_shortley:
            # Condon-Shortley phase term is already included in
            # the special.spherical_harmonic function
            # so need to divide by (-1)^m
            basis[:, acn] /= (-1) ** degree
    return basis


def spherical_harmonic_basis_real(
    n_max, coords, normalization="n3d", channel_convention="acn",
    condon_shortley=True
):
    r"""
    Calculates the real valued spherical harmonic basis matrix.
    See also :func:`spherical_harmonic_basis`.

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
    coords : :doc:`pf.Coordinates <pyfar:classes/pyfar.coordinates>`
        Coordinate object with sampling points for which the basis matrix is
        calculated
    normalization : str, optional
        Normalization convention, either 'n3d', 'maxN' or 'sn3d'.
        The default is 'n3d'.
        (maxN is only supported up to 3rd order)
    channel_convention : str, optional
        Channel ordering convention, either 'acn' or 'fuma'.
        The default is 'acn'.
        (FuMa is only supported up to 3rd order)
    condon_shortley : bool, optional

    Returns
    -------
    Y : ndarray, float
        Real valued spherical harmonic basis matrix.


    """
    n_coeff = (n_max + 1) ** 2

    basis = np.zeros((coords.csize, n_coeff), dtype=float)

    for acn in range(n_coeff):
        if channel_convention == "fuma":
            order, degree = fuma_to_nm(acn)
        else:
            order, degree = acn_to_nm(acn)
        basis[:, acn] = _special.spherical_harmonic_real(
            order, degree, coords.colatitude, coords.azimuth
        )
        if normalization == "sn3d":
            basis[:, acn] *= n3d_to_sn3d_norm(degree, order)
        elif normalization == "maxN":
            basis[:, acn] *= n3d_to_maxn(acn)
        if not condon_shortley:
            # Condon-Shortley phase term is already included in
            # the special.spherical_harmonic function
            # so need to divide by (-1)^m
            basis[:, acn] /= (-1) ** degree

    return basis


def acn_to_nm(acn):
    r"""
    Calculate the order n and degree m from the linear coefficient index.

    The linear index corresponds to the Ambisonics Channel Convention [#]_.

    .. math::

        n = \lfloor \sqrt{\mathrm{acn} + 1} \rfloor - 1

        m = \mathrm{acn} - n^2 -n

    Parameters
    ----------
    acn : ndarray, int
        Linear index

    Returns
    ----------
    n : ndarray, int
        Spherical harmonic order
    m : ndarray, int
        Spherical harmonic degree

    """
    acn = np.asarray(acn, dtype=int)

    n = np.ceil(np.sqrt(acn + 1)) - 1
    m = acn - n ** 2 - n

    n = n.astype(int, copy=False)
    m = m.astype(int, copy=False)

    return n, m


def nm_to_acn(n, m):
    r"""
    Calculate the linear index coefficient for a order n and degree m,

    The linear index corresponds to the Ambisonics Channel Convention [#]_.

    .. math::

        \mathrm{acn} = n^2 + n + m

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

    return n ** 2 + n + m


def spherical_harmonic_basis_gradient(n_max, coords):
    r"""
    Calulcates the unit sphere gradients of the complex spherical harmonics.


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
    coords : :doc:`pf.Coordinates <pyfar:classes/pyfar.coordinates>`
        Pyfar ``Coordinate`` object with sampling points for which the basis matrix is
        calculated

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
    >>> coords = spharpy.samplings.icosahedron()
    >>> grad_theta, grad_phi = spharpy.spherical.spherical_harmonic_basis_gradient(n_max, coords)


    """  # noqa: 501
    if not isinstance(coords, pf.Coordinates):
        axis = np.where(coords.shape == 3)[0][0]
        if axis == 0:
            coords = coords.T
        coords = pf.Coordinates(coords[:, 0], coords[:, 1], coords[:, 2])

    n_points = coords.csize
    n_coeff = (n_max + 1) ** 2
    theta = coords.colatitude
    phi = coords.azimuth
    grad_theta = np.zeros((n_points, n_coeff), dtype=complex)
    grad_phi = np.zeros((n_points, n_coeff), dtype=complex)

    for acn in range(n_coeff):
        n, m = acn_to_nm(acn)

        grad_theta[:, acn] = _special.spherical_harmonic_derivative_theta(
            n, m, theta, phi
        )
        grad_phi[:, acn] = _special.spherical_harmonic_gradient_phi(
            n, m, theta, phi)

    return grad_theta, grad_phi


def spherical_harmonic_basis_gradient_real(n_max, coords):
    r"""
    Calulcates the unit sphere gradients of the real valued spherical hamonics.

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
    coords : :doc:`pf.Coordinates <pyfar:classes/pyfar.coordinates>`
        Coordinate object with sampling points for which the basis matrix is
        calculated

    Returns
    -------
    grad_theta : ndarray, float
        Gradient with respect to the co-latitude angle.
    grad_phi : ndarray, float
        Gradient with respect to the azimuth angle.

    """  # noqa: 501
    if not isinstance(coords, pf.Coordinates):
        axis = np.where(coords.shape == 3)[0][0]
        if axis == 0:
            coords = coords.T
        coords = pf.Coordinates(coords[:, 0], coords[:, 1], coords[:, 2])
    n_points = coords.csize
    n_coeff = (n_max + 1) ** 2
    theta = coords.colatitude
    phi = coords.azimuth
    grad_theta = np.zeros((n_points, n_coeff), dtype=float)
    grad_phi = np.zeros((n_points, n_coeff), dtype=float)

    for acn in range(n_coeff):
        n, m = acn_to_nm(acn)

        grad_theta[:, acn] = _special.spherical_harmonic_derivative_theta_real(
            n, m, theta, phi
        )
        grad_phi[:, acn] = _special.spherical_harmonic_gradient_phi_real(
            n, m, theta, phi
        )

    return grad_theta, grad_phi


def modal_strength(n_max, kr, arraytype="rigid"):
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
    n_coeff = (n_max + 1) ** 2
    n_bins = kr.shape[0]

    modal_strength_mat = np.zeros((n_bins, n_coeff, n_coeff), dtype=complex)

    for n in range(n_max + 1):
        bn = _modal_strength(n, kr, arraytype)
        for m in range(-n, n + 1):
            acn = n * n + n + m
            modal_strength_mat[:, acn, acn] = bn

    return np.squeeze(modal_strength_mat)


def _modal_strength(n, kr, config):
    """Helper function for the calculation of the modal strength for
    plane waves"""
    if config == "open":
        ms = 4 * np.pi * pow(1.0j, n) * _special.spherical_bessel(n, kr)
    elif config == "rigid":
        ms = (
            4
            * np.pi
            * pow(1.0j, n + 1)
            / _special.spherical_hankel(n, kr, derivative=True)
            / kr ** 2
        )
    elif config == "cardioid":
        ms = (
            4
            * np.pi
            * pow(1.0j, n)
            * (
                _special.spherical_bessel(n, kr)
                - 1.0j * _special.spherical_bessel(n, kr, derivative=True)
            )
        )
    else:
        raise ValueError("Invalid configuration.")

    return ms


def aperture_vibrating_spherical_cap(n_max, rad_sphere, rad_cap):
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
    rad_sphere : double, ndarray
        Radius of the sphere
    rad_cap : double
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
    n_sh = (n_max + 1) ** 2

    aperture = np.zeros((n_sh, n_sh), dtype=float)

    aperture[0, 0] = (1 - arg) * 2 * np.pi
    for n in range(1, n_max + 1):
        legendre_minus = special.legendre(n - 1)(arg)
        legendre_plus = special.legendre(n + 1)(arg)
        legendre_term = legendre_minus - legendre_plus
        for m in range(-n, n + 1):
            acn = nm_to_acn(n, m)
            aperture[acn, acn] = legendre_term * 4 * np.pi / (2 * n + 1)

    return aperture


def radiation_from_sphere(
    n_max, rad_sphere, k, distance, density_medium=1.2, speed_of_sound=343.0
):
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
    n_sh = (n_max + 1) ** 2

    k = np.atleast_1d(k)
    n_bins = k.shape[0]
    radiation = np.zeros((n_bins, n_sh, n_sh), dtype=complex)

    for n in range(n_max + 1):
        hankel = _special.spherical_hankel(n, k * distance, kind=2)
        hankel_prime = _special.spherical_hankel(
            n, k * rad_sphere, kind=2, derivative=True
        )
        radiation_order = (-1j * hankel / hankel_prime *
                           density_medium * speed_of_sound)
        for m in range(-n, n + 1):
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
    n_sh = (n_max + 1) ** 2
    sid_n = sph_identity_matrix(n_max, "n-nm").T @ np.arange(0, n_max + 1)
    sid_m = np.zeros(n_sh, dtype=int)
    idx_n = 0
    for n in range(1, n_max + 1):
        for m in range(1, n + 1):
            sid_m[idx_n + 2 * m - 1] = n - m + 1
            sid_m[idx_n + 2 * m] = -(n - m + 1)
        sid_m[idx_n + 2 * n + 1] = 0
        idx_n += 2 * n + 1

    return sid_n, sid_m


def sid2acn(n_max):
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


def sph_identity_matrix(n_max, type="n-nm"):
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
    n_sh = (n_max + 1) ** 2

    if type != "n-nm":
        raise NotImplementedError

    identity_matrix = np.zeros((n_max + 1, n_sh), dtype=int)

    for n in range(n_max + 1):
        m = np.arange(-n, n + 1)
        linear_nm = nm_to_acn(np.tile(n, m.shape), m)
        identity_matrix[n, linear_nm] = 1

    return identity_matrix
