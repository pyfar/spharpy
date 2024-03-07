import numpy as np
import scipy.special as special
import spharpy.special as _special
import logging as logger
from pyfar import Coordinates
from functools import lru_cache
from spharpy.samplings.helpers import calculate_sampling_weights


class SphHarm:
    def __init__(self, n_max, coords, basis_type='complex', channel_convention='acn', inv_transform_type=None,
                 normalization='n3d'):
        r"""
        Create a spherical harmonics class for transformation between spherical harmonics and signals.

        Parameters
        ----------
        n_max : int,
            Maximum spherical harmonic order
        coords : :doc:`pf.Coordinates <pyfar:classes/pyfar.coordinates>`
            Coordinate object with sampling points for which the basis matrix is
            calculated
        basis_type : str, optional
            Type of spherical harmonic basis, either 'complex' or 'real'. The default is 'complex'.
        channel_convention : str, optional
            Channel ordering convention, either 'acn' or 'fuma'. The default is 'acn'.
            (FuMa is only supported up to 3rd order)
        inv_transform_type : str, optional
            Inverse transform type, either 'pseudo_inverse' or 'quadrature'. The default is None.
        normalization : str, optional
            Normalization convention, either 'n3d', 'maxN' or 'sn3d'. The default is 'n3d'.
            (maxN is only supported up to 3rd order)
        """
        self.n_max = n_max
        self.coords = coords  # coordinates : :doc:`pf.Coordinates <pyfar:classes/pyfar.coordinates>`
        self.basis_type = basis_type
        self.inv_transform_type = inv_transform_type
        self.channel_convention = channel_convention  # channel ordering convention
        self.normalization = normalization  # gain normalization convention
        self.basis = None
        self.basis_gradient_theta, self.basis_gradient_phi = (None, None)
        self._basis_inv = None
        self._basis_inv_gradient_theta, self._basis_inv_gradient_phi = (None, None)

    def _validate_parameters(self):
        if not isinstance(self.coords, Coordinates):
            raise TypeError("coords must be a pyfar.Coordinates object")
        assert self.n_max >= 0, "n_max must be a positive integer"
        assert self.channel_convention in ['acn', 'fuma'], ("Invalid channel convention, currently only 'acn' "
                                                            "and 'fuma' are supported")
        assert self.normalization in ['n3d', 'maxN', 'sn3d'], ("Invalid normalization, "
                                                               "currently only 'n3d', 'maxN', 'sn3d' are "
                                                               "supported")
        assert self.basis_type in ['complex', 'real'], ("Invalid basis type, currently only 'complex' and 'real' "
                                                        "are supported")
        assert self.inv_transform_type in ['pseudo_inverse', 'quadrature', None], (
            "Invalid inverse transform type, "
            "currently only 'pseudo_inverse' "
            "and 'quadrature' are supported")

    @lru_cache(maxsize=128)
    def _compute_basis(self):
        logger.info("Computing basis matrix for n_max=%d", self.n_max)
        if self.basis_type == 'complex':
            self.basis = spherical_harmonic_basis(self.n_max, self.coords,
                                                  self.normalization, self.channel_convention)
        elif self.basis_type == 'real':
            self.basis = spherical_harmonic_basis_real(self.n_max, self.coords,
                                                       self.normalization, self.channel_convention)
        else:
            raise ValueError("Invalid signal type, should be either 'complex' or 'real'")

    def compute_basis(self):
        try:
            return self._compute_basis()
        except Exception as e:
            raise ValueError("Error computing basis:", e) from e

    @lru_cache(maxsize=128)
    def _compute_basis_gradient(self):
        logger.info("Computing basis gradient for n_max=%d", self.n_max)
        if self.normalization in ['maxN', 'sn3d'] or self.channel_convention == 'fuma':
            raise ValueError("Gradient computation not supported for MaxN, SN3D normalization or FuMa channel ordering")
        else:
            if self.basis_type == 'complex':
                grad_theta, grad_phi = spherical_harmonic_basis_gradient(self.n_max, self.coords)
                self.basis_gradient_theta, self.basis_gradient_phi = (grad_theta, grad_phi)
            elif self.basis_type == 'real':
                grad_theta, grad_phi = spherical_harmonic_basis_gradient_real(self.n_max, self.coords)
                self.basis_gradient_theta, self.basis_gradient_phi = (grad_theta, grad_phi)
            else:
                raise ValueError("Invalid signal type, should be either 'complex' or 'real'")

    def compute_basis_gradient(self):
        try:
            return self._compute_basis_gradient()
        except Exception as e:
            raise ValueError("Error computing basis gradient:", e) from e

    @property
    def basis_inv(self):
        if self._basis_inv is None:
            self.compute_inverse()
        return self._basis_inv

    def compute_inverse(self, inv_transform_type=None, weights=None):
        """
        Compute the inverse transform matrix for the specified transform type

        The inverse transform matrix is calculated based on the specified `inv_transform_type`.
        If 'pseudo_inverse' is chosen, the Moore-Penrose pseudo-inverse is used.
        If 'quadrature' is chosen, the inverse is computed using the conjugate transpose of
        the basis matrix multiplied by 4 * π * weights.

        Parameters
        ----------
        inv_transform_type : {'pseudo_inverse', 'quadrature'}, optional
            Type of inverse transform to compute.
        weights : array-like, optional
            Sampling weights for the quadrature transform. Required if 'quadrature' is chosen.

        Returns
        -------
        None
        """
        if self.basis is None:
            self.compute_basis()
        if inv_transform_type != self.inv_transform_type:
            assert inv_transform_type in ['pseudo_inverse', 'quadrature'], (
                "Invalid inverse transform type, "
                "currently only 'pseudo_inverse' "
                "and 'quadrature' are supported")
            self.inv_transform_type = inv_transform_type
        # print("computing inverse basis using ", self.inv_transform_type)
        if self.inv_transform_type is None:
            ValueError("Inverse transform type not specified")
        elif self.inv_transform_type == 'pseudo_inverse':
            self._basis_inv = np.linalg.pinv(self.basis)
        elif self.inv_transform_type == 'quadrature':
            if weights is None:
                print("Warning: No weights specified for quadrature transform,"
                      "calculating weights using voronoi tessellation of sphere")
                weights = calculate_sampling_weights(self.coords)
            self._basis_inv = np.conj(self.basis).T * (weights)

    @property
    def basis_inv_gradient_theta(self):
        if self._basis_inv_gradient_theta is None or self._basis_inv_gradient_phi is None:
            self.compute_inverse_gradient()
        return self._basis_inv_gradient_theta

    @property
    def basis_inv_gradient_phi(self):
        if self._basis_inv_gradient_theta is None or self._basis_inv_gradient_phi is None:
            self.compute_inverse_gradient()
        return self._basis_inv_gradient_phi

    def compute_inverse_gradient(self, inv_transform_type=None, weights=None):
        if inv_transform_type != self.inv_transform_type:
            self.inv_transform_type = inv_transform_type
        if self.inv_transform_type is None:
            ValueError("Inverse transform type not specified")
        elif self.inv_transform_type == 'pseudo_inverse':
            self._basis_inv_gradient_theta = np.linalg.pinv(self.basis_gradient_theta)
            self._basis_inv_gradient_phi = np.linalg.pinv(self.basis_gradient_phi)
        elif self.inv_transform_type == 'quadrature':
            if weights is None:
                weights = calculate_sampling_weights(self.coords)
            self._basis_inv_gradient_theta = np.conj(self.basis_gradient_theta).T * (4 * np.pi * weights)
            self._basis_inv_gradient_phi = np.conj(self.basis_gradient_phi).T * (4 * np.pi * weights)

    def transform(self, signal):
        if self.basis is None:
            self.compute_basis()
        return np.dot(self.basis.T, signal)

    def transform_gradient(self, signal):
        if self.basis_gradient_theta is None or self.basis_gradient_phi is None:
            self.compute_basis_gradient()
        return np.dot(self.basis_gradient_theta.T, signal), np.dot(self.basis_gradient_phi.T, signal)

    def inverse_transform(self, coefficients):
        if self.basis_inv is None:
            self.compute_inverse()
        return np.dot(self.basis_inv.T, coefficients)

    def inverse_transform_gradient(self, coefficients):
        if self.basis_inv_gradient_theta is None or self.basis_inv_gradient_phi is None:
            self.compute_inverse_gradient()
        return np.dot(self.basis_inv_gradient_theta, coefficients), np.dot(self.basis_inv_gradient_phi, coefficients)

    def interpolate(self, non_uniform_coords):
        # Implement interpolation algorithm using the basis matrix
        # TODO: Implement interpolation algorithm using the basis matrix
        pass

    def filter(self, coefficients, band_type, cutoff_order):
        # Implement filtering algorithm using the basis matrix
        # TODO: Implement filtering
        pass

    def plot_basis_functions(self, order, mode):
        # Implement plotting function for basis functions
        # TODO: Implement plotting function for basis functions
        pass

    def plot_reconstructed_signal(self, coefficients):
        # Implement plotting function for reconstructed signal
        # TODO: Implement plotting function for reconstructed signal
        pass


def to_maxN_norm(acn):
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
    maxN = [np.sqrt(1 / 2),
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
            np.sqrt(8 / 35)]
    return maxN[acn]


def to_sn3d_norm(m, n):
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


def fuma2nm(fuma):
    r"""
    Calculate the spherical harmonic order n and degree m for a linear
    coefficient index, according to the FuMa (Furse-Malham) Channel Ordering Convention [2]_.

    FuMa = WXYZ | RSTUV | KLMNOPQ
    ACN = WYZX | VTRSU | QOMKLNP


    References
    ----------
    [2]  D. Malham, "Higher order Ambisonic systems” Space in Music – Music in Space (Mphil thesis).
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
        raise ValueError("Invalid FuMa channel index, must be between 0 and 15 (supported up to 3rd order)")

    acn = fuma_mapping[fuma]
    n, m = acn2nm(acn)  # Assuming you have the acn2nm function defined
    return n, m


def spherical_harmonic_basis(n_max, coords, normalization='n3d', channel_convention='acn'):
    r"""
    Calculates the complex valued spherical harmonic basis matrix.

    The spherical harmonic functions are fully normalized (N3D) and include the
    Condon-Shotley phase term :math:`(-1)^m` [#]_, [#]_.

    .. math::

        Y_n^m(\theta, \phi) = \sqrt{\frac{2n+1}{4\pi}
        \frac{(n-m)!}{(n+m)!}} P_n^m(\cos \theta) e^{i m \phi}

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

    Returns
    -------
    Y : ndarray, complex
        Complex spherical harmonic basis matrix

    Examples
    --------

    >>> import spharpy
    >>> n_max = 2
    >>> coords = spharpy.samplings.icosahedron()
    >>> Y = spharpy.spherical.spherical_harmonic_basis(n_max, coords)

    """

    n_coeff = (n_max + 1) ** 2

    basis = np.zeros((coords.csize, n_coeff), dtype=complex)

    for acn in range(n_coeff):
        if channel_convention == 'fuma':
            order, degree = fuma2nm(acn)
        else:
            order, degree = acn2nm(acn)
        basis[:, acn] = _special.spherical_harmonic(
            order,
            degree,
            coords.colatitude,
            coords.azimuth)
        if normalization == 'sn3d':
            basis[:, acn] *= to_sn3d_norm(degree, order)
        elif normalization == 'maxN':
            basis[:, acn] *= to_maxN_norm(acn)

    return basis


def spherical_harmonic_basis_real(n_max, coords, normalization='n3d', channel_convention='acn'):
    r"""
    Calculates the real valued spherical harmonic basis matrix.

    The spherical harmonic functions are fully normalized (N3D) and follow
    the AmbiX phase convention [#]_.

    .. math::

        Y_n^m(\theta, \phi) = \sqrt{\frac{2n+1}{4\pi}
        \frac{(n-|m|)!}{(n+|m|)!}} P_n^{|m|}(\cos \theta)
        \begin{cases}
            \displaystyle \cos(|m|\phi),  & \text{if $m \ge 0$} \newline
            \displaystyle \sin(|m|\phi) ,  & \text{if $m < 0$}
        \end{cases}

    References
    ----------
    .. [#]  C. Nachbar, F. Zotter, E. Deleflie, and A. Sontacchi, “Ambix - A
            Suggested Ambisonics Format (revised by F. Zotter),” International
            Symposium on Ambisonics and Spherical Acoustics,
            vol. 3, pp. 1-11, 2011.


    Parameters
    ----------
    n_max : int
        Spherical harmonic order
    coords : :doc:`pf.Coordinates <pyfar:classes/pyfar.coordinates>`
        Coordinate object with sampling points for which the basis matrix is
        calculated
    normalization : str, optional
        Normalization convention, either 'n3d', 'maxN' or 'sn3d'. The default is 'n3d'.
        (maxN is only supported up to 3rd order)
    channel_convention : str, optional
        Channel ordering convention, either 'acn' or 'fuma'. The default is 'acn'.
        (FuMa is only supported up to 3rd order)

    Returns
    -------
    Y : ndarray, float
        Real valued spherical harmonic basis matrix.


    """
    n_coeff = (n_max + 1) ** 2

    basis = np.zeros((coords.csize, n_coeff), dtype=float)

    for acn in range(n_coeff):
        if channel_convention == 'fuma':
            order, degree = fuma2nm(acn)
        else:
            order, degree = acn2nm(acn)
        basis[:, acn] = _special.spherical_harmonic_real(
            order,
            degree,
            coords.colatitude,
            coords.azimuth)
        if normalization == 'sn3d':
            basis[:, acn] *= to_sn3d_norm(degree, order)
        elif normalization == 'maxN':
            basis[:, acn] *= to_maxN_norm(acn)

    return basis


def acn2nm(acn):
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

    Returns
    ----------
    n : ndarray, int
        Spherical harmonic order
    m : ndarray, int
        Spherical harmonic degree

    """
    acn = np.asarray(acn, dtype=int)

    n = (np.ceil(np.sqrt(acn + 1)) - 1)
    m = acn - n ** 2 - n

    n = n.astype(int, copy=False)
    m = m.astype(int, copy=False)

    return n, m


def nm2acn(n, m):
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

    return n ** 2 + n + m


def spherical_harmonic_basis_gradient(n_max, coords):
    r"""
    Calulcates the unit sphere gradients of the complex spherical harmonics.

    The spherical harmonic functions are fully normalized (N3D) and include the
    Condon-Shotley phase term :math:`(-1)^m` [#]_.

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
        Coordinate object with sampling points for which the basis matrix is
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
    >>> Y_theta, Y_phi = spharpy.spherical.spherical_harmonic_basis_gradient(
            n_max, coords)

    """
    n_points = coords.csize
    n_coeff = (n_max + 1) ** 2
    theta = coords.colatitude
    phi = coords.azimuth
    grad_theta = np.zeros((n_points, n_coeff), dtype=complex)
    grad_phi = np.zeros((n_points, n_coeff), dtype=complex)

    for acn in range(n_coeff):
        n, m = acn2nm(acn)

        grad_theta[:, acn] = \
            _special.spherical_harmonic_derivative_theta(
                n, m, theta, phi)
        grad_phi[:, acn] = \
            _special.spherical_harmonic_gradient_phi(
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

    """
    n_points = coords.csize
    n_coeff = (n_max + 1) ** 2
    theta = coords.colatitude
    phi = coords.azimuth
    grad_theta = np.zeros((n_points, n_coeff), dtype=float)
    grad_phi = np.zeros((n_points, n_coeff), dtype=float)

    for acn in range(n_coeff):
        n, m = acn2nm(acn)

        grad_theta[:, acn] = \
            _special.spherical_harmonic_derivative_theta_real(
                n, m, theta, phi)
        grad_phi[:, acn] = \
            _special.spherical_harmonic_gradient_phi_real(
                n, m, theta, phi)

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
    if config == 'open':
        ms = 4 * np.pi * pow(1.0j, n) * _special.spherical_bessel(n, kr)
    elif config == 'rigid':
        ms = 4 * np.pi * pow(1.0j, n + 1) / \
             _special.spherical_hankel(n, kr, derivative=True) / (kr) ** 2
    elif config == 'cardioid':
        ms = 4 * np.pi * pow(1.0j, n) * \
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
            acn = nm2acn(n, m)
            aperture[acn, acn] = legendre_term * 4 * np.pi / (2 * n + 1)

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
    n_sh = (n_max + 1) ** 2

    k = np.atleast_1d(k)
    n_bins = k.shape[0]
    radiation = np.zeros((n_bins, n_sh, n_sh), dtype=complex)

    for n in range(n_max + 1):
        hankel = _special.spherical_hankel(n, k * distance, kind=2)
        hankel_prime = _special.spherical_hankel(
            n, k * rad_sphere, kind=2, derivative=True)
        radiation_order = -1j * hankel / hankel_prime * \
                          density_medium * speed_of_sound
        for m in range(-n, n + 1):
            acn = nm2acn(n, m)
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
    sid_n = sph_identity_matrix(n_max, 'n-nm').T @ np.arange(0, n_max + 1)
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
    linear_sid = nm2acn(sid_n, sid_m)
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
    n_sh = (n_max + 1) ** 2

    if type != 'n-nm':
        raise NotImplementedError

    identity_matrix = np.zeros((n_max + 1, n_sh), dtype=int)

    for n in range(n_max + 1):
        m = np.arange(-n, n + 1)
        linear_nm = nm2acn(np.tile(n, m.shape), m)
        identity_matrix[n, linear_nm] = 1

    return identity_matrix


if __name__ == '__main__':
    from samplings import equiangular
#
    coords = equiangular(n_max=6, radius=0.3)
    Sphharm = SphHarm(n_max=6, coords=coords)
    Sphharm.compute_basis() # initialize basis matrix
    Sphharm.compute_basis_gradient() # initialize basis gradient matrices
    Sphharm.compute_inverse( inv_transform_type = 'quadrature') # initialize inverse transform matrix
    Sphharm.compute_inverse_gradient() # initialize inverse gradient matrices
    Pnm_quad = Sphharm.transform(np.ones(196))
    sig1 = Sphharm.inverse_transform(Pnm_quad)
    Sphharm.compute_inverse( inv_transform_type = 'pseudo_inverse') # initialize inverse transform matrix
    Y = Sphharm.basis
    Yinv = Sphharm.basis_inv
    Pnm_pinv = Sphharm.transform(np.ones(196))
    sig2 = Sphharm.inverse_transform(Pnm_pinv)
    print("sig1 close to original signal: ", np.allclose(sig1, np.ones(196)))
    print("sig2 close to original signal: ", np.allclose(sig2, np.ones(196)))
