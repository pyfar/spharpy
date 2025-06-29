import numpy as np
import pyfar as pf
import spharpy as sy


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
        - ``'auto'``: ``'quadrature'`` if `coordinates.quadrature`
           is True otherwise ``'quadrature'``. If `coordinates` is not
           SamplingSphere an error is returned.



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
    inverse_method : {'auto', 'quadrature', 'pseudo_inverse'}, default='auto'
        Method for computing the inverse transform:

        - ‘auto’: use ‘quadrature’ when applicable, otherwise ‘pseudo_inverse’.
        - ‘quadrature’: compute the inverse via numerical quadrature.
        - ‘pseudo_inverse’: compute the inverse via a pseudo-inverse approximation.
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
        inverse_method="auto",
        condon_shortley="auto",
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
            # If basis_type hasn't been set yet, assume "complex" by default,
            # but in practice __init__ sets basis_type before condon_shortley.
            if self.basis_type == "complex":
                resolved = True
            else:
                resolved = False
            value = resolved
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
        # If the user passes "auto", require SamplingSphere and resolve it
        if isinstance(value, str) and value == "auto":
            if not isinstance(self.coordinates,sy.SamplingSphere):
                raise ValueError("'auto' is only valid if `coordinates` is a SamplingSphere.")
            if isinstance(self.coordinates, sy.SamplingSphere) and self.coordinates.quadrature:
                value = "quadrature"
            else:
                value = "pseudo_inverse"
        elif value == "quadrature":
            if not isinstance(self.coordinates, sy.SamplingSphere) or \
            not self.coordinates.quadrature:
                raise ValueError("'quadrature' requires `coordinates` to be " \
                "a SamplingSphere and coordinates.quadrature to be True.")
        elif value != "pseudo_inverse":
            raise ValueError("Invalid inverse_method. Allowed: 'pseudo_inverse', " \
            "'quadrature', or 'auto'.")

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
            function = sy.spherical.spherical_harmonic_basis
        elif self.basis_type == "real":
            function = sy.spherical.spherical_harmonic_basis_real
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
                function = sy.spherical.spherical_harmonic_basis_gradient
            elif self.basis_type == "real":
                function = sy.spherical.spherical_harmonic_basis_gradient_real
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
        _inv_flag = self.inverse_method
        if _inv_flag == "pseudo_inverse":
            self._basis_inv = np.linalg.pinv(self._basis)
        elif _inv_flag == "quadrature":
            self._basis_inv = np.einsum('ij,i->ji', np.conj(self._basis),
                                         self.coordinates.weights)

    def _reset_compute_attributes(self):
        """Reset the computed attributes for the SphericalHarmonics class in
        case of changes in the parameters."""
        self._basis = None
        self._basis_gradient_theta = None
        self._basis_gradient_phi = None
        self._basis_inv = None

