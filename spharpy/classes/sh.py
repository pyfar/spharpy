"""
Documentation for the SphericalHarmonics class, will be added in an other PR.
"""
import numpy as np
import pyfar as pf
import spharpy
from abc import ABC, abstractmethod


class _SphericalHarmonicBase(ABC):
    """Base class defining properties to parametrize spherical harmonics.

    This base class serves as a base for all classes requiring a definition of
    the spherical harmonics without explicitly setting a spherical harmonic
    order. This class is intended for cases where the spherical harmonic order
    is implemented as a read-only property in child classes, for example when
    the order is implicitly defined by other parameters or inferred from data.

    Attributes
    ----------
    basis_type : str
        Type of spherical harmonic basis, either ``'real'`` or ``'complex'``.
        The default is ``'real'``.
    normalization : str, optional
        Normalization convention, either ``'N3D'``, ``'NM'``, ``'maxN'``,
        ``'SN3D'``, or ``'SNM'``. The default is ``'N3D'``. Note that
        ``'maxN'`` is only supported up to 3rd order.
    channel_convention : str, optional
        Channel ordering convention, either ``'ACN'`` or ``'FuMa'``.
        The default is ``'ACN'``. Note that ``'FuMa'`` is only supported up to
        3rd order.
    condon_shortley : bool, str, optional
        Condon-Shortley phase term. If ``True``, Condon-Shortley is included,
        if ``False`` it is not included. The default is ``'auto'``, which
        corresponds to ``True`` for type ``complex`` and ``False`` for type
        ``real``.
    """

    def __init__(
            self,
            basis_type="real",
            normalization="N3D",
            channel_convention="ACN",
            condon_shortley="auto",
        ):
        self._basis_type = None
        self._channel_convention = None
        self._condon_shortley = None
        self._normalization = None

        # basis_type needs to be initialized first, since the default for the
        # Condon-Shortley phase depends on the basis type
        self.basis_type = basis_type
        self.condon_shortley = condon_shortley
        # n_max needs to be initialized before channel_convention and
        # normalization, since both have restrictions on n_max
        self.channel_convention = channel_convention
        self.normalization = normalization

    @property
    @abstractmethod
    def n_max(self):
        """Get or set the spherical harmonic order."""

    @property
    def condon_shortley(self):
        """Get or set the Condon-Shortley phase term."""
        return self._condon_shortley

    @condon_shortley.setter
    def condon_shortley(self, value):
        """Get or set the Condon-Shortley phase term."""
        if isinstance(value, str):
            if value != 'auto':
                raise ValueError(
                    "condon_shortley must be a bool or the string 'auto'")

            value = self.basis_type == "complex"
        elif not isinstance(value, bool):
            raise ValueError(
                "condon_shortley must be a bool or the string 'auto'")

        if self._condon_shortley != value:
            self._condon_shortley = value
            self._on_property_change()

    @property
    def basis_type(self):
        """Get or set the spherical harmonic basis type."""
        return self._basis_type

    @basis_type.setter
    def basis_type(self, value):
        """Get or set the spherical harmonic basis type."""
        if value not in ["complex", "real"]:
            raise ValueError(
                "Invalid basis type, only 'complex' and 'real' are supported")

        if self._basis_type != value:
            self._basis_type = value
            self._on_property_change()

    @property
    def channel_convention(self):
        """Get or set the channel ordering convention."""
        return self._channel_convention

    @channel_convention.setter
    def channel_convention(self, value):
        """Get or set the channel order convention."""
        if value not in ["ACN", "FuMa"]:
            raise ValueError("Invalid channel convention, "
                             "currently only 'ACN' "
                             "and 'FuMa' are supported")

        if value == "FuMa" and self.n_max > 3:
            raise ValueError(
                "n_max > 3 is not allowed with 'FuMa' channel convention")

        if self._channel_convention != value:
            self._channel_convention = value
            self._on_property_change()

    @property
    def normalization(self):
        """Get or set the normalization convention."""
        return self._normalization

    @normalization.setter
    def normalization(self, value):
        """Get or set the normalization convention."""
        if value not in ["N3D", "NM", "maxN", "SN3D", "SNM"]:
            raise ValueError(
                "Invalid normalization, "
                "currently only 'N3D', 'NM', 'maxN', 'SN3D', 'SNM' are "
                "supported",
            )

        if value == "maxN" and self.n_max > 3:
            raise ValueError(
                "n_max > 3 is not allowed with 'maxN' normalization")

        if self._normalization != value:
            self._normalization = value
            self._on_property_change()

    def _on_property_change(self):  # noqa: B027
        """Method called when a class property changes.
        This method can be overridden in child classes to re-compute dependent
        properties.
        """
        pass


class SphericalHarmonicDefinition(_SphericalHarmonicBase):
    """Class storing the (discrete) definition of spherical harmonics.

    This class can serve as a container to create related objects, e.g.,
    spherical harmonic basis matrices for given sampling points, transforms,
    or other spherical harmonic related data and computations.

    Attributes
    ----------
    n_max : int, optional
        Maximum spherical harmonic order. Must be an integer greater or equal
        to 0. The default is ``0``.
    basis_type : str
        Type of spherical harmonic basis, either ``'complex'`` or
        ``'real'``. The default is ``'real'``.
    normalization : str, optional
        Normalization convention, either ``'N3D'``, ``'NM'``, ``'SN3D'``,
        ``'SNM'``, or ``'maxN'``. ``'maxN'`` is only supported up to 3rd order.
        The default is ``'N3D'``.
    channel_convention : str, optional
        Channel ordering convention, either ``'ACN'`` or ``'FuMa'``.
        ``'FuMa'`` is only supported up to 3rd order.
        The default is ``'ACN'``.
    condon_shortley : bool, str, optional
         Whether to include the Condon-Shortley phase term. If ``True``,
        Condon-Shortley is included, if ``False`` it is not
        included. The default is ``'auto'``, which corresponds to
        ``True`` for complex `basis_type` and ``False`` for real `basis_type`.
    """

    def __init__(
            self,
            n_max=0,
            basis_type="real",
            normalization="N3D",
            channel_convention="ACN",
            condon_shortley="auto",
        ):
        self._n_max = 0
        super().__init__(
            basis_type=basis_type,
            normalization=normalization,
            channel_convention=channel_convention,
            condon_shortley=condon_shortley,
        )
        self.n_max = n_max

    @property
    def n_max(self):
        """Get or set the spherical harmonic order."""
        return self._n_max

    @n_max.setter
    def n_max(self, value : int):
        """Get or set the spherical harmonic order."""
        if value < 0 or value % 1 != 0:
            raise ValueError("n_max must be a positive integer")
        if self.channel_convention == "FuMa" and value > 3:
            raise ValueError(
                "n_max > 3 is not allowed with 'FuMa' channel convention")
        if self.normalization == "maxN" and value > 3:
            raise ValueError(
                "n_max > 3 is not allowed with 'maxN' normalization")

        if self._n_max != value:
            self._n_max = value
            self._on_property_change()


class SphericalHarmonics(SphericalHarmonicDefinition):
    r"""
    Compute spherical harmonic basis matrices, their inverses, and gradients.

    The the basis functions that are used to build the matrices can be
    configured with the parameters described below. See the
    :py:mod:`spherical harmonic documentation<spharpy.classes.sh>` for a
    detailed description.

    Parameters
    ----------
    n_max : int
        Maximum spherical harmonic order. Must be an integer greater or equal
        to 0.
    coordinates : :py:class:`~pyfar.Coordinates`, :py:class:`SamplingSphere`
        The sampling points for which the matrices are computed.
    basis_type : str, optional
        Type of spherical harmonic basis, either ``'complex'`` or
        ``'real'``. The default is ``'real'``.
    normalization : str, optional
        Normalization convention, either ``'N3D'``, ``'NM'``, ``'SN3D'``,
        ``'SNM'``, or ``'maxN'``. ``'maxN'`` is only supported up to 3rd order.
        The default is ``'N3D'``.
    channel_convention : str, optional
        Channel ordering convention, either ``'ACN'`` or ``'FuMa'``.
        ``'FuMa'`` is only supported up to 3rd order.
        The default is ``'ACN'``.
    inverse_method : str, ``None``, optional
        Method for computing the inverse transform:

        - ``'quadrature'``: compute the inverse via numerical quadrature. Note
          that this requires `coordinates` to be a :py:class:`SamplingSphere`
          object with weights summing to :math:`4\pi`.
        - ``'pseudo_inverse'``: compute the inverse via the pseudo-inverse.
          Note that this requires `coordinates` to be a
          :py:class:`SamplingSphere` object.
        - ``None``: denotes that the inverse is not defined and cannot be
          computed.
        - ``'auto'``: If coordinates are a :py:class:`SamplingSphere`, use
          ``'quadrature'`` when applicable and ``'pseudo_inverse'`` otherwise.
          If coordinates are a :py:class:`~pyfar.Coordinates` object ``None``
          is used.

        The default is ``'auto'``.

    condon_shortley : bool or str, optional
        Whether to include the Condon-Shortley phase term. If ``True``,
        Condon-Shortley is included, if ``False`` it is not
        included. The default is ``'auto'``, which corresponds to
        ``True`` for complex `basis_type` and ``False`` for real `basis_type`.
    """

    def __init__(
        self,
        n_max,
        coordinates,
        basis_type="real",
        normalization="N3D",
        channel_convention="ACN",
        inverse_method="auto",
        condon_shortley="auto",
    ):
        super().__init__(
            n_max=n_max,
            basis_type=basis_type,
            normalization=normalization,
            channel_convention=channel_convention,
            condon_shortley=condon_shortley,
        )

        # initialize private attributes
        self._coordinates = pf.Coordinates()
        self._inverse_method = None
        self._reset_compute_attributes()

        self.coordinates = coordinates
        self.inverse_method = inverse_method

    @classmethod
    def from_definition(cls, definition, coordinates, inverse_method="auto"):
        r"""
        Create SphericalHarmonics instance from SphericalHarmonicDefinition.

        Parameters
        ----------
        definition : SphericalHarmonicDefinition
            The spherical harmonic definition.
        coordinates : :py:class:`~pyfar.Coordinates`, :py:class:`SamplingSphere`
            The sampling points for which the matrices are computed.
        inverse_method : str, ``None``, optional
            Method for computing the inverse transform:

            - ``'quadrature'``: compute the inverse via numerical quadrature.
              Note that this requires `coordinates` to be a
              :py:class:`SamplingSphere` object with weights summing to
              :math:`4\pi`.
            - ``'pseudo_inverse'``: compute the inverse via the pseudo-inverse.
              Note that this requires `coordinates` to be a
              :py:class:`SamplingSphere` object.
            - ``None``: denotes that the inverse is not defined and cannot be
              computed.
              - ``'auto'``: If coordinates are a :py:class:`SamplingSphere`,
              use ``'quadrature'`` when applicable and ``'pseudo_inverse'``
              otherwise. If coordinates are a :py:class:`~pyfar.Coordinates`
              object ``None`` is used.

            The default is ``'auto'``.

        Returns
        -------
        SphericalHarmonics
            A new SphericalHarmonics instance initialized with the parameters
            from the definition object.
        """  # noqa: E501

        if type(definition) is not SphericalHarmonicDefinition:
            raise TypeError('definition must be a SphericalHarmonicDefinition')

        return cls(definition.n_max, coordinates, definition.basis_type,
                   definition.normalization, definition.channel_convention,
                   inverse_method, definition.condon_shortley)

    @property
    def coordinates(self):
        """Get or set the coordinates."""
        return self._coordinates

    @coordinates.setter
    def coordinates(self, value):
        """Get or set the coordinates."""
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
    def inverse_method(self):
        """Get or set the inverse transform method."""
        return self._inverse_method

    @inverse_method.setter
    def inverse_method(self, value):
        """Get or set the inverse transform method."""
        if value not in ["pseudo_inverse", "quadrature", "auto", None]:
            raise ValueError(
                "Invalid inverse_method. Allowed: 'pseudo_inverse', "
                "'quadrature', or 'auto'.")

        if value == self._inverse_method:
            return

        if value is None:
            self._inverse_method = value
            return

        if type(self.coordinates) is pf.Coordinates:
            if value == "auto":
                self.inverse_method = None
                return

            raise ValueError(
                "The inverse method can only be set if the coordinates "
                "are a provided as SamplingSphere.")

        elif type(self.coordinates) is spharpy.SamplingSphere:
            if value == "auto":

                value = (
                    "quadrature"
                    if self.coordinates.quadrature
                    else "pseudo_inverse"
                )

            elif value == "quadrature" and not self.coordinates.quadrature:
                raise ValueError(
                    "'quadrature' requires `coordinates` to be a '"
                    "SamplingSphere and coordinates.quadrature to be "
                    "True.")

        self._reset_compute_attributes()
        self._inverse_method = value

    @property
    def basis(self):
        """Get the spherical harmonic basis matrix."""
        if self._basis is None:
            self._compute_basis()
        return self._basis

    @property
    def basis_gradient_theta(self):
        """
        Get the gradient of the basis matrix with respect to the
        `colatitude`/`elevation` angle.

        See the :py:mod:`~pyfar.classes.coordinates` documentation for the
        definition of the angles.
        """
        if self._basis_gradient_theta is None:
            self._compute_basis_gradient()
        return self._basis_gradient_theta

    @property
    def basis_gradient_phi(self):
        """
        Get the gradient of the basis matrix with respect to the
        `azimuth`/`polar` angle.

        See the :py:mod:`~pyfar.classes.coordinates` documentation for the
        definition of the angles.
        """
        if self._basis_gradient_phi is None:
            self._compute_basis_gradient()
        return self._basis_gradient_phi

    def _compute_basis(self):
        """
        Compute the basis matrix for the SphericalHarmonics class.
        """
        if self.basis_type == "complex":
            function = spharpy.spherical.spherical_harmonic_basis
        elif self.basis_type == "real":
            function = spharpy.spherical.spherical_harmonic_basis_real
        else:
            raise ValueError(
                "Invalid basis type, should be either 'complex' or 'real'")
        self._basis = function(
            self.n_max, self.coordinates,
            self.normalization, self.channel_convention)

    def _compute_basis_gradient(self):
        """
        Compute the gradient of the basis matrix for the SphericalHarmonics
        class.
        """
        if any((self.normalization in ["maxN", "SN3D"],
                self.channel_convention == "fuma")):
            raise ValueError(
            f"Gradient computation not supported for normalization "
            f"'{self.normalization}' and "
            f"channel convention '{self.channel_convention}'.")

        if self.basis_type == "complex":
            function = spharpy.spherical.spherical_harmonic_basis_gradient
        elif self.basis_type == "real":
           function = spharpy.spherical.spherical_harmonic_basis_gradient_real
        else:
            raise ValueError(
                "Invalid basis type, should be either 'complex' or 'real'")
        self._basis_gradient_theta, self._basis_gradient_phi = function(
            self.n_max, self.coordinates)

    @property
    def basis_inv(self):
        """Get the inverse basis matrix."""
        if self._inverse_method is None:
            raise ValueError("The inverse method is not defined.")

        if self._basis is None:
            self._compute_basis()
        if self._basis_inv is None:
            self._compute_inverse()
        return self._basis_inv

    def _compute_inverse(self):
        """
        Compute the inverse basis matrix for the SphericalHarmonics class.
        """
        if self._basis is None:
            self._compute_basis()
        _inv_flag = self.inverse_method
        if _inv_flag == "pseudo_inverse":
            self._basis_inv = np.linalg.pinv(self._basis)
        elif _inv_flag == "quadrature":
            self._basis_inv = np.einsum(
                'ij,i->ji', np.conj(self._basis), self.coordinates.weights)

    def _reset_compute_attributes(self):
        """Reset the computed attributes for the SphericalHarmonics class in
        case of changes in the parameters.
        """
        self._basis = None
        self._basis_gradient_theta = None
        self._basis_gradient_phi = None
        self._basis_inv = None

    def _on_property_change(self):
        """Reset computed attributes on property changes."""
        self._reset_compute_attributes()
