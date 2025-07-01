r"""
The following documents the definition of the spherical harmonics used in 
SpherocalHarmonic and SphericalHarmonicSignal:

The spherical harmonic Ynm is defined as:

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
   .. plot::
        :include-source:
        :format: python
        :align: center
        :scale: 75

        import matplotlib.pyplot as plt
        import numpy as np
        import matplotlib as mpl
        from matplotlib.colorbar import Colorbar
        import spharpy
        from mpl_toolkits.mplot3d import Axes3D
        from mpl_toolkits.mplot3d import axes3d

        n_max = 3
        sampling = spharpy.samplings.equal_area(25, condition_num=np.inf)
        Y_real = spharpy.spherical.spherical_harmonic_basis_real(n_max, sampling)

        #fig = plt.figure(figsize=(12, 8)) 
        from matplotlib import cm

        fig = plt.figure()
        ax = fig.gca(projection='3d')
        X, Y, Z = axes3d.get_test_data(0.005)
        ax.plot_surface(X, Y, Z, rstride=8, cstride=8, alpha=0.3)
        cset = ax.contourf(X, Y, Z, zdir='z', offset=-100, cmap=cm.coolwarm)
        cset = ax.contourf(X, Y, Z, zdir='x', offset=-40, cmap=cm.coolwarm)
        cset = ax.contourf(X, Y, Z, zdir='y', offset=40, cmap=cm.coolwarm)

        ax.set_xlabel('X'); ax.set_xlim(-40, 40)
        ax.set_ylabel('Y'); ax.set_ylim(-40, 40)
        ax.set_zlabel('Z'); ax.set_zlim(-100, 100)

        plt.show()

        gs = plt.GridSpec(4, 5, height_ratios=[1, 1, 1, 0.1], width_ratios=[1, 1, 1, 1, 1])
        #for acn in range((n_max+1)**2):
        #    n, m = spharpy.spherical.acn_to_nm(acn)
        #    idx_m = int(np.floor(n_max/2+1)) + m
        #    #ax = plt.subplot(gs[n, idx_m], projection='3d')
        #    ax = fig.add_subplot(gs[n, idx_m])
        #    balloon = spharpy.plot.balloon_wireframe(sampling, Y_real[:, acn], phase=True, colorbar=False, ax=ax)
        #     ax.set_title('$Y_{' + str(n) + '}^{' + str(m) + '}(\\theta, \\phi)$')
        #     plt.axis('off')

        # cax = plt.subplot(gs[n_max+1, :])

        # cnorm = plt.Normalize(0, 2*np.pi)
        # cmappable = mpl.cm.ScalarMappable(cnorm, spharpy.plot.phase_twilight())
        # cmappable.set_array(np.linspace(0, 2*np.pi, 128))

        # cb = Colorbar(ax=cax, mappable=cmappable, orientation='horizontal', ticklocation='bottom')
        # cb.set_label('Phase in rad')
        # cb.set_ticks(np.linspace(0, 2*np.pi, 5))
        # cb.set_ticklabels(['$0$', '$\pi/2$', '$\pi$', '$3\pi/2$', '$2\pi$'])

The spherical harmonics are orthogonal on the unit sphere, i.e.,

.. math::
   \int_{sphere} Y_{nm} Y_{n'm'}* d\omega = \delta_{nn'} \delta_{mm'}

where:

- :math:`*` denotes complex conjugation
- :math:`\delta_{nn'}` is the Kronecker delta function
- :math:`d\omega` is the solid angle element
- The integral is over the entire sphere

The SphericalHarmonic and SphericalHarmonicSignal classes support conversion between real and
complex-valued basis functions, as well as between channel conventions, normalizations and 
phase conventions. This flexibility enables compatibility with multiple spherical harmonic
implementations and standards.
Popular spherical harmonic configurations include:

The Ambix convention used for example by SOFA or MPEG, see [#]_: 
                   :py:attr:`~spharpy.SphericalHarmonics.basis_type` = `real`, 
                   :py:attr:`~spharpy.SphericalHarmonics.channel_convention` = `ACN`,
                   :py:attr:`~spharpy.SphericalHarmonics.normalization` = `N3D`, 
                   :py:attr:`~spharpy.SphericalHarmonics.condon_shortley` = False

A convention used by [#]_ or [#]_:
   :py:attr:`~spharpy.SphericalHarmonics.basis_type` = `complex`, 
   :py:attr:`~spharpy.SphericalHarmonics.channel_convention` = `ACN`,
   :py:attr:`~spharpy.SphericalHarmonics.normalization` = `N3D`, 
   :py:attr:`~spharpy.SphericalHarmonics.condon_shortley` = True

Convention defined in [#]_: 
   :py:attr:`~spharpy.SphericalHarmonics.basis_type` = `complex`, 
   :py:attr:`~spharpy.SphericalHarmonics.channel_convention` = `ACN`,
   :py:attr:`~spharpy.SphericalHarmonics.normalization` = `N3D`, 
   :py:attr:`~spharpy.SphericalHarmonics.condon_shortley` = True

.. [#] F. Zotter, M. Frank, "Ambisonics A Practical 3D Audio Theory for Recording, Studio Production, Sound Reinforcement, and
       Virtual Reality", (2019), Springer-Verlag
.. [#] B. Rafely, "Fundamentals of Spherical Array Processing", (2015), Springer-Verlag
.. [#] E.G. Williams, "Fourier Acoustics", (1999), Academic Press
.. [#]  J. Ahrens, "Analytic Methods of Sound Field Synthesis", (2012), Springer-Verlag
"""

import numpy as np
import pyfar as pf
import spharpy as sy
from pyfar import Signal
from spharpy.spherical import renormalize, change_channel_convention


class SphericalHarmonics:
    r"""
    Compute spherical harmonic basis matrices, their inverses, and gradients.


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


class SphericalHarmonicSignal(Signal):
    """Create audio object with spherical harmonics coefficients in time or
    frequency domain.

    Objects of this class contain spherical harmonics coefficients which are
    directly convertible between time and frequency domain (equally spaced
    samples and frequency bins), the channel conventions ACN and FUMA, as
    well as the normalizations N3D, SN3D, or MaxN, see [#]_. The definition of
    the spherical harmonics basis functions is based on the scipy convention
    which includes the Condon-Shortley phase, [#]_, [#]_.


    Parameters
    ----------
    data : ndarray, double
        Raw data of the spherical harmonics signal in the time or
        frequency domain. The data should have at least 3 dimensions,
        according to the 'C' memory layout, e.g. data of
        ``shape = (1, 4, 1024)`` has 1 channel with 4 spherical harmonic
        coefficients with 1024 samples or frequency
        bins each. Time data is converted to ``float``. Frequency is
        converted to ``complex`` and must be provided as single
        sided spectra, i.e., for all frequencies between 0 Hz and
        half the sampling rate.
    sampling_rate : double
        Sampling rate in Hz
    basis_type : str
        Type of spherical harmonic basis, either ``'complex'`` or
        ``'real'``.
    normalization : str
        Normalization convention, either ``'n3d'``, ``'maxN'`` or
        ``'sn3d'``. (maxN is only supported up to 3rd order)
    channel_convention : str
        Channel ordering convention, either ``'acn'`` or ``'fuma'``.
        (FuMa is only supported up to 3rd order)
    condon_shortley : bool
        Flag to indicate if the Condon-Shortley phase term is included
        (``True``) or not (``False``).
    n_samples : int, optional
        Number of time domain samples. Required if domain is ``'freq'``.
        The default is ``None``, which assumes an even number of samples
        if the data is provided in the frequency domain.
    domain : ``'time'``, ``'freq'``, optional
        Domain of data. The default is ``'time'``
    fft_norm : str, optional
        The normalization of the Discrete Fourier Transform (DFT). Can be
        ``'none'``, ``'unitary'``, ``'amplitude'``, ``'rms'``, ``'power'``,
        or ``'psd'``. See :py:func:`~pyfar.dsp.fft.normalization` and [#]_
        for more information. The default is ``'none'``, which is typically
        used for energy signals, such as impulse responses.
    comment : str
        A comment related to `data`. The default is ``None``.
    is_complex : bool, optional
        Specifies if the underlying time domain data are complex
        or real-valued. If ``True`` and `domain` is ``'time'``, the
        input data will be cast to complex. The default is ``False``.

    References
    ----------
    .. [#] F. Zotter, M. Frank, "Ambisonics A Practical 3D Audio Theory
            for Recording, Studio Production, Sound Reinforcement, and
            Virtual Reality", (2019), Springer-Verlag
    .. [#] B. Rafely, "Fundamentals of Spherical Array Processing", (2015),
            Springer-Verlag
    .. [#] E.G. Williams, "Fourier Acoustics", (1999), Academic Press

    """
    def __init__(
            self,
            data,
            sampling_rate,
            basis_type,
            normalization,
            channel_convention,
            condon_shortley,
            n_samples=None,
            domain='time',
            fft_norm='none',
            comment="",
            is_complex=False):
        """
        Create SphericalHarmonicSignal with data, and sampling rate.
        """
        # check dimensions
        if len(data.shape) < 3:
            raise ValueError("Invalid number of dimensions. Data should have "
                             "at least 3 dimensions.")

        # set n_max
        n_max = np.sqrt(data.shape[-2])-1
        if n_max - int(n_max) != 0:
            raise ValueError("Invalid number of SH channels: "
                             f"{data.shape[-2]}. It must match (n_max + 1)^2.")
        self._n_max = int(n_max)

        # set basis_type
        if basis_type not in ["complex", "real"]:
            raise ValueError("Invalid basis type, only "
                             "'complex' and 'real' are supported")
        self._basis_type = basis_type

        # set normalization
        if normalization not in ["n3d", "maxN", "sn3d"]:
            raise ValueError("Invalid normalization, has to be 'sn3d', "
                             f"'n3d', or 'maxN', but is {normalization}")
        self._normalization = normalization

        # set channel_convention
        if channel_convention not in ["acn", "fuma"]:
            raise ValueError("Invalid channel convention, has to be 'acn' "
                             f"or 'fuma', but is {channel_convention}")
        self._channel_convention = channel_convention

        # set Condon Shortley
        if not isinstance(condon_shortley, bool):
            raise ValueError("Condon_shortley has to be a bool.")
        self._condon_shortley = condon_shortley

        Signal.__init__(self, data, sampling_rate=sampling_rate,
                        n_samples=n_samples, domain=domain, fft_norm=fft_norm,
                        comment=comment, is_complex=is_complex)

    @property
    def n_max(self):
        """Get the maximum spherical harmonic order."""
        return self._n_max

    @property
    def basis_type(self):
        """Get the type of the spherical harmonic basis."""
        return self._basis_type

    @property
    def normalization(self):
        """
        Get or set and apply the normalization of the spherical harmonic
        coefficients.
        """
        return self._normalization

    @normalization.setter
    def normalization(self, value):
        """
        Get or set and apply the normalization of the spherical harmonic
        coefficients.
        """
        if self.normalization is not value:
            self._data = renormalize(self._data, self.channel_convention,
                                     self.normalization, value, axis=-2)
            self._normalization = value

    @property
    def condon_shortley(self):
        """Get info whether to include the Condon-Shortley phase term."""
        return self._condon_shortley

    @property
    def channel_convention(self):
        """
        Get or set and apply the channel convention of the spherical harmonic
        coefficients.
        """
        return self._channel_convention

    @channel_convention.setter
    def channel_convention(self, value):
        """
        Get or set and apply the channel convention of the spherical harmonic
        coefficients.
        """
        if self.channel_convention is not value:
            self._data = change_channel_convention(self._data,
                                                   self.channel_convention,
                                                   value, axis=-2)
            self._channel_convention = value