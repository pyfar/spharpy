Spherical Harmonic
----------------------

.. automodule:: spharpy.classes.audio

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

Ambix, SOFA, MPEG: 
                   :py:attr:`~spharpy.SphericalHarmonic.basis_type` = `real`, 
                   :py:attr:`~spharpy.SphericalHarmonic.channel_convention` = `ACN`,
                   :py:attr:`~spharpy.SphericalHarmonic.normalization` = `N3D`, 
                   :py:attr:`~spharpy.SphericalHarmonic.condon_shortley` = False
.. [#] F. Zotter, M. Frank, "Ambisonics A Practical 3D Audio Theory for Recording, Studio Production, Sound Reinforcement, and
       Virtual Reality", (2019), Springer-Verlag

Rafaely:
   :py:attr:`~spharpy.SphericalHarmonic.basis_type` = `complex`, 
   :py:attr:`~spharpy.SphericalHarmonic.channel_convention` = `ACN`,
   :py:attr:`~spharpy.SphericalHarmonic.normalization` = `N3D`, 
   :py:attr:`~spharpy.SphericalHarmonic.condon_shortley` = True

.. [#] B. Rafely, "Fundamentals of Spherical Array Processing", (2015), Springer-Verlag
.. [#] E.G. Williams, "Fourier Acoustics", (1999), Academic Press

Ahrens: 
   :py:attr:`~spharpy.SphericalHarmonic.basis_type` = `complex`, 
   :py:attr:`~spharpy.SphericalHarmonic.channel_convention` = `ACN`,
   :py:attr:`~spharpy.SphericalHarmonic.normalization` = `N3D`, 
   :py:attr:`~spharpy.SphericalHarmonic.condon_shortley` = True
.. [#] J. Ahrens, "Analytic Methods of Sound Field Synthesis", (2012), Springer-Verlag

**Classes:**

.. autosummary::

   SphericalHarmonic
   SphericalHarmonicSignal

.. autoclass:: spharpy.SphericalHarmonic
   :members:
   :undoc-members:
   :inherited-members:

.. autoclass:: spharpy.SphericalHarmonicSignal
   :members:
   :undoc-members:
   :inherited-members:


