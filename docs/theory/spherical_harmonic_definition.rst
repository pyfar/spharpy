Spherical Harmonic Definitions
==============================

Within spharpy, the spherical harmonics are referred to using the symbol
:math:`Y_n^m(\theta, \phi)`, where :math:`n` is the spherical harmonic order,
:math:`m` is the degree, and :math:`\theta` and :math:`\phi` are the
colatitude and azimuth angles, respectively.
This coordinate convention corresponds to the ``spherical_colatitude``
coordinate convention in ``pyfar``
(see also the `pyfar documentation page <https://pyfar.readthedocs.io/en/stable/classes/pyfar.coordinates.html>`_).

Spharpy supports multiple definitions of the spherical harmonics also
referred to as conventions.
These definitions differ in their choice of real or complex valued functions
and in the magnitude normalization and phase convention.

General definition
------------------

In a general form, the spherical harmonics can be expressed as

.. math::

    Y_n^m(\theta, \phi) = S^m N_n^m P_n^{m}(\cos\theta)  A^m(\phi)

where :math:`N_n^m` is the magnitude normalization, :math:`S^m` is the phase
convention, :math:`P_n^{m}(\cos\theta)` are the associated Legendre functions,
and :math:`A^m(\phi)` is the azimuthal function.

Magnitude normalization
^^^^^^^^^^^^^^^^^^^^^^^

All implementations form an orthogonal basis on the sphere.
Accordingly, the inner product of two spherical harmonics yields the weighted
product of Kronecker delta symbols :math:`\delta_{nn'} \delta_{mm'}`

.. math::

    \iint Y_n^m(\theta, \phi) Y_{n'}^{m'*}(\theta, \phi)
    \sin\theta d\theta d\phi = C_n \delta_{nn'} \delta_{mm'}

where :math:`C_n` depends on the chosen normalization convention.

Magnitude normalizations implemented in spharpy are:

.. table::
    :widths: auto

    ===========  =========================================  =========================  ===============================================================
    Identifier   Description                                :math:`C`                  :math:`N_n^m`
    ===========  =========================================  =========================  ===============================================================
    ``'N3D'``    full normalization on the sphere           1                          :math:`\sqrt{\frac{2n+1}{4\pi} \frac{(n-m)!}{(n+m)!}}`
    ``'SN3D'``   semi-normalized on the sphere              :math:`\frac{1}{2n+1}`     :math:`\sqrt{ \frac{1}{4 \pi} \frac{(n-m)!}{(n+m)!}}`
    ``'NM'``     monopole moment normalization              :math:`4\pi`               :math:`\sqrt{(2n+1) \frac{(n-m)!}{(n+m)!}}`
    ``'SNM'``    monopole moment semi-normalized            :math:`\frac{4\pi}{2n+1}`  :math:`\sqrt{\frac{(n-m)!}{(n+m)!}}`
    ``'maxN'``   maximum magnitude normalization            N/A                        N/A
    ===========  =========================================  =========================  ===============================================================

Note that for the ``maxN`` normalization, no general closed-form expression
for the constant :math:`C` exists. The constant is chosen such that the
maximum magnitude of the spherical harmonics on the sphere is equal to 1,
i.e., :math:`\max_{\theta, \phi} |Y_n^m(\theta, \phi)| = 1`.

The name monopole moment normalization refers to the fact that the magnitude of the
spherical harmonic of order :math:`n=0` is normalized to 1 for this convention.
In contrast, for fully normalized and semi normalized spherical harmonics, the
magnitude of the spherical harmonic of order :math:`n=0` is normalized to
:math:`\sqrt{1/4\pi}`.

Note that in some literature, the normalization and Legendre functions are
expressed using the absolute value of the degree :math:`|m|` instead of :math:`m`.
This has however no effect on the resulting spherical harmonics as long as the
absolute value is used consistently for the normalization and associated Legendre
function, since

.. math::
    N_n^{|m|}P_n^{|m|} = N_n^{m}P_n^{m}.

Phase convention
^^^^^^^^^^^^^^^^

Spharpy implements two different phase conventions, also referred to as the
Condon-Shortley phase term

.. math::
    S_n^m = (-1)^m.

By default, the Condon-Shortley phase term is included for complex spherical
harmonics and not included for real spherical harmonics, which corresponds to
the common convention in the fields of acoustics.
Note that the Condon-Shortley phase term is included in the definition of the
associated Legendre functions in the :py:mod:`scipy.special` module, which is used
for the computation of the spherical harmonics in spharpy.

Real and complex valued definition of the azimuthal function
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The azimuthal function :math:`A^m(\phi)` can be defined in a real or complex-valued
form, which leads to the real and complex spherical harmonics.
For complex valued spherical harmonics, the azimuthal function is defined as

.. math::
    A^m(\phi) = e^{im\phi}

For real valued spherical harmonics, the azimuthal function is defined as

.. math::
    A^m(\phi) = \begin{cases}
        \sqrt{2}\cos(m\phi), & m > 0 \\
        1, & m = 0 \\
        \sqrt{2}\sin(m\phi), & m < 0
    \end{cases}


Examples
--------

For reference, the following plots show the spherical harmonics up to third order.
The color map encodes the phase of the spherical harmonics and the radius encodes
the magnitude.
The source code of the plot function is hidden for better readability, but can be
downloaded using the link below.

.. plot::
    :include-source: false
    :context: close-figs


    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    from matplotlib.colorbar import Colorbar
    import spharpy
    from spharpy.plot import balloon_wireframe


    def plot_basis_functions(Y, sampling, n_max=2):
        fig = plt.figure(figsize=(8, 6))
        gs = plt.GridSpec(
            n_max+2, 2*n_max+1, height_ratios=np.r_[np.ones(n_max+1), 0.1])
        axs = []

        view_angle = (30, 30)

        for acn in range((n_max+1)**2):
            n, m = spharpy.spherical.acn_to_nm(acn)
            idx_m = (2*n_max + 1)//2 + m
            ax = plt.subplot(gs[n, idx_m], projection='3d')

            balloon = balloon_wireframe(
                sampling, Y[:, acn], cmap_encoding='phase', colorbar=False,
                ax=ax)
            ax.set_title('$Y_{' + str(n) + '}^{' + str(m) + '}(\\theta, \\phi)$')
            plt.axis('off')

            ax.view_init(*view_angle)
            axs.append(ax)


        ax = plt.subplot(gs[0, 0], projection='3d')
        ax.plot([0, 1], [0, 0], [0, 0], color='k')
        ax.plot([0, 0], [0, 1], [0, 0], color='k')
        ax.plot([0, 0], [0, 0], [0, 1], color='k')
        ax.text(0, 0, 1.1, 'z')
        ax.text(0, 1.1, 0, 'y')
        ax.text(1.3, -.05, 0, 'x')

        ax.set_box_aspect(np.ones(3))
        ax.view_init(*view_angle)
        plt.axis('off')

        cax = plt.subplot(gs[n_max+1, 1:-1])

        cnorm = plt.Normalize(0, 2*np.pi)
        cmappable = mpl.cm.ScalarMappable(cnorm, spharpy.plot.phase_twilight())
        cmappable.set_array(np.linspace(0, 2*np.pi, 128))

        cb = Colorbar(
            ax=cax, mappable=cmappable,
            orientation='horizontal', ticklocation='bottom')
        cb.set_label('Phase in rad')
        cb.set_ticks(np.linspace(0, 2*np.pi, 5))
        cb.set_ticklabels([r'$0$', r'$\pi/2$', r'$\pi$', r'$3\pi/2$', r'$2\pi$'])
        plt.tight_layout()

        return axs, gs


The default definition of the complex valued spherical harmonics corresponds to
the 'N3D' normalization, the inclusion of the Condon-Shortley phase term.


.. plot::
    :include-source: true
    :context: close-figs

    import spharpy

    n_max = 2
    sampling = spharpy.samplings.equal_area(0, n_points=500)

    Y_nm = spharpy.SphericalHarmonics(
        n_max=n_max,
        coordinates=sampling,
        basis_type='complex',
        normalization='N3D',
        condon_shortley='auto'
    ).basis

    axs, gs = plot_basis_functions(Y_nm, sampling)


The default definition of the real valued spherical harmonics corresponds
to the ``'N3D'`` normalization and the exclusion of the Condon-Shortley phase term.

.. plot::
    :include-source: true
    :context: close-figs

    Y_nm = spharpy.SphericalHarmonics(
        n_max=n_max, coordinates=sampling,
        basis_type='real',
        normalization='N3D',
        condon_shortley='auto'
    ).basis

    axs, gs = plot_basis_functions(Y_nm, sampling)


As an example, the following plot shows the real valued spherical harmonics
with the inclusion of the Condon-Shortley phase term. Note that the phase
is rotated by :math:`\pi` for all spherical harmonics with odd degree :math:`m`
compared to the plot above.


.. plot::
    :include-source: true
    :context: close-figs

    Y_nm = spharpy.SphericalHarmonics(
        n_max=n_max, coordinates=sampling,
        basis_type='real', condon_shortley=True).basis

    axs, gs = plot_basis_functions(Y_nm, sampling)


References
==========

.. [1] E. G. Williams, Fourier Acoustics, 1st ed. Academic Press, 1999.
.. [2] B. Rafaely, Fundamentals of Spherical Array Processing, 2st ed.,
       vol. 8. in Springer Topics in Signal Processing, vol. 8. Springer-Verlag GmbH Berlin Heidelberg, 2015.
.. [3] F. Zotter, “Analysis and synthesis of sound-radiation with spherical arrays,” Doctoral Disseration, KUG, Graz, 2009.
.. [4] F. Zotter and M. Frank, Ambisonics: A Practical 3D Audio Theory for Recording, Studio Production,
       Sound Reinforcement, and Virtual Reality, vol. 19. in Springer Topics in Signal Processing, vol. 19.
       Springer International Publishing, 2019.
