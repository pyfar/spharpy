=======
History
=======

0.5.0 (2023-03-22)
------------------
* Introduce support for `pyfar.Coordinates` alongside spharpy's implementation (PR #24). This is the initial step of a full transition in future versions.
* Minor refactorings of tests (PR #25)
* Updates to the documentation (PR #26)
* Bugfix: Round values smaller than eps in `sph2cart` (PR #27)

0.4.2 (2023-01-13)
------------------
* Bugfix: Replace deprecated numpy wrappers for built-in python types (PR #20)
* Refactor axes handling in plots, required as getting the current axis while specifying a specific projection is deprecated (PR #21)
* Drop support for Python 3.7 (PR #23)

0.4.1 (2022-04-21)
------------------
* Bugfix for the aperture function of vibrating spherical caps

0.4.0 (2022-02-21)
------------------
* SH Rotation class derived from scipy rotations
* Implement Wigner-D Rotations
* SmoothBivariateSpline Interpolation on the sphere
* New phase colormap for balloon plots
* Switch to CircleCI for unit-testing
* Updated readthedocs.org
* Minor documentation fixes and flake8 consistency changes

0.3.3 (2021-01-13)
------------------
* Replace broken link in hyperinterpolation sampling download.

0.3.2 (2021-01-29)
------------------
* Bugfix in modal strength calculation for rigid spheres


0.3.1 (2020-12-09)
------------------
* Update equal area partitioning documentation


0.3.0 (2020-09-04)
------------------
* Removed C extensions for spherical harmonics and special functions
* Spherical harmonic basis gradient
* Interior point sampling for eigenvalue stabilization
* Weight calculation for beamforming algorithms
* Added more map projection plots and map data interpolation
* Several bugfixes for plot functions


0.2.0 (2019-07-31)
------------------
* Last release version with C extension
* Refactored C extensions with OpenMP support
* Sampling class
* Spherical Voronoi diagrams


0.1.2 (2018-09-19)
------------------

* Added Geocentric to Cartesian coordinate transform


0.1.1 (2018-06-20)
------------------

* Added Coordinate class object
* New plot functions
* Bugfixes


0.1.0 (2018-02-18)
------------------

* First release
