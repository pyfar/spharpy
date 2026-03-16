=======
History
=======

1.0.0 (2026-03-16)
------------------

Note: `spharpy` version 1.0.0 introduces several breaking changes which are not backwards compatible with prior versions.
These changes harmonize the package with the pyfar ecosystem. Basic functionality such as the `Coordinate` class are fully moved to the `pyfar` package which serves as a common base for all packages of the ecosystem.


These are:

- `RotationSH` has been re-designed (PR #226). Changes include
  - `RotationSH` has been renamed to `SphericalHarmonicRotation`
  - The class no longer inherits from `scipy.spatial.transform.Rotation`, instead it inherits from `pyfar.Rotation`, which is based on `scipy.spatial.transform.Rotation` using composition. This was necessary due to the re-write of `scipy.spatial.transform.Rotation` in scipy v1.17.0.
  - The class no longer stores spherical harmonic orders, instead a `SphericalHarmonicDefinition` object is required when calling the respective class method to calculate the spherical harmonic rotation matrix.
- Removed the class `spharpy.samplings.Coordinates`. Instead, refer to `pyfar.Coordinates` for storing coordinates. All functions which previously used `spharpy.samplings.Coordinates` now use `pyfar.Coordinates`.
- `spharpy.SamplingSphere` has been rewritten and now implements checks if
  - all points are placed on the same radius, given a specified tolerance,
  - the area weights sum correctly to the integral of the unit sphere.
- The `indexing` module has been removed, and the functionality has been moved to the `spherical` module.
  - In `spharpy.spherical.sph_identity_matrix` the argument `type` has been renamed to `matrix_type` to avoid conflict with the built-in `type` function.
- In the `spherical` module, the functions
  - `acn2nm` and `nm2acn` have been renamed to `acn_to_nm` and `nm_to_acn`, respectively.
- The `spharpy.plot` module has been re-written and now has a more consistent API with the `pyfar.plot` module
  - Added support for pyfar's plot styles.
  - Add support for always passing and returning `matplotlib.axes.Axes` objects to all plotting functions, which allows for more flexible plotting and composition of plots.
  - Always return matplotlib `Colorbar` and mappables for each plot, which is consistent with the `pyfar.plot` module.
  - Omit the implicit interpolation of data in the map projection plots. Users should now explicitly interpolate the data to a regular grid using `spharpy.interpolate.griddata` or similar functions before plotting if necessary.
- The `spharpy.samplings` module has been re-written
  - `spharpy.samplings.cube_equidistand` has been renamed to `spharpy.samplings.equidistant_cuboid`
  - `spherical_t_design` has been renamed to `t_design`
  - `equalarea` has been renamed to `equal_area`
  - All functions now return a refactored `spharpy.SamplingSphere` object.
  - In the `spharpy.samplings.utils` module, the helper functions `cart2sph`, `latlon2cart`, and `sph2cart` have been removed, since they were moved to `pyfar`.
  - The function `spharpy.samplings.utils.cart2latlon` has been refactored to use a `pyfar.Coordiantes` (or child class) object as input instead of an array containing cartesian coordinates.

Changed:
^^^^^^^^

- Removed support for Python 3.10 and earlier.
- Require pyfar >= 0.8.0.
- Require scipy >= 1.17.0.
- The `special` and `spatial` sub-modules are now directly imported as `spharpy.special` and `spharpy.spatial`.


Added:
^^^^^^

- `spharpy.SphericalHarmonicDefintion`: A class encapsulating parameters for the definition of spherical harmonics:  (#)
- `spharpy.SphericalHarmonics`: A class to lazily compute the spherical harmonics for a given definition. For `spharpy.SamplingSphere` the class also calculates inverse basis matrices.
- `spharpy.SphericalHarmonicSignal`, `spharpy.SphericalHarmonicFrequencyData`, and `spharpy.SphericalHarmonicTimeData`: classes to store audio data in the spherical harmonic domain. These classes are derived from `pyfar.Signal`, `pyfar.FrequencyData`, and `pyfar.TimeData` respectively, and thus support all methods implemented for these classes. (PR #166)
- Added support for the following parametrization of the functions for calculation of spherical harmonics defined in the `spharpy.spherical` module:
  - normalizations (N3D, SN3D, NM, SNM, and  maxN),
  - the Condon-Shortley phase convention,
  - the FuMa channel ordering convention.
- Added `spharpy.spherical.renormalize` function to convert between different spherical harmonic normalization schemes.
- Added `spharpy.spherical.change_channel_convention` to convert between different channel ordering conventions.
- Added functions `nm_to_fuma` and `fuma_to_nm` to convert to and from FuMa channel ordering
- Shared plot preparation in the `spharpy.plot` module, which is used by all plot functions.
- Tests for all plots and baseline images for manual comparison of the plots in the test suite.
- Added the following spherical sampling grids:
  - `spharpy.samplings.lebedev`
  - `spharpy.samplings.equal_angle`
  - `spharpy.samplings.great_circle`
  - `spharpy.samplings.fliege`

Documentation
^^^^^^^^^^^^^
- Extended documentation and added examples for all functions in the following modules:
  - `beamforming`,
  - `spatial`,
  - `interpolate`,
  - `special`,
  - `spherical`,

0.6.3 (2026-03-16)
------------------
- Remove support for Python 3.8. (PR #111)
- Require matplotlib >= v3.10.3 for properly working contour map plots. (PR #117)
- Require scipy < v1.17.0 due to the deprecations of `scipy.special.sph_harm`  and related functions. (PR #180)
- Minor updates to documentation and package info. (PRs #112, #130, #144)

0.6.2 (2025-03-16)
------------------
- Added sampling for the eigenmike em64 (PR #84)
- Updated requirements to pyfar <v0.8.0
- Minor numpy deprecation updates (PR #88,107)
- Updated to toml based definition, as well as updates to docs (PR #105)

0.6.1 (2024-03-20)
------------------
- Update of the documentation theme and adaption of the new example gallery, no changes to the code base.

0.6.0 (2023-10-17)
------------------
This is the last minor release of spharpy before full integration into the pyfar ecosystem.
The next release will be version 1.0.0 which will depend on pyfar >= 0.6.0 and rely on pyfar's `pyfar.Coordinates` class as a replacement for spharpy's `spharpy.samplings.Coordinates` class.

**Changes:**

* Add normalization for distortionless response beamforming weights (PR #30)
* Minor refactoring in beamforming weight calculations (PR #30)
* Change dependency to pyfar versions < 0.6.0 (PR #35)
* Adaptations to the repository move to pyfar/spharpy (PR #32)

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
