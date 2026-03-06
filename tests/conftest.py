import pytest
import pyfar as pf
import numpy as np
from spharpy import SamplingSphere
from spharpy.samplings import equal_area


@pytest.fixture
def make_coordinates():
    """Fixture factory which can be used to return coordinate storing objects.

    Works with every child class of pyfar.Coordinates which supports the
    from_spherical_colatitude method. The default implementation is
    pyfar.Coordinates.
    """
    class Factory:
        @staticmethod
        def create_coordinates(
                implementation=pf.Coordinates,
                rad=1, theta=np.pi/2, phi=np.pi/2,
            ):
            return implementation.from_spherical_colatitude(phi, theta, rad)
    return Factory


def icosahedron_points():
    """Return the coordinate points of an icosahedron in spherical coordinates.

    Returns
    -------
    rad : float, ndarray
        The radius in meters.
    theta : float, ndarray
        The colatitude angle in radians.
    phi : float, ndarray
        The azimuth angle in radians.
    """
    gamma_R_r = np.arccos(np.cos(np.pi/3) / np.sin(np.pi/5))
    gamma_R_rho = np.arccos(1/(np.tan(np.pi/5) * np.tan(np.pi/3)))

    theta = np.tile(np.array([
        np.pi - gamma_R_rho,
        np.pi - gamma_R_rho - 2*gamma_R_r,
        2*gamma_R_r + gamma_R_rho,
        gamma_R_rho]), 5)
    theta = np.sort(theta)
    phi = np.arange(0, 2*np.pi, 2*np.pi/5)
    phi = np.concatenate((np.tile(phi, 2), np.tile(phi + np.pi/5, 2)))
    rad = np.ones(20)

    return rad, theta, phi


@pytest.fixture
def icosahedron():
    """Return the coordinate points of an icosahedron in spherical coordinates.

    Returns
    -------
    rad : float, ndarray
        The radius in meters.
    theta : float, ndarray
        The colatitude angle in radians.
    phi : float, ndarray
        The azimuth angle in radians.
    """
    rad, theta, phi = icosahedron_points()
    return rad, theta, phi


@pytest.fixture
def icosahedron_sampling():
    """Return the coordinate points of an icosahedron in spherical coordinates.

    Returns
    -------
    rad : float, ndarray
        The radius in meters.
    theta : float, ndarray
        The colatitude angle in radians.
    phi : float, ndarray
        The azimuth angle in radians.
    """
    rad, theta, phi = icosahedron_points()
    weights = np.ones_like(rad) * 4 * np.pi / len(rad)
    return SamplingSphere.from_spherical_colatitude(
        phi, theta, rad, weights=weights, n_max=2)


@pytest.fixture
def download_sampling():
    def download_sampling(kind, degree):
        if kind in ['extremal', 'hyperinterpolation']:
            from spharpy.samplings.samplings import _sph_extremal_load_data
            return _sph_extremal_load_data(degree)
        elif kind == 't-design':
            from spharpy.samplings.samplings import _sph_t_design_load_data
            return _sph_t_design_load_data(degree)

    return download_sampling


@pytest.fixture
def equal_area_sampling():
    """Return 500 point equal area test data."""
    coords = equal_area(n_max=0, n_points=500)
    data = np.sin(coords.azimuth) * np.cos(coords.elevation)
    return coords, data
