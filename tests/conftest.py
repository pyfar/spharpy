import pytest
import pyfar as pf
import numpy as np


@pytest.fixture
def make_coordinates():

    class Factory:
        @staticmethod
        def create_coordinates(rad=1, theta=np.pi/2, phi=np.pi/2):
            return pf.Coordinates.from_spherical_colatitude(
                phi, theta, rad
            )

    yield Factory


@pytest.fixture
def icosahedron():
    """Return the coordinate points of an icosahedron in spherical coordinates

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
