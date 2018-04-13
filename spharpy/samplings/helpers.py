"""
Helper functions for coordinate operations
"""

import numpy as np


def sph2cart(r, theta, phi):
    """Transforms from spherical to Cartesian coordinates.
    Spherical coordinates follow the common convention in Physics/Mathematics
    Theta denotes the elevation angle with theta = 0 at the north pole and theta = pi
    at the south pole
    Phi is the azimuth angle counting from phi = 0 at the x-axis in positive direction
    (counter clockwise rotation).

    .. math::

        x = r \\sin(\\theta) \\cos(\\phi),

        y = r \\sin(\\theta) \\sin(\\phi),

        z = r \\cos(\\theta)

    Parameters
    ----------
    r : ndarray, number
    theta : ndarray, number
    phi : ndarray, number

    Returns
    -------
    x : ndarray, number
    y : ndarray, number
    z : ndarray, number

    """
    x = r*np.sin(theta)*np.cos(phi)
    y = r*np.sin(theta)*np.sin(phi)
    z = r*np.cos(theta)
    return x, y, z

def cart2sph(x, y, z):
    """
    Transforms from Cartesian to spherical coordinates.
    Spherical coordinates follow the common convention in Physics/Mathematics
    Theta denotes the elevation angle with theta = 0 at the north pole and theta = pi
    at the south pole
    Phi is the azimuth angle counting from phi = 0 at the x-axis in positive direction
    (counter clockwise rotation).

    .. math::

        r = \\sqrt{x^2 + y^2 + z^2},

        \\theta = \\arccos(\\frac{z}{r}),

        \\phi = \\arctan(\\frac{y}{x})

        0 < \\theta < \\pi,

        0 < \\phi < 2 \\pi


    Notes
    -----
    To ensure proper handling of the radiant for the azimuth angle, the arctan2
    implementatition from numpy is used here.

    Parameters
    ----------
    x : ndarray, number
    y : ndarray, number
    z : ndarray, number

    Returns
    -------
    r : ndarray, number
    theta : ndarray, number
    phi : ndarray, number

    """
    rad = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(z/rad)
    phi = np.mod(np.arctan2(y, x), 2*np.pi)
    return rad, theta, phi

def cart2latlon(x, y, z):
    """Transforms from Cartesian coordinates to latitude and longitude coordinates

    .. math::

        r = \\sqrt{x^2 + y^2 + z^2},

        \\theta = \\pi/2 - \\arccos(\\frac{z}{r}),

        \\phi = \\arctan(\\frac{y}{x})

        -\\pi/2 < \\theta < \\pi/2,

        -\\pi < \\phi < \\pi


    Parameters
    ----------
    x : ndarray, number
        x-axis coordinates
    y : ndarray, number
        y-axis coordinates
    z : ndarray, number
        z-axis coordinates

    Returns
    -------
    height : ndarray, number
        The radius is rendered as height information
    latitude : ndarray, number
        Geocentric latitude angle
    longitude : ndarray, number
        Geocentric longitude angle

    """
    height = np.sqrt(x**2 + y**2 + z**2)
    latitude = np.pi/2 - np.arccos(z/height)
    longitude = np.arctan2(y, x)
    return height, latitude, longitude
