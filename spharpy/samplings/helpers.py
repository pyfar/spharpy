"""
Helper functions for coordinate operations
"""

import numpy as np
from scipy.spatial import cKDTree, SphericalVoronoi


def sph2cart(r, theta, phi):
    """Transforms from spherical to Cartesian coordinates.
    Spherical coordinates follow the common convention in Physics/Mathematics
    Theta denotes the elevation angle with theta = 0 at the north pole and
    theta = pi at the south pole. Phi is the azimuth angle counting from
    phi = 0 at the x-axis in positive direction
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
    Theta denotes the elevation angle with theta = 0 at the north pole and
    theta = pi at the south pole. Phi is the azimuth angle counting from
    phi = 0 at the x-axis in positive direction
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
    """Transforms from Cartesian coordinates to Geocentric coordinates

    .. math::

        h = \\sqrt{x^2 + y^2 + z^2},

        \\theta = \\pi/2 - \\arccos(\\frac{z}{r}),

        \\phi = \\arctan(\\frac{y}{x})

        -\\pi/2 < \\theta < \\pi/2,

        -\\pi < \\phi < \\pi

    where :math:`h` is the heigth, :math:`\\theta` is the latitude angle
    and :math:`\\phi` is the longitude angle

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


def latlon2cart(height, latitude, longitude):
    """Transforms from Geocentric coordinates to Cartesian coordinates

    .. math::

        x = h \\cos(\\theta) \\cos(\\phi),

        y = h \\cos(\\theta) \\sin(\\phi),

        z = h \\sin(\\theta)

        -\\pi/2 < \\theta < \\pi/2,

        -\\pi < \\phi < \\pi

    where :math:`h` is the heigth, :math:`\\theta` is the latitude angle
    and :math:`\\phi` is the longitude angle

    Parameters
    ----------
    height : ndarray, number
        The radius is rendered as height information
    latitude : ndarray, number
        Geocentric latitude angle
    longitude : ndarray, number
        Geocentric longitude angle

    Returns
    -------
    x : ndarray, number
        x-axis coordinates
    y : ndarray, number
        y-axis coordinates
    z : ndarray, number
        z-axis coordinates

    """

    x = height * np.cos(latitude) * np.cos(longitude)
    y = height * np.cos(latitude) * np.sin(longitude)
    z = height * np.sin(latitude)

    return x, y, z


def spherical_voronoi(sampling, round_decimals=13, center=0.0):
    """Calculate a Voronoi diagram on the sphere for the given samplings
    points.

    Parameters
    ----------
    sampling : SamplingSphere
        Sampling points on a sphere
    round_decimals : int
        Number of decimals to be rounded to.
    center : double
        Center point of the voronoi diagram.

    Returns
    -------
    voronoi : SphericalVoronoi
        Spherical voronoi diagram as implemented in scipy.

    """
    points = sampling.cartesian.T
    radius = np.unique(np.round(sampling.radius, decimals=round_decimals))
    if len(radius) > 1:
        raise ValueError("All sampling points need to be on the \
                same radius.")
    voronoi = SphericalVoronoi(points, radius, center)

    return voronoi


def calculate_sampling_weights(sampling, round_decimals=12):
    """Calculate the sampling weights for numeric integration.

    Parameters
    ----------
    sampling : SamplingSphere
        Sampling points on a sphere
    round_decimals : int, optional

    apply : boolean, optional
        Whether or not to store the weights into the class object

    Returns
    -------
    weigths : ndarray, np.double
        Sampling weights

    """
    sv = spherical_voronoi(sampling, round_decimals=round_decimals)
    sv.sort_vertices_of_regions()

    unique_verts, idx_uni = np.unique(
        np.round(sv.vertices, decimals=10),
        axis=0,
        return_index=True)

    searchtree = cKDTree(unique_verts)
    area = np.zeros(sampling.n_points, np.double)

    for idx, region in enumerate(sv.regions):
        _, idx_nearest = searchtree.query(sv.vertices[np.array(region)])
        mask_unique = np.sort(np.unique(idx_nearest, return_index=True)[1])
        mask_new = idx_uni[idx_nearest[mask_unique]]

        area[idx] = _poly_area(sv.vertices[mask_new])

    area = area / np.sum(area) * 4 * np.pi

    return area


def _unit_normal(a, b, c):
    x = np.linalg.det(
        [[1, a[1], a[2]],
         [1, b[1], b[2]],
         [1, c[1], c[2]]])
    y = np.linalg.det(
        [[a[0], 1, a[2]],
         [b[0], 1, b[2]],
         [c[0], 1, c[2]]])
    z = np.linalg.det(
        [[a[0], a[1], 1],
         [b[0], b[1], 1],
         [c[0], c[1], 1]])

    magnitude = np.sqrt(x**2 + y**2 + z**2)

    return (x/magnitude, y/magnitude, z/magnitude)


def _poly_area(poly):
    # area of polygon poly
    if len(poly) < 3:
        # not a plane - no area
        return 0
    total = [0.0, 0.0, 0.0]
    N = len(poly)
    for i in range(N):
        vi1 = poly[i]
        vi2 = poly[np.mod((i+1), N)]
        prod = np.cross(vi1, vi2)
        total[0] += prod[0]
        total[1] += prod[1]
        total[2] += prod[2]
    result = np.dot(total, _unit_normal(poly[0], poly[1], poly[2]))
    return np.abs(result/2)
