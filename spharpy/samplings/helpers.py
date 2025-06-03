"""
Helper functions for coordinate operations.
"""

import numpy as np
from scipy.spatial import cKDTree, SphericalVoronoi


def coordinates2latlon(coordinates):
    r"""Transforms from Cartesian coordinates to Geocentric coordinates.

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
    coordinates : ndarray, number
        Coordinates Object which cartesian coordinates are taken to convert to
        Geocentric coordinates

    Returns
    -------
    height : ndarray, number
        The radius is rendered as height information
    latitude : ndarray, number
        Geocentric latitude angle
    longitude : ndarray, number
        Geocentric longitude angle

    """
    x = coordinates.x
    y = coordinates.y
    z = coordinates.z
    height = np.sqrt(x**2 + y**2 + z**2)
    latitude = np.pi/2 - np.arccos(z/height)
    longitude = np.arctan2(y, x)
    return height, latitude, longitude


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
    points = sampling.cartesian
    points = points if points.shape[-1] == 3 else points.T
    radius = np.unique(np.round(sampling.radius, decimals=round_decimals))
    if len(radius) > 1:
        raise ValueError("All sampling points need to be on the \
                same radius.")
    voronoi = SphericalVoronoi(points, radius[0], center)

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
    weigths : ndarray, float
        Sampling weights

    """
    sv = spherical_voronoi(sampling, round_decimals=round_decimals)
    sv.sort_vertices_of_regions()

    unique_verts, idx_uni = np.unique(
        np.round(sv.vertices, decimals=10),
        axis=0,
        return_index=True)

    searchtree = cKDTree(unique_verts)
    if hasattr(sampling, 'csize'):
        area = np.zeros(sampling.csize, float)
    else:
        area = np.zeros(sampling.n_points, float)

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
