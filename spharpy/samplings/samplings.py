"""
Collection of sampling schemes for the sphere
"""

import urllib3
import numpy as np
from spharpy.samplings.coordinates import Coordinates


def hyperinterpolation(n_max):
    """Gives the points of a Hyperinterpolation sampling grid
    after Sloan and Womersley [1]_.

    Notes
    -----
    This implementation uses precalculated sets of points which are downloaded
    from Womersley's homepage [2]_.

    References
    ----------
    .. [1]  I. H. Sloan and R. S. Womersley, “Extremal Systems of Points and
            Numerical Integration on the Sphere,” Advances in Computational
            Mathematics, vol. 21, no. 1/2, pp. 107–125, 2004.
    .. [2]  http://web.maths.unsw.edu.au/~rsw/Sphere/Extremal/New/index.html

    Parameters
    ----------
    n_max : integer
        Spherical harmonic order of the sampling

    Returns
    -------
    coordinates : Coordinates
        Coordinate object containing all sampling points
    weights : ndarray
        Weights for each sampling point
    """
    n_sh = (n_max+1)**2
    filename = "/Womersley/md%02d.%04d" % (n_max, n_sh)
    url = "http://www.ita-toolbox.org/Griddata"
    fileurl = url + filename

    http = urllib3.PoolManager()
    http_data = http.urlopen('GET', fileurl)

    if http_data.status == 200:
        file_data = http_data.data.decode()
    else:
        raise ConnectionError("Connection error. Please check your internet \
                connection.")

    file_data = np.fromstring(file_data,
                              dtype='double',
                              sep=' ').reshape((n_sh, 4))
    coordinates = Coordinates(file_data[:, 0],
                              file_data[:, 1],
                              file_data[:, 2])
    weights = file_data[:, 3]

    return coordinates, weights


def spherical_t_design(order, n_points=None, symmetric=False):
    """Return the sampling positions for a spherical t-design [1]_ .

    For a given order t

    .. math::

        L = \\lceil \\frac{(t+1)^2}{2} \\rceil+1,

    points will be generated, except for t = 3, 5, 7, 9, 11.

    Notes
    -----
    This function downloads a pre-calculated set of points from
    Rob Womersley's homepage [2]_

    References
    ----------

    .. [1]  C. An, X. Chen, I. H. Sloan, and R. S. Womersley, “Well Conditioned
            Spherical Designs for Integration and Interpolation on the
            Two-Sphere,” SIAM Journal on Numerical Analysis, vol. 48, no. 6,
            pp. 2135–2157, Jan. 2010.
    .. [2]  http://web.maths.unsw.edu.au/~rsw/Sphere/EffSphDes/sf.html


    Parameters
    ----------
    order : integer
        T-design order
    n_points : integer, optional
        Number of sampling points
    symmetric : boolean
        Return symmetric (antipodal) t-designs

    Returns
    -------
    coordinates : Coordinates
        Coordinate object containing all sampling points
    """
    n_points = np.int(np.ceil((order + 1)**2 / 2) + 1)
    n_points_exceptions = {3:8, 5:18, 7:32, 9:50, 11:72}
    if order in n_points_exceptions:
        n_points = n_points_exceptions[order]

    filename = "sf%03d.%05d" % (order, n_points)
    url = "http://web.maths.unsw.edu.au/~rsw/Sphere/Points/SF/SF29-Nov-2012/"
    fileurl = url + filename

    http = urllib3.PoolManager()
    http_data = http.urlopen('GET', fileurl)

    if http_data.status == 200:
        file_data = http_data.data.decode()
    elif http_data.status == 404:
        raise FileNotFoundError("File was not found. Check if the design you \
                are trying to calculate is a valid t-design.")
    else:
        raise ConnectionError("Connection error. Please check your internet \
                connection.")

    points = np.fromstring(file_data,
                           dtype=np.double,
                           sep=' ').reshape((n_points, 3)).T
    coordinates = Coordinates(points[:, 0], points[:, 1], points[:, 2])
    return coordinates



# def healpix(n_max):
#     """
#     Sampling based on the healpix pixelisation.
#     """
#     n_sh = (n_max+1)**2
#     n_side = np.int(2**(np.ceil(np.log2(np.sqrt(n_sh/12)))))
#
#     n_pix = np.arange(healpy.nside2npix(n_side))
#     theta, phi = healpy.pix2ang(n_side, n_pix)
#     rad = np.ones(theta.size)
#
#     return rad, theta, phi


def dodecahedron():
    """Generate a sampling based on the center points of the twelve dodecahedron faces.

    Returns
    -------
    rad : ndarray
        Radius of the sampling points
    theta : ndarray
        Elevation angle in the range [0, pi]
    phi : ndarray
        Azimuth angle in the range [0, 2 pi]
    Returns
    -------
    coordinates : Coordinates
        Coordinate object containing all sampling points
    """

    dihedral = 2*np.arcsin(np.cos(np.pi/3)/np.sin(np.pi/5))
    R = np.tan(np.pi/3)*np.tan(dihedral/2)
    rho = np.cos(np.pi/5)/np.sin(np.pi/10)

    theta1 = np.arccos((np.cos(np.pi/5)/np.sin(np.pi/5))/np.tan(np.pi/3))

    a2 = 2*np.arccos(rho/R)

    theta2 = theta1+a2
    theta3 = np.pi - theta2
    theta4 = np.pi - theta1

    phi1 = 0
    phi2 = 2*np.pi/3
    phi3 = 4*np.pi/3

    theta = np.concatenate((np.tile(theta1, 3),
                            np.tile(theta2, 3),
                            np.tile(theta3, 3),
                            np.tile(theta4, 3)))
    phi = np.tile(np.array([phi1,
                            phi2,
                            phi3,
                            phi1 + np.pi/3,
                            phi2 + np.pi/3,
                            phi3 + np.pi/3]), 2)
    rad = np.ones(np.size(theta))

    coordinates = Coordinates.from_spherical(rad, theta, phi)
    return coordinates


def icosahedron():
    """Generate a sampling based on the center points of the twenty \
            icosahedron faces.

    Returns
    -------
    coordinates : Coordinates
        Coordinate object containing all sampling points
    """
    gamma_R_r = np.arccos(np.cos(np.pi/3) / np.sin(np.pi/5))
    gamma_R_rho = np.arccos(1/(np.tan(np.pi/5) * np.tan(np.pi/3)))

    theta = np.tile(np.array([np.pi - gamma_R_rho,
                              np.pi - gamma_R_rho - 2*gamma_R_r,
                              2*gamma_R_r + gamma_R_rho,
                              gamma_R_rho]), 5)
    theta = np.sort(theta)
    phi = np.arange(0, 2*np.pi, 2*np.pi/5)
    phi = np.concatenate((np.tile(phi, 2), np.tile(phi + np.pi/5, 2)))

    rad = np.ones(20)
    coordinates = Coordinates.from_spherical(rad, theta, phi)
    return coordinates


def equiangular(n_max):
    """Generate an equiangular sampling of the sphere.

    Paramters
    ---------
    n_max : integer
        Spherical harmonic order of the sampling

    Returns
    -------
    coordinates : Coordinates
        Coordinate object containing all sampling points
    weights : ndarray
        Quadrature weights for each point

    TODO: implement test function and check weights
    """
    n_theta = np.round((n_max+1)*2)
    n_phi = n_theta
    theta_angles = np.arange(np.pi/(n_theta*2),
                             np.pi,
                             np.pi/n_theta)
    phi_angles = np.arange(0,
                           2*np.pi,
                           2*np.pi/n_phi)
    theta, phi = np.meshgrid(theta_angles, phi_angles)
    rad = np.ones(theta.size)

    # calculate weights
    L = 2*np.arange(0, n_max + 1) + 1
    factor_phi = 2*np.pi/n_phi
    factor_theta = 2/n_theta
    factor_sin = 4/np.pi * np.sin(theta_angles) * \
        (1/L @ L[np.newaxis].T @ theta_angles[np.newaxis])
    weights = np.tile(factor_phi * factor_theta * np.pi/2 * factor_sin, n_phi)

    coordinates = Coordinates.from_spherical(rad,
                                             theta.reshape(-1),
                                             phi.reshape(-1))

    return coordinates, weights


def gaussian(n_max):
    """Generate sampling of the sphere based on the Gaussian quadrature.

    Paramters
    ---------
    n_max : integer
        Spherical harmonic order of the sampling

    Returns
    -------
    coordinates : Coordinates
        Coordinate object containing all sampling points
    weights : ndarray
        Quadrature weights for each point

    """
    legendre, weights = np.polynomial.legendre.leggauss(n_max+1)
    theta_angles = np.arccos(legendre)
    n_phi = np.round((n_max+1)*2)
    phi_angles = np.arange(0,
                           2*np.pi,
                           2*np.pi/n_phi)
    theta, phi = np.meshgrid(theta_angles, phi_angles)
    rad = np.ones(theta.size)
    weights = np.tile(weights*np.pi/(n_max+1), 2*(n_max+1))

    coordinates = Coordinates.from_spherical(rad,
                                             theta.reshape(-1),
                                             phi.reshape(-1))

    return coordinates, weights


def eigenmike_em32():
    """Microphone positions of the Eigenmike em32 by mhacoustics according to the
    Eigenstudio user manual on the homepage [1]_.

    References
    ----------
    .. [1]  Eigenstudio User Manual, https://mhacoustics.com/download


    Returns
    -------
    coordinates : Coordinates
        Coordinate object containing all sampling points

    """
    rad = np.ones(32)
    theta = np.array([69.0, 90.0, 111.0, 90.0, 32.0, 55.0,
                      90.0, 125.0, 148.0, 125.0, 90.0, 55.0,
                      21.0, 58.0, 121.0, 159.0, 69.0, 90.0,
                      111.0, 90.0, 32.0, 55.0, 90.0, 125.0,
                      148.0, 125.0, 90.0, 55.0, 21.0, 58.0,
                      122.0, 159.0]) * np.pi / 180
    phi = np.array([0.0, 32.0, 0.0, 328.0, 0.0, 45.0,
                    69.0, 45.0, 0, 315.0, 291.0, 315.0,
                    91.0, 90.0, 90.0, 89.0, 180.0, 212.0,
                    180.0, 148.0, 180.0, 225.0, 249.0, 225.0,
                    180.0, 135.0, 111.0, 135.0, 269.0, 270.0,
                    270.0, 271.0]) * np.pi / 180

    coordinates = Coordinates.from_spherical(rad, theta, phi)
    return coordinates


def icosahedron_ke4():
    """Microphone positions of the KE4 spherical microphone array.
    The microphone marked as "1" defines the positive x-axis.

    Returns
    -------
    coordinates : Coordinates
        Coordinate object containing all sampling points

    """

    theta = np.array([1.565269801525254, 2.294997457752220, 1.226592351686568,
                      1.226592351686568, 1.696605844149543, 2.409088979442091,
                      2.409088979442090, 1.696605844149543, 0.506378251408914,
                      0.506378251408914, 2.635214402180879, 1.444986809440250,
                      0.732503674147703, 0.732503674147703, 1.444986809440250,
                      2.635214402180879, 1.915000301903226, 0.846595195837573,
                      1.915000301903225, 1.576322852064539])

    phi = np.array([0.000000000000000, 6.283185307179586, 0.660263824203969,
                    5.622921482975618, 5.055791576545843, 5.241315660913181,
                    1.041869646266405, 1.227393730633743, 0.826693168093822,
                    5.456492139085764, 3.968285821683615, 4.368986384223536,
                    4.183462299856198, 2.099723007323388, 1.914198922956050,
                    2.314899485495971, 2.481328829385824, 3.141592653589793,
                    3.801856477793762, 3.141592653589793])

    rad = np.ones(20) * 0.065
    coordinates = Coordinates.from_spherical(rad, theta, phi)

    return coordinates
