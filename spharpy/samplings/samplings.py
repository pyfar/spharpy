"""
Collection of sampling schemes for the sphere
"""
import urllib3
import numpy as np
from spharpy.samplings.coordinates import Coordinates, SamplingSphere
import spharpy

from ._eqsp import point_set as eq_point_set


def cube_equidistant(n_points):
    """Create a cuboid sampling with equidistant spacings in x, y, and z.
    The cube will have dimensions 1 x 1 x 1

    Parameters
    ----------
    n_points : int, tuple
        Number of points in the sampling. If a single value is given, the
        number of sampling positions will be the same in every axis. If a tuple
        is given, the number of points will be set as (n_x, n_y, n_z)

    Returns
    -------
    sampling : Coordinates
        Sampling positions as Coordinate object

    """
    if np.size(n_points) == 1:
        n_x = n_points
        n_y = n_points
        n_z = n_points
    elif np.size(n_points) == 3:
        n_x = n_points[0]
        n_y = n_points[1]
        n_z = n_points[2]
    else:
        raise ValueError("The number of points needs to be either an integer \
                or a tuple with 3 elements.")

    x = np.linspace(-1, 1, n_x)
    y = np.linspace(-1, 1, n_y)
    z = np.linspace(-1, 1, n_z)

    x_grid, y_grid, z_grid = np.meshgrid(x, y, z)

    return Coordinates(x_grid.flatten(), y_grid.flatten(), z_grid.flatten())


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
    sampling: SamplingSphere
        SamplingSphere object containing all sampling points
    """
    n_sh = (n_max+1)**2
    filename = "md%03d.%05d" % (n_max, n_sh)
    url = "https://web.maths.unsw.edu.au/~rsw/Sphere/S2Pts/MD/"
    fileurl = url + filename

    http = urllib3.PoolManager(cert_reqs=False)
    http_data = http.urlopen('GET', fileurl)

    if http_data.status == 200:
        file_data = http_data.data.decode()
    else:
        raise ConnectionError("Connection error. Please check your internet \
                connection.")

    file_data = np.fromstring(
        file_data, dtype='double', sep=' ').reshape((n_sh, 4))
    sampling = SamplingSphere(
        file_data[:, 0],
        file_data[:, 1],
        file_data[:, 2])
    sampling.weights = file_data[:, 3]

    return sampling


def spherical_t_design(n_max, criterion='const_energy'):
    r"""Return the sampling positions for a spherical t-design [3]_ .
    For a spherical harmonic order N, a t-Design of degree `:math: t=2N` for
    constant energy or `:math: t=2N+1` additionally ensuring a constant angular
    spread of energy is required [4]_. For a given degree t

    .. math::

        L = \lceil \frac{(t+1)^2}{2} \rceil+1,

    points will be generated, except for t = 3, 5, 7, 9, 11, 13, and 15.
    T-designs allow for a inverse spherical harmonic transform matrix
    calculated as `:math: D = \frac{4\pi}{L} \mathbf{Y}^\mathrm{H}`.

    Parameters
    ----------
    degree : integer
        T-design degree
    criterion : 'const_energy', 'const_angular_spread'
        Design criterion ensuring only a constant energy or additionally
        constant angular spread of energy

    Returns
    -------
    sampling : SamplingSphere
        SamplingSphere object containing all sampling points

    Notes
    -----
    This function downloads a pre-calculated set of points from
    Rob Womersley's homepage [5]_ .

    References
    ----------

    .. [3]  C. An, X. Chen, I. H. Sloan, and R. S. Womersley, “Well Conditioned
            Spherical Designs for Integration and Interpolation on the
            Two-Sphere,” SIAM Journal on Numerical Analysis, vol. 48, no. 6,
            pp. 2135–2157, Jan. 2010.
    .. [4]  F. Zotter, M. Frank, and A. Sontacchi, “The Virtual T-Design
            Ambisonics-Rig Using VBAP,” in Proceedings on the Congress on
            Sound and Vibration, 2010.
    .. [5]  http://web.maths.unsw.edu.au/~rsw/Sphere/EffSphDes/sf.html

    """
    if criterion == 'const_energy':
        degree = 2*n_max
    elif criterion == 'const_angular_spread':
        degree = 2*n_max + 1
    else:
        raise ValueError("Invalid design criterion.")

    n_points = int(np.ceil((degree + 1)**2 / 2) + 1)
    n_points_exceptions = {3: 8, 5: 18, 7: 32, 9: 50, 11: 72, 13: 98, 15: 128}
    if degree in n_points_exceptions:
        n_points = n_points_exceptions[degree]

    filename = "sf%03d.%05d" % (degree, n_points)
    url = "http://web.maths.unsw.edu.au/~rsw/Sphere/Points/SF/SF29-Nov-2012/"
    fileurl = url + filename

    http = urllib3.PoolManager(
        cert_reqs=False)
    http_data = http.urlopen('GET', fileurl)

    if http_data.status == 200:
        file_data = http_data.data.decode()
    elif http_data.status == 404:
        raise FileNotFoundError("File was not found. Check if the design you \
                are trying to calculate is a valid t-design.")
    else:
        raise ConnectionError("Connection error. Please check your internet \
                connection.")

    points = np.fromstring(
        file_data,
        dtype=float,
        sep=' ').reshape((n_points, 3)).T

    return SamplingSphere.from_array(points)


def dodecahedron():
    """Generate a sampling based on the center points of the twelve
    dodecahedron faces.

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
    sampling : SamplingSphere
        SamplingSphere object containing all sampling points
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

    theta = np.concatenate((
        np.tile(theta1, 3), np.tile(theta2, 3),
        np.tile(theta3, 3), np.tile(theta4, 3)))
    phi = np.tile(np.array([
        phi1, phi2, phi3,
        phi1 + np.pi/3, phi2 + np.pi/3, phi3 + np.pi/3]), 2)

    rad = np.ones(np.size(theta))

    return SamplingSphere.from_spherical(rad, theta, phi)


def icosahedron():
    """Generate a sampling based on the center points of the twenty \
            icosahedron faces.

    Returns
    -------
    sampling : SamplingSphere
        SamplingSphere object containing all sampling points
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

    return SamplingSphere.from_spherical(rad, theta, phi)


def equiangular(n_max):
    """Generate an equiangular sampling of the sphere.

    Paramters
    ---------
    n_max : integer
        Spherical harmonic order of the sampling

    Returns
    -------
    sampling : SamplingSphere
        SamplingSphere object containing all sampling points

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

    sampling = SamplingSphere.from_spherical(
        rad, theta.reshape(-1), phi.reshape(-1))
    sampling.weights = weights

    return sampling


def gaussian(n_max):
    """Generate sampling of the sphere based on the Gaussian quadrature.

    Paramters
    ---------
    n_max : integer
        Spherical harmonic order of the sampling

    Returns
    -------
    sampling : SamplingSphere
        SamplingSphere object containing all sampling points

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

    sampling = SamplingSphere.from_spherical(
        rad, theta.reshape(-1), phi.reshape(-1))
    sampling.weights = weights
    return sampling


def eigenmike_em32():
    """Microphone positions of the Eigenmike em32 by mhacoustics according to
    the Eigenstudio user manual on the homepage [6]_.

    References
    ----------
    .. [6]  Eigenstudio User Manual, https://mhacoustics.com/download


    Returns
    -------
    sampling : SamplingSphere
        SamplingSphere object containing all sampling points

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

    return SamplingSphere.from_spherical(rad, theta, phi)


def eigenmike_em64():
    """Microphone positions of the Eigenmike em64 by mhacoustics.

    according to
    the Eigenmuke user manual on the homepage [#]_.

    References
    ----------
    .. [#]  Eigenmike em64 User Manual, https://eigenmike.com/sites/default/files/documentation-2024-09/getting%20started%20Guide%20to%20em64%20and%20ES3%20R01H.pdf

    Returns
    -------
    sampling : SamplingSphere
        SamplingSphere object containing all sampling points

    """  # noqa: E501

    rad = np.ones(64)*0.042

    theta = np.deg2rad(np.array([
        16.7656, 21.9677, 42.3941, 13.2817, 22.6728, 52.6925, 37.806, 43.3944,
        43.9386, 70.3132, 33.2231, 60.0257, 56.4763, 67.4936, 93.2735, 48.423,
        78.0793, 62.0685, 38.7171, 63.8004, 70.1946, 96.246, 81.0992, 106.094,
        67.7533, 91.7061, 39.9985, 68.7726, 60.8869, 82.2833, 63.0247, 89.794,
        137.5166, 139.7604, 135.2133, 160.3628, 162.577, 142.0685, 161.1987,
        162.577, 115.536, 86.2594, 116.0164, 95.3313, 90.0637, 111.4549,
        85.8671, 130.8398, 102.5775, 142.6375, 117.032, 117.5631, 115.8884,
        89.69, 118.4478, 93.9338, 106.3875, 81.0511, 135.9764, 142.6771,
        120.6556, 133.8834, 116.3591, 107.464,
    ]))

    phi = np.deg2rad(np.array([
        197.4561, 115.734, 81.911, 313.3592, 43.1785, 46.7324, 335.9958,
        14.5398, 204.4547, 206.542, 247.3219, 233.817, 264.5437, 99.6669,
        104.6842, 120.9227, 126.513, 148.2368, 162.6381, 178.5498, 21.2715,
        25.7834, 47.8607, 55.9075, 71.4285, 78.4921, 293.221, 290.5683,
        318.1354, 334.0042, 352.0227, 0, 174.0335, 212.7205, 251.9179,
        150.6471, 240.8266, 293.0625, 331.0098, 60.8266, 226.9135, 233.9255,
        193.6382, 209.6696, 183.169, 163.7105, 156.9524, 139.4318, 135.9729,
        102.3273, 112.5511, 83.1464, 307.7078, 309.1392, 278.2519, 282.9735,
        253.147, 260.0688, 59.7394, 14.2241, 32.4901, 334.0753, 2.0842,
        335.0677,
    ]))

    weights = np.array([
        0.954, 0.9738, 1.0029, 1.0426, 1.0426, 1.0024, 0.9738, 0.954, 1.009,
        0.9932, 1.0024, 1.0324, 0.954, 1.0024, 1.0079, 1.0268, 1.0151, 0.9463,
        1.012, 1.0253, 1.009, 0.9932, 1.0324, 1.0151, 0.954, 1.0079, 1.0029,
        1.0024, 1.0268, 0.9463, 1.012, 1.0253, 0.954, 0.9738, 1.0029, 1.0426,
        1.0426, 1.0024, 0.954, 0.9738, 1.0268, 1.0151, 1.012, 0.9463, 1.0253,
        1.009, 0.9932, 1.0024, 1.0324, 1.0029, 0.954, 1.0024, 1.0324, 1.0151,
        0.954, 1.0079, 1.0024, 1.0079, 1.0268, 1.012, 0.9463, 1.009, 1.0253,
        0.9932,
    ]) / 64*4*np.pi

    sampling = SamplingSphere.from_spherical(
        rad, theta, phi, n_max=6)
    sampling.weights = weights

    return sampling


def icosahedron_ke4():
    """Microphone positions of the KE4 spherical microphone array.
    The microphone marked as "1" defines the positive x-axis.

    Returns
    -------
    sampling : SamplingSphere
        SamplingSphere object containing all sampling points

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

    return SamplingSphere.from_spherical(rad, theta, phi)


def equalarea(n_max, condition_num=2.5, n_points=None):
    """Sampling based on partitioning into faces with equal area [9]_.

    Parameters
    ----------
    n_max : int
        Spherical harmonic order
    condition_num : double
        Desired maximum condition number of the spherical harmonic basis matrix
    n_points : int, optional
        Number of points to start the condition number optimization. If set to
        None n_points will be (n_max+1)**2

    Returns
    -------
    sampling : SamplingSphere
        SamplingSphere object containing all sampling points

    References
    ----------
    .. [9]  P. Leopardi, “A partition of the unit sphere into regions of equal
            area and small diameter,” Electronic Transactions on Numerical
            Analysis, vol. 25, no. 12, pp. 309–327, 2006.

    """
    if not n_points:
        n_points = (n_max+1)**2

    while True:
        point_set = eq_point_set(2, n_points)
        sampling = SamplingSphere(point_set[0], point_set[1], point_set[2])

        if condition_num == np.inf:
            break
        Y = spharpy.spherical.spherical_harmonic_basis(n_max, sampling)
        cond = np.linalg.cond(Y)
        if cond < condition_num:
            break
        n_points += 1

    sampling.n_max = n_max
    return sampling


def spiral_points(n_max, condition_num=2.5, n_points=None):
    """Sampling based on a spiral distribution of points on a sphere [10]_.

    Parameters
    ----------
    n_max : int
        Spherical harmonic order
    condition_num : double
        Desired maximum condition number of the spherical harmonic basis matrix
    n_points : int, optional
        Number of points to start the condition number optimization. If set to
        None n_points will be (n_max+1)**2

    Returns
    -------
    sampling : SamplingSphere
        SamplingSphere object containing all sampling points

    References
    ----------

    .. [10]  E. a. Rakhmanov, E. B. Saff, and Y. M. Zhou, “Minimal Discrete
            Energy on the Sphere,” Mathematical Research Letters, vol. 1,
            no. 6, pp. 647–662, 1994.

    """
    if n_points is None:
        n_points = (n_max+1)**2

    def _spiral_points(n_points):
        """Helper function doing the actual calculation of the points"""
        r = np.zeros(n_points)
        h = np.zeros(n_points)
        theta = np.zeros(n_points)
        phi = np.zeros(n_points)

        p = 1/2
        a = 1 - 2*p/(n_points-3)
        b = p*(n_points+1)/(n_points-3)
        r[0] = 0
        theta[0] = np.pi
        phi[0] = 0
        # Then for k stepping by 1 from 2 to n-1:
        for k in range(1, n_points-1):
            kStrich = a*k + b
            h[k] = -1 + 2*(kStrich-1)/(n_points-1)
            r[k] = np.sqrt(1-h[k]**2)
            theta[k] = np.arccos(h[k])
            phi[k] = np.mod(
                (phi[k-1]) + 3.6/np.sqrt(n_points)*2/(r[k-1]+r[k]), 2*np.pi)
        # Finally:
        theta[n_points-1] = 0
        phi[n_points-1] = 0

        return theta, phi

    while True:
        theta, phi = _spiral_points(n_points)
        sampling = SamplingSphere.from_spherical(np.ones(n_points), theta, phi)
        if condition_num == np.inf:
            break
        Y = spharpy.spherical.spherical_harmonic_basis(n_max, sampling)
        cond = np.linalg.cond(Y)
        if cond < condition_num:
            break
        n_points += 1

    sampling.n_max = n_max

    return sampling
