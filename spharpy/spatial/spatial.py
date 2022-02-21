import numpy as np
import scipy.spatial as sspat


def greens_function_plane_wave(
        source_points,
        receiver_points,
        wave_number,
        gradient=False):
    """The matrix describing the propagation of a plane wave from a direction
    of arrival defined by the azimuth and elevation angles of the source points
    to the receiver points. The phase sign convention reflects a direction of
    arrival from the source position.

    Parameters
    ----------
    source_points : Coordinates
        The source points defining the direction of incidence for the plane
        wave. Note that the radius on which the source is positioned has no
        relevance.
    receiver_points : Coordinates
        The receiver points.
    wave_number : double
        The wave number of the wave
    speed_of_sound : double
        The speed of sound
    gradient : bool
        If True, the gradient will be returned as well


    Returns
    -------
    M : ndarray, complex, shape(n_receiver, n_sources)
        The plane wave propagation matrix

    """
    e_doa = source_points.cartesian / \
        np.linalg.norm(source_points.cartesian, axis=0)
    k_vec = np.squeeze(wave_number*e_doa)

    # avoid using the complex exponential since it's slower than sin and cos
    arg = receiver_points.cartesian.T @ k_vec
    plane_wave_matrix = np.cos(arg) + 1j*np.sin(arg)

    if not gradient:
        return plane_wave_matrix
    else:
        plane_wave_gradient_matrix_x = (plane_wave_matrix * 1j*k_vec[0])
        plane_wave_gradient_matrix_y = (plane_wave_matrix * 1j*k_vec[1])
        plane_wave_gradient_matrix_z = (plane_wave_matrix * 1j*k_vec[2])
        return plane_wave_matrix, \
            [plane_wave_gradient_matrix_x,
             plane_wave_gradient_matrix_y,
             plane_wave_gradient_matrix_z]


def greens_function_point_source(sources, receivers, k, gradient=False):
    r"""Green's function for point sources in free space in matrix form. The
    phase sign convention corresponds to a direction of propagation away from
    the source at position $r_s$.

    .. math::

        G(k) = \\frac{e^{- k\\|\\mathbf{r_s} - \\mathbf{r_r}\\|}}
            {4 \\pi \\|\\mathbf{r_s} - \\mathbf{r_r}\\|}

    Parameters
    ----------
    source : Coordinates
        source points as Coordinates object
    receivers : Coordinates
        receiver points as Coordinates object
    k : ndarray, double
        wave number

    Returns
    -------
    G : ndarray, double
        Green's function

    """
    dist = sspat.distance.cdist(receivers.cartesian.T, sources.cartesian.T)
    dist = np.squeeze(dist)
    cexp = np.cos(k*dist) - 1j*np.sin(k*dist)
    G = cexp/dist/4/np.pi

    if not gradient:
        return G
    else:
        def lambda_cdiff(u, v):
            return u-v

        diff_x = sspat.distance.cdist(
            np.atleast_2d(receivers.x).T,
            np.atleast_2d(sources.x).T,
            lambda_cdiff)
        diff_y = sspat.distance.cdist(
            np.atleast_2d(receivers.y).T,
            np.atleast_2d(sources.y).T,
            lambda_cdiff)
        diff_z = sspat.distance.cdist(
            np.atleast_2d(receivers.z).T,
            np.atleast_2d(sources.z).T,
            lambda_cdiff)

        G_dx = G/dist * np.squeeze(diff_x) * (-1j-1/dist)
        G_dy = G/dist * np.squeeze(diff_y) * (-1j-1/dist)
        G_dz = G/dist * np.squeeze(diff_z) * (-1j-1/dist)

        return G, (G_dx, G_dy, G_dz)
