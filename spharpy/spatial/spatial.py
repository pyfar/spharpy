import numpy as np
import scipy.spatial as sspat


def greens_function_plane_wave(
        source_points,
        receiver_points,
        wave_number,
        gradient=False):
    r"""Green's function for plane waves in free space in matrix form.
    The matrix describing the propagation of a plane wave from a direction
    of arrival defined by the azimuth and elevation angles of the source points
    to the receiver points. The phase sign convention reflects a direction of
    arrival from the source position.

    .. math::

        G(k) = e^{i \mathbf{k}^\mathrm{T} \mathbf{r}}

    Parameters
    ----------
    source_points : :py:class:`pyfar.Coordinates`
        The source points defining the direction of incidence for the plane
        wave. Note that the radius on which the source is positioned has no
        relevance.
    receiver_points : :py:class:`pyfar.Coordinates`
        The receiver points.
    wave_number : float, complex
        The wave number. A complex wave number can be used for evanescent
        waves.
    gradient : bool
        If True, the gradient will be returned as well


    Returns
    -------
    M : ndarray, complex
        The plane wave propagation matrix with shape [n_receiver, n_sources].

    Examples
    --------

    Plot Green's function in the x-y plane for a plane wave with a direction
    of incidence defined by the vector :math:`[x, y, z] = [2, 1, 0]`.

    .. plot::

        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> from pyfar import Coordinates
        >>> import spharpy
        ...
        >>> spat_res = 30
        >>> x_min, x_max = -10, 10
        >>> xx, yy = np.meshgrid(
        >>>     np.linspace(x_min, x_max, spat_res),
        >>> np.linspace(x_min, x_max, spat_res))
        >>> receivers = Coordinates(
        ...     xx.flatten(), yy.flatten(), np.zeros(spat_res**2))
        >>> doa = Coordinates(2, 1, 0)
        ...
        >>> k = 1
        >>> plane_wave_matrix = spharpy.spatial.greens_function_plane_wave(
        ...     doa, receivers, k)
        >>> plt.figure()
        >>> plt.contourf(
        ...     xx, yy, np.real(plane_wave_matrix.reshape(spat_res, spat_res)),
        ...     cmap='RdBu_r', levels=100)
        >>> plt.colorbar()
        >>> ax = plt.gca()
        >>> ax.set_aspect('equal')

    For evanescent waves, a complex wave number can be used

    .. plot::

        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> from pyfar import Coordinates
        >>> import spharpy
        ...
        >>> spat_res = 30
        >>> x_min, x_max = -10, 10
        >>> xx, yy = np.meshgrid(
        >>>     np.linspace(x_min, x_max, spat_res),
        >>> np.linspace(x_min, x_max, spat_res))
        >>> receivers = Coordinates(
        ...     xx.flatten(), yy.flatten(), np.zeros(spat_res**2))
        >>> doa = Coordinates(2, 1, 0)
        ...
        >>> k = 1-.1j
        >>> plane_wave_matrix = spharpy.spatial.greens_function_plane_wave(
        ...     doa, receivers, k, gradient=False)
        >>> plt.contourf(
        ...     xx, yy, np.real(plane_wave_matrix.reshape(spat_res, spat_res)),
        ...     cmap='RdBu_r', levels=100)
        >>> plt.colorbar()
        >>> ax = plt.gca()
        >>> ax.set_aspect('equal')


    """
    e_doa = source_points.cartesian.T / \
        np.linalg.norm(source_points.cartesian.T, axis=0)
    k_vec = np.squeeze(wave_number*e_doa)

    arg = receiver_points.cartesian @ k_vec
    plane_wave_matrix = np.cos(arg) + 1j*np.sin(arg)

    if gradient is False:
        return plane_wave_matrix

    plane_wave_gradient_matrix_x = (plane_wave_matrix * 1j*k_vec[0])
    plane_wave_gradient_matrix_y = (plane_wave_matrix * 1j*k_vec[1])
    plane_wave_gradient_matrix_z = (plane_wave_matrix * 1j*k_vec[2])
    return plane_wave_matrix, \
        [plane_wave_gradient_matrix_x,
         plane_wave_gradient_matrix_y,
         plane_wave_gradient_matrix_z]


def greens_function_point_source(sources, receivers, k, gradient=False):
    r"""Green's function for point sources in free space in matrix form.
    The phase sign convention corresponds to a direction of propagation away
    from the source at position :math:`r_s`.

    .. math::

        G(k) = \frac{e^{-i k\|\mathbf{r_s} - \mathbf{r_r}\|}}
            {4 \pi \|\mathbf{r_s} - \mathbf{r_r}\|}

    Parameters
    ----------

    source : :py:class:`pyfar.Coordinates`
        source points as Coordinates object
    receivers : :py:class:`pyfar.Coordinates`
        receiver points as Coordinates object
    k : ndarray, float
        The wave number

    Returns
    -------
    G : ndarray, double
        Green's function


    Examples
    --------

    Plot Green's function in the x-y plane for a point source at
    :math:`[x, y, z] = [10, 15, 0]`.

    .. plot::

        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> from pyfar import Coordinates
        >>> import spharpy
        ...
        >>> spat_res = 30
        >>> x_min, x_max = -10, 10
        >>> xx, yy = np.meshgrid(
        >>>     np.linspace(x_min, x_max, spat_res),
        >>> np.linspace(x_min, x_max, spat_res))
        >>> receivers = Coordinates(
        ...     xx.flatten(), yy.flatten(), np.zeros(spat_res**2))
        >>> doa = Coordinates(10, 15, 0)
        >>> plane_wave_matrix = spharpy.spatial.greens_function_point_source(
        ...     doa, receivers, 1, gradient=False)
        >>> plt.contourf(
        ...     xx, yy, np.real(plane_wave_matrix.reshape(spat_res, spat_res)),
        ...     cmap='RdBu_r', levels=100)
        >>> plt.colorbar()
        >>> ax = plt.gca()
        >>> ax.set_aspect('equal')

    """
    dist = sspat.distance.cdist(receivers.cartesian, sources.cartesian)
    dist = np.squeeze(dist)
    cexp = np.cos(k*dist) - 1j*np.sin(k*dist)
    G = cexp/dist/4/np.pi

    if gradient is False:
        return G

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
