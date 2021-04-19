import numpy as np
import scipy.special as spspecial
from spharpy.special import spherical_bessel_zeros
from spharpy.samplings import Coordinates


def interior_stabilization_points(kr_max, resolution_factor=1):
    """ Find points inside the interior domain of an open  spherical microphone
    array that stabilize the array array response at the eigenfrequencies of
    the array. The algorithm is based on [7]_ and implemented following the
    Matlab code provided by Gilles Chardon on his homepage at [8]_.
    The stabilization points are independent of the sampling of the sphere
    and can therefore be combined with arbitrary spherical samplings.

    Parameters
    ----------
    kr_max : float
        The maximum kr value to be considered. This will define
        the upper frequency limit of the array.
    resolution_factor : int
        Factor to increase the spatial resolution of the grid
        used to estimate the stabilization points.

    Returns
    -------
    sampling_interior : Coordinates
        Coordinates of the stabilization points

    References
    ----------
    .. [7]  G. Chardon, W. Kreuzer, und M. Noisternig, "Design of spatial
            microphone arrays for sound field interpolation", IEEE Journal of
            Selected Topics in Signal Processing
    .. [8]  https://gilleschardon.fr/jstsp_array/

    """
    x, y, z = find_interior_points(kr_max, resolution_factor=resolution_factor)

    sampling_interior = Coordinates(x, y, z)

    return sampling_interior


def find_eigenfrequencies(kr_max):
    """
    Find the eigenfrequencies for the sphere from the spherical
    Bessel function.
    """
    jn_zeros = spherical_bessel_zeros(kr_max, kr_max)

    eigenfrequencies = []
    for idx in range(0, kr_max + 1):
        roots = jn_zeros[idx]
        roots = roots[roots < kr_max]
        if roots.size != 0:
            eigenfrequencies.append(roots)

    mults = np.arange(1, 2*len(eigenfrequencies))

    return eigenfrequencies, mults


def calculate_eigenspaces(kr_max, theta, phi, rad):
    """Calculate the eigenspaces for the corresponding eigenfrequencies of
    the sphere

    Parameters
    ----------
    k_max : float
        The largest wave number to be included
    theta : array, float
        Azimuth angle
    phi : array, float
        Elevation angle
    rad : array, float
        Radius

    Returns
    -------
    eigenspaces : list, ndarray, float
        List containing all eigenspaces

    """
    eigenfrequencies, mults = find_eigenfrequencies(kr_max)

    subspaces = []
    for u in range(len(eigenfrequencies)):
        for root in eigenfrequencies[u]:
            subspaces.append(sph_modes_matrix(u, root, theta, phi, rad))

    return subspaces, mults


def sph_modes_matrix(n_max, k, theta, phi, rad):
    """Build the matrix containing all spherical harmonic modes of the domain
    inside an open sphere for a specific order n_max.

    Parameters
    ----------
    n_max : int
        Spherical harmonic order
    k : float
        Wave number
    theta : array, float
        Azimuth angle
    phi : array, float
        Elevation angle
    rad : array, float
        Radius

    Returns
    -------
    modes : array, float
        A matrix with dimension [(...) x (2*n_max+1)] containing all spherical
        harmonic modes.

    Note
    ----
    This function returns only coefficients for one order, but all degrees.

    """
    n_coefficients = 2*n_max+1
    meshgrid_shape = theta.shape
    B = spspecial.spherical_jn(n_max, rad.flatten()*k) * 4*np.pi * (1j)**n_max
    B = np.reshape(B, meshgrid_shape)
    M = np.zeros((*meshgrid_shape, n_coefficients), dtype=np.complex)
    for m in range(-n_max, n_max+1):
        Y_m = spspecial.sph_harm(m, n_max, theta.flatten(), phi.flatten())
        M[:, :, :, m+n_max] = B * np.reshape(Y_m, meshgrid_shape)

    return M


def ball_dot(S1, S2, radius, phi):
    wphi = np.sin(phi)
    wr = radius**2
    w = wr*wphi
    d = np.sum(np.conj(S1) * S2 * w) / np.sum(w)
    return d


def find_interior_points(k_max, resolution_factor=1):
    resolution = 50 * resolution_factor

    vec_theta = np.linspace(0, 2 * np.pi, resolution*2)
    vec_phi = np.linspace(0, np.pi, resolution)
    vec_rad = np.linspace(0, 1, resolution)

    phi, theta, rad = np.meshgrid(vec_phi, vec_theta, vec_rad)
    meshgrid_shape = (resolution*2, resolution, resolution)

    subspaces, mults = calculate_eigenspaces(k_max, theta, phi, rad)
    max_mult = mults.max()

    idx_sel = []
    for w in range(0, max_mult):
        maxes = np.ones((*meshgrid_shape, len(subspaces))) * 1e3

        for idx_space in range(0, len(subspaces)):
            if subspaces[idx_space].size > 0:
                maxes[:, :, :, idx_space] = 0.0

                for v in range(0, subspaces[idx_space].shape[3]):
                    vector = subspaces[idx_space][:, :, :, v]
                    for www in range(0, v):
                        vector = vector - ball_dot(
                                        subspaces[idx_space][:, :, :, www],
                                        vector,
                                        rad,
                                        phi) * \
                                    subspaces[idx_space][:, :, :, www]
                    vector = vector / np.sqrt(np.abs(ball_dot(vector,
                                                              vector,
                                                              rad,
                                                              phi)))

                    subspaces[idx_space][:, :, :, v] = vector
                    maxes[:, :, :, idx_space] += np.abs(vector) ** 2

        minmax = np.min(maxes, axis=3)
        argmax = np.argmax(minmax)
        argmax_unravel = np.unravel_index(argmax, meshgrid_shape)
        idx_sel.append(argmax)

        for idx_space in range(0, len(subspaces)):
            if subspaces[idx_space].shape[3] <= 1:
                subspaces[idx_space] = np.array([[[[]]]])
            else:
                value_end = 0
                www = -1
                while value_end == 0:
                    www += 1
                    vend = subspaces[idx_space][:, :, :, www].copy()
                    value_end = vend[argmax_unravel]

                subspaces[idx_space][:, :, :, www] = \
                    subspaces[idx_space][:, :, :, -1]
                subspaces[idx_space][:, :, :, -1] = vend

                for v in range(0, subspaces[idx_space].shape[3] - 1):
                    vv = subspaces[idx_space][:, :, :, v]
                    value_vv = vv[argmax_unravel]
                    subspaces[idx_space][:, :, :, v] = \
                        vv - vend / value_end * value_vv

                subspaces[idx_space] = subspaces[idx_space][:, :, :, :-1]

    idx_sel_unravel = np.unravel_index(idx_sel, meshgrid_shape)
    theta_opt = theta[idx_sel_unravel]
    phi_opt = phi[idx_sel_unravel]
    rad_opt = rad[idx_sel_unravel]

    x = rad_opt * np.cos(theta_opt) * np.sin(phi_opt)
    y = rad_opt * np.sin(theta_opt) * np.sin(phi_opt)
    z = rad_opt * np.cos(phi_opt)

    return x, y, z
