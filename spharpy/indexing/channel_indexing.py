"""
Commonly used channel indexing functions used in Ambisonics
"""

import numpy as np
import spharpy


def sid(n_max):
    """Calculates the SID indices up to a maximum spherical harmonic order n_max.
    TODO: Add citation Daniel

    Parameters
    ----------
    n_max : int

    Returns
    -------
    sid : numpy array

    """
    n_sh = (n_max+1)**2
    sid_n = sph_identity_matrix(n_max, 'n-nm').T @ np.arange(0, n_max+1)
    sid_m = np.zeros(n_sh, dtype=np.int)
    idx_n = 0
    for n in range(1, n_max+1):
        for m in range(1, n+1):
            sid_m[idx_n + 2*m-1] = n-m+1
            sid_m[idx_n + 2*m] = -(n-m+1)
        sid_m[idx_n + 2*n + 1] = 0
        idx_n += 2*n+1

    return sid_n, sid_m


def sid2acn(n_max):
    """Convert from SID channel indexing as proposed by Daniel in
    TODO: add citation
    Returns the indices to achieve a corresponding linear acn indexing.

    Parameters
    ----------
    sid : numpy array

    Returns
    -------
    acn : numpy array

    """
    sid_n, sid_m = spharpy.indexing.sid(n_max)
    linear_sid = spharpy.spherical.nm2acn(sid_n, sid_m)
    sort_index = np.argsort(linear_sid)

    return sort_index


def sph_identity_matrix(n_max, type='n-nm'):
    """Calculate a spherical harmonic identity matrix.
    TODO: Implement the other identity matrices

    Parameters
    ----------
    n_max : TODO
    type : TODO, optional

    Returns
    -------
    identity_matrix : numpy array

    """
    n_sh = (n_max+1)**2

    if type != 'n-nm':
        raise NotImplementedError

    identity_matrix = np.zeros((n_max+1, n_sh), dtype=np.int)
    # linear_n0 = np.cumsum(np.arange(0, 2*(n_max+1), 2))

    for n in range(0, n_max+1):
        m = np.arange(-n, n+1)
        linear_nm = spharpy.spherical.nm2acn(np.tile(n, m.shape), m)
        identity_matrix[n, linear_nm] = 1

    return identity_matrix
