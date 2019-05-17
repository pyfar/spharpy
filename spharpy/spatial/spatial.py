import numpy as np


def greens_function(source, receivers, k):
    """Green's function in matrix form

    .. math::

        G(k) = \\frac{e^{-k\\|\\mathbf{r_s} - \\mathbf{r_r}\\|}}{4 \\pi \\|\\mathbf{r_s} - \\mathbf{r_r}\\|}

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
    R_vec = receivers.cartesian - source.cartesian
    R = np.linalg.norm(R_vec, axis=0)
    G = np.exp(-1j*k*R)/R/4/np.pi
    return G


def greens_function_gradient(source, receivers, k):
    """Gradient of Green's function in Cartesian coordinates.

    .. math::

        \\nabla G(k)

        G(k) = \\frac{e^{-k\\|\\mathbf{r_s} - \\mathbf{r_r}\\|}}{4 \\pi \\|\\mathbf{r_s} - \\mathbf{r_r}\\|}


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
    grad_G : ndarray, double
        Gradient of Green's function

    """
    R_vec = receivers.cartesian - source.cartesian
    R = np.linalg.norm(R_vec, axis=0)
    outer = np.exp(-1j * k * R) / (4*np.pi*R**2) * (-1j*k - 1/R)
    inner = R_vec
    grad_G = inner @ np.diag(outer)
    return grad_G
