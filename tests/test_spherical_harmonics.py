"""
Tests for spherical harmonic basis and related functions
"""
import spharpy.spherical as sh
import numpy as np
import numpy.testing as npt
import pytest


@pytest.mark.parametrize("implementation", ['spharpy', 'pyfar'])
@pytest.mark.parametrize("normalization", ['n3d', 'maxN', 'sn3d'])
@pytest.mark.parametrize("channel_convention", ['acn', 'fuma'])
@pytest.mark.parametrize("phase_convention", [None, 'Condon-Shortley'])
def test_spherical_harmonic(make_coordinates, implementation,
                            normalization, channel_convention,
                            phase_convention):
    n_max = 1
    theta = np.array([np.pi/2, np.pi/2, 0], dtype=float)
    phi = np.array([0, np.pi/2, 0], dtype=float)
    rad = np.ones(3, dtype=float)

    coords = make_coordinates.create_coordinates(
        implementation, rad, theta, phi)

    Y = np.genfromtxt(f'./tests/data/Y_cmplx_{phase_convention}_'
                      f'{normalization}_{channel_convention}.csv',
                      dtype=complex,
                      delimiter=',')

    basis = sh.spherical_harmonic_basis(n_max, coords,
                                        normalization=normalization,
                                        channel_convention=channel_convention,
                                        phase_convention=phase_convention)

    np.testing.assert_allclose(Y, basis, atol=1e-13)


@pytest.mark.parametrize("implementation", ['spharpy', 'pyfar'])
def test_spherical_harmonic_n10(make_coordinates, implementation):
    n_max = 10
    theta = np.array([np.pi/2, np.pi/2, 0], dtype=float)
    phi = np.array([0, np.pi/2, 0], dtype=float)

    coords = make_coordinates.create_coordinates(
        implementation, np.ones_like(theta), theta, phi)

    Y = np.genfromtxt(
        './tests/data/sh_basis_cplx_n10.csv',
        delimiter=',',
        dtype=complex)

    basis = sh.spherical_harmonic_basis(n_max, coords,
                                        phase_convention='Condon-Shortley')

    np.testing.assert_allclose(Y, basis, atol=1e-13)


@pytest.mark.parametrize("implementation", ['spharpy', 'pyfar'])
@pytest.mark.parametrize("phase_convention", [None, 'Condon-Shortley'])
@pytest.mark.parametrize("channel_convention", ['acn', 'fuma'])
@pytest.mark.parametrize("normalization", ['n3d', 'sn3d'])
def test_spherical_harmonics_real(make_coordinates, implementation,
                                  normalization, channel_convention,
                                  phase_convention):
    n_max = 1
    theta = np.array([np.pi/2, np.pi/2, 0], dtype=float)
    phi = np.array([0, np.pi/2, 0], dtype=float)
    rad = np.ones(3, dtype=float)

    coords = make_coordinates.create_coordinates(
        implementation, rad, theta, phi)

    Y = np.genfromtxt(f'./tests/data/Y_real_{phase_convention}_'
                      f'{normalization}_{channel_convention}.csv',
                      dtype=float,
                      delimiter=',')
    basis = sh.spherical_harmonic_basis_real(n_max, coords,
                                             normalization,
                                             channel_convention,
                                             phase_convention=phase_convention)
    np.testing.assert_allclose(basis, Y, atol=1e-13)


@pytest.mark.parametrize("implementation", ['spharpy', 'pyfar'])
def test_orthogonality(make_coordinates, implementation):
    """
    Check if the orthonormality condition of the spherical harmonics is
    fulfilled
    """
    n_max = 82
    theta = np.array([np.pi/2, np.pi/2, 0, np.pi/2], dtype=float)
    phi = np.array([0, np.pi/2, 0, np.pi/4], dtype=float)
    n_points = phi.size
    n_points = len(theta)
    coords = make_coordinates.create_coordinates(
        implementation, np.ones_like(theta), theta, phi)
    basis = sh.spherical_harmonic_basis(n_max, coords)

    inner = (basis @ np.conjugate(basis.T))
    fact = 4 * np.pi / (n_max + 1) ** 2
    orth = np.diagonal(fact * inner)
    np.testing.assert_allclose(
        orth, np.ones(n_points, dtype=complex), rtol=1e-10)


@pytest.mark.parametrize("implementation", ['spharpy', 'pyfar'])
def test_orthogonality_real(make_coordinates, implementation):
    """
    Check if the orthonormality condition of the reavl valued spherical
    harmonics is fulfilled
    """
    n_max = 82
    theta = np.array([np.pi / 2, np.pi / 2, 0, np.pi / 2], dtype='double')
    phi = np.array([0, np.pi / 2, 0, np.pi / 4], dtype='double')
    n_points = phi.size

    coords = make_coordinates.create_coordinates(
        implementation, np.ones_like(theta), theta, phi)

    basis = sh.spherical_harmonic_basis_real(n_max, coords)

    inner = (basis @ np.conjugate(basis.T))
    fact = 4 * np.pi / (n_max + 1) ** 2
    orth = np.diagonal(fact * inner)
    np.testing.assert_allclose(orth, np.ones(n_points), rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("implementation", ['spharpy', 'pyfar'])
def test_spherical_harmonic_basis_gradient(make_coordinates, implementation):
    n_max = 15
    theta = np.array([np.pi/2, np.pi/2, 0, np.pi/2, np.pi/4])
    phi = np.array([0, np.pi/2, 0, np.pi/4, np.pi/4])

    coords = make_coordinates.create_coordinates(
        implementation, np.ones_like(theta), theta, phi)

    grad_ele, grad_azi = \
        sh.spherical_harmonic_basis_gradient(n_max, coords)

    desire_ele = np.genfromtxt(
        './tests/data/Y_grad_ele.csv',
        dtype=complex,
        delimiter=',')
    npt.assert_allclose(grad_ele, desire_ele, rtol=1e-10, atol=1e-10)

    desire_azi = np.genfromtxt(
        './tests/data/Y_grad_azi.csv',
        dtype=complex,
        delimiter=',')
    npt.assert_allclose(grad_azi, desire_azi, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("implementation", ['spharpy', 'pyfar'])
def test_spherical_harmonic_basis_gradient_real(
        make_coordinates, implementation):
    n_max = 15
    theta = np.array([np.pi/2, np.pi/2, 0, np.pi/2, np.pi/4])
    phi = np.array([0, np.pi/2, 0, np.pi/4, np.pi/4])

    coords = make_coordinates.create_coordinates(
        implementation, np.ones_like(theta), theta, phi)

    grad_ele, grad_azi = \
        sh.spherical_harmonic_basis_gradient_real(n_max, coords)

    desire_ele = np.genfromtxt(
        './tests/data/Y_grad_real_ele.csv',
        dtype=complex,
        delimiter=',')
    npt.assert_allclose(grad_ele, desire_ele, rtol=1e-10, atol=1e-10)

    desire_azi = np.genfromtxt(
        './tests/data/Y_grad_real_azi.csv',
        dtype=complex,
        delimiter=',')
    npt.assert_allclose(grad_azi, desire_azi, rtol=1e-10, atol=1e-10)
