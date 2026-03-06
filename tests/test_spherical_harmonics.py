"""
Tests for spherical harmonic basis and related functions.
"""
import spharpy.spherical as sh
import numpy as np
import numpy.testing as npt
import pytest
from pyfar import Coordinates
from spharpy import SamplingSphere


@pytest.mark.parametrize("coordinate_object", [Coordinates, SamplingSphere])
@pytest.mark.parametrize("normalization", ['N3D', 'NM', 'maxN', 'SN3D', 'SNM'])
@pytest.mark.parametrize("channel_convention", ['ACN', 'FuMa'])
@pytest.mark.parametrize("condon_shortley", [True, False, 'auto'])
def test_spherical_harmonic(make_coordinates, coordinate_object,
                            normalization, channel_convention,
                            condon_shortley):
    n_max = 1
    theta = np.array([np.pi/2, np.pi/2, 0], dtype=float)
    phi = np.array([0, np.pi/2, 0], dtype=float)
    rad = np.ones(3, dtype=float)

    coords = make_coordinates.create_coordinates(
        coordinate_object, rad, theta, phi)

    phase_conv_id = 'None' if not condon_shortley else 'Condon-Shortley'

    norm_id = normalization
    if normalization == 'NM' or normalization == 'N3D':
        norm_id = 'n3d'
    if normalization == 'SNM' or normalization == 'SN3D':
        norm_id = 'sn3d'

    Y = np.genfromtxt(f'./tests/data/Y_cmplx_{phase_conv_id}_'
                      f'{norm_id}_{channel_convention.lower()}.csv',
                      dtype=complex,
                      delimiter=',')
    if normalization in ('NM', 'SNM'):
        Y *= np.sqrt(4 * np.pi)

    basis = sh.spherical_harmonic_basis(n_max, coords,
                                        normalization=normalization,
                                        channel_convention=channel_convention,
                                        condon_shortley=condon_shortley)

    np.testing.assert_allclose(Y, basis, atol=1e-13)


@pytest.mark.parametrize("coordinate_object", [Coordinates, SamplingSphere])
@pytest.mark.parametrize("condon_shortley", [True, False, 'auto'])
@pytest.mark.parametrize("channel_convention", ['ACN', 'FuMa'])
@pytest.mark.parametrize("normalization", ['N3D', 'SN3D'])
def test_spherical_harmonics_real(make_coordinates, coordinate_object,
                                  normalization, channel_convention,
                                  condon_shortley):
    n_max = 1
    theta = np.array([np.pi/2, np.pi/2, 0], dtype=float)
    phi = np.array([0, np.pi/2, 0], dtype=float)
    rad = np.ones(3, dtype=float)

    coords = make_coordinates.create_coordinates(
        coordinate_object, rad, theta, phi)

    phase_conv_id = 'Condon-Shortley' if condon_shortley is True else 'None'

    norm_id = normalization
    if normalization == 'NM' or normalization == 'N3D':
        norm_id = 'n3d'
    if normalization == 'SNM' or normalization == 'SN3D':
        norm_id = 'sn3d'

    Y = np.genfromtxt(f'./tests/data/Y_real_{phase_conv_id}_'
                      f'{norm_id}_{channel_convention.lower()}.csv',
                      dtype=float,
                      delimiter=',')

    if normalization in ('NM', 'SNM'):
        Y *= np.sqrt(4 * np.pi)

    basis = sh.spherical_harmonic_basis_real(n_max, coords,
                                             normalization,
                                             channel_convention,
                                             condon_shortley=condon_shortley)
    np.testing.assert_allclose(basis, Y, atol=1e-13)


@pytest.mark.parametrize("coordinate_object", [Coordinates, SamplingSphere])
def test_spherical_harmonics_invalid_nmax(make_coordinates, coordinate_object):
    n_max = 4
    theta = np.array([np.pi/2, np.pi/2, 0], dtype=float)
    phi = np.array([0, np.pi/2, 0], dtype=float)
    rad = np.ones(3, dtype=float)

    coords = make_coordinates.create_coordinates(
        coordinate_object, rad, theta, phi)

    with pytest.raises(ValueError,
                       match='MaxN normalization is only'
                             ' supported up to 3rd order.'):
        sh.spherical_harmonic_basis(n_max, coords,
                                    normalization='maxN')
    with pytest.raises(ValueError,
                       match='MaxN normalization is only'
                             ' supported up to 3rd order.'):
        sh.spherical_harmonic_basis_real(n_max, coords,
                                         normalization='maxN')


@pytest.mark.parametrize("coordinate_object", [Coordinates, SamplingSphere])
def test_spherical_harmonics_invalid_fuma(make_coordinates, coordinate_object):
    n_max = 4
    theta = np.array([np.pi/2, np.pi/2, 0], dtype=float)
    phi = np.array([0, np.pi/2, 0], dtype=float)
    rad = np.ones(3, dtype=float)

    coords = make_coordinates.create_coordinates(
        coordinate_object, rad, theta, phi)

    with pytest.raises(ValueError,
                       match='FuMa channel convention is only'
                             ' supported up to 3rd order.'):
        sh.spherical_harmonic_basis(n_max, coords,
                                    channel_convention='FuMa')
    with pytest.raises(ValueError,
                       match='FuMa channel convention is only'
                             ' supported up to 3rd order.'):
        sh.spherical_harmonic_basis_real(n_max, coords,
                                         channel_convention='FuMa')


@pytest.mark.parametrize("coordinate_object", [SamplingSphere, Coordinates])
def test_spherical_harmonics_invalid_condon_shortley(
        make_coordinates, coordinate_object,
    ):
    n_max = 4
    theta = np.array([np.pi/2, np.pi/2, 0], dtype=float)
    phi = np.array([0, np.pi/2, 0], dtype=float)
    rad = np.ones(3, dtype=float)

    coords = make_coordinates.create_coordinates(
        coordinate_object, rad, theta, phi)

    with pytest.raises(ValueError,
                       match="Condon_shortley has to be a bool, or 'auto'."):
        sh.spherical_harmonic_basis(n_max, coords,
                                    condon_shortley='xx')
    with pytest.raises(ValueError,
                       match="Condon_shortley has to be a bool, or 'auto'."):
        sh.spherical_harmonic_basis_real(n_max, coords,
                                         condon_shortley='xx')


@pytest.mark.parametrize("coordinate_object", [Coordinates, SamplingSphere])
def test_spherical_harmonic_default_n10(make_coordinates, coordinate_object):
    """Test the default parameters of SH basis function generator. This
    simultaneously tests if the methods still match the implementation
    up to spharpy 0.6.2.
    """
    n_max = 10
    theta = np.array([np.pi/2, np.pi/2, 0], dtype=float)
    phi = np.array([0, np.pi/2, 0], dtype=float)

    coords = make_coordinates.create_coordinates(
        coordinate_object, np.ones_like(theta), theta, phi)

    Y = np.genfromtxt(
        './tests/data/sh_basis_cplx_n10.csv',
        delimiter=',',
        dtype=complex)

    basis = sh.spherical_harmonic_basis(n_max, coords)

    np.testing.assert_allclose(Y, basis, atol=1e-13)


@pytest.mark.parametrize("coordinate_object", [Coordinates, SamplingSphere])
def test_spherical_harmonics_real_n10_default(
        make_coordinates, coordinate_object,
    ):
    """Test the default parameters of SH basis function generator. This
    simultaneously tests if the methods still match the implementation
    up to spharpy 0.6.2.
    """
    n_max = 10
    theta = np.array([np.pi/2, np.pi/2, 0, np.pi/2], dtype=float)
    phi = np.array([0, np.pi/2, 0, np.pi/4], dtype=float)
    rad = np.ones(4)

    coords = make_coordinates.create_coordinates(
        coordinate_object, rad, theta, phi)

    Y = np.genfromtxt('./tests/data/sh_basis_real.csv',
                      dtype=float,
                      delimiter=',')
    basis = sh.spherical_harmonic_basis_real(n_max, coords)
    np.testing.assert_allclose(basis, Y, atol=1e-13)


@pytest.mark.parametrize("coordinate_object", [Coordinates, SamplingSphere])
def test_orthogonality(make_coordinates, coordinate_object):
    """
    Check if the orthonormality condition of the spherical harmonics is
    fulfilled.
    """
    n_max = 82
    theta = np.array([np.pi/2, np.pi/2, 0, np.pi/2], dtype=float)
    phi = np.array([0, np.pi/2, 0, np.pi/4], dtype=float)
    n_points = phi.size
    n_points = len(theta)
    coords = make_coordinates.create_coordinates(
        coordinate_object, np.ones_like(theta), theta, phi)
    basis = sh.spherical_harmonic_basis(n_max, coords)

    inner = (basis @ np.conjugate(basis.T))
    fact = 4 * np.pi / (n_max + 1) ** 2
    orth = np.diagonal(fact * inner)
    np.testing.assert_allclose(
        orth, np.ones(n_points, dtype=complex), rtol=1e-10)


@pytest.mark.parametrize("coordinate_object", [Coordinates, SamplingSphere])
def test_orthogonality_real(make_coordinates, coordinate_object):
    """
    Check if the orthonormality condition of the reavl valued spherical
    harmonics is fulfilled.
    """
    n_max = 82
    theta = np.array([np.pi / 2, np.pi / 2, 0, np.pi / 2], dtype='double')
    phi = np.array([0, np.pi / 2, 0, np.pi / 4], dtype='double')
    n_points = phi.size

    coords = make_coordinates.create_coordinates(
        coordinate_object, np.ones_like(theta), theta, phi)

    basis = sh.spherical_harmonic_basis_real(n_max, coords)

    inner = (basis @ np.conjugate(basis.T))
    fact = 4 * np.pi / (n_max + 1) ** 2
    orth = np.diagonal(fact * inner)
    np.testing.assert_allclose(orth, np.ones(n_points), rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("coordinate_object", [Coordinates, SamplingSphere])
def test_spherical_harmonic_basis_gradient(
        make_coordinates, coordinate_object,
    ):
    n_max = 15
    theta = np.array([np.pi/2, np.pi/2, 0, np.pi/2, np.pi/4])
    phi = np.array([0, np.pi/2, 0, np.pi/4, np.pi/4])

    coords = make_coordinates.create_coordinates(
        coordinate_object, np.ones_like(theta), theta, phi)

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


@pytest.mark.parametrize("coordinate_object", [Coordinates, SamplingSphere])
def test_spherical_harmonic_basis_gradient_real(
        make_coordinates, coordinate_object,
    ):
    n_max = 15
    theta = np.array([np.pi/2, np.pi/2, 0, np.pi/2, np.pi/4])
    phi = np.array([0, np.pi/2, 0, np.pi/4, np.pi/4])

    coords = make_coordinates.create_coordinates(
        coordinate_object, np.ones_like(theta), theta, phi)

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


@pytest.mark.parametrize("coordinate_object", [Coordinates, SamplingSphere])
def test_spherical_harmonics_gradient_invalid_nmax(
        make_coordinates, coordinate_object,
    ):
    n_max = 15
    theta = np.array([np.pi/2, np.pi/2, 0, np.pi/2, np.pi/4])
    phi = np.array([0, np.pi/2, 0, np.pi/4, np.pi/4])

    coords = make_coordinates.create_coordinates(
        coordinate_object, np.ones_like(theta), theta, phi)

    with pytest.raises(ValueError,
                       match='MaxN normalization is only'
                             ' supported up to 3rd order.'):
        sh.spherical_harmonic_basis_gradient(n_max, coords,
                                             normalization='maxN')
    with pytest.raises(ValueError,
                       match='MaxN normalization is only'
                             ' supported up to 3rd order.'):
        sh.spherical_harmonic_basis_gradient_real(n_max, coords,
                                                  normalization='maxN')


@pytest.mark.parametrize("coordinate_object", [Coordinates, SamplingSphere])
def test_spherical_harmonics_gradient_invalid_fuma(
        make_coordinates, coordinate_object,
    ):
    n_max = 15
    theta = np.array([np.pi/2, np.pi/2, 0, np.pi/2, np.pi/4])
    phi = np.array([0, np.pi/2, 0, np.pi/4, np.pi/4])

    coords = make_coordinates.create_coordinates(
        coordinate_object, np.ones_like(theta), theta, phi)

    with pytest.raises(ValueError,
                       match='FuMa channel convention is only'
                             ' supported up to 3rd order.'):
        sh.spherical_harmonic_basis_gradient(n_max, coords,
                                             channel_convention='FuMa')
    with pytest.raises(ValueError,
                       match='FuMa channel convention is only'
                             ' supported up to 3rd order.'):
        sh.spherical_harmonic_basis_gradient_real(n_max, coords,
                                                  channel_convention='FuMa')
