"""
Tests for special functions
"""

import pytest
from spharpy import special
from spharpy import samplings

import numpy as np
import numpy.testing as npt


def genfromtxt_complex(filename, delimiter=','):
    """generate complex numpy array from csv file."""
    data_str = np.genfromtxt(filename, delimiter=delimiter, dtype=str)
    mapping = np.vectorize(lambda t: complex(t.replace('i', 'j')))
    return mapping(data_str)


class TestBessel(object):
    def test_shape(self):
        n = np.array([0, 1])
        z = np.linspace(0.1, 5, 10)
        res = special.spherical_bessel(n, z)

        shape = (2, 10)
        assert shape == res.shape

    def test_val(self):
        z = np.linspace(0, 10, 25)
        n = [0, 1, 2]
        res = special.spherical_bessel(n, z)
        truth = np.genfromtxt('./tests/data/bessel.csv', delimiter=',')
        npt.assert_allclose(res, truth)


class TestBesselPrime(object):
    def test_shape(self):
        n = np.array([0, 1])
        z = np.linspace(0.1, 5, 10)
        res = special.spherical_bessel(n, z, derivative=True)

        shape = (2, 10)
        assert shape == res.shape

    def test_val(self):
        z = np.linspace(0.1, 10, 25)
        n = [0, 1, 2]
        res = special.spherical_bessel(n, z, derivative=True)
        truth = np.genfromtxt('./tests/data/bessel_diff.csv', delimiter=',')
        npt.assert_allclose(res, truth)


class TestHankel(object):
    def test_shape(self):
        n = np.array([0, 1])
        z = np.linspace(0.1, 5, 10)
        res = special.spherical_hankel(n, z, kind=1)

        shape = (2, 10)
        assert shape == res.shape

    def test_kind_exception(self):
        with pytest.raises(ValueError):
            special.spherical_hankel([0], [1], kind=3)

    def test_val_second_kind(self):
        z = np.linspace(0.1, 5, 25)
        n = np.array([0, 1, 2])
        res = special.spherical_hankel(n, z, kind=2)
        truth = genfromtxt_complex('./tests/data/hankel_2.csv', delimiter=',')
        npt.assert_allclose(res, truth)

    def test_val_first_kind(self):
        z = np.linspace(0.1, 5, 25)
        n = np.array([0, 1, 2])
        res = special.spherical_hankel(n, z, kind=1)
        truth = genfromtxt_complex('./tests/data/hankel_1.csv', delimiter=',')
        npt.assert_allclose(res, truth)


class TestHankelPrime(object):
    def test_shape(self):
        n = np.array([0, 1])
        z = np.linspace(0.1, 5, 10)
        res = special.spherical_hankel(n, z, kind=1, derivative=True)

        shape = (2, 10)
        assert shape == res.shape

    def test_kind_exception(self):
        with pytest.raises(ValueError):
            special.spherical_hankel([0], [1], kind=3, derivative=True)

    def test_val_second_kind(self):
        z = np.linspace(0.1, 5, 25)
        n = [0, 1, 2]
        res = special.spherical_hankel(n, z, kind=2, derivative=True)
        truth = genfromtxt_complex('./tests/data/hankel_2_diff.csv', delimiter=',')
        npt.assert_allclose(res, truth)

    def test_val_first_kind(self):
        z = np.linspace(0.1, 5, 25)
        n = [0, 1, 2]
        res = special.spherical_hankel(n, z, kind=1, derivative=True)
        truth = genfromtxt_complex('./tests/data/hankel_1_diff.csv', delimiter=',')
        npt.assert_allclose(res, truth)


@pytest.mark.parametrize(['m'], [(-1, ), (0, ), (1, )])
def test_spherical_harmonic_complex(m):
    """
    Test first order complex valued spherical harmonics for selected angels.
    """
    # six positions: front, left, back, right, top, bottom
    pi = np.pi
    azimuth = np.array([0, pi / 2, pi, 3 * pi / 2, 0, 0])
    colatitude = np.array([pi / 2, pi / 2, pi / 2, pi / 2, 0, pi])

    # Manually computed desired values according to
    # Rafaely (2019), Fundamentals of Spherical Array Processing, Table 1.1
    if m == -1:
        desired = np.sqrt(3 / (8 * np.pi)) * \
            np.sin(colatitude) * np.exp(-1j * azimuth)
    if m == 0:
        desired = np.sqrt(3 / (4 * np.pi)) * \
            np.cos(colatitude)
    if m == 1:
        desired = -np.sqrt(3 / (8 * np.pi)) * \
            np.sin(colatitude) * np.exp(1j * azimuth)

    # compute and compare actual values
    actual = special.spherical_harmonic(1, m, colatitude, azimuth)
    npt.assert_almost_equal(actual, desired, 10)


def test_spherical_harmonic_complex_degree_out_of_range():
    """Test if zero is returned if the degree m is larger than the order n"""
    n = 1
    m = [-2, 2]

    npt.assert_equal(special.spherical_harmonic(n, m, 0, 0),
                     np.array([0, 0], dtype=complex))


def test_spherical_harmonic_derivative_theta():
    n_max = 5
    theta = np.array([np.pi/2, np.pi/2, 0, np.pi/2, np.pi/4])
    phi = np.array([0, np.pi/2, 0, np.pi/4, np.pi/4])

    desired_all = np.genfromtxt(
        './tests/data/Y_grad_ele.csv',
        dtype=complex,
        delimiter=',')

    for acn in range((n_max+1)**2):
        n = int((np.ceil(np.sqrt(acn + 1)) - 1))
        m = int(acn - n**2 - n)

        actual = special.spherical_harmonic_derivative_theta(n, m, theta, phi)
        desired = desired_all[:, acn]

        npt.assert_allclose(actual, desired, rtol=1e-10, atol=1e-10)


def test_spherical_harmonic_derivative_phi():
    n_max = 5
    theta = np.array([np.pi/2, np.pi/2, 0, np.pi/2, np.pi/4])
    phi = np.array([0, np.pi/2, 0, np.pi/4, np.pi/4])

    desired_all = np.genfromtxt(
        './tests/data/Y_diff_azi.csv',
        dtype=complex,
        delimiter=',')

    for acn in range((n_max+1)**2):
        n = int((np.ceil(np.sqrt(acn + 1)) - 1))
        m = int(acn - n**2 - n)

        actual = special.spherical_harmonic_derivative_phi(n, m, theta, phi)
        desired = desired_all[:, acn]

        npt.assert_allclose(actual, desired, rtol=1e-10, atol=1e-10)


def test_spherical_harmonic_gradient_phi():
    n_max = 5
    theta = np.array([np.pi/2, np.pi/2, 0, np.pi/2, np.pi/4])
    phi = np.array([0, np.pi/2, 0, np.pi/4, np.pi/4])

    desired_all = np.genfromtxt(
        './tests/data/Y_grad_azi.csv',
        dtype=complex,
        delimiter=',')

    for acn in range((n_max+1)**2):
        n = int((np.ceil(np.sqrt(acn + 1)) - 1))
        m = int(acn - n**2 - n)

        actual = special.spherical_harmonic_gradient_phi(n, m, theta, phi)
        desired = desired_all[:, acn]

        npt.assert_allclose(actual, desired, rtol=1e-10, atol=1e-10)


def test_spherical_harmonic_derivative_theta_real():
    n_max = 5
    theta = np.array([np.pi/2, np.pi/2, 0, np.pi/2, np.pi/4])
    phi = np.array([0, np.pi/2, 0, np.pi/4, np.pi/4])

    desired_all = np.genfromtxt(
        './tests/data/Y_grad_real_ele.csv',
        dtype=float,
        delimiter=',')

    for acn in range((n_max+1)**2):
        n = int((np.ceil(np.sqrt(acn + 1)) - 1))
        m = int(acn - n**2 - n)

        actual = special.spherical_harmonic_derivative_theta_real(
            n, m, theta, phi)
        desired = desired_all[:, acn]

        npt.assert_allclose(actual, desired, rtol=1e-10, atol=1e-10)


def test_spherical_harmonic_derivative_phi_real():
    n_max = 5
    theta = np.array([np.pi/2, np.pi/2, 0, np.pi/2, np.pi/4])
    phi = np.array([0, np.pi/2, 0, np.pi/4, np.pi/4])

    desired_all = np.genfromtxt(
        './tests/data/Y_diff_real_azi.csv',
        dtype=float,
        delimiter=',')

    for acn in range((n_max+1)**2):
        n = int((np.ceil(np.sqrt(acn + 1)) - 1))
        m = int(acn - n**2 - n)

        actual = special.spherical_harmonic_derivative_phi_real(
            n, m, theta, phi)
        desired = desired_all[:, acn]

        npt.assert_allclose(actual, desired, rtol=1e-10, atol=1e-10)


def test_spherical_harmonic_gradient_phi_real():
    n_max = 5
    theta = np.array([np.pi/2, np.pi/2, 0, np.pi/2, np.pi/4])
    phi = np.array([0, np.pi/2, 0, np.pi/4, np.pi/4])

    desired_all = np.genfromtxt(
        './tests/data/Y_grad_real_azi.csv',
        dtype=float,
        delimiter=',')

    for acn in range((n_max+1)**2):
        n = int((np.ceil(np.sqrt(acn + 1)) - 1))
        m = int(acn - n**2 - n)

        actual = special.spherical_harmonic_gradient_phi_real(n, m, theta, phi)
        desired = desired_all[:, acn]

        npt.assert_allclose(actual, desired, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize(['condon_shortley'], [(True, ), (False, )])
@pytest.mark.parametrize(['m'], [(-1, ), (0, ), (1, )])
def test_legendre(condon_shortley, m):
    """
    Test values of the Legendre functions for first order and all degrees.
    """
    z = np.linspace(-1, 1, 11)

    # Manually computed desired values according to
    # Rafaely (2019), Fundamentals of Spherical Array Processing, Table 1.3
    if m == -1:
        desired = .5 * np.sqrt(1 - z**2)
    if m == 0:
        desired = z.copy()
    if m == 1:
        desired =  -np.sqrt(1 - z**2)

    # remove Condon-Shortley phase
    if not condon_shortley and m % 2:
        desired *= -1

    # compute and compare actual values
    actual = special.legendre_function(1, m, z, condon_shortley)
    npt.assert_almost_equal(actual, desired, 10)


@pytest.mark.parametrize(['m'], [(-2, ), (2, )])
def test_legendre_degree_out_of_range(m):
    """Test if zero is returned if the degree m is larger than the order n"""
    n = 1
    z = np.linspace(-1, 1, 11)

    npt.assert_equal(special.legendre_function(n, m, z),
                     np.zeros_like(z))
