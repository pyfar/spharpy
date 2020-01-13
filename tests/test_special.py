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


def test_spherical_harmonic_derivative_theta():
    n_max = 5
    theta = np.array([np.pi/2, np.pi/2, 0, np.pi/2, np.pi/4])
    phi = np.array([0, np.pi/2, 0, np.pi/4, np.pi/4])
    n_points = np.size(theta)

    desired_all = np.genfromtxt(
        './tests/data/Y_grad_ele.csv',
        dtype=np.complex,
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
    n_points = np.size(theta)

    desired_all = np.genfromtxt(
        './tests/data/Y_diff_azi.csv',
        dtype=np.complex,
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
    n_points = np.size(theta)

    desired_all = np.genfromtxt(
        './tests/data/Y_grad_azi.csv',
        dtype=np.complex,
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
    n_points = np.size(theta)

    desired_all = np.genfromtxt(
        './tests/data/Y_grad_real_ele.csv',
        dtype=np.double,
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
    n_points = np.size(theta)

    desired_all = np.genfromtxt(
        './tests/data/Y_diff_real_azi.csv',
        dtype=np.double,
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
    n_points = np.size(theta)

    desired_all = np.genfromtxt(
        './tests/data/Y_grad_real_azi.csv',
        dtype=np.double,
        delimiter=',')

    for acn in range((n_max+1)**2):
        n = int((np.ceil(np.sqrt(acn + 1)) - 1))
        m = int(acn - n**2 - n)

        actual = special.spherical_harmonic_gradient_phi_real(n, m, theta, phi)
        desired = desired_all[:, acn]

        npt.assert_allclose(actual, desired, rtol=1e-10, atol=1e-10)
