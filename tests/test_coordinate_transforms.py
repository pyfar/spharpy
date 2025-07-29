"""Tests for coordinate transforms."""

import pytest
import numpy as np
import pyfar as pf
import spharpy.samplings as samplings
from spharpy.samplings import spherical_voronoi


def test_cart2latlon_array():
    x = np.array([1, -1, 0, 0, 0, 0])
    y = np.array([0, 0, 1, -1, 0, 0])
    z = np.array([0, 0, 0, 0, 1, -1])
    coords = pf.Coordinates(x, y, z)
    rr, tt, pp = samplings.helpers.coordinates2latlon(coords)

    rad = np.ones(6)
    theta = np.array([0, 0, 0, 0, np.pi/2, -np.pi/2])
    phi = np.array([0, np.pi, np.pi/2, -np.pi/2, 0, 0])

    np.testing.assert_allclose(rad, rr, atol=1e-15)
    np.testing.assert_allclose(phi, pp, atol=1e-15)
    np.testing.assert_allclose(theta, tt, atol=1e-15)


def test_sph_voronoi():
    s = samplings.dodecahedron()
    verts = np.array([[ 8.72677996e-01, -3.56822090e-01,  3.33333333e-01],
                      [ 3.33333333e-01, -5.77350269e-01,  7.45355992e-01],
                      [ 7.45355992e-01, -5.77350269e-01, -3.33333333e-01],
                      [ 8.72677996e-01,  3.56822090e-01,  3.33333333e-01],
                      [-8.72677996e-01, -3.56822090e-01, -3.33333333e-01],
                      [-1.27322004e-01, -9.34172359e-01,  3.33333333e-01],
                      [-7.45355992e-01, -5.77350269e-01,  3.33333333e-01],
                      [ 1.27322004e-01, -9.34172359e-01, -3.33333333e-01],
                      [-3.33333333e-01, -5.77350269e-01, -7.45355992e-01],
                      [-8.72677996e-01,  3.56822090e-01, -3.33333333e-01],
                      [ 0.00000000e+00,  0.00000000e+00, -1.00000000e+00],
                      [ 6.66666667e-01, -1.91105568e-16, -7.45355992e-01],
                      [ 7.45355992e-01,  5.77350269e-01, -3.33333333e-01],
                      [-3.33333333e-01,  5.77350269e-01, -7.45355992e-01],
                      [ 1.27322004e-01,  9.34172359e-01, -3.33333333e-01],
                      [-6.66666667e-01,  2.46373130e-16,  7.45355992e-01],
                      [ 0.00000000e+00,  0.00000000e+00,  1.00000000e+00],
                      [ 3.33333333e-01,  5.77350269e-01,  7.45355992e-01],
                      [-1.27322004e-01,  9.34172359e-01,  3.33333333e-01],
                      [-7.45355992e-01,  5.77350269e-01,  3.33333333e-01]])

    sv = spherical_voronoi(s)
    np.testing.assert_allclose(
        np.sort(np.sum(verts, axis=-1)),
        np.sort(np.sum(sv.vertices, axis=-1)),
        atol=1e-6,
        rtol=1e-6)
