""" Tests for spatial sampling functions """

import pytest
import numpy as np
import spharpy.samplings as samplings
from spharpy.samplings.coordinates import Coordinates


def test_hyperinterpolation():
    n_max = 1
    coords, weights = samplings.hyperinterpolation(n_max)
    assert coords.radius.size == (n_max+1)**2


def test_spherical_t_design():
    order = 2
    coords = samplings.spherical_t_design(order)
    assert isinstance(coords, Coordinates)


def test_dodecahedron():
    coords = samplings.dodecahedron()
    assert isinstance(coords, Coordinates)


def test_icosahedron():
    coords = samplings.icosahedron()
    assert isinstance(coords, Coordinates)


def test_equiangular():
    n_max = 1
    coords = samplings.equiangular(n_max)[0]
    assert isinstance(coords, Coordinates)


def test_gaussian():
    n_max = 1
    coords = samplings.gaussian(n_max)[0]
    assert isinstance(coords, Coordinates)


def test_em32():
    coords = samplings.eigenmike_em32()
    assert isinstance(coords, Coordinates)


def test_icosahedron_ke4():
    coords = samplings.icosahedron_ke4()
    assert isinstance(coords, Coordinates)
