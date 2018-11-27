""" Tests for spatial sampling functions """

import pytest
import numpy as np
import spharpy.samplings as samplings
from spharpy.samplings.coordinates import Coordinates, SamplingSphere


def test_hyperinterpolation():
    n_max = 1
    sampling = samplings.hyperinterpolation(n_max)
    assert sampling.radius.size == (n_max+1)**2


def test_spherical_t_design():
    order = 2
    coords = samplings.spherical_t_design(order)
    assert isinstance(coords, SamplingSphere)


def test_dodecahedron():
    sampling = samplings.dodecahedron()
    assert isinstance(sampling, SamplingSphere)


def test_icosahedron():
    sampling = samplings.icosahedron()
    assert isinstance(sampling, SamplingSphere)


def test_equiangular():
    n_max = 1
    sampling = samplings.equiangular(n_max)
    assert isinstance(sampling, SamplingSphere)


def test_gaussian():
    n_max = 1
    sampling = samplings.gaussian(n_max)
    assert isinstance(sampling, SamplingSphere)


def test_em32():
    sampling = samplings.eigenmike_em32()
    assert isinstance(sampling, SamplingSphere)


def test_icosahedron_ke4():
    sampling = samplings.icosahedron_ke4()
    assert isinstance(sampling, SamplingSphere)
