"""
Test spharpy SH definition against common definitions of others.

data/common_sh_defintions.m generated SH matrices for definitions from
Boaz Rafaely (which equals that of Williams and Polits) and Jens Ahrens using
the Matlab code provided by the authors themselves and stores data in the
mat-file of the same name.
"""
from spharpy import SphericalHarmonics
from spharpy import SamplingSphere
import pytest
import scipy as sp
import numpy as np
import numpy.testing as npt
import os

# load data generated with reference implementations
reference = sp.io.loadmat(os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    'data', 'common_sh_definitions.mat'))

# create SamplingSphere from reference data
n_max = np.max(reference["acn_order_n"])
sampling = SamplingSphere().from_spherical_colatitude(
    reference["azimuth"].flatten(), reference["colatitude"].flatten(), 1,
    n_max, radius_tolerance=1e-3)

# get reference SH matrices
Y_rafaely = reference["Y_rafaely"]
Y_ahrens = reference["Y_ahrens"]
Y_sofa = reference["Y_sofa"]


@pytest.mark.parametrize(
        ("basis_type", "normalization", "channel_convention",
         "condon_shortley", "Y"),
        [("complex", "n3d", "acn", False, Y_rafaely),
         ("complex", "n3d", "acn", True, Y_ahrens),
         ("real", "n3d", "acn", False, Y_sofa),])
def test_definitions(
    basis_type, normalization, channel_convention, condon_shortley, Y):

    Y_spharpy = SphericalHarmonics(
        n_max, sampling, basis_type, normalization, channel_convention,
        "auto", condon_shortley)

    npt.assert_almost_equal(Y_spharpy.basis, Y, 3)
