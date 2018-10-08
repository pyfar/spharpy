"""
Tests for the aperture function of a spherical loudspeaker array
"""

import sys
from filehandling import read_matrix_from_mat

sys.path.append('./')

import pytest
import spharpy.spherical as sh
import numpy as np

def test_aperture_diag():
    n_max = 10
    rad_cap = 0.1
    rad_sphere = 1
    aperture = sh.aperture_vibrating_spherical_cap(n_max, rad_sphere, rad_cap)
    reference = read_matrix_from_mat('./tests/data/aperture_function.mat')
    np.testing.assert_allclose(aperture, reference)
