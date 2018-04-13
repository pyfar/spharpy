""" Tests for spatial sampling functions """

import pytest
import numpy as np
import spharpy.samplings as samplings

def test_hyperinterpolation():
    n_max = 1
    rad, theta, phi, weights = samplings.hyperinterpolation(n_max)
    assert rad.size == (n_max+1)**2
