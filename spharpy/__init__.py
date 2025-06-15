# -*- coding: utf-8 -*-

"""Top-level package for spharpy."""

__author__ = """The pyfar developers"""
__email__ = 'info@pyfar.org'
__version__ = '0.6.2'

from .classes.coordinates import SamplingSphere
from .classes.audio import SphericalHarmonicSignal
from . import spherical
from . import samplings
from . import plot
from . import transforms
from . import beamforming
from . import interpolate
from . import spatial
from . import special


__all__ = [
    'SphericalHarmonicSignal',
    'spherical',
    'samplings',
    'plot',
    'transforms',
    'beamforming',
    'interpolate',
    'spatial',
    'special',
    'SamplingSphere',
]
