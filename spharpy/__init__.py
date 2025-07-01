# -*- coding: utf-8 -*-

"""Top-level package for spharpy."""

__author__ = """The pyfar developers"""
__email__ = 'info@pyfar.org'
__version__ = '0.6.2'

from .classes.sh import SphericalHarmonics
from .classes.sh import SphericalHarmonicSignal
from .classes.coordinates import SamplingSphere
from . import spherical
from . import samplings
from . import plot
from . import transforms
from . import beamforming
from . import interpolate
from . import spatial
from . import special


__all__ = [
    'SphericalHarmonics',
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
