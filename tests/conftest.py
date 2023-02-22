import pytest
from spharpy.samplings import Coordinates
import pyfar as pf
import numpy as np


@pytest.fixture
def make_coordinates():

    class Factory:
        @staticmethod
        def create_coordinates(
                implementation='spharpy', rad=1, theta=np.pi/2, phi=np.pi/2):

            if implementation == 'pyfar':
                return pf.Coordinates(
                    phi, theta, rad, domain='sph', convention='top_colat'
                )
            elif implementation == 'spharpy':
                return Coordinates.from_spherical(rad, theta, phi)

    yield Factory
