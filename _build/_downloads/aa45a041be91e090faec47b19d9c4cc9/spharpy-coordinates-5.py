import pyfar as pf
import numpy as np
coords = pf.Coordinates.from_spherical_elevation(
    np.arange(-30, 30, 5)*np.pi/180, 0, 1)
result = coords.find_slice('azimuth', 'deg', 0, 5, show=True)
