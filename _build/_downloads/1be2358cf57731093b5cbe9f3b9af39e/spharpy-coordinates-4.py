import pyfar as pf
import numpy as np
coords = pf.Coordinates.from_spherical_elevation(
    0, np.arange(-90, 91, 10)*np.pi/180, 1)
result = coords.find_nearest_sph(0, np.pi/2, 1, 45, show=True)
