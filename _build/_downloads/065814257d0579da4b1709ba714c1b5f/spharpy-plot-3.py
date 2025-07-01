import spharpy
import numpy as np
coords = spharpy.samplings.equal_area(n_max=0, n_points=500)
data = np.sin(2*coords.colatitude) * np.cos(2*coords.azimuth)
spharpy.plot.contour(coords, data)
