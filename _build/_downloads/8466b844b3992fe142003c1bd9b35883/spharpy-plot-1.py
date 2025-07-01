import spharpy
import numpy as np
coords = spharpy.samplings.equal_area(n_max=0, n_points=500)
data = np.sin(coords.colatitude) * np.cos(coords.azimuth)
spharpy.plot.balloon(coords, data)
