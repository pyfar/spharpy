import numpy as np
import matplotlib.pyplot as plt
from pyfar import Coordinates
import spharpy
spat_res = 30
x_min, x_max = -10, 10
xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, spat_res),
np.linspace(x_min, x_max, spat_res))
receivers = Coordinates(
    xx.flatten(), yy.flatten(), np.zeros(spat_res**2))
doa = Coordinates(10, 15, 0)
plane_wave_matrix = spharpy.spatial.greens_function_point_source(
    doa, receivers, 1, gradient=False)
plt.contourf(
    xx, yy, np.real(plane_wave_matrix.reshape(spat_res, spat_res)),
    cmap='RdBu_r', levels=100)
plt.colorbar()
ax = plt.gca()
ax.set_aspect('equal')
