import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cmap

import spharpy
import cartopy.crs as ccrs

sampling = spharpy.samplings.equalarea(20, condition_num=np.inf)
# sampling = spharpy.samplings.hyperinterpolation(15)
Y = spharpy.spherical.spherical_harmonic_basis_real(15, sampling)

# plt.figure()
# spharpy.plot.contour_map(sampling, (Y[:, spharpy.spherical.nm2acn(4, 2)]), cmap=cmap.Spectral_r)
#
# plt.figure()
# spharpy.plot.contour_map(sampling, (Y[:, 4]), cmap=cmap.RdBu_r, projection=ccrs.Mollweide())


# plt.figure(figsize=(5, 5))
# plot = spharpy.plot.balloon_wireframe(sampling, Y[:, 1], phase=False, cmap=cmap.RdBu_r)



# plt.figure(figsize=(5,5))
# plot = spharpy.plot.balloon(sampling, Y[:, 1], phase=False, cmap=cmap.RdBu_r)


# plt.figure(figsize=(5,5))
# plot = spharpy.plot.balloon_wireframe(sampling, Y[:, 7], phase=False, cmap=cmap.RdBu_r, colorbar=False)


# spharpy.plot.contour_map(sampling, Y[:, 7], cmap=cmap.RdBu_r, projection=ccrs.Sinusoidal())
plt.figure()
spharpy.plot.contour(sampling, Y[:, 7], cmap=cmap.RdBu_r)

plt.figure()
spharpy.plot.contour_map(sampling, Y[:, 7], cmap=cmap.RdBu_r)

plt.figure()
spharpy.plot.pcolor_map(sampling, Y[:, 7], cmap=cmap.RdBu_r, limits=[-0.3, 1])

# plt.figure()
# spharpy.plot.contour_map(sampling, Y[:, 2], cmap=cmap.RdBu_r, projection=ccrs.Sinusoidal())
#
# plt.figure()
# spharpy.plot.contour_map(sampling, Y[:, 2], cmap=cmap.RdBu_r, projection=ccrs.PlateCarree())



#
# plt.figure(figsize=(5,5))
# spharpy.plot.balloon_interp(sampling, Y[:, 1])

#
# Y = spharpy.spherical.spherical_harmonic_basis(15, sampling)
#
# plt.figure(figsize=(5,5))
# plot = spharpy.plot.balloon_wireframe(sampling, Y[:, 1], phase=False,colorbar=False)
#
# plt.figure(figsize=(5,5))
# plot = spharpy.plot.balloon(sampling, Y[:, 1], phase=True)
#
#
