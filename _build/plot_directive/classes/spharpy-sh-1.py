import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colorbar import Colorbar
import spharpy

n_max = 3
sampling = spharpy.samplings.equal_area(25, condition_num=np.inf)
Y_real = spharpy.spherical.spherical_harmonic_basis_real(n_max, sampling)

fig = plt.figure(figsize=(12, 8))
gs = plt.GridSpec(4, 5, height_ratios=[1, 1, 1, 0.1], width_ratios=[1, 1, 1, 1, 1])
for acn in range((n_max+1)**2):
    n, m = spharpy.spherical.acn_to_nm(acn)
    idx_m = int(np.floor(n_max/2+1)) + m
    ax = plt.subplot(gs[n, idx_m], projection='3d')
    balloon = spharpy.plot.balloon_wireframe(sampling, Y_real[:, acn], phase=True, colorbar=False, ax=ax)
    ax.set_title('$Y_{' + str(n) + '}^{' + str(m) + '}(\\theta, \\phi)$')
    plt.axis('off')
cax = plt.subplot(gs[n_max+1, :])

cnorm = plt.Normalize(0, 2*np.pi)
cmappable = mpl.cm.ScalarMappable(cnorm, spharpy.plot.phase_twilight())
cmappable.set_array(np.linspace(0, 2*np.pi, 128))

cb = Colorbar(ax=cax, mappable=cmappable, orientation='horizontal', ticklocation='bottom')
cb.set_label('Phase in rad')
cb.set_ticks(np.linspace(0, 2*np.pi, 5))
cb.set_ticklabels(['$0$', '$\pi/2$', '$\pi$', '$3\pi/2$', '$2\pi$'])