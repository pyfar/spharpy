import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import spharpy
from matplotlib.collections import LineCollection

phi = np.linspace(0, 2*np.pi, 512)
func = np.exp(1j*phi)

fig, axs = plt.subplots(1, 2, figsize=(12, 4))

points = np.array([np.real(func), np.imag(func)]).T.reshape(-1, 1, 2)
segments = np.concatenate([points[:-1], points[1:]], axis=1)

norm = plt.Normalize(phi.min(), phi.max())
lc = LineCollection(segments, cmap=cm.twilight, norm=norm)
# # Set the values used for colormapping
lc.set_array(phi)
lc.set_linewidth(2)
line = axs[0].add_collection(lc)
axs[0].set_ylim((-1, 1))
axs[0].set_xlim((-1, 1))
axs[0].axis('equal')
fig.colorbar(line, ax=axs[0], label=r"$\varphi$ [rad]")
axs[0].grid(True)
axs[0].set_title("Matplotlib twilight")
axs[0].set_xlabel(r'$\Re\{e^{i\varphi}\}$')
axs[0].set_ylabel(r'$\Im\{e^{i\varphi}\}$')


lc = LineCollection(segments, cmap=spharpy.plot.phase_twilight(), norm=norm)
# # Set the values used for colormapping
lc.set_array(phi)
lc.set_linewidth(2)
line = axs[1].add_collection(lc)
axs[1].set_ylim((-1, 1))
axs[1].set_xlim((-1, 1))
axs[1].axis('equal')
fig.colorbar(line, ax=axs[1], label=r"$\varphi$ [rad]")
axs[1].grid(True)
axs[1].set_title("Spharpy twilight_phase")
axs[1].set_xlabel(r'$\Re\{e^{i\varphi}\}$')
axs[1].set_ylabel(r'$\Im\{e^{i\varphi}\}$')
