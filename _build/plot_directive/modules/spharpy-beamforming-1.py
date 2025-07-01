import spharpy
import pyfar as pf
import matplotlib.pyplot as plt
import numpy as np
N = 7
steering = pf.Coordinates.from_spherical_colatitude(
    np.linspace(0, 2*np.pi, 500), np.ones(500)*np.pi/2,
    np.ones(500))
Y_steering = spharpy.spherical.spherical_harmonic_basis_real(
    N, steering)
a_nm = spharpy.spherical.spherical_harmonic_basis_real(
    N, pf.Coordinates(1, 0, 0))
R = 10**(50/20)
d_nm = spharpy.beamforming.dolph_chebyshev_weights(
    N, R, design_criterion='sidelobe')
beamformer = np.squeeze(Y_steering @ np.diag(d_nm) @ a_nm.T)
ax = plt.axes(projection='polar')
ax.plot(steering.azimuth, 20*np.log10(np.abs(beamformer)))
ax.set_rticks([-50, -25, 0])
ax.set_theta_zero_location('N')
ax.set_xlabel('Azimuth (deg)')
