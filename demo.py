import spharpy as sph
import numpy as np
import pyfar as pf


n_max = 1
theta = np.array([np.pi/2, np.pi/2, 0], dtype=float)
phi = np.array([0, np.pi/2, 0], dtype=float)
rad = np.ones(3, dtype=float)

coords = pf.Coordinates(phi, theta, rad, domain='sph',
                        convention='top_colat')

np.set_printoptions(precision=14, formatter={'float': lambda x: f"{x:.14e}"})


phase_conventions = [None, 'Condon-Shortley']
normalizations = ['n3d', 'maxN', 'sn3d']
channel_conventions = ['acn', 'fuma']

for p in phase_conventions:
    for n in normalizations:
        for c in channel_conventions:
            print(p, n, c)
            basis = sph.spherical.spherical_harmonic_basis(
                n_max, coordinates=coords, channel_convention=c,
                normalization=n, phase_convention=p)

            np.savetxt(f'./tests/data/Y_cmplx_{p}_{n}_{c}.csv', basis,
                       delimiter=',', fmt='%.13e')

            Y = np.genfromtxt(f'./tests/data/Y_cmplx_{p}_{n}_{c}.csv',
                              dtype=complex, delimiter=',')

            np.testing.assert_allclose(Y, basis, atol=1e-13)

            basis = sph.spherical.spherical_harmonic_basis_real(
                n_max, coordinates=coords, channel_convention=c,
                normalization=n, phase_convention=p)

            np.savetxt(f'./tests/data/Y_real_{p}_{n}_{c}.csv', basis,
                       delimiter=',', fmt='%.13e')

            Y = np.genfromtxt(f'./tests/data/Y_real_{p}_{n}_{c}.csv',
                              dtype=float, delimiter=',')

            np.testing.assert_allclose(Y, basis, atol=1e-13)
