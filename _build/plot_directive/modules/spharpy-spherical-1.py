import spharpy
import matplotlib.pyplot as plt
n_max = 2
E = spharpy.spherical.sph_identity_matrix(n_max, type='n-nm')
plt.matshow(E, cmap=plt.get_cmap('Greys'))
plt.gca().set_aspect('equal')
