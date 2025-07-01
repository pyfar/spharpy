import pyfar as pf
coords = pf.Coordinates(np.arange(-5, 5), 0, 0)
result = coords.find_nearest_k(0, 0, 0, show=True)
