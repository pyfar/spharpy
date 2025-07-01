import pyfar as pf
coords = pf.Coordinates(np.arange(-5, 5), 0, 0)
result = coords.find_nearest_cart(2, 0, 0, 0.5, show=True)
