import pyfar as pf
coords = pf.Coordinates(np.arange(6), 0, 0)
find = pf.Coordinates([2, 3], 0, 0)
index = coords.find_within(find, 1)
coords.show(index[0])
