import pyfar as pf
coords = pf.samplings.sph_lebedev(sh_order=10)
find = pf.Coordinates(1, 0, 0)
index = coords.find_within(find, 1)
coords.show(index)
