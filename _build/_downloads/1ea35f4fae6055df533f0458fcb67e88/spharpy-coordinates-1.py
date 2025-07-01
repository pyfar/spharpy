import pyfar as pf
coords = pf.samplings.sph_lebedev(sh_order=10)
to_find = pf.Coordinates(1, 0, 0)
index, distance = coords.find_nearest(to_find)
coords.show(index)
distance
# Expected:
## 0.0
