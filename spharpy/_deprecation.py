from spharpy.samplings import Coordinates
from spharpy.samplings import SamplingSphere
import pyfar as pf


def convert_coordinates(coordinates):
    coords_type = type(coordinates)
    if coords_type is not pf.Coordinates:
        return coordinates
    if coordinates.sh_order is None:
        return Coordinates.from_pyfar(coordinates)
    else:
        return SamplingSphere.from_pyfar(coordinates)
