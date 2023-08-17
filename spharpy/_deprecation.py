from spharpy.samplings import Coordinates
import pyfar as pf


def convert_coordinates(coordinates):
    coords_type = type(coordinates)
    if coords_type is not Coordinates:
        return coordinates
    return pf.Coordinates(coordinates.x, coordinates.y, coordinates.z)
