from spharpy.samplings import Coordinates
import pyfar as pf


def convert_coordinates_to_pyfar(coordinates):
    coords_type = type(coordinates)
    if coords_type is not Coordinates:
        return coordinates
    return pf.Coordinates(coordinates.x, coordinates.y, coordinates.z)


def convert_coordinates(coordinates):
    coords_type = type(coordinates)
    if coords_type is not pf.Coordinates:
        return coordinates
    return Coordinates.from_pyfar(coordinates)
