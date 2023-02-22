import warnings
from spharpy.samplings import Coordinates as SphCoordinates
from spharpy.samplings import SamplingSphere as SphSamplingSphere
import pyfar as pf

def convert_coordinates(coordinates):
    coords_type = type(coordinates)
    if coords_type is pf.Coordinates:
        if coordinates.sh_order is None:
            return SphCoordinates.from_pyfar(coordinates)
        else:
            return SphSamplingSphere.from_pyfar(coordinates)
    else:
        return coordinates
    # if coords_type is not SphCoordinates and coords_type is not SphSamplingSphere:
    #     return coordinates
    # warnings.warn(
    #     'The use of spharpy Coordinates will be deprecated soon. '
    #     'Use pyfar.Coordinates instead',
    #     DeprecationWarning)
    return coordinates.to_pyfar()
