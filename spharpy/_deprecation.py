import warnings
from spharpy.samplings import Coordinates as SphCoordinates
from spharpy.samplings import SamplingSphere


def coordinate_deprecation_warning(coordinates):
    coords_type = type(coordinates)
    if coords_type is SphCoordinates or coords_type is SamplingSphere:
        warnings.warn(
            'The use of spharpy Coordinates will be deprecated soon. '
            'Use pyfar.Coordinates instead',
            DeprecationWarning)
        return coordinates.to_pyfar()
    else:
        return coordinates
