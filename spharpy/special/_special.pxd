cdef extern from "boost/math/special_functions.hpp" namespace "boost::math":
    double sph_bessel(int n, double z) nogil;
    double sph_bessel_prime(int n, double z) nogil;
    complex sph_hankel_2(int n, double z) nogil;
    complex sph_hankel_1(int n, double z) nogil;
    double sph_neumann(int n, double z) nogil;
    double sph_neumann_prime(int n, double z) nogil;
    double legendre_p(int n, double x) nogil;


cdef inline complex sph_hankel_2_prime(int n, double z) nogil:
    """Derivative of the spherical hankel function of second kind.
    Not defined in the boost libs, therefore defined here."""
    cdef complex hankel_2_prime
    hankel_2_prime = sph_hankel_2(n-1, z) - (n+1)/z * sph_hankel_2(n, z)
    return hankel_2_prime


cdef inline complex sph_hankel_1_prime(int n, double z) nogil:
    """Derivative of the spherical hankel function of first kind.
    Not defined in the boost libs, therefore defined here."""
    cdef complex hankel_1_prime
    hankel_1_prime = sph_hankel_1(n-1, z) - (n+1)/z * sph_hankel_1(n, z)
    return hankel_1_prime
