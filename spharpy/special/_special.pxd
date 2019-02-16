cdef extern from "boost/math/special_functions.hpp" namespace "boost::math":
    double sph_bessel(int n, double z) nogil;
    double sph_bessel_prime(int n, double z) nogil;
    complex sph_hankel_2(int n, double z) nogil;
    complex sph_hankel_1(int n, double z) nogil;
    double sph_neumann(int n, double z) nogil;
    double sph_neumann_prime(int n, double z) nogil;
    double legendre_p(int n, double x) nogil;
    double legendre_p(int n, int m, double x) nogil;
    # double tgamma_ratio(double a, double b) nogil;
    double tgamma_delta_ratio(double a, double delta) nogil;



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


cdef inline double legendre_p_no_cs_phase(int n, int m, double x) nogil:
    """The associated Legendre functions without the Condon-Shortley phase
    term (-1)^m.
    """
    cdef double legendre = legendre_p(n, m, x)

    if m&1:
        legendre = -legendre

    return legendre
