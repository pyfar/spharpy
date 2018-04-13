#ifndef SPECICAL_FUNCTIONS_H
#define SPECICAL_FUNCTIONS_H

#include <boost/math/special_functions.hpp>
#define _USE_MATH_DEFINES
#include <math.h>
#include <complex.h>

std::complex<double> sph_hankel_2_prime(int n, double z);
std::complex<double> sph_hankel_1_prime(int n, double z);
std::complex<double> sph_hankel_2(int n, double z);
std::complex<double> sph_hankel_1(int n, double z);

double legendre_polynomial(int n, double x);
double legendre_function(int n, int m, double x);


#endif // SPECICAL_FUNCTIONS_H
