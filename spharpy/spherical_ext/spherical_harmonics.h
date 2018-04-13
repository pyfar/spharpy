//
// Created by klein on 2/20/2017.
//

#ifndef SPHERICAL_HARMONICS_H
#define SPHERICAL_HARMONICS_H

#include <boost/math/special_functions/legendre.hpp>
#include <boost/math/special_functions/spherical_harmonic.hpp>

unsigned pyramid2linear(int order, int degree);
unsigned linear2pyramid_order(int linear_index);
int linear2pyramid_degree(int linear_index, unsigned order);
std::complex<double> spherical_harmonic_function_cpp(unsigned order, int degree, double theta, double phi);
void spherical_harmonics_basis_cpp(unsigned maxOrder, unsigned nArgs, double* theta, double* phi, std::complex<double>* y);

std::complex<double>* make_spherical_harmonics_basis(unsigned Nmax, double *theta, double *phi, unsigned n_points);
double* make_spherical_harmonics_basis_real(unsigned n_max, double *theta, double *phi, unsigned n_points);
double spherical_harmonic_function_real_cpp(unsigned n, int m, double theta, double phi);

#endif //SPHERICAL_HARMONICS_H
