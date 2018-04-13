#ifndef BESSEL_FUNCTIONS_H
#define BESSEL_FUNCTIONS_H

#include <boost/math/special_functions.hpp>
#define _USE_MATH_DEFINES
#include <math.h>
#include <complex.h>
#include "special_functions.h"


std::complex<double> *make_modal_strength(int n_max, double *kr, int n_bins, int arrtype);

std::complex<double> modal_strength_open(int n, double kr);
std::complex<double> modal_strength_rigid(int n, double kr);
std::complex<double> modal_strength_cardioid(int n, double kr);

std::complex<double> factor_open(int n);

#endif //BESSEL_FUNCTIONS_H
