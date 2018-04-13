//
// Created by klein on 2/20/2017.
//

#include <boost/math/special_functions/legendre.hpp>
#include <boost/math/special_functions/spherical_harmonic.hpp>
#include <cmath>
#include "spherical_harmonics.h"

unsigned pyramid2linear(int order, int degree)
{
    unsigned linear_index = (unsigned) order * order + order + degree;
    return linear_index;
}

unsigned linear2pyramid_order(int linear_index)
{
    unsigned order = (unsigned) std::ceil(std::sqrt((double) linear_index +1 )) -1;
    return order;
}

int linear2pyramid_degree(int linear_index, unsigned order)
{
    int degree = linear_index - order * order - order;
    return degree;
}

std::complex<double> spherical_harmonic_function_cpp(unsigned order, int degree, double theta, double phi)
{
	if ((unsigned)abs(degree) > order){
		return 0;
	}
    using boost::math::spherical_harmonic;
    std::complex<double> result = spherical_harmonic(order, degree, theta, phi);
    return result;
}

void spherical_harmonics_basis_cpp(unsigned maxOrder, unsigned nArgs, double* theta, double* phi, std::complex<double> *y)
{
    unsigned nCoefficients = (maxOrder+1)*(maxOrder+1);
    unsigned linearCoefficient = 0;
    for (unsigned aa=0; aa<nArgs; aa++) {
        for (unsigned ii=0; ii < nCoefficients; ii++) {
            unsigned order = linear2pyramid_order(ii);
            int degree = linear2pyramid_degree(ii, order);
            y[linearCoefficient] = spherical_harmonic_function_cpp(order, degree, theta[aa], phi[aa]);
            linearCoefficient++;
        }
    }
    return;
}

std::complex<double>* make_spherical_harmonics_basis(unsigned Nmax, double *theta, double *phi, unsigned npoints){
	std::complex<double> *basis = new std::complex<double>[(Nmax+1)*(Nmax+1)*npoints*sizeof(*basis)];
	spherical_harmonics_basis_cpp(Nmax, npoints, theta, phi, basis);
	return basis;
}


double spherical_harmonic_function_real_cpp(unsigned n, int m, double theta, double phi)
{
	double Y_nm = 0.0;
	if (m == 0)
	{
		using boost::math::spherical_harmonic_r;
		Y_nm = spherical_harmonic_r(n, (m), theta, phi);
	}
	else if (m > 0)
	{
		using boost::math::spherical_harmonic_r;
		Y_nm = spherical_harmonic_r(n, (m), theta, phi) * sqrt(2);
	}
	else if (m < 0)
	{
		using boost::math::spherical_harmonic_i;
		Y_nm = spherical_harmonic_i(n, (m), theta, phi) * sqrt(2) * pow(-1, m+1);
	}
	return Y_nm * pow(-1, m);
}


double* make_spherical_harmonics_basis_real(unsigned n_max, double *theta, double *phi, unsigned n_points){
	double *basis = new double[(n_max+1)*(n_max+1)*n_points*sizeof(*basis)];
	unsigned nCoefficients = (n_max+1)*(n_max+1);
    unsigned linearCoefficient = 0;
    for (unsigned aa=0; aa<n_points; aa++) {
        for (unsigned ii=0; ii < nCoefficients; ii++) {
            unsigned order = linear2pyramid_order(ii);
            int degree = linear2pyramid_degree(ii, order);
            basis[linearCoefficient] = spherical_harmonic_function_real_cpp(order, degree, theta[aa], phi[aa]);
            linearCoefficient++;
        }
    }
    return basis;
}
