#include "special_functions.h"

std::complex<double> sph_hankel_2_prime(int n, double z){
	return boost::math::sph_hankel_2(n-1, z) - (n+1)/z * boost::math::sph_hankel_2(n, z);
}

std::complex<double> sph_hankel_1_prime(int n, double z){
	return boost::math::sph_hankel_1(n-1, z) - (n+1)/z * boost::math::sph_hankel_1(n, z);
}

std::complex<double> sph_hankel_2(int n, double z){
	return boost::math::sph_hankel_2(n, z);
}

std::complex<double> sph_hankel_1(int n, double z){
	return boost::math::sph_hankel_1(n, z);
}



double legendre_polynomial(int n, double x){
	return boost::math::legendre_p(n, x);
}

/*
 * Associated Legendre function of order n and degree m evaluated at x
 * */
double legendre_function(int n, int m, double x){
	return boost::math::legendre_p(n, m, x);
}
