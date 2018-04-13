#include "bessel_functions.h"
#include "spherical_harmonics.h"

const std::complex<double> i(0.0,1.0);

std::complex<double> *make_modal_strength(int n_max, double *kr, int n_bins, int arrtype){
	std::complex<double> *bn;
	int n_sh = (n_max+1)*(n_max+1);
	bn = new std::complex<double>[n_bins*n_sh*n_sh*sizeof(*bn)];
	
	std::complex<double> bn_order;

	for (int k = 0; k < n_bins; k++){
		for (int n = 0; n <= n_max; n++){
			if (arrtype == 0){
				bn_order = modal_strength_open(n, kr[k]);
			}
			else if(arrtype == 1){
				bn_order = modal_strength_rigid(n, kr[k]);
			}
			else if(arrtype == 2){
				bn_order = modal_strength_cardioid(n, kr[k]);
			}
			else{
				bn_order = -1;
			}
			for (int m = -n; m <= n; ++m){
				int acn = pyramid2linear(n, m);
				int idxarr = acn + n_sh*(k*n_sh + acn);
				bn[idxarr] = bn_order;
			}
		}
	}
	return bn;
}

std::complex<double> modal_strength_open(int n, double kr){
	return boost::math::sph_bessel(n, kr) * factor_open(n);
}

std::complex<double> modal_strength_rigid(int n, double kr){
    std::complex<double> fact = 4 * M_PI * pow(i, n-1) / (kr*kr);
	return fact / sph_hankel_2_prime(n, kr);
}

std::complex<double> modal_strength_cardioid(int n, double kr){
    return (boost::math::sph_bessel(n, kr) - i * boost::math::sph_bessel_prime(n, kr)) * factor_open(n);
}

std::complex<double> factor_open(int n){
    return 4 * M_PI* pow(i, n);
}


