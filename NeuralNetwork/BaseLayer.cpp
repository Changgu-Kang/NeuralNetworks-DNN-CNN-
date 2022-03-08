#include "BaseLayer.h"


void iniWrandom(double* w, int w_size)
{
	// If you want to use uniform distribution,
	for (int i = 0; i < w_size; i++)
	{
		w[i] = (double)rand() / RAND_MAX * 0.1;
		w[i] *= 0.1;

		if (isnan(w[i]))
		{
			printf("W ini break\n");
		}
	}
}

void iniWxavier(double* w, int w_size, int n_size)
{
	double n = n_size;
	double factor = 2.0;
	double limit = sqrt(3.0*factor / n);
	unsigned int seed = std::chrono::steady_clock::now().time_since_epoch().count();
	std::default_random_engine e(seed);
	std::uniform_real_distribution<double> distrR(-limit, limit);			
	//double trunc_stddev = sqrt(1.3 * factor / n);
	//std::normal_distribution<double> distrR(0.0, trunc_stddev);

	for (int i = 0; i < w_size; i++) { w[i] = distrR(e); }

}