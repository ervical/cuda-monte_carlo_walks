#include "random.cuh"
#include "vector.cuh"

__device__ random::random(curandState_t& state)
: state(state)
{}

__device__ float random::operator()(void) {
	return curand_uniform(&state);
}

__device__ float random::operator()(float a, float b) {
	return a + (b - a) * curand_uniform(&state);
}

__device__ float random::operator()(float a) {
	return (*this)(0, a);
}

__device__ float random::get_gamma(int m) {
	float prod = 1;
	float add = 0;
	if (m % 2 == 0) {
		for (int i = 1; i <= m / 2; i++) {
			prod *= (*this)();
		}
		prod = -logf(prod);
	}
	else {
		prod = get_gamma(m - 1);
		add = -logf((*this)()) * powf(cosf((*this)(2 * M_PI)), 2);
	}
	return prod + add;
}

__device__ vector random::get_dir_space() {
	float z = (*this)(-1, 1);
	float r = sqrtf(1 - z*z);
	float alpha = (*this)(2 * M_PI);
	float x = r * cosf(alpha);
	float y = r * sinf(alpha);
	return vector(x, y, z);
}

__device__ vector random::get_dir_semispace(const vector& inner_normal) {
	vector v(0, 0, 0);
	v = get_dir_space();
	if ((v * inner_normal) <
		0) {
		v = -v;
	}
	return v;
}

__device__ float random::get_rho() {
	return powf((*this)(), 1.0f / 3);
}