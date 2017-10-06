#pragma once
#define _USE_MATH_DEFINES
#include "device_launch_parameters.h"
#include "curand_kernel.h"

class vector;

class random {

public:
	curandState_t state;

	__device__ random(curandState_t& state);

	// Равномерное распределение на [0, 1)
	__device__ float operator()(void);

	// Равномерное распределение на [a, b)
	__device__ float operator()(float a, float b);

	// Равномерное распределение на [0, a)
	__device__ float operator()(float a);

	// Гамма распределение с параметром m/2
	__device__ float get_gamma(int m);

	// Изотропный вектор в пространстве
	__device__ vector get_dir_space();

	// Изотропный вектор в полупространстве отностительно внутренней нормали inner_normal
	__device__ vector get_dir_semispace(const vector& inner_normal);

	// Распределение, соответствующее плотности 3x^2
	__device__ float get_rho();
};

