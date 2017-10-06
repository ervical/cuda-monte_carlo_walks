#pragma once
#define _USE_MATH_DEFINES
#include "device_launch_parameters.h"
#include "curand_kernel.h"

class vector;

class random {

public:
	curandState_t state;

	__device__ random(curandState_t& state);

	// ����������� ������������� �� [0, 1)
	__device__ float operator()(void);

	// ����������� ������������� �� [a, b)
	__device__ float operator()(float a, float b);

	// ����������� ������������� �� [0, a)
	__device__ float operator()(float a);

	// ����� ������������� � ���������� m/2
	__device__ float get_gamma(int m);

	// ���������� ������ � ������������
	__device__ vector get_dir_space();

	// ���������� ������ � ���������������� ������������� ���������� ������� inner_normal
	__device__ vector get_dir_semispace(const vector& inner_normal);

	// �������������, ��������������� ��������� 3x^2
	__device__ float get_rho();
};

