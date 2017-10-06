#pragma once
#include "device_launch_parameters.h"

class vector {
public:
	float x, y, z;

	__device__ __host__ vector(float x, float y, float z);
	__device__ __host__ vector(const vector& other);
	__device__ __host__ vector operator=(const vector& other);

	// ����� �������
	__device__ float get_length() const;

	// ����������������� ������
	__device__ vector get_normal() const;
};

// ��������� ������ �� ����� �����
__device__ vector operator*(float a, const vector& v);

// �������� ��������
__device__ vector operator+(const vector& v1, const vector& v2);

// �������� ��������
__device__ vector operator-(const vector& v1, const vector& v2);

// ������� �����
__device__ vector operator-(const vector& v);

// ��������� ������������
__device__ float operator*(const vector& v1, const vector& v2);

// ����������� ���� ����� ��� ���� ������� 
typedef vector point;