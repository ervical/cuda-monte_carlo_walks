#pragma once
#include "device_launch_parameters.h"

class vector {
public:
	float x, y, z;

	__device__ __host__ vector(float x, float y, float z);
	__device__ __host__ vector(const vector& other);
	__device__ __host__ vector operator=(const vector& other);

	// Длина вектора
	__device__ float get_length() const;

	// Нормализированный вектор
	__device__ vector get_normal() const;
};

// Умножение вектор на число слева
__device__ vector operator*(float a, const vector& v);

// Сложение векторов
__device__ vector operator+(const vector& v1, const vector& v2);

// Разность векторов
__device__ vector operator-(const vector& v1, const vector& v2);

// Унарный минус
__device__ vector operator-(const vector& v);

// Скалярное произведение
__device__ float operator*(const vector& v1, const vector& v2);

// Определение типа точки как типа вектора 
typedef vector point;