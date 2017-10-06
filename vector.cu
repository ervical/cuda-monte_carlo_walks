#include "vector.cuh"
#include "math.h"

__device__ __host__ void copy(const vector& from, vector& to) {
	to.x = from.x;
	to.y = from.y;
	to.z = from.z;
}

__device__ __host__ vector::vector(float x, float y, float z)
: x(x)
, y(y)
, z(z)
{}

__device__ __host__ vector::vector(const vector& other) {
	copy(other, *this);
}
__device__ __host__ vector vector::operator=(const vector& other) {
	if (this != &other) {
		copy(other, *this);
	}
	return *this;
}

__device__ float vector::get_length() const {
	return sqrtf((*this) * (*this));
}

__device__ vector vector::get_normal() const {
	return 1 / get_length() * (*this);
}

__device__ vector operator*(float a, const vector& v) {
	return vector(a * v.x, a * v.y, a * v.z);
}

__device__ vector operator+(const vector& v1, const vector& v2) {
	return vector(v1.x + v2.x, v1.y + v2.y, v1.z + v2.z);
}

__device__ vector operator-(const vector& v1, const vector& v2) {
	return v1 + (-v2);
}

__device__ vector operator-(const vector& v) {
	return -1 * v;
}

__device__ float operator*(const vector& v1, const vector& v2) {
	return v1.x*v2.x + v1.y*v2.y + v1.z*v2.z;
}