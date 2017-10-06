#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "random.cuh"
#include "vector.cuh"
#include "curand_kernel.h"

#include <iostream>
#include <ctime>

#define BLOCKS 16
#define THREADS 16
#define N_PER_THREAD 1000

#define ID blockDim.x * blockIdx.x + threadIdx.x

// Вспомогательные структуры метода Монте-Карло

struct monte_carlo_args {
	const point p0;
	const float t0;

	__device__ __host__ monte_carlo_args(const point& p0, float t0)
		: p0(p0)
		, t0(t0)
	{}

};

struct monte_carlo_result {
	const float value;
	const float error;

	__host__ monte_carlo_result(float value, float error)
		: value(value)
		, error(error)
	{}
};

// Область (Шар с радиусом RAD)

#define RAD 10

__device__ float dist(const point& x) {
	return RAD - x.get_length();
}

__device__ float dist(const point& x, const vector& omega) {
	float b = x*omega;
	float c = x*x - RAD * RAD;
	float d_4 = b*b - c;
	float t1 = -b + sqrtf(d_4);
	float t2 = -b - sqrtf(d_4);
	return fmaxf(t1, t2);
}

__device__ float semi_dist(const point& x, const vector& omega) {
	return -2 * (x * omega);
}

// Задача

__device__ float f(const point& p, float t) {
	return 0;
}

__device__ float PHI(const point& p, float t) {
	return t / 6;
}

__device__ float phi(const point& p) {
	return 0;
}


// Блуждание по границе

__device__ float u1(const point& p, float t, random& rnd) {
	float gamma = rnd.get_gamma(3);
	float theta = rnd();
	vector omega = rnd.get_dir_space();
	float r = dist(p, omega);
	if (gamma <= r*r / (4 * t * theta)) {
		return t * f(p + 2 * sqrtf(gamma * t * theta) * omega, t - t * theta);
	} else {
		return 0;
	}
}

__device__ float u1_border(const point& p, float t, random& rnd) {
	float gamma = rnd.get_gamma(3);
	float theta = rnd();
	vector omega1 = rnd.get_dir_semispace(-p.get_normal());
	float r = semi_dist(p, omega1);
	if (gamma <= r*r / (4 * t * theta)) {
		return t * f(p + 2 * sqrtf(gamma * t * theta) * omega1, t - t * theta) / 2;
	}
	else {
		return 0;
	}
}

__device__ float u2(const point& p, float t, random& rnd) {
	float gamma = rnd.get_gamma(3);
	vector omega = rnd.get_dir_space();
	float r = dist(p, omega);
	if (gamma <= r*r / (4 * t)) {
		return phi(p + 2 * sqrtf(gamma * t) * omega);
	}
	else {
		return 0;
	}
}

__device__ float u2_border(const point& p, float t, random& rnd) {
	float gamma = rnd.get_gamma(3);
	vector omega1 = rnd.get_dir_semispace(-p.get_normal());
	float r = semi_dist(p, omega1);
	if (gamma <= r*r / (4 * t)) {
		return phi(p + 2 * sqrtf(gamma * t) * omega1) / 2;
	}
	else {
		return 0;
	}
}

__device__ float u3(const point& p0, float t0, random& rnd) {
	vector omega = rnd.get_dir_space();
	float r = dist(p0, omega);
	float gamma = rnd.get_gamma(3);
	point x = p0 + r*omega;
	float t = t0 - r*r / (4 * gamma);
	float sum = 0;
	int k = 0;
	while (t > 0) {
		sum += powf(-1, k) * 2 * (PHI(x, t) - u1_border(x, t, rnd) - u2_border(x, t, rnd));
		k++;
		omega = rnd.get_dir_semispace(-x.get_normal());
		r = semi_dist(x, omega);
		gamma = rnd.get_gamma(3);
		x = x + r*omega;
		t = t - r*r / (4 * gamma);
	}
	return sum;
}

__device__ float walk_in_border(const point& p0, float t0, random& rnd) {
	if (dist(p0) < 0) {
		return 0;
	}
	if (t0 == 0) {
		return phi(p0);
	}
	if (dist(p0) == 0) {
		return PHI(p0, t0);
	}
	return u1(p0, t0, rnd) + u2(p0, t0, rnd) + u3(p0, t0, rnd);
}

// Блуждание по цилиндрам

__device__ float walk_in_cylinder(const point& p0, float t0, random& rnd) {

	point p = p0;
	float t = t0;
	float xi = 0;

	if (t == 0) {
		return phi(p);
	}

	while (true) {
		float r = dist(p);
		if (r < 10E-5f) {
			xi += PHI(p, t);
			break;
		}

		float tetta = rnd();
		float gamma = rnd.get_gamma(5);
		float ro = rnd.get_rho();
		vector omega = rnd.get_dir_space();

		if (0 < 6 * t && 6 * t < r * r) {
			if (gamma < r * r / (4 * t * tetta)) {
				xi += t * f(p + 2 * ro * sqrtf(t * tetta * gamma) * omega, t - t * tetta);
			}
			if (gamma <= r * r / (4 * t)) {
				xi += phi(p + 2 * ro * sqrtf(t * gamma) * omega);
				break;
			}
			else {
				if (tetta > 3.0f / (2 * gamma)) {
					p = p + r * ro * omega;
				}
				else {
					p = p + r * omega;
				}
				t = t - r * r / (4 * gamma);
			}

		}
		else {
			if (gamma <= 3.0f / (2 * tetta)) {
				xi += r * r / 6 * f(p + r * ro * sqrtf(2 * tetta * gamma / 3) * omega, t - r * r / 6 * tetta);
			}
			if (gamma > 3.0f / 2 && tetta > 3.0f / (2 * gamma)) {
				p = p + r * ro * omega;
				t = t - r * r / (4 * gamma);
			}
			if (gamma > 3.0f / 2 && tetta <= 3.0f / (2 * gamma)) {
				p = p + r * omega;
				t = t - r * r / (4 * gamma);
			}
			if (gamma <= 3.0f / 2) {
				p = p + r * ro * sqrtf(2 * gamma / 3) * omega;
				t = t - r * r / 6;
			}
		}
	}
	return xi;
}

// Монте-Карло

__device__ float get_xi(monte_carlo_args * args, random& rnd) {
	return walk_in_border(args->p0, args->t0, rnd);
	//return walk_in_cylinder(args->p0, args->t0, rnd);
}

__global__ void monte_carlo(curandState * states, float * d_sum, float * d_sum_sqr, monte_carlo_args * args) {

	// Разделяемая память для суммирования оценки решения и ее квадратов
	__shared__ float tmp_sum[THREADS];
	__shared__ float tmp_sum_sqr[THREADS];

	// Обертка датчика случайных чисел
	random rnd(states[ID]);

	// Вычисление N_PER_THREAD оценок решения
	for (int i = 0; i < N_PER_THREAD; i++) {
		float xi = get_xi(args, rnd); // (11.1)
		tmp_sum[threadIdx.x] += xi;
		tmp_sum_sqr[threadIdx.x] += xi * xi;
	}
	
	// Усреднение по N_PER_THREAD
	tmp_sum[threadIdx.x] /=  N_PER_THREAD;
	tmp_sum_sqr[threadIdx.x] /=  N_PER_THREAD;
	__syncthreads();

	// Редукция
	int i = THREADS / 2;
	while (i != 0) {
		if (threadIdx.x < i) {
			tmp_sum[threadIdx.x] += tmp_sum[threadIdx.x + i];
			tmp_sum_sqr[threadIdx.x] += tmp_sum_sqr[threadIdx.x + i];	
		}
		if (i > 32) {
			__syncthreads();
		}
		i /= 2;
	}

	// Сохранение средного по THREADS * N_PER_THREAD  в данном блоке
	if (threadIdx.x == 0) {
		d_sum[blockIdx.x] = tmp_sum[0] / THREADS;
		d_sum_sqr[blockIdx.x] = tmp_sum_sqr[0] / THREADS;
	}
}

monte_carlo_result get_result(float * d_sum, float * d_sum_sqr) {

	// Копирование результатов в оперативную память
	float h_sum[BLOCKS];
	float h_sum_sqr[BLOCKS];
	cudaMemcpy(h_sum, d_sum, sizeof(float)* BLOCKS, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_sum_sqr, d_sum_sqr, sizeof(float)* BLOCKS, cudaMemcpyDeviceToHost);

	// Усреднение по блокам
	float sum = 0;
	float sum_sqr = 0;
	for (int i = 0; i < BLOCKS; i++) {
		sum += h_sum[i];
		sum_sqr += h_sum_sqr[i];
	}
	sum /= BLOCKS;
	sum_sqr /= BLOCKS;

	// Результат
	float value = sum;
	
	// Оценка погрешности
	float error = 3.0f * sqrtf((sum_sqr - value * value) / BLOCKS / THREADS / N_PER_THREAD);

	return monte_carlo_result(value, error);
}

__global__ void device_random_init(curandState* states, unsigned int seed = 0) {
	curand_init(seed, ID, 0, &states[ID]);
}

// Основаная программа
// Вычиление для поля 20x20, с шаром (RAD=10) в центре
int main() {
	freopen("out.txt", "wt", stdout);
		for (int x = -10; x <= 10; x++) {
			for (int y = -10; y <= 10; y++) {
				curandState * d_states;
				cudaMalloc((void**)&d_states, sizeof(curandState)* BLOCKS * THREADS);
				device_random_init << < BLOCKS, THREADS >> >(d_states);
				cudaDeviceSynchronize();

				float * d_sum;
				float * d_sum_sqr;
				cudaMalloc((void**)&d_sum, sizeof(float)* BLOCKS);
				cudaMalloc((void**)&d_sum_sqr, sizeof(float)* BLOCKS);

				monte_carlo_args h_args(point(x, y, 0), 80.f);
				monte_carlo_args * d_args;
				cudaMalloc((void**)&d_args, sizeof(monte_carlo_args));
				cudaMemcpy(d_args, &h_args, sizeof(monte_carlo_args), cudaMemcpyHostToDevice);

				double start = clock();
				monte_carlo << < BLOCKS, THREADS >> >(d_states, d_sum, d_sum_sqr, d_args);
				cudaDeviceSynchronize();
				double stop = clock();
				double elapsed = (stop - start) / CLOCKS_PER_SEC;

				monte_carlo_result result = get_result(d_sum, d_sum_sqr);
				std::cout << result.value << " ";

				cudaFree(d_states);
				cudaFree(d_sum);
				cudaFree(d_sum_sqr);
				cudaFree(d_args);
				cudaDeviceReset();
			}
			std::cout << std::endl;
		}
	return 0;
}