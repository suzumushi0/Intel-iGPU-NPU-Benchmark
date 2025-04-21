
#include <omp.h>
#include <iostream>
#include <random>
#include <chrono>


// product of N x N square matrix and its transpose

//#define N 32
//#define N 64
//#define N 128
//#define N 256
//#define N 512
#define N 1'024
//#define N 2'048
//#define N 4'096
//#define N 8'192

//#define XXX

#ifndef XXX
#if (N <= 512)
constexpr int iter {8 * 8 * 16 * 4};
#elif (N == 1'024)
constexpr int iter {8 * 8 * 16};
#elif (N == 2'048)
constexpr int iter {8 * 8};
#elif (N == 4'096)
constexpr int iter {8};
#elif (N ==  8'192)
constexpr int iter {1};
#endif
#else
constexpr int iter {1};
#endif

constexpr int vector_size {256};	// SIMD vector size (bit)
constexpr int simd {vector_size / 8};

constexpr int B {simd / 4};

#pragma omp declare target
alignas (simd) int a_int32 [N][N], bT_int32 [N][N], c_int32 [N][N];
alignas (simd) float a_fp32 [N][N], bT_fp32 [N][N], c_fp32 [N][N];
#pragma omp end declare target

std::random_device seed;						// noise seed
std::default_random_engine noise_engine;		// noise engine 
std::uniform_int_distribution<int> noise_int32 {INT_MIN, INT_MAX};
std::uniform_real_distribution<float> noise_fp32 {0.0f, 1.0f};

std::chrono::system_clock::time_point tp [4];
double gops, elapsed_time;


int main (int argc, char* argv [])
{
	noise_engine.seed (seed ());

	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			a_int32 [i][j] = noise_int32 (noise_engine);
			bT_int32 [i][j] = noise_int32 (noise_engine);
			a_fp32 [i][j] = noise_fp32 (noise_engine);
			bT_fp32 [i][j] = noise_fp32 (noise_engine);
		}
	}

	// Reductions for c_int32 and so on are not necessary, 
	// because the outermost loop variable i is assigned to the thread identification

	tp [0] = std::chrono::system_clock::now (); 

	for (int r = 0; r < iter; r++) {
		#pragma omp target update to (a_int32, bT_int32)
		#pragma omp target teams distribute parallel for collapse (2)		
		for (int i = 0; i < N; i += B)
			for (int j = 0; j < N; j += B)
				for (int k = 0; k < N; k += B)
					for (int ii = i; ii < i + B; ii++)
						for (int jj = j; jj < j + B; jj++) {
							int sum = 0;
							for (int kk = k; kk < k + B; kk++)
								sum += a_int32 [ii][kk] * bT_int32 [jj][kk];
							c_int32 [ii][jj] += sum;
						}
		#pragma omp target update from (c_int32)
	}

	tp [1] = std::chrono::system_clock::now ();


	tp [2] = std::chrono::system_clock::now (); 
	
	for (int r = 0; r < iter; r++) {
		#pragma omp target update to (a_fp32, bT_fp32)
		#pragma omp target teams distribute parallel for collapse (2)
		for (int i = 0; i < N; i += B)
			for (int j = 0; j < N; j += B)
				for (int k = 0; k < N; k += B)
					for (int ii = i; ii < i + B; ii++)
						for (int jj = j; jj < j + B; jj++) {
							float sum = 0.0f;
							for (int kk = k; kk < k + B; kk++)
								sum += a_fp32 [ii][kk] * bT_fp32 [jj][kk];
							c_fp32 [ii][jj] += sum;
						}
		#pragma omp target update from (c_fp32)
	}
	tp [3] = std::chrono::system_clock::now (); 


#ifdef XXX
	#pragma omp single
	{
		for (int i = 0; i < N; i++)
			for (int j = 0; j < N; j++) {
				int sum = 0;
				for (int k = 0; k < N; k++)
					sum += a_int32 [i][k] * bT_int32 [j][k];
				if (sum != c_int32 [i][j])
					std::cout << "error\n";
			}

		for (int i = 0; i < N; i++)
			for (int j = 0; j < N; j++) {
				float sum = 0.0f;
				for (int k = 0; k < N; k++)
					sum += a_fp32 [i][k] * bT_fp32 [j][k];
				if (std::abs (sum - c_fp32 [i][j]) / sum > 1e-5f)
					std::cout << sum << "\t" << sum - c_fp32 [i][j] <<"\n";
			}
	}
#endif

	std::cout << __FILE__ << "\n";
#ifdef __INTEL_LLVM_COMPILER	// Intel C++ Compiler
	std::cout << "Intel_C++_Compiler\n";
#elif _MSC_VER					// Visual C++
	std::cout << "Visual_C++\n";
#endif
#ifdef __AVX2__
	std::cout << "AVX2\n";
#elif __AVX__
	std::cout << "AVX\n";
#else
	std::cout << "SSE2\n";
#endif
#ifdef _OPENMP
	std::cout << "OpenMP\n";
#else
	std::cout << "\n";
#endif

	std::cout << "Array_size_(KB)\t" << N * N * 4 / 1'024 << "\n";
	std::cout << "Matrix_size\t" << N << "x" <<  N <<"\n";

	std::cout << "GOPS_(Giga_Operations_Per_Second)\n";
	gops = 2.0 / 1'000 * N * N * N * iter  /  static_cast<double>(std::chrono::duration_cast<std::chrono::microseconds>(tp [1] - tp [0]).count()); 
	std::cout << "int32\t" << gops << "\n";
	gops = 2.0 / 1'000 * N * N * N * iter  /  static_cast<double>(std::chrono::duration_cast<std::chrono::microseconds>(tp [3] - tp [2]).count()); 
	std::cout << "fp32\t" << gops << "\n";

	std::cout << "Elapsed_time_(ms)\n";
	elapsed_time = static_cast<double>(std::chrono::duration_cast<std::chrono::milliseconds>(tp [1] - tp [0]).count()); 
	std::cout << "int32\t" << elapsed_time << "\n";
	elapsed_time = static_cast<double>(std::chrono::duration_cast<std::chrono::milliseconds>(tp [3] - tp [2]).count()); 
	std::cout << "fp32\t" << elapsed_time << "\n";

}
