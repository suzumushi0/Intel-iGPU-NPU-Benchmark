
#include <omp.h>
#include <iostream>
#include <random>
#include <chrono>


// product of N x N square matrices

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
#ifdef _OPENMP
#if (N <= 128)
constexpr int iter {16 * 8 * 8 * 8 * 8 * 8};
#elif (N == 256)
constexpr int iter {16 * 8 * 8 * 8 * 8};
#elif (N == 512)
constexpr int iter {16 * 8 * 8 * 8};
#elif (N == 1'024)
constexpr int iter {16 * 8 * 8};
#elif (N == 2'048)
constexpr int iter {16 * 8};
#elif (N == 4'096)
constexpr int iter {16};
#elif (N == 8'192)
constexpr int iter {2};
#endif
#else
#if (N == 32)
constexpr int iter {8 * 8 * 8 * 8* 8 * 8 * 8 * 8};
#elif (N == 64)
constexpr int iter {8 * 8 * 8 * 8 * 8 * 8 * 8};
#elif (N == 128)
constexpr int iter {8 * 8 * 8 * 8 * 8 * 8};
#elif (N == 256)
constexpr int iter {8 * 8 * 8 * 8 * 8};
#elif (N == 512)
constexpr int iter {8 * 8 * 8 * 8};
#elif (N == 1'024)
constexpr int iter {8 * 8 * 8};
#elif (N == 2'048)
constexpr int iter {8 * 8};
#elif (N == 4'096)
constexpr int iter {8};
#elif (N == 8'192)
constexpr int iter {1};
#endif
#endif
#else
constexpr int iter {1};
#endif

constexpr int vector_size {256};	// SIMD vector size (bit)
constexpr int simd {vector_size / 8};

constexpr int B {simd / 4};

alignas (simd) int a_int32 [N][N], b_int32 [N][N], c_int32 [N][N];
alignas (simd) float a_fp32 [N][N], b_fp32 [N][N], c_fp32 [N][N];

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
			b_int32 [i][j] = noise_int32 (noise_engine);
			a_fp32 [i][j] = noise_fp32 (noise_engine);
			b_fp32 [i][j] = noise_fp32 (noise_engine);
		}
	}

	// Reductions for c_int32 and so on are not necessary, 
	// because the outermost loop variable i is assigned to the thread identification

	tp [0] = std::chrono::system_clock::now (); 

	for (int r = 0; r < iter; r++) {
		#pragma omp parallel for
		for (int i = 0; i < N; i += B)
			for (int k = 0; k < N; k += B)
				for (int j = 0; j < N; j += B)
					for (int ii = i; ii < i + B; ii++)
						for (int kk = k; kk < k + B; kk++)
							for (int jj = j; jj < j + B; jj++)
								c_int32 [ii][jj] += a_int32 [ii][kk] * b_int32 [kk][jj];
	}

	tp [1] = std::chrono::system_clock::now (); 

	tp [2] = std::chrono::system_clock::now (); 

	for (int r = 0; r < iter; r++) {
		#pragma omp parallel for
		for (int i = 0; i < N; i += B)
			for (int k = 0; k < N; k += B)
				for (int j = 0; j < N; j += B)
					for (int ii = i; ii < i + B; ii++)
						for (int kk = k; kk < k + B; kk++)
							for (int jj = j; jj < j + B; jj++)
								c_fp32 [ii][jj] += a_fp32 [ii][kk] * b_fp32 [kk][jj];
	}

	tp [3] = std::chrono::system_clock::now (); 

#ifdef XXX
	#pragma omp single
	{
		for (int i = 0; i < N; i++)
			for (int j = 0; j < N; j++) {
				int sum = 0;
				for (int k = 0; k < N; k++)
					sum += a_int32 [i][k] * b_int32 [k][j];
				if (sum != c_int32 [i][j])
					std::cout << "error\n";
			}

		for (int i = 0; i < N; i++)
			for (int j = 0; j < N; j++) {
				float sum = 0.0f;
				for (int k = 0; k < N; k++)
					sum += a_fp32 [i][k] * b_fp32 [k][j];
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
