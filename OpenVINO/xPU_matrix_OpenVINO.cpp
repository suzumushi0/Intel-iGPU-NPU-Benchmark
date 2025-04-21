#include <openvino/openvino.hpp> 
#include <openvino/op/matmul.hpp>

#include <iostream>
#include <random>
#include <chrono>


// product of N x N square matrix and its transpose

//#define CPU
//#define GPU
#define NPU
//#define GPU16

//#define N 32
//#define N 64
//#define N 128
//#define N 256
//#define N 512
//#define N 1'024
//#define N 2'048
#define N 4'096
//#define N 8'192

//#define XXX

#if defined (CPU)
#define xPU "CPU"
#elif defined (GPU)
#define xPU "GPU"
#elif defined (NPU)
#define xPU "NPU"
#elif defined (GPU16)
#define xPU "GPU"
#endif

#ifndef XXX
#ifdef CPU
#if (N <= 128)
constexpr int iter {32 * 8 * 8 * 8 * 8 * 8};
#elif (N == 256)
constexpr int iter {32 * 8 * 8 * 8 * 8};
#elif (N == 512)
constexpr int iter {32 * 8 * 8 * 8};
#elif (N == 1'024)
constexpr int iter {32 * 8 * 8};
#elif (N == 2'048)
constexpr int iter {32 * 8};
#elif (N == 4'096)
constexpr int iter {32};
#elif (N == 8'192)
constexpr int iter {4};
#endif
#else
#if (N <= 512)
constexpr int iter {32 * 8 * 16 * 4};
#elif (N == 1'024)
constexpr int iter {32 * 8 * 16};
#elif (N == 2'048)
constexpr int iter {32 * 8};
#elif (N == 4'096)
constexpr int iter {32};
#elif (N == 8'192)
constexpr int iter {4};
#endif
#endif
#else
constexpr int iter {1};
#endif

constexpr int vector_size {256};	// SIMD vector size (bit)
constexpr int simd {vector_size / 8};

alignas (simd) float a_fp32 [N][N], bT_fp32 [N][N];

std::random_device seed;						// noise seed
std::default_random_engine noise_engine;		// noise engine 
std::uniform_real_distribution<float> noise_fp32 {0.0f, 1.0f};

std::chrono::system_clock::time_point tp [2];
double gops, elapsed_time;


int main (int argc, char* argv [])
{
	noise_engine.seed (seed ());

	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			a_fp32 [i][j] = noise_fp32 (noise_engine);
			bT_fp32 [i][j] = noise_fp32 (noise_engine);
		}
	}

	// Model definition
	auto A_fp32 = std::make_shared<ov::op::v0::Parameter> (ov::element::f32, ov::Shape {N, N});
	auto BT_fp32 = std::make_shared<ov::op::v0::Parameter> (ov::element::f32, ov::Shape {N, N});
	auto matmul_A_BT = std::make_shared<ov::op::v0::MatMul> (A_fp32, BT_fp32, false, true);
	auto result = std::make_shared<ov::op::v0::Result> (matmul_A_BT);
	auto model = std::make_shared<ov::Model> (ov::ResultVector {result}, ov::ParameterVector {A_fp32, BT_fp32}, "MatMul");

	// Create OpenVINO runtime core
	ov::Core ov_core;
#ifdef GPU
	ov_core.set_property ("GPU", ov::hint::inference_precision (ov::element::f32));
#endif

	// Convert the model to OpenVINO intermediate representation 
	ov::CompiledModel ov_ir = ov_core.compile_model (model, xPU);

	// Create an inference request 
	ov::InferRequest infer_request = ov_ir.create_infer_request ();

	tp [0] = std::chrono::system_clock::now (); 

	for (int r = 0; r < iter; r++) {
		// Setup input data
		infer_request.set_tensor (A_fp32, ov::Tensor (ov::element::f32, ov::Shape {N, N}, a_fp32));
		infer_request.set_tensor (BT_fp32, ov::Tensor (ov::element::f32, ov::Shape {N, N}, bT_fp32));

		// Inference
		infer_request.infer ();

		// Process the inference results
		ov::Tensor output = infer_request.get_output_tensor ();
		const float* output_data = output.data<float> ();

#ifdef XXX
		for (int i = 0; i < N; i++)
			for (int j = 0; j < N; j++) {
				float sum = 0.0f;
				for (int k = 0; k < N; k++)
					sum += a_fp32 [i][k] * bT_fp32 [j][k];
#if defined (CPU)
				if (std::abs (sum - output_data [i * N + j]) / sum > 1e-6f)
#elif defined (GPU)
				if (std::abs (sum - output_data [i * N + j]) / sum > 1e-5f)
#elif defined (NPU)
				if (std::abs (sum - output_data [i * N + j]) / sum > 1e-3f)
#elif defined (GPU16)
				if (std::abs (sum - output_data [i * N + j]) / sum > 5e-2f)
#endif
					std::cout << sum << "\t" << sum - output_data [i * N + j] <<"\n";
			}
#endif
	}

	tp [1] = std::chrono::system_clock::now (); 

	std::cout << __FILE__ << "\n";
	std::cout << xPU << "\n";
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

	std::cout << "Array_size_(KB)\t" << N * N * 4 / 1'024 << "\n";
	std::cout << "Matrix_size\t" << N << "x" <<  N <<"\n";

	std::cout << "GOPS_(Giga_Operations_Per_Second)\n";
	gops = 2.0 / 1'000 * N * N * N * iter  /  static_cast<double>(std::chrono::duration_cast<std::chrono::microseconds>(tp [1] - tp [0]).count()); 
	std::cout << "fp32\t" << gops << "\n";

	std::cout << "Elapsed_time_(ms)\n";
	elapsed_time = static_cast<double>(std::chrono::duration_cast<std::chrono::milliseconds>(tp [1] - tp [0]).count()); 
	std::cout << "fp32\t" << elapsed_time << "\n";

}
