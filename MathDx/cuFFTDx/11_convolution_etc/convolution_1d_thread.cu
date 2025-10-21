#include <iostream>
#include <vector>

#include <cuda_runtime_api.h>
#include <cufftdx.hpp>

#include "../common/common.hpp"
#include "../common/mixed_io.hpp"

#define CUFFTDX_EXAMPLE_DETAIL_DEBUG_CONV_1D_THREAD

inline constexpr unsigned int warm_up_runs = 5;
inline constexpr unsigned int performance_runs = 20;

template <unsigned int fft_size, class T>
__global__ void scaling_kernel(T *data,
                               const unsigned int input_size,
                               const unsigned int ept) {

  static constexpr float scale = 1.0 / fft_size;

  T temp;
  unsigned int index = blockDim.x * blockIdx.x + threadIdx.x;

  for (int i = 0; i < ept; i++) {
    if (index < input_size) {
      temp = data[index];
      temp.x *= scale;
      temp.y *= scale;
      data[index] = temp;
      index += blockDim.x * gridDim.x;
    }
  }
}

template <typename FFT, unsigned int fft_size, class T>
example::fft_results<T>
cufft_conv_1d(T *input, T *output, const unsigned int bs, cudaStream_t stream) {
  using complex_type = typename example::make_cufft_compatible<typename FFT::value_type>::type;

  static_assert(sizeof(T) == sizeof(complex_type), "");
  static_assert(std::alignment_of_v<T> == std::alignment_of_v<complex_type>,
                "");

  complex_type *cufft_input = reinterpret_cast<complex_type *>(input);
  complex_type *cufft_output = reinterpret_cast<complex_type *>(output);

  static constexpr unsigned int block_dim_scaling_kernel = 1024;

  const unsigned int total_fft_size = fft_size * bs;
  const unsigned int cuda_blocks =
      (total_fft_size + block_dim_scaling_kernel - 1) /
      block_dim_scaling_kernel;

  // Create cuFFT plan
  cufftHandle plan_forward, plan_inverse;
  CUFFT_CHECK_AND_EXIT(cufftPlan1d(
      &plan_forward, fft_size,
      std::is_same_v<complex_type, cufftComplex> ? CUFFT_C2C : CUFFT_Z2Z, bs));
  CUFFT_CHECK_AND_EXIT(cufftPlan1d(
      &plan_inverse, fft_size,
      std::is_same_v<complex_type, cufftComplex> ? CUFFT_C2C : CUFFT_Z2Z, bs));

  CUFFT_CHECK_AND_EXIT(cufftSetStream(plan_forward, stream));
  CUFFT_CHECK_AND_EXIT(cufftSetStream(plan_inverse, stream));

  // Create execute
  auto cufft_execution = [&](cudaStream_t stream) {
    if constexpr (std::is_same_v<complex_type, cufftComplex>) {
      CUFFT_CHECK_AND_EXIT(
          cufftExecC2C(plan_forward, cufft_input, cufft_output, CUFFT_FORWARD));
    } else if constexpr (std::is_same_v<complex_type, cufftDoubleComplex>) {
      CUFFT_CHECK_AND_EXIT(
          cufftExecZ2Z(plan_forward, cufft_input, cufft_output, CUFFT_FORWARD));
    }
    scaling_kernel<fft_size>
        <<<cuda_blocks, block_dim_scaling_kernel, 0, stream>>>(
            cufft_output, total_fft_size, 1);
    if constexpr (std::is_same_v<complex_type, cufftComplex>) {
      CUFFT_CHECK_AND_EXIT(cufftExecC2C(plan_inverse, cufft_output,
                                        cufft_output, CUFFT_INVERSE));
    } else if constexpr (std::is_same_v<complex_type, cufftDoubleComplex>) {
      CUFFT_CHECK_AND_EXIT(cufftExecZ2Z(plan_inverse, cufft_output,
                                        cufft_output, CUFFT_INVERSE));
    }
  };

  // Correctness run
  cufft_execution(stream);
  CUDA_CHECK_AND_EXIT(cudaPeekAtLastError());
  CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());

  // Copy total FFT results to host
  std::vector<typename FFT::value_type> output_host(total_fft_size);
  CUDA_CHECK_AND_EXIT(cudaMemcpy(output_host.data(), cufft_output,
                                 total_fft_size * sizeof(complex_type),
                                 cudaMemcpyDeviceToHost));
  CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());

  // Performance measurements
  auto time = example::measure_execution_ms(cufft_execution, warm_up_runs,
                                            performance_runs, stream);

  CUFFT_CHECK_AND_EXIT(cufftDestroy(plan_forward));
  CUFFT_CHECK_AND_EXIT(cufftDestroy(plan_inverse));

  return example::fft_results<T>{output_host, (time / performance_runs)};
}

template <class FFT, class IFFT>
__global__ void conv1d_thread_kernel(typename FFT::value_type *input,
                                     typename FFT::value_type *output) {
  using complex_type = typename FFT::value_type;
  using scalar_type = typename complex_type::value_type;

  // Local array for thread
  complex_type thread_data[FFT::storage_size];

  // Load data from global memory to registers.
  // thread_data should have all input data in order.
  unsigned int index = threadIdx.x * FFT::elements_per_thread;
  for (size_t i = 0; i < FFT::elements_per_thread; i++) {
    thread_data[i] = input[index + i];
  }

  // Execute FFT
  FFT().execute(thread_data);

  // Scale values (normalizing with 1/N)
  scalar_type scale = 1.0 / cufftdx::size_of<FFT>::value;
  for (unsigned int i = 0; i < FFT::elements_per_thread; i++) {
    thread_data[i].x *= scale;
    thread_data[i].y *= scale;
  }

  // Execute IFFT
  IFFT().execute(thread_data);

  // Save results
  for (size_t i = 0; i < FFT::elements_per_thread; i++) {
    output[index + i] = thread_data[i];
  }
}

template <class FFT, class IFFT, class T>
example::fft_results<T> cufftdx_conv_1d(T *input, T *output,
                                        const unsigned int num_threads,
                                        cudaStream_t stream) {
  using complex_type = typename FFT::value_type;

  complex_type *cufftdx_input = reinterpret_cast<complex_type *>(input);
  complex_type *cufftdx_output = reinterpret_cast<complex_type *>(output);

  // Correctness run
  conv1d_thread_kernel<FFT, IFFT>
      <<<1, num_threads, 0, stream>>>(cufftdx_input, cufftdx_output);
  CUDA_CHECK_AND_EXIT(cudaPeekAtLastError());
  CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());

  // Copy total FFT results to host
  const unsigned int total_fft_size = FFT::input_length * num_threads;
  std::vector<T> output_host(total_fft_size,
                             {std::numeric_limits<float>::quiet_NaN(),
                              std::numeric_limits<float>::quiet_NaN()});
  CUDA_CHECK_AND_EXIT(cudaMemcpy(output_host.data(), cufftdx_output,
                                 total_fft_size * sizeof(complex_type),
                                 cudaMemcpyDeviceToHost));
  CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());

  // Performance measurements
  auto time = example::measure_execution_ms(
      [&](cudaStream_t stream) {
        conv1d_thread_kernel<FFT, IFFT>
            <<<1, num_threads, 0, stream>>>(cufftdx_input, cufftdx_output);
      },
      warm_up_runs, performance_runs, stream);

  return example::fft_results<T>{output_host, (time / performance_runs)};
}

// In this example a one-dimensional C2C convolution is perform by a CUDA
// thread.
//
// Number of 'threads_count' threads are run, and each thread calculates
// 'fft_size'-point C2C float precision FFT. Data is generated on host, copied
// to device buffer, and then results are copied back to host.
int main(int, char **) {
  using namespace cufftdx;

  using precision_type = float;
  using complex_type = complex<precision_type>;

  // Size of FFT
  static constexpr unsigned int fft_size = 64;

  // Number of threads to execute FFTs (= batches)
  static constexpr unsigned int threads_count = 2;

  // FFT is defined, its: size, type, direction, precision. Thread() operator
  // informs that FFT will be executed on thread level.
  using FFT_base = decltype(Thread() + Size<fft_size>() +
                            Type<fft_type::c2c>() + Precision<precision_type>());
  using FFT = decltype(FFT_base() + Direction<fft_direction::forward>());
  using IFFT = decltype(FFT_base() + Direction<fft_direction::inverse>());

  std::cout << "[FFT 1D C2C Convolution (Thread Execution)]" << std::endl;
  std::cout << "===================================================\n";
  std::cout << " FFT size: " << fft_size << std::endl;
  std::cout << " Batches (= Number of threads): " << threads_count << std::endl;
  static const std::string precision_str =
      (typeid(precision_type) == typeid(float))
          ? "float"
          : (typeid(precision_type) == typeid(__half))
                ? "half"
                : (typeid(precision_type) == typeid(double)) ? "double"
                                                             : "unknown";
  std::cout << " Precision: " << precision_str << "\n";

  // Host data
  auto input_size = threads_count * fft_size;
  auto input_size_bytes = input_size * sizeof(complex_type);
  std::vector<complex_type> host_data(input_size);

  for (size_t i = 0; i < host_data.size(); i++) {
    host_data[i] = complex_type{precision_type(i), -precision_type(i)};
  }

#ifdef CUFFTDX_EXAMPLE_DETAIL_DEBUG_CONV_1D_THREAD
  std::cout << "input:\n";
  for (size_t i = 0; i < host_data.size(); i++) {
    std::cout << host_data[i].x << " " << host_data[i].y << std::endl;
  }
#endif

  // Device buffers
  complex_type *input;
  complex_type *output;

  CUDA_CHECK_AND_EXIT(cudaMalloc(&input, input_size_bytes));
  CUDA_CHECK_AND_EXIT(cudaMalloc(&output, input_size_bytes));

  // Copy host to device
  CUDA_CHECK_AND_EXIT(cudaMemcpy(input, host_data.data(), input_size_bytes,
                                 cudaMemcpyHostToDevice));

  cudaStream_t stream;
  CUDA_CHECK_AND_EXIT(cudaStreamCreate(&stream));

  // cuFFTDx convolution
  auto cufftdx_results =
      cufftdx_conv_1d<FFT, IFFT>(input, output, threads_count, stream);

  // cuFFT convolution for correctness check
  auto cufft_results =
      cufft_conv_1d<FFT, fft_size>(input, output, threads_count, stream);

  CUDA_CHECK_AND_EXIT(cudaStreamDestroy(stream));
  CUDA_CHECK_AND_EXIT(cudaFree(input));
  CUDA_CHECK_AND_EXIT(cudaFree(output));

#ifdef CUFFTDX_EXAMPLE_DETAIL_DEBUG_CONV_1D_THREAD
  std::cout << "output of cuFFTDx:\n";
  for (size_t i = 0; i < cufftdx_results.output.size(); i++) {
    std::cout << cufftdx_results.output[i].x << " "
              << cufftdx_results.output[i].y << std::endl;
  }

  std::cout << "output of cuFFT:\n";
  for (size_t i = 0; i < cufft_results.output.size(); i++) {
    std::cout << cufft_results.output[i].x << " " << cufft_results.output[i].y
              << std::endl;
  }
#endif

  // Check correctness
  bool success = true;
  std::cout << "===================================================";
  std::cout << "\nCorrectness results:\n";
  // Check if cuFFTDx results are correct
  {
    auto fft_error = example::fft_signal_error::calculate_for_complex_values(
        cufftdx_results.output, cufft_results.output);
    std::cout << " cuFFTDx (vs cuFFT) \n";
    std::cout << "  - L2 error: " << fft_error.l2_relative_error << "\n";
    std::cout << "  - Peak error (index: " << fft_error.peak_error_index
              << "): " << fft_error.peak_error << "\n";
    std::cout << "  - Peak relative error (index: "
              << fft_error.peak_error_index
              << "): " << fft_error.peak_error_relative << "\n";
    if (success) {
      success = (fft_error.l2_relative_error < 0.001);
    }
  }

  // Print performance results
  if (success) {
    std::cout << "===================================================";
    std::cout << "\nPerformance results:\n";

    std::cout << std::setw(28) << " cuFFTDx: " << cufftdx_results.avg_time_in_ms
              << " [ms]\n";
    std::cout << std::setw(28) << " cuFFT: " << cufft_results.avg_time_in_ms
              << " [ms]\n";
    std::cout << "===================================================";
    std::cout << "\nSuccess (validation)\n";
  } else {
    std::cout << "===================================================";
    std::cout << "\nFailure (validation)\n";
    std::exit(1);
  }
}
