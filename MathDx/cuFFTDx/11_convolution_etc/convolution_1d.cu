#include <iostream>
#include <vector>

#include <cuda_runtime_api.h>
#include <cufft.h>
#include <cufftdx.hpp>

#include "../common/block_io.hpp"
#include "../common/common.hpp"

inline constexpr unsigned int warm_up_runs = 5;
inline constexpr unsigned int performance_runs = 20;

#define CUFFTDX_EXAMPLE_DETAIL_DEBUG_CONV_1D

template <unsigned int fft_size>
__global__ void scaling_kernel(cufftComplex *data,
                               const unsigned int input_size,
                               const unsigned int ept) {

  static constexpr float scale = 1.0 / fft_size;

  cufftComplex temp;
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

template <unsigned int fft_size, class T>
example::fft_results<T>
cufft_conv_1d(T *input, T *output, const unsigned int bs, cudaStream_t stream) {
  using complex_type = cufftComplex;
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
  CUFFT_CHECK_AND_EXIT(cufftPlan1d(&plan_forward, fft_size, CUFFT_C2C, bs));
  CUFFT_CHECK_AND_EXIT(cufftPlan1d(&plan_inverse, fft_size, CUFFT_C2C, bs));

  CUFFT_CHECK_AND_EXIT(cufftSetStream(plan_forward, stream));
  CUFFT_CHECK_AND_EXIT(cufftSetStream(plan_inverse, stream));

  // Correctness run
  CUFFT_CHECK_AND_EXIT(
      cufftExecC2C(plan_forward, cufft_input, cufft_output, CUFFT_FORWARD));
  scaling_kernel<fft_size>
      <<<cuda_blocks, block_dim_scaling_kernel, 0, stream>>>(cufft_output,
                                                             total_fft_size, 1);
  CUFFT_CHECK_AND_EXIT(
      cufftExecC2C(plan_inverse, cufft_output, cufft_output, CUFFT_INVERSE));
  CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());

  // Copy total FFT results to host
  std::vector<T> output_host(total_fft_size,
                             {std::numeric_limits<float>::quiet_NaN(),
                              std::numeric_limits<float>::quiet_NaN()});
  CUDA_CHECK_AND_EXIT(cudaMemcpy(output_host.data(), cufft_output,
                                 total_fft_size * sizeof(complex_type),
                                 cudaMemcpyDeviceToHost));
  CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());

  // Performance measurements
  auto time = example::measure_execution_ms(
      [&](cudaStream_t /* stream */) {
        CUFFT_CHECK_AND_EXIT(cufftExecC2C(plan_forward, cufft_input,
                                          cufft_output, CUFFT_FORWARD));
        scaling_kernel<fft_size>
            <<<cuda_blocks, block_dim_scaling_kernel, 0, stream>>>(
                cufft_output, total_fft_size, 1);
        CUFFT_CHECK_AND_EXIT(cufftExecC2C(plan_inverse, cufft_output,
                                          cufft_output, CUFFT_INVERSE));
      },
      warm_up_runs, performance_runs, stream);

  CUFFT_CHECK_AND_EXIT(cufftDestroy(plan_forward));
  CUFFT_CHECK_AND_EXIT(cufftDestroy(plan_inverse));

  return example::fft_results<T>{output_host, (time / performance_runs)};
}

template <class FFT, class IFFT>
__launch_bounds__(FFT::max_threads_per_block) __global__
    void convolution_kernel(typename FFT::value_type *input,
                            typename FFT::value_type *output,
                            typename FFT::workspace_type workspace) {
  using complex_type = typename FFT::value_type;
  using scalar_type = typename complex_type::value_type;

  // Local array for thread
  complex_type thread_data[FFT::storage_size];

  // ID of FFT in CUDA block, in range [0; FFT::ffts_per_block)
  const unsigned int local_fft_id = threadIdx.y;
  // Load data from global memory to registers
  example::io<FFT>::load(input, thread_data, local_fft_id);

  // Execute FFT
  extern __shared__ __align__(alignof(float4)) complex_type shared_mem[];
  FFT().execute(thread_data, shared_mem, workspace);

  // Scale values (point-wise operation: normalizing with 1/N)
  scalar_type scale = 1.0 / cufftdx::size_of<FFT>::value;
  for (unsigned int i = 0; i < FFT::elements_per_thread; i++) {
    thread_data[i].x *= scale;
    thread_data[i].y *= scale;
  }

  // Execute inverse FFT
  IFFT().execute(thread_data, shared_mem);

  // Save results
  example::io<FFT>::store(thread_data, output, local_fft_id);
}

template <class FFT, class IFFT, class T>
example::fft_results<T> cufftdx_conv_1d(T *input, T *output,
                                        const unsigned int bs_div_fpb,
                                        cudaStream_t stream) {
  using complex_type = typename FFT::value_type;
  static constexpr unsigned int fft_size = cufftdx::size_of<FFT>::value;

  // Checks FFT is correctly defined
  static_assert(sizeof(T) == sizeof(complex_type), "Type size mismatch");
  static_assert(std::alignment_of_v<T> == std::alignment_of_v<complex_type>,
                "Type alignment mismatch");

  complex_type *cufftdx_input = reinterpret_cast<complex_type *>(input);
  complex_type *cufftdx_output = reinterpret_cast<complex_type *>(output);

  // Increase max shared memory if needed
  CUDA_CHECK_AND_EXIT(cudaFuncSetAttribute(
      convolution_kernel<FFT, IFFT>,
      cudaFuncAttributeMaxDynamicSharedMemorySize, FFT::shared_memory_size));

  // Create workspaces for FFTs
  cudaError_t error_code;
  auto workspace = cufftdx::make_workspace<FFT>(error_code, stream);
  CUDA_CHECK_AND_EXIT(error_code);

  // Correctness run (NOTE: batches for cuFFTDx = batch_size / ffts_per_block)
  convolution_kernel<FFT, IFFT>
      <<<bs_div_fpb, FFT::block_dim, FFT::shared_memory_size, stream>>>(
          cufftdx_input, cufftdx_output, workspace);
  CUDA_CHECK_AND_EXIT(cudaGetLastError());
  CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());

  // Copy total FFT results to host
  static const size_t total_fft_size =
      bs_div_fpb * FFT::ffts_per_block * fft_size;
  static const size_t total_fft_size_bytes =
      total_fft_size * sizeof(complex_type);
  std::vector<complex_type> output_host(
      total_fft_size, {std::numeric_limits<float>::quiet_NaN(),
                       std::numeric_limits<float>::quiet_NaN()});
  CUDA_CHECK_AND_EXIT(cudaMemcpy(output_host.data(), cufftdx_output,
                                 total_fft_size_bytes, cudaMemcpyDeviceToHost));
  CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());

  // Performance measurements
  auto time = example::measure_execution_ms(
      [&](cudaStream_t stream) {
        convolution_kernel<FFT, IFFT>
            <<<bs_div_fpb, FFT::block_dim, FFT::shared_memory_size, stream>>>(
                cufftdx_input, cufftdx_output, workspace);
      },
      warm_up_runs, performance_runs, stream);

  return example::fft_results<T>{output_host, (time / performance_runs)};
}

// This example demonstrates how to use cuFFTDx to perform a C2C convolution
// using one-dimensional FFTs.
//
// One block is run, it calculates two 128-point convolutions by first doing
// forward FFT, then applying pointwise operation, and ending with inverse FFT.
// Data is generated on host, copied to device buffer, and then results are
// copied back to host.
template <unsigned int Arch> void conv_1d() {
  using namespace cufftdx;

  using precision_type = float;
  using complex_type = complex<precision_type>;

  // Size of single FFT
  static constexpr unsigned int fft_size = 64;

  // Number of FFTs (convolutions)
  // NOTE: Should be multiple of ffts_per_block
  static constexpr unsigned int batches = 2;

  constexpr bool use_suggested = false; // Whether to use suggested values

  static constexpr unsigned int custom_elements_per_thread = 8;
  static constexpr unsigned int custom_ffts_per_block = 2;

  // Declaration of cuFFTDx run
  using fft_incomplete =
      decltype(Block() + Size<fft_size>() + Type<fft_type::c2c>() +
               Precision<precision_type>() + SM<Arch>());
  using fft_base =
      decltype(fft_incomplete() + Direction<fft_direction::forward>());
  using ifft_base =
      decltype(fft_incomplete() + Direction<fft_direction::inverse>());

  static constexpr unsigned int elements_per_thread =
      use_suggested ? fft_base::elements_per_thread
                    : custom_elements_per_thread;
  static constexpr unsigned int ffts_per_block =
      use_suggested ? fft_base::suggested_ffts_per_block
                    : custom_ffts_per_block;

  using FFT = decltype(fft_base() + ElementsPerThread<elements_per_thread>() +
                       FFTsPerBlock<ffts_per_block>());
  using IFFT = decltype(ifft_base() + ElementsPerThread<elements_per_thread>() +
                        FFTsPerBlock<ffts_per_block>());

  static_assert(batches % ffts_per_block == 0,
                "batches must be multiple of ffts_per_block");

  std::cout << "[FFT 1D C2C Convolution (Block Execution)]" << std::endl;
  std::cout << "===================================================\n";
  std::cout << " cuFFTDx configuration: "
            << "\n";
  std::cout << "  - Elements per thread: " << elements_per_thread << "\n";
  std::cout << "  - FFTs per block: " << ffts_per_block << "\n";
  std::cout << "---------------------------------------------------\n";
  std::cout << " FFT size: " << fft_size << "\n";
  std::cout << " Batches: " << batches << "\n";
  static const std::string precision_str =
      (typeid(precision_type) == typeid(float))
          ? "float"
          : (typeid(precision_type) == typeid(__half))
                ? "half"
                : (typeid(precision_type) == typeid(double)) ? "double"
                                                             : "unknown";
  std::cout << " Precision: " << precision_str << "\n";

  // Host data
  auto input_size = batches * fft_size;
  auto input_size_bytes = input_size * sizeof(complex_type);
  std::vector<complex_type> host_data(input_size);
  for (size_t i = 0; i < input_size; i++) {
    host_data[i] = complex_type{precision_type(i), -precision_type(i)};
  }

#ifdef CUFFTDX_EXAMPLE_DETAIL_DEBUG_CONV_1D
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

  // Copy host data to device buffer
  CUDA_CHECK_AND_EXIT(cudaMemcpy(input, host_data.data(), input_size_bytes,
                                 cudaMemcpyHostToDevice));

  cudaStream_t stream;
  CUDA_CHECK_AND_EXIT(cudaStreamCreate(&stream));

  // cuFFTDx convolution
  auto cufftdx_results = cufftdx_conv_1d<FFT, IFFT>(
      input, output, batches / ffts_per_block, stream);

  // cuFFT convolution as reference
  auto cufft_results = cufft_conv_1d<fft_size>(input, output, batches, stream);

  // TODO: cuFFT convolution with Callback

  CUDA_CHECK_AND_EXIT(cudaStreamDestroy(stream));
  CUDA_CHECK_AND_EXIT(cudaFree(input));
  CUDA_CHECK_AND_EXIT(cudaFree(output));

#ifdef CUFFTDX_EXAMPLE_DETAIL_DEBUG_CONV_1D
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

template <unsigned int Arch> struct conv_1d_functor {
  void operator()() { return conv_1d<Arch>(); }
};

int main(int, char **) { return example::sm_runner<conv_1d_functor>(); }
