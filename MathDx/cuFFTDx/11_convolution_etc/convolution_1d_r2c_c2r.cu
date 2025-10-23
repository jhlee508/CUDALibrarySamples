#include <iostream>
#include <vector>

#include <cuda_runtime_api.h>
#include <cufftdx.hpp>

#include "../common/block_io.hpp"
#include "../common/common.hpp"
#include "../common/random.hpp"

inline constexpr unsigned int warm_up_runs = 5;
inline constexpr unsigned int performance_runs = 20;

// #define CUFFTDX_EXAMPLE_DETAIL_DEBUG_CONV_1D

template <unsigned int fft_size, class T>
__global__ void scaling_kernel(T *data, const unsigned int input_size,
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

template <class FFTR2C, class FFTC2R>
__launch_bounds__(FFTR2C::max_threads_per_block) __global__
    void convolution_r2c_c2r_kernel(
        cufftdx::precision_of_t<FFTR2C> *input,
        cufftdx::precision_of_t<FFTC2R> *output,
        typename FFTR2C::workspace_type workspace_r2c,
        typename FFTC2R::workspace_type workspace_c2r) {
  using complex_type = typename FFTR2C::value_type;
  using scalar_type = typename complex_type::value_type;

  // Local array for thread
  complex_type thread_data[FFTR2C::storage_size];

  // ID of FFT in CUDA block, in range [0; FFTR2C::ffts_per_block)
  const unsigned int local_fft_id = threadIdx.y;
  // Load data from global memory to registers
  example::io<FFTR2C>::load(input, thread_data, local_fft_id);

  // Execute FFT
  extern __shared__ __align__(alignof(float4)) complex_type shared_mem[];
  FFTR2C().execute(thread_data, shared_mem, workspace_r2c);

  // Scale values (point-wise operation: normalizing with 1/N)
  scalar_type scale = 1.0 / cufftdx::size_of<FFTR2C>::value;
  for (unsigned int i = 0; i < FFTR2C::elements_per_thread; i++) {
    thread_data[i].x *= scale;
    thread_data[i].y *= scale;
  }

  // Execute inverse FFT
  FFTC2R().execute(thread_data, shared_mem, workspace_c2r);

  // Save results
  example::io<FFTC2R>::store(thread_data, output, local_fft_id);
}

template <typename FFTR2C, unsigned int fft_size, class RealType,
          class ComplexType>
example::fft_results<RealType>
cufft_conv_1d_r2c_c2r(RealType *real_input, RealType *real_output,
                      ComplexType *complex_intermediate, unsigned int bs,
                      cudaStream_t stream) {
  // Complex type should be cufftComplex/cufftDoubleComplex
  using complex_type =
      std::conditional_t<std::is_same_v<cufftdx::precision_of_t<FFTR2C>, float>,
                         cufftComplex, cufftDoubleComplex>;
  // Real type should be cufftReal/cufftDoubleReal
  using real_type =
      std::conditional_t<std::is_same_v<cufftdx::precision_of_t<FFTR2C>, float>,
                         cufftReal, cufftDoubleReal>;

  static_assert(sizeof(RealType) == sizeof(real_type), "Type size mismatch");
  static_assert(std::alignment_of_v<RealType> == std::alignment_of_v<real_type>,
                "Type alignment mismatch");

  static_assert(sizeof(ComplexType) == sizeof(complex_type),
                "Type size mismatch");
  static_assert(std::alignment_of_v<ComplexType> ==
                    std::alignment_of_v<complex_type>,
                "Type alignment mismatch");

  static_assert(sizeof(complex_type) % sizeof(real_type) == 0,
                "Complex type size must be multiple of real type size");
  static_assert(
      std::alignment_of_v<complex_type> % std::alignment_of_v<real_type> == 0,
      "Complex type alignment must be multiple of real type alignment");

  real_type *cufft_r2c_input = reinterpret_cast<real_type *>(real_input);
  complex_type *cufft_r2c_output =
      reinterpret_cast<complex_type *>(complex_intermediate);
  complex_type *cufft_c2r_input = cufft_r2c_output; // In-place
  real_type *cufft_c2r_output = reinterpret_cast<real_type *>(real_output);

  static constexpr unsigned int block_dim_scaling_kernel = 1024;

  const unsigned int scaling_size = FFTR2C::output_length * bs;
  const unsigned int cuda_blocks =
      (scaling_size + block_dim_scaling_kernel - 1) / block_dim_scaling_kernel;

  // Create cuFFT plan
  cufftHandle planF, planI;
  CUFFT_CHECK_AND_EXIT(cufftPlan1d(
      &planF, fft_size,
      std::is_same_v<real_type, cufftReal> ? CUFFT_R2C : CUFFT_D2Z, bs));
  CUFFT_CHECK_AND_EXIT(cufftPlan1d(
      &planI, fft_size,
      std::is_same_v<real_type, cufftReal> ? CUFFT_C2R : CUFFT_Z2D, bs));

  CUFFT_CHECK_AND_EXIT(cufftSetStream(planF, stream));
  CUFFT_CHECK_AND_EXIT(cufftSetStream(planI, stream));

  // Create execute
  auto cufft_execution = [&](cudaStream_t stream) {
    if constexpr (std::is_same_v<complex_type, cufftComplex>) {
      CUFFT_CHECK_AND_EXIT(
          cufftExecR2C(planF, cufft_r2c_input, cufft_r2c_output));
    } else {
      CUFFT_CHECK_AND_EXIT(
          cufftExecD2Z(planF, cufft_r2c_input, cufft_r2c_output));
    }
    scaling_kernel<fft_size>
        <<<cuda_blocks, block_dim_scaling_kernel, 0, stream>>>(cufft_r2c_output,
                                                               scaling_size, 1);
    if constexpr (std::is_same_v<complex_type, cufftComplex>) {
      CUFFT_CHECK_AND_EXIT(
          cufftExecC2R(planI, cufft_c2r_input, cufft_c2r_output));
    } else {
      CUFFT_CHECK_AND_EXIT(
          cufftExecZ2D(planI, cufft_c2r_input, cufft_c2r_output));
    }
  };

  // Correctness run
  cufft_execution(stream);
  CUDA_CHECK_AND_EXIT(cudaPeekAtLastError());
  CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());

  // Copy total FFT results to host
  const unsigned int total_fft_size = fft_size * bs;
  std::vector<real_type> output_host(
      total_fft_size, std::numeric_limits<real_type>::quiet_NaN());
  CUDA_CHECK_AND_EXIT(cudaMemcpy(output_host.data(), cufft_c2r_output,
                                 total_fft_size * sizeof(real_type),
                                 cudaMemcpyDeviceToHost));
  CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());

  // Performance measurements
  auto time = example::measure_execution_ms(cufft_execution, warm_up_runs,
                                            performance_runs, stream);

  CUFFT_CHECK_AND_EXIT(cufftDestroy(planF));
  CUFFT_CHECK_AND_EXIT(cufftDestroy(planI));

  return example::fft_results<RealType>{output_host, (time / performance_runs)};
}

template <class FFTR2C, class FFTC2R, class T>
example::fft_results<T> cufftdx_conv_1d_r2c_c2r(T *real_input, T *real_output,
                                                unsigned int bs_div_fpb,
                                                cudaStream_t stream) {
  using complex_type = typename FFTR2C::value_type;
  using real_type = typename complex_type::value_type;
  static constexpr unsigned int fft_size = cufftdx::size_of<FFTR2C>::value;

  // Check FFTs are correctly defined
  static_assert(sizeof(T) == sizeof(real_type), "Type size mismatch");
  static_assert(std::alignment_of_v<T> == std::alignment_of_v<real_type>,
                "Type alignment mismatch");

  real_type *cufftdx_input = reinterpret_cast<real_type *>(real_input);
  real_type *cufftdx_output = reinterpret_cast<real_type *>(real_output);

  // Increase max shared memory if needed
  const auto shared_memory_size =
      std::max(FFTR2C::shared_memory_size, FFTC2R::shared_memory_size);
  // Increase max shared memory if needed
  CUDA_CHECK_AND_EXIT(cudaFuncSetAttribute(
      convolution_r2c_c2r_kernel<FFTR2C, FFTC2R>,
      cudaFuncAttributeMaxDynamicSharedMemorySize, shared_memory_size));

  // Create workspaces for FFTs
  cudaError_t error_code;
  auto workspaceR2C = cufftdx::make_workspace<FFTR2C>(error_code, stream);
  CUDA_CHECK_AND_EXIT(error_code);
  auto workspaceC2R = cufftdx::make_workspace<FFTC2R>(error_code, stream);
  CUDA_CHECK_AND_EXIT(error_code);

  // Create execute
  auto cufftdx_execution = [&](cudaStream_t stream) {
    convolution_r2c_c2r_kernel<FFTR2C, FFTC2R>
        <<<bs_div_fpb, FFTR2C::block_dim, shared_memory_size, stream>>>(
            cufftdx_input, cufftdx_output, workspaceR2C, workspaceC2R);
  };

  // Correctness run
  cufftdx_execution(stream);
  CUDA_CHECK_AND_EXIT(cudaPeekAtLastError());
  CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());

  // Copy total FFT results to host
  static const size_t total_fft_size =
      bs_div_fpb * FFTR2C::ffts_per_block * fft_size;
  static const size_t total_fft_size_bytes = total_fft_size * sizeof(real_type);
  std::vector<real_type> output_host(
      total_fft_size, {std::numeric_limits<real_type>::quiet_NaN()});

  CUDA_CHECK_AND_EXIT(cudaMemcpy(output_host.data(), cufftdx_output,
                                 total_fft_size_bytes, cudaMemcpyDeviceToHost));
  CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());

  // Performance run
  auto time = example::measure_execution_ms(cufftdx_execution, warm_up_runs,
                                            performance_runs, stream);

  CUDA_CHECK_AND_EXIT(cudaPeekAtLastError());
  CUDA_CHECK_AND_EXIT(cudaStreamSynchronize(stream));

  return example::fft_results<T>{output_host, (time / performance_runs)};
}

// This example demonstrates how to use cuFFTDx to perform a R2C-C2R
// convolution using one-dimensional FFTs.
//
// One block is run, it calculates two 128-point convolutions by first doing
// forward FFT, then applying pointwise operation, and ending with inverse FFT.
// Data is generated on host, copied to device buffer, and then results are
// copied back to host.
template <unsigned int Arch> void conv_1d_r2c_c2r() {
  using namespace cufftdx;

  using precision_type = float;
  using complex_type = complex<precision_type>;

  // Size of single FFT
  static constexpr unsigned int fft_size = 64;

  // Number of FFTs (convolutions)
  static constexpr unsigned int batches = 2;

  constexpr bool use_suggested = false; // Whether to use suggested values

  static constexpr unsigned int custom_elements_per_thread = 8;
  static constexpr unsigned int custom_ffts_per_block = 2;

  // FFT_base defined common options for FFT and IFFT. FFT_base is not a
  // complete FFT description. In order to complete FFT description directions
  // are specified: forward for FFT, inverse for IFFT.
  using FFT_base = decltype(Block() + Size<fft_size>() +
                            Precision<precision_type>() + SM<Arch>());
  using real_options =
      RealFFTOptions<complex_layout::natural, real_mode::folded>;
  using FFTR2C_base =
      decltype(FFT_base() + Type<fft_type::r2c>() +
               Direction<fft_direction::forward>() + real_options());
  using FFTC2R_base =
      decltype(FFT_base() + Type<fft_type::c2r>() +
               Direction<fft_direction::inverse>() + real_options());

  static constexpr unsigned int elements_per_thread =
      use_suggested ? FFTR2C_base::suggested_elements_per_thread
                    : custom_elements_per_thread;
  static constexpr unsigned int ffts_per_block =
      use_suggested ? FFTR2C_base::suggested_ffts_per_block
                    : custom_ffts_per_block;

  using FFTR2C =
      decltype(FFTR2C_base() + ElementsPerThread<elements_per_thread>() +
               FFTsPerBlock<ffts_per_block>());
  using FFTC2R =
      decltype(FFTC2R_base() + ElementsPerThread<elements_per_thread>() +
               FFTsPerBlock<ffts_per_block>());

  using real_type = precision_of_t<FFTR2C>;

  static_assert(batches % ffts_per_block == 0,
                "Number of batches must be multiple of ffts_per_block");

  std::cout << "[FFT 1D R2C-C2R Convolution (Block Execution)]" << std::endl;
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
  auto real_input_size = batches * fft_size;
  auto real_input_size_bytes = real_input_size * sizeof(real_type);
  auto complex_intermediate_size = batches * (fft_size / 2 + 1);
  auto complex_intermediate_size_bytes =
      complex_intermediate_size * sizeof(complex_type);
#ifdef CUFFTDX_EXAMPLE_DETAIL_DEBUG_CONV_1D
  std::vector<real_type> host_data(real_input_size);
  for (size_t i = 0; i < real_input_size; i++) {
    host_data[i] = precision_type(i);
  }
  std::cout << "input:\n";
  for (size_t i = 0; i < real_input_size; i++) {
    std::cout << host_data[i] << std::endl;
  }
#else
  auto host_data =
      example::get_random_real_data<precision_type>(real_input_size, -1, 1);
#endif

  // Device buffers
  real_type *real_input;
  real_type *real_output;
  complex_type *complex_intermediate;
  CUDA_CHECK_AND_EXIT(cudaMalloc(&real_input, real_input_size_bytes));
  CUDA_CHECK_AND_EXIT(cudaMalloc(&real_output, real_input_size_bytes));
  CUDA_CHECK_AND_EXIT(
      cudaMalloc(&complex_intermediate, complex_intermediate_size_bytes));

  // Copy host data to device buffer
  CUDA_CHECK_AND_EXIT(cudaMemcpy(real_input, host_data.data(),
                                 real_input_size_bytes,
                                 cudaMemcpyHostToDevice));

  cudaStream_t stream;
  CUDA_CHECK_AND_EXIT(cudaStreamCreate(&stream));

  // cuFFTDx 1D R2C-C2R convolution
  auto cufftdx_results = cufftdx_conv_1d_r2c_c2r<FFTR2C, FFTC2R>(
      real_input, real_output, batches / ffts_per_block, stream);

  // cuFFT 1D R2C-C2R convolution for validation
  auto cufft_results = cufft_conv_1d_r2c_c2r<FFTR2C, fft_size>(
      real_input, real_output, complex_intermediate, batches, stream);

  CUDA_CHECK_AND_EXIT(cudaFree(real_input));
  CUDA_CHECK_AND_EXIT(cudaFree(real_output));
  CUDA_CHECK_AND_EXIT(cudaFree(complex_intermediate));
  CUDA_CHECK_AND_EXIT(cudaStreamDestroy(stream));

#ifdef CUFFTDX_EXAMPLE_DETAIL_DEBUG_CONV_1D
  std::cout << "output of cuFFTDx:\n";
  for (size_t i = 0; i < real_input_size; i++) {
    std::cout << cufftdx_results.output[i] << std::endl;
  }

  std::cout << "output of cuFFT:\n";
  for (size_t i = 0; i < real_input_size; i++) {
    std::cout << cufft_results.output[i] << std::endl;
  }
#endif

  // Check correctness
  bool success = true;
  std::cout << "===================================================";
  std::cout << "\nCorrectness results:\n";
  // Check if cuFFTDx results are correct
  {
    auto fft_error = example::fft_signal_error::calculate_for_real_values(
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

template <unsigned int Arch> struct conv1d_functor {
  void operator()() { return conv_1d_r2c_c2r<Arch>(); }
};

int main(int, char **) { return example::sm_runner<conv1d_functor>(); }