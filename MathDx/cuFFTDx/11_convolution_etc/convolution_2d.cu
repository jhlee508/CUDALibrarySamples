#include <iomanip>
#include <iostream>
#include <limits>
#include <vector>

#include <cuda_runtime_api.h>
#include <cufftdx.hpp>

#include "../common/common.hpp"
#include "../common/random.hpp"

#include "io_strided_conv2d_smem.hpp"
#include "kernels_conv2d.hpp"
#include "reference_conv2d.hpp"

inline constexpr unsigned int cufftdx_example_warm_up_runs = 5;
inline constexpr unsigned int cufftdx_example_performance_runs = 20;

// #define CUFFTDX_EXAMPLE_DETAIL_DEBUG_CONV_2D

using namespace example;

// Fused 2D FFT convolution (C2C) with pre/post-processing.
// Pipeline: pre-processing -> front 2D FFT -> filter (element-wise) -> back 2D
// FFT.
template <int Batches, bool IsForwardConv, class FFTXPartial, class FFTYPartial,
          class LoadFunctor, class FilterFunctor, class StoreFunctor,
          typename ValueType>
auto cufftdx_2d_convolution(ValueType *input, ValueType *output,
                            cudaStream_t stream) {
  using namespace cufftdx;
  using id_op = example::identity;

  using precision = cufftdx::precision_of_t<FFTXPartial>;
  constexpr bool is_double = std::is_same_v<precision, double>;
  using vector_type = std::conditional_t<is_double, double2, float2>;
  using value_type = ValueType;

  // Sizes
  static constexpr unsigned int fft_size_x =
      cufftdx::size_of<FFTXPartial>::value;
  static constexpr unsigned int fft_size_y =
      cufftdx::size_of<FFTYPartial>::value;
  static constexpr unsigned int flat_batch_size = fft_size_x * fft_size_y;

  // Directions
  using FFTX = decltype(FFTXPartial() + Direction < IsForwardConv
                            ? fft_direction::forward
                            : fft_direction::inverse > ());
  using IFFTX = decltype(FFTXPartial() + Direction < IsForwardConv
                             ? fft_direction::inverse
                             : fft_direction::forward > ());

  using FFTY = decltype(FFTYPartial() + Direction < IsForwardConv
                            ? fft_direction::forward
                            : fft_direction::inverse > ());
  using IFFTY = decltype(FFTYPartial() + Direction < IsForwardConv
                             ? fft_direction::inverse
                             : fft_direction::forward > ());

  static constexpr auto x_fpb = FFTX::ffts_per_block;
  static constexpr auto y_fpb = FFTY::ffts_per_block;

  static constexpr unsigned int x_batches = flat_batch_size / fft_size_x;
  static constexpr unsigned int y_batches = flat_batch_size / fft_size_y;

  // Front/back FFTs require same memory access (C2C), so "Front"=true
  using io_x = io_strided_conv2d_smem<dimension::x, true, Batches, FFTX, IFFTX,
                                      FFTY, IFFTY>;
  using io_y = io_strided_conv2d_smem<dimension::y, true, Batches, FFTX, IFFTX,
                                      FFTY, IFFTY>;

  cudaError_t err;
  auto workspace_x = cufftdx::make_workspace<FFTX>(err, stream);
  auto workspace_y = cufftdx::make_workspace<FFTY>(err, stream);

  // Shared memory requirements
  constexpr int x_max_bytes = io_x::get_shared_bytes();
  constexpr int y_max_bytes = io_y::get_shared_bytes();

  auto set_kernel_shared_size = [](auto kernel, int size) {
    CUDA_CHECK_AND_EXIT(cudaFuncSetAttribute(
        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, size));
  };

  // Kernels:
  // Y-front does load_functor on the fly (pre-proc fused)
  auto kernel_y_front = fft_kernel<FFTY, io_y, LoadFunctor, id_op, value_type>;
  set_kernel_shared_size(kernel_y_front, y_max_bytes);

  // X is the strided dimension where we fuse the element-wise filter in
  // frequency domain
  auto kernel_x = convolution_kernel<FFTX, IFFTX, FilterFunctor, io_x>;
  set_kernel_shared_size(kernel_x, x_max_bytes);

  // Y-back does store_functor on the fly (post-proc fused)
  auto kernel_y_back = fft_kernel<IFFTY, io_y, id_op, StoreFunctor, value_type>;
  set_kernel_shared_size(kernel_y_back, y_max_bytes);

  // Execute cuFFTDx in Y–X order (Y is contiguous, X is the outer/strided)
  auto cufftdx_execution = [&](cudaStream_t s) {
    // Grid  : (Total Subbatches / FPB, Batches, 1)
    // Block : (Size / EPT, FPB)
    kernel_y_front<<<dim3{example::div_up(y_batches, y_fpb), Batches, 1},
                     FFTY::block_dim, y_max_bytes, s>>>(y_batches, input,
                                                        output, workspace_y);

    kernel_x<<<dim3{example::div_up(x_batches, x_fpb), Batches, 1},
               FFTX::block_dim, x_max_bytes, s>>>(x_batches, output, output,
                                                  workspace_x);

    kernel_y_back<<<dim3{example::div_up(y_batches, y_fpb), Batches, 1},
                    FFTY::block_dim, y_max_bytes, s>>>(y_batches, output,
                                                       output, workspace_y);
  };

  // Correctness run
  cufftdx_execution(stream);
  CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());

  // Copy results to host
  const size_t flat_fft_size = fft_size_x * fft_size_y;
  const size_t flat_fft_size_bytes = flat_fft_size * sizeof(vector_type);
  std::vector<vector_type> output_host(
      Batches * flat_fft_size, {std::numeric_limits<precision>::quiet_NaN(),
                                std::numeric_limits<precision>::quiet_NaN()});

  CUDA_CHECK_AND_EXIT(cudaMemcpy(output_host.data(), output,
                                 Batches * flat_fft_size_bytes,
                                 cudaMemcpyDeviceToHost));
  CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());

  // Performance measurements
  auto time_ms = example::measure_execution_ms(
      cufftdx_execution, cufftdx_example_warm_up_runs,
      cufftdx_example_performance_runs, stream);

  return example::fft_results<vector_type>{
      output_host, (time_ms / cufftdx_example_performance_runs)};
}

template <int Arch> int conv_2d() {
  using namespace cufftdx;

  // 2D Convolution configuration
  static constexpr unsigned int batches = 2;

  // X: outer (strided)
  static constexpr unsigned int fft_size_x = 16;
  static constexpr unsigned int x_ept = 8;
  static constexpr unsigned int x_fpb = 1;

  // Y: contiguous
  static constexpr unsigned int fft_size_y = 16;
  static constexpr unsigned int y_ept = 8;
  static constexpr unsigned int y_fpb = 1;

  // Functors fused into FFTs
  using load_functor = example::rational_scaler<1, 1>;
  using filter_functor = example::rational_scaler<1, 1>;
  using store_functor = example::rational_scaler<1, fft_size_x * fft_size_y>;

  // Is this forward or inverse convolution?
  static constexpr bool is_forward = false;

  // Use double (as in your 3D sample) for apples-to-apples with cuFFT reference
  constexpr bool is_double_precision = true;
  using precision = std::conditional_t<is_double_precision, double, float>;

  using fftx_partial =
      decltype(Block() + Size<fft_size_x>() + Type<fft_type::c2c>() +
               ElementsPerThread<x_ept>() + FFTsPerBlock<x_fpb>() +
               Precision<precision>() + SM<Arch>());

  using ffty_partial =
      decltype(Block() + Size<fft_size_y>() + Type<fft_type::c2c>() +
               ElementsPerThread<y_ept>() + FFTsPerBlock<y_fpb>() +
               Precision<precision>() + SM<Arch>());

  using value_type = cufftdx::complex<precision>; // = complex_type

  std::cout << "[FFT 2D C2C Convolution (Block Execution)]" << std::endl;
  std::cout << "===================================================\n";
  std::cout << " cuFFTDx configuration: "
            << "\n";
  std::cout << "  - Y-dim: "
            << "\n";
  std::cout << "   - Elements per thread: " << y_ept << "\n";
  std::cout << "   - FFTs per block: " << y_fpb << "\n";
  std::cout << "  - X-dim: "
            << "\n";
  std::cout << "   - Elements per thread: " << x_ept << "\n";
  std::cout << "   - FFTs per block: " << x_fpb << "\n";
  std::cout << "---------------------------------------------------\n";
  std::cout << " FFT size: (" << fft_size_x << ", " << fft_size_y << ")\n";
  std::cout << " Batches: " << batches << "\n";
  static const std::string precision_str =
      (typeid(precision) == typeid(float))
          ? "float"
          : (typeid(precision) == typeid(__half))
                ? "half"
                : (typeid(precision) == typeid(double)) ? "double" : "unknown";
  std::cout << " Precision: " << precision_str << "\n";

  // Host input
  static const unsigned int flat_fft_size = fft_size_x * fft_size_y;

#ifdef CUFFTDX_EXAMPLE_DETAIL_DEBUG_CONV_2D
  static constexpr size_t input_size = batches * flat_fft_size;
  std::vector<value_type> host_input(input_size);
  std::cout << "input:\n";
  for (size_t i = 0; i < host_input.size(); i++) {
    host_input[i] = value_type{precision(i), -precision(i)};
    std::cout << host_input[i].x << " " << host_input[i].y << "\n";
  }
#else
  auto host_input = example::get_random_complex_data<precision>(
      batches * flat_fft_size, -1, 1);
#endif

  // Device buffers
  value_type *input = nullptr;
  value_type *output = nullptr;
  const auto flat_fft_size_bytes = flat_fft_size * sizeof(value_type);
  CUDA_CHECK_AND_EXIT(cudaMalloc(&input, batches * flat_fft_size_bytes));
  CUDA_CHECK_AND_EXIT(cudaMalloc(&output, batches * flat_fft_size_bytes));
  CUDA_CHECK_AND_EXIT(cudaMemcpy(input, host_input.data(),
                                 batches * flat_fft_size_bytes,
                                 cudaMemcpyHostToDevice));
  CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());

  cudaStream_t stream;
  CUDA_CHECK_AND_EXIT(cudaStreamCreate(&stream));

  // cuFFTDx convolution 2D
  auto cufftdx_results =
      cufftdx_2d_convolution<batches, is_forward, fftx_partial, ffty_partial,
                             load_functor, filter_functor, store_functor>(
          input, output, stream);

  // cuFFT convolution 2D as reference
  auto cufft_results = cufft_2d_convolution<false, is_forward, load_functor,
                                            filter_functor, store_functor>(
      fft_size_x, fft_size_y, batches, input, output, stream,
      cufftdx_example_warm_up_runs, cufftdx_example_performance_runs);

  // Free CUDA stream and CUDA buffers
  CUDA_CHECK_AND_EXIT(cudaStreamDestroy(stream));
  CUDA_CHECK_AND_EXIT(cudaFree(input));
  CUDA_CHECK_AND_EXIT(cudaFree(output));

#ifdef CUFFTDX_EXAMPLE_DETAIL_DEBUG_CONV_2D
  std::cout << "output of cuFFTDx:\n";
  for (size_t i = 0; i < cufftdx_results.output.size(); i++) {
    std::cout << cufftdx_results.output[i].x << " "
              << cufftdx_results.output[i].y << "\n";
  }
  std::cout << "output of cuFFT:\n";
  for (size_t i = 0; i < cufft_results.output.size(); i++) {
    std::cout << cufft_results.output[i].x << " " << cufft_results.output[i].y
              << "\n";
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

    std::cout << std::setw(28) << "cuFFTDx: " << cufftdx_results.avg_time_in_ms
              << " [ms]\n";
    std::cout << std::setw(28) << "cuFFT: " << cufft_results.avg_time_in_ms
              << " [ms]\n";
    std::cout << "===================================================";
    std::cout << "\nSuccess (validation)\n";
    return 0;
  } else {
    std::cout << "===================================================";
    std::cout << "\nFailure (validation)\n";
    std::exit(1);
  }
}

template <unsigned int Arch> struct conv_2d_functor {
  int operator()() { return conv_2d<Arch>(); }
};

int main(int, char **) { return example::sm_runner<conv_2d_functor>(); }
