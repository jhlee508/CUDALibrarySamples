
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

// Fused 2D FFT convolution (R2C path): pre-processing -> front 2D FFT (R2C) ->
// frequency-domain filter -> back 2D FFT (C2R) -> post-processing Front/back
// directions are fixed here to Forward (R2C) then Inverse (C2R).
template <int Batches, class FFTXPartial, class FFTYPartial, class LoadFunctor,
          class FilterFunctor, class StoreFunctor, typename InputType>
auto cufftdx_2d_convolution_r2c_c2r(InputType *input, InputType *output,
                                    cudaStream_t stream) {
  using namespace cufftdx;
  using id_op = example::identity;

  // Compose concrete FFT descriptors with directions/types
  using FFTX = decltype(
      FFTXPartial() + Direction<fft_direction::forward>()); // c2c forward on X
  using IFFTX = decltype(
      FFTXPartial() + Direction<fft_direction::inverse>()); // c2c inverse on X

  // Y is contiguous and real-transform dimension (R2C/C2R)
  using FFTY = decltype(
      FFTYPartial() + Type<fft_type::r2c>() +
      RealFFTOptions<complex_layout::natural, cufftdx::real_mode::normal>());
  using IFFTY = decltype(
      FFTYPartial() + Type<fft_type::c2r>() +
      RealFFTOptions<complex_layout::natural, cufftdx::real_mode::normal>());

  using input_type = InputType;                 // real scalar (float/double)
  using value_type = typename FFTY::value_type; // complex type produced by R2C
  using precision = cufftdx::precision_of_t<FFTXPartial>; // float or double

  // Sizes
  static constexpr unsigned int x_size = cufftdx::size_of<FFTXPartial>::value;
  static constexpr unsigned int y_size = cufftdx::size_of<FFTYPartial>::value;
  static constexpr unsigned int y_output_length =
      FFTY::output_length; // y_size/2 + 1 for R2C

  // Flat sizes
  static constexpr unsigned int flat_fft_size = x_size * y_size; // real domain
  static constexpr unsigned int total_complex_elems =
      x_size * y_output_length; // frequency domain (R2C)

  // Inter buffer holds complex frequency-domain data
  value_type *inter = nullptr;
  const auto flat_inter_size_bytes = total_complex_elems * sizeof(value_type);
  CUDA_CHECK_AND_EXIT(cudaMalloc(&inter, Batches * flat_inter_size_bytes));

  // Batching factors (sub-batches per dimension)
  static constexpr auto x_fpb = FFTX::ffts_per_block;
  static constexpr auto y_fpb = FFTY::ffts_per_block;

  static constexpr unsigned int x_batches = total_complex_elems / x_size;
  static constexpr unsigned int y_batches =
      total_complex_elems / y_output_length;

  // IO adaptors
  using io_x_front =
      example::io_strided_conv2d_smem<dimension::x, true, Batches, FFTX, IFFTX,
                                      FFTY, IFFTY>;
  using io_x_back =
      example::io_strided_conv2d_smem<dimension::x, false, Batches, FFTX, IFFTX,
                                      FFTY, IFFTY>;
  using io_y_front =
      example::io_strided_conv2d_smem<dimension::y, true, Batches, FFTX, IFFTX,
                                      FFTY, IFFTY>;
  using io_y_back =
      example::io_strided_conv2d_smem<dimension::y, false, Batches, FFTX, IFFTX,
                                      FFTY, IFFTY>;

  // Workspaces
  cudaError_t err;
  auto workspace_x = cufftdx::make_workspace<FFTX>(err, stream);
  auto workspace_y = cufftdx::make_workspace<FFTY>(err, stream);

  // Shared memory requirements
  constexpr int x_max_bytes = io_x_front::get_shared_bytes();
  constexpr int y_max_bytes = io_y_front::get_shared_bytes();

  auto set_kernel_shared_size = [](auto kernel, int size) {
    CUDA_CHECK_AND_EXIT(cudaFuncSetAttribute(
        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, size));
  };

  // Kernels:
  // Y-front: real -> complex (R2C), can fuse LoadFunctor
  auto kernel_y_front =
      fft_kernel<FFTY, io_y_front, LoadFunctor, id_op, input_type, value_type>;
  set_kernel_shared_size(kernel_y_front, y_max_bytes);

  // X: convolution in frequency domain (complex), fuse FilterFunctor
  auto kernel_x =
      convolution_kernel<FFTX, IFFTX, FilterFunctor, io_x_front, io_x_back>;
  set_kernel_shared_size(kernel_x, x_max_bytes);

  // Y-back: complex -> real (C2R), can fuse StoreFunctor
  auto kernel_y_back =
      fft_kernel<IFFTY, io_y_back, id_op, StoreFunctor, value_type, input_type>;
  set_kernel_shared_size(kernel_y_back, y_max_bytes);

  // Launch in Y(front) -> X(conv) -> Y(back) order
  auto cufftdx_execution = [&](cudaStream_t s) {
    // Grid: (subbatches/FPB, Batches, 1), Block: (Size/EPT, FPB)
    kernel_y_front<<<dim3{example::div_up(y_batches, y_fpb), Batches, 1},
                     FFTY::block_dim, y_max_bytes, s>>>(y_batches, input, inter,
                                                        workspace_y);

    kernel_x<<<dim3{example::div_up(x_batches, x_fpb), Batches, 1},
               FFTX::block_dim, x_max_bytes, s>>>(x_batches, inter, inter,
                                                  workspace_x);

    kernel_y_back<<<dim3{example::div_up(y_batches, y_fpb), Batches, 1},
                    IFFTY::block_dim, y_max_bytes, s>>>(y_batches, inter,
                                                        output, workspace_y);
  };

  // Correctness run
  cufftdx_execution(stream);
  CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());

  // Copy results to host (real domain size)
  const size_t flat_input_size_bytes = flat_fft_size * sizeof(input_type);
  std::vector<input_type> output_host(Batches * flat_fft_size,
                                      example::get_nan<input_type>());
  CUDA_CHECK_AND_EXIT(cudaMemcpy(output_host.data(), output,
                                 Batches * flat_input_size_bytes,
                                 cudaMemcpyDeviceToHost));
  CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());

  // Performance
  auto time_ms = example::measure_execution_ms(
      cufftdx_execution, cufftdx_example_warm_up_runs,
      cufftdx_example_performance_runs, stream);

  CUDA_CHECK_AND_EXIT(cudaFree(inter));

  return example::fft_results<input_type>{
      output_host, (time_ms / cufftdx_example_performance_runs)};
}

template <int Arch> int conv_2d() {
  using namespace cufftdx;

  // Precision (real input type is precision)
  constexpr bool is_double_precision = false;
  using precision = std::conditional_t<is_double_precision, double, float>;

  // 2D Convolution configuration
  static constexpr unsigned int batches = 1;

  // X (outer / strided)
  static constexpr unsigned int fft_size_x = 16384;
  static constexpr unsigned int x_ept = 32;
  static constexpr unsigned int x_fpb = 1;

  // Y (contiguous, real-transform dimension)
  static constexpr unsigned int fft_size_y = 16384;
  static constexpr unsigned int y_ept = 32;
  static constexpr unsigned int y_fpb = 1;

  // Functors fused into FFTs
  using load_functor = example::rational_scaler<1, 1>;
  using filter_functor = example::rational_scaler<1, 1>;
  using store_functor = example::rational_scaler<1, fft_size_x * fft_size_y>;

  // cuFFTDx description types (partials)
  using fftx_partial =
      decltype(Block() + Size<fft_size_x>() + Type<fft_type::c2c>() +
               ElementsPerThread<x_ept>() + FFTsPerBlock<x_fpb>() +
               Precision<precision>() + SM<Arch>());

  using ffty_partial =
      decltype(Block() + Size<fft_size_y>() + ElementsPerThread<y_ept>() +
               FFTsPerBlock<y_fpb>() + Precision<precision>() + SM<Arch>());

  using input_type = precision; // real

  std::cout << "[FFT 2D R2C-C2R Convolution (Block Execution)]" << std::endl;
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
  std::vector<input_type> host_input(input_size);
  std::cout << "input:\n";
  for (size_t i = 0; i < host_input.size(); i++) {
    host_input[i] = precision(i);
    std::cout << host_input[i] << std::endl;
  }
#else
  auto host_input =
      example::get_random_real_data<input_type>(batches * flat_fft_size, -1, 1);
#endif

  // Device buffers
  input_type *input = nullptr;
  input_type *output = nullptr;
  const auto flat_input_size_bytes = flat_fft_size * sizeof(input_type);
  CUDA_CHECK_AND_EXIT(cudaMalloc(&input, batches * flat_input_size_bytes));
  CUDA_CHECK_AND_EXIT(cudaMalloc(&output, batches * flat_input_size_bytes));
  CUDA_CHECK_AND_EXIT(cudaMemcpy(input, host_input.data(),
                                 batches * flat_input_size_bytes,
                                 cudaMemcpyHostToDevice));
  CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());

  cudaStream_t stream;
  CUDA_CHECK_AND_EXIT(cudaStreamCreate(&stream));

  // Run cuFFTDx 2D R2C fused convolution
  auto cufftdx_results =
      cufftdx_2d_convolution_r2c_c2r<batches, fftx_partial, ffty_partial,
                                     load_functor, filter_functor,
                                     store_functor>(input, output, stream);

  // Run cuFFT reference (2D, real, forward convolution path)
  auto cufft_results =
      cufft_2d_convolution<true, true, // IsReal=true, IsForward=true
                           load_functor, filter_functor, store_functor>(
          fft_size_x, fft_size_y, batches, input, output, stream,
          cufftdx_example_warm_up_runs, cufftdx_example_performance_runs);

  // Cleanup
  CUDA_CHECK_AND_EXIT(cudaStreamDestroy(stream));
  CUDA_CHECK_AND_EXIT(cudaFree(input));
  CUDA_CHECK_AND_EXIT(cudaFree(output));

#ifdef CUFFTDX_EXAMPLE_DETAIL_DEBUG_CONV_2D
  std::cout << "output of cuFFTDx:\n";
  for (size_t i = 0; i < cufftdx_results.output.size(); i++) {
    std::cout << cufftdx_results.output[i] << std::endl;
  }
  std::cout << "output of cuFFT:\n";
  for (size_t i = 0; i < cufft_results.output.size(); i++) {
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
