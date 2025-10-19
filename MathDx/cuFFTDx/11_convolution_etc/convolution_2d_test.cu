#include <iomanip>
#include <iostream>
#include <limits>
#include <vector>

#include <cuda_runtime_api.h>
#include <cufft.h>
#include <cufftdx.hpp>

#include "../common/common.hpp"
#include "../common/block_io.hpp"
#include "../common/block_io_strided.hpp"

inline constexpr unsigned int warm_up_runs = 5;
inline constexpr unsigned int performance_runs = 20;

// #define CUFFTDX_EXAMPLE_DETAIL_DEBUG_CONV_2D

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

template <unsigned int fft_size_x, unsigned int fft_size_y, class T>
example::fft_results<T>
cufft_conv_2d(T *input, T *output, const unsigned int bs, cudaStream_t stream) {
  using complex_type = cufftComplex;
  static_assert(sizeof(T) == sizeof(complex_type), "Type size mismatch");
  static_assert(std::alignment_of_v<T> == std::alignment_of_v<complex_type>,
                "Type alignment mismatch");

  complex_type *cufft_input = reinterpret_cast<complex_type *>(input);
  complex_type *cufft_output = reinterpret_cast<complex_type *>(output);

  static constexpr unsigned int block_dim_scaling_kernel = 1024;

  constexpr unsigned int flat_fft_size = fft_size_x * fft_size_y;
  const unsigned int total_fft_size = bs * flat_fft_size;
  const unsigned int cuda_blocks =
      (total_fft_size + block_dim_scaling_kernel - 1) /
      block_dim_scaling_kernel;

  // Configs of cuFFT plan
  // For 2D FFT
  int n = 2;

  // The dims[0] is outermost (= biggest strides)
  int dims[n] = {fft_size_x, fft_size_y};

  // Element index is computes as below (row-major):
  // input[bs][fft_size_x][fft_size_y]:
  // = input[bs * (fft_size_x * fft_size_y) + x * fft_size_y + y]
  // = input[bs * idist + (x * inembed[1] + y) * istride]
  int istride = 1;
  int idist = fft_size_x * fft_size_y;
  int *inembed = NULL;

  // output[bs][fft_size_x][fft_size_y]:
  // = output[bs * (fft_size_x * fft_size_y) + x * fft_size_y + y]
  // = output[bs * odist + (x * onembed[1] + y) * ostride]
  int ostride = 1;
  int odist = fft_size_x * fft_size_y;
  int *onembed = NULL;

  // Create cuFFT plan
  cufftHandle plan_forward, plan_inverse;
  CUFFT_CHECK_AND_EXIT(cufftPlanMany(&plan_forward, n, dims, inembed, istride,
                                     idist, onembed, ostride, odist, CUFFT_C2C,
                                     bs));
  CUFFT_CHECK_AND_EXIT(cufftPlanMany(&plan_inverse, n, dims, onembed, ostride,
                                     odist, inembed, istride, idist, CUFFT_C2C,
                                     bs));
  CUFFT_CHECK_AND_EXIT(cufftSetStream(plan_forward, stream));
  CUFFT_CHECK_AND_EXIT(cufftSetStream(plan_inverse, stream));

  // Correctness run
  CUFFT_CHECK_AND_EXIT(
      cufftExecC2C(plan_forward, cufft_input, cufft_output, CUFFT_FORWARD));
  scaling_kernel<flat_fft_size>
      <<<cuda_blocks, block_dim_scaling_kernel, 0, stream>>>(cufft_output,
                                                             total_fft_size, 1);
  CUFFT_CHECK_AND_EXIT(
      cufftExecC2C(plan_inverse, cufft_output, cufft_output, CUFFT_INVERSE));
  CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());

  // Copy total FFT result to host
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
        scaling_kernel<flat_fft_size>
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

template <class FFT, class IFFT, unsigned int Stride, unsigned int SizeY>
__launch_bounds__(FFT::max_threads_per_block) __global__
    void kernel_x(typename FFT::value_type *input,
                  typename FFT::value_type *output,
                  typename FFT::workspace_type workspaceF,
                  typename IFFT::workspace_type workspaceI) {
  using complex_type = typename FFT::value_type;
  using scalar_type = typename complex_type::value_type;

  // Local array for thread
  complex_type thread_data[FFT::storage_size];

  // ID of FFT in CUDA block, in range [0, FFT::ffts_per_block)
  const unsigned int local_fft_id = threadIdx.y;
  example::io_strided<FFT>::load_strided<Stride, SizeY>(input, thread_data, local_fft_id);
  
  // Execute FFT (part of 2D FFT-IFFT)
  extern __shared__ __align__(alignof(float4)) complex_type shared_mem[];
  FFT().execute(thread_data, shared_mem, workspaceF);

  // Note: You can do any point-wise operation in here.
  // E.g., Scale values (point-wise operation: normalizing with 1/N)
  scalar_type scale = 1.0 / cufftdx::size_of<FFT>::value * cufftdx::size_of<IFFT>::value;
  for (unsigned int i = 0; i < FFT::elements_per_thread; i++) {
    thread_data[i].x *= scale;
    thread_data[i].y *= scale;
  }

  // Execute IFFT (part of 2D FFT-IFFT)
  IFFT().execute(thread_data, shared_mem, workspaceI);

  // Save results
  example::io_strided<IFFT>::store_strided<Stride, SizeY>(thread_data, output, local_fft_id);
}

template <class FFT>
__launch_bounds__(FFT::max_threads_per_block) __global__
    void kernel_y(typename FFT::value_type *input,
                  typename FFT::value_type *output,
                  typename FFT::workspace_type workspace) {
  using complex_type = typename FFT::value_type;

  // Local array for thread
  complex_type thread_data[FFT::storage_size];

  // ID of FFT in CUDA block, in range [0; FFT::ffts_per_block)
  const unsigned int local_fft_id = threadIdx.y;
  // Load data from global memory to registers
  example::io<FFT>::load(input, thread_data, local_fft_id);

  // Execute FFT
  extern __shared__ __align__(alignof(float4)) complex_type shared_mem[];
  FFT().execute(thread_data, shared_mem, workspace);

  // Save results
  example::io<FFT>::store(thread_data, output, local_fft_id);
}

template <class FFTY, class IFFTY, class FFTX, class IFFTX, class T>
example::fft_results<T> cufftdx_conv_2d(T *input, T *output,
                                        const unsigned int bs,
                                        cudaStream_t stream) {
  using namespace cufftdx;
  using complex_type = typename FFTY::value_type;

  // Checks FFT is correctly defined
  static_assert(std::is_same_v<cufftdx::precision_of_t<FFTX>,
                               cufftdx::precision_of_t<FFTY>>,
                "FFTY and FFTX must have the same precision type");
  static_assert(
      std::is_same_v<typename FFTX::value_type, typename FFTY::value_type>,
      "FFTY and FFTX must have the same complex type");
  static_assert(sizeof(T) == sizeof(complex_type), "Type size mismatch");
  static_assert(std::alignment_of_v<T> == std::alignment_of_v<complex_type>,
                "Type alignment mismatch");

  // Checks IFFT is correctly defined
  static_assert(std::is_same_v<cufftdx::precision_of_t<IFFTX>,
                               cufftdx::precision_of_t<IFFTY>>,
                "IFFTY and IFFTX must have the same precision type");
  static_assert(
      std::is_same_v<typename IFFTX::value_type, typename IFFTY::value_type>,
      "IFFTY and IFFTX must have the same complex type");
  static_assert(sizeof(T) == sizeof(complex_type), "Type size mismatch");
  static_assert(std::alignment_of_v<T> == std::alignment_of_v<complex_type>,
                "Type alignment mismatch");

  static constexpr unsigned int fft_size_x = cufftdx::size_of<FFTX>::value;
  static constexpr unsigned int fft_size_y = cufftdx::size_of<FFTY>::value;
  static constexpr unsigned int flat_fft_size = fft_size_x * fft_size_y;

  static constexpr auto fpb_x = FFTX::ffts_per_block;
  static constexpr auto fpb_y = FFTY::ffts_per_block;

  static const unsigned int bs_x = bs / fpb_x;
  static const unsigned int bs_y = bs / fpb_y;

  complex_type *cufftdx_input = reinterpret_cast<complex_type *>(input);
  complex_type *cufftdx_output = reinterpret_cast<complex_type *>(output);

  // Set shared memory requirements
  constexpr int max_shared_mem_x = FFTX::shared_memory_size;
  constexpr int max_shared_mem_y = FFTY::shared_memory_size;
  CUDA_CHECK_AND_EXIT(cudaFuncSetAttribute(
      kernel_y<FFTY>, cudaFuncAttributeMaxDynamicSharedMemorySize,
      max_shared_mem_y));
  CUDA_CHECK_AND_EXIT(cudaFuncSetAttribute(
      kernel_x<FFTX, IFFTX, FFTY::output_length, FFTY::output_length>, cudaFuncAttributeMaxDynamicSharedMemorySize,
      max_shared_mem_x));
  CUDA_CHECK_AND_EXIT(cudaFuncSetAttribute(
      kernel_y<IFFTY>, cudaFuncAttributeMaxDynamicSharedMemorySize,
      max_shared_mem_y));

  // Create workspaces for FFTs
  cudaError_t error_code;
  auto workspace_xF = cufftdx::make_workspace<FFTX>(error_code, stream);
  CUDA_CHECK_AND_EXIT(error_code);
  auto workspace_yF = cufftdx::make_workspace<FFTY>(error_code, stream);
  CUDA_CHECK_AND_EXIT(error_code);
  auto workspace_xI = cufftdx::make_workspace<IFFTX>(error_code, stream);
  CUDA_CHECK_AND_EXIT(error_code);
  auto workspace_yI = cufftdx::make_workspace<IFFTY>(error_code, stream);
  CUDA_CHECK_AND_EXIT(error_code);

  // Execute conv2d in FFTY -> FFTX-IFFTX -> IFFTY order
  auto cufftdx_execution = [&](cudaStream_t stream) {
    // FFTY forward
    kernel_y<FFTY>
        <<<bs_y / fpb_y, FFTY::block_dim, max_shared_mem_y, stream>>>(
            cufftdx_input, cufftdx_output, workspace_yF);

    // FFTX forward & FFTX inverse (= convolution)
    kernel_x<FFTX, IFFTX, FFTY::output_length, FFTY::output_length>
        <<<bs_x / fpb_x, FFTX::block_dim, max_shared_mem_x, stream>>>(
            cufftdx_output, cufftdx_output, workspace_xF, workspace_xI);

    // FFTY inverse
    kernel_y<IFFTY>
        <<<bs_y / fpb_y, IFFTY::block_dim, max_shared_mem_y, stream>>>(
            cufftdx_output, cufftdx_output, workspace_yI);
  };

  // Correctness run
  cufftdx_execution(stream);
  CUDA_CHECK_AND_EXIT(cudaStreamSynchronize(stream));

  // Copy total FFT result to host
  const unsigned int total_fft_size = bs * flat_fft_size;
  std::vector<complex_type> output_host(
      total_fft_size, {std::numeric_limits<float>::quiet_NaN(),
                       std::numeric_limits<float>::quiet_NaN()});
  CUDA_CHECK_AND_EXIT(cudaMemcpy(output_host.data(), cufftdx_output,
                                 total_fft_size * sizeof(complex_type),
                                 cudaMemcpyDeviceToHost));
  CUDA_CHECK_AND_EXIT(cudaStreamSynchronize(stream));

  // Performance measurements
  auto time = example::measure_execution_ms(cufftdx_execution, warm_up_runs,
                                            performance_runs, stream);

  return example::fft_results<complex_type>{output_host,
                                            (time / performance_runs)};
}

// C2C convolution 2D example
template <unsigned int Arch> void conv_2d() {
  using namespace cufftdx;

  using precision_type = float;
  using complex_type = complex<precision_type>;

  // FFT sizes & cuFFTDx parameters
  static constexpr unsigned int fft_size_x = 16;
  static constexpr unsigned int fft_size_y = 16;
  static constexpr unsigned int ept_y = 8;
  static constexpr unsigned int fpb_y = 1;
  static constexpr unsigned int ept_x = 8;
  static constexpr unsigned int fpb_x = 1;

  // Number of FFTs (convolutions)
  static constexpr unsigned int batches = 2;

  // Suggested EPT & FPB values
  constexpr bool use_suggested = false; // Whether to use suggested values

  // Declaration of cuFFTDx run
  using fft_base = decltype(Block() + Type<fft_type::c2c>() +
                            Precision<precision_type>() + SM<Arch>());
  using fft_y_base = decltype(fft_base() + Size<fft_size_y>());
  using fft_x_base = decltype(fft_base() + Size<fft_size_x>());

  using fft_y_fwd =
      decltype(fft_y_base() + Direction<fft_direction::forward>());
  using fft_y_inv =
      decltype(fft_y_base() + Direction<fft_direction::inverse>());
  using fft_x_fwd =
      decltype(fft_x_base() + Direction<fft_direction::forward>());
  using fft_x_inv =
      decltype(fft_x_base() + Direction<fft_direction::inverse>());

  static constexpr unsigned int elements_per_thread_y =
      use_suggested ? fft_y_fwd::elements_per_thread : ept_y;
  static constexpr unsigned int ffts_per_block_y =
      use_suggested ? fft_y_fwd::suggested_ffts_per_block : fpb_y;
  static constexpr unsigned int elements_per_thread_x =
      use_suggested ? fft_x_fwd::elements_per_thread : ept_x;
  static constexpr unsigned int ffts_per_block_x =
      use_suggested ? fft_x_fwd::suggested_ffts_per_block : fpb_x;

  using FFTY =
      decltype(fft_y_fwd() + ElementsPerThread<elements_per_thread_y>() +
               FFTsPerBlock<ffts_per_block_y>());
  using IFFTY =
      decltype(fft_y_inv() + ElementsPerThread<elements_per_thread_y>() +
               FFTsPerBlock<ffts_per_block_y>());
  using FFTX =
      decltype(fft_x_fwd() + ElementsPerThread<elements_per_thread_x>() +
               FFTsPerBlock<ffts_per_block_x>());
  using IFFTX =
      decltype(fft_x_inv() + ElementsPerThread<elements_per_thread_x>() +
               FFTsPerBlock<ffts_per_block_x>());

  static_assert(batches % (FFTY::ffts_per_block) == 0,
                "Error: batches % FFTY::ffts_per_block != 0");
  static_assert(batches % (FFTX::ffts_per_block) == 0,
                "Error: batches % FFTX::ffts_per_block != 0");

  std::cout << "[FFT 2D C2C Convolution (Block Execution)]" << std::endl;
  std::cout << "===================================================\n";
  std::cout << " cuFFTDx configuration: "
            << "\n";
  std::cout << "  - Y-dim: "
            << "\n";
  std::cout << "   - Elements per thread: " << elements_per_thread_y << "\n";
  std::cout << "   - FFTs per block: " << ffts_per_block_y << "\n";
  std::cout << "  - X-dim: "
            << "\n";
  std::cout << "   - Elements per thread: " << elements_per_thread_x << "\n";
  std::cout << "   - FFTs per block: " << ffts_per_block_x << "\n";
  std::cout << "---------------------------------------------------\n";
  std::cout << " FFT size: (" << fft_size_x << ", " << fft_size_y << ")\n";
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
  static constexpr size_t flat_fft_size = fft_size_x * fft_size_y;
  static constexpr size_t input_size = batches * flat_fft_size;
  static constexpr size_t input_size_bytes = input_size * sizeof(complex_type);
  std::vector<complex_type> host_data(input_size);
  for (size_t i = 0; i < input_size; i++) {
    host_data[i] = complex_type{precision_type(i), -precision_type(i)};
  }

#ifdef CUFFTDX_EXAMPLE_DETAIL_DEBUG_CONV_2D
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

  // Copy host data to device
  CUDA_CHECK_AND_EXIT(cudaMemcpy(input, host_data.data(), input_size_bytes,
                                 cudaMemcpyHostToDevice));

  cudaStream_t stream;
  CUDA_CHECK_AND_EXIT(cudaStreamCreate(&stream));

  // cuFFTDx convolution 2D
  auto cufftdx_results =
      cufftdx_conv_2d<FFTY, IFFTY, FFTX, IFFTX>(input, output, batches, stream);

  // cuFFT convolution 2D as reference
  auto cufft_results =
      cufft_conv_2d<fft_size_x, fft_size_y>(input, output, batches, stream);

  // Free CUDA stream and CUDA buffers
  CUDA_CHECK_AND_EXIT(cudaStreamDestroy(stream));
  CUDA_CHECK_AND_EXIT(cudaFree(input));
  CUDA_CHECK_AND_EXIT(cudaFree(output));

#ifdef CUFFTDX_EXAMPLE_DETAIL_DEBUG_CONV_2D
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

template <unsigned int Arch> struct conv_2d_functor {
  void operator()() { return conv_2d<Arch>(); }
};

int main(int, char **) { return example::sm_runner<conv_2d_functor>(); }