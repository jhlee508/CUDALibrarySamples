#include <iomanip>
#include <iostream>
#include <limits>
#include <vector>

#include <cuda_runtime_api.h>
#include <cufft.h>
#include <cufftdx.hpp>

#include "../common/common.hpp"

inline constexpr unsigned int warm_up_runs = 5;
inline constexpr unsigned int performance_runs = 20;

#define CUFFTDX_EXAMPLE_DETAIL_DEBUG_CONV_2D

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
example::fft_results<T> cufft_conv_2d(T *data, const unsigned int bs,
                                      cudaStream_t stream) {
  using complex_type = cufftComplex;
  static_assert(sizeof(T) == sizeof(complex_type), "Type size mismatch");
  static_assert(std::alignment_of_v<T> == std::alignment_of_v<complex_type>,
                "Type alignment mismatch");

  complex_type *cufft_data = reinterpret_cast<complex_type *>(data);

  static constexpr unsigned int block_dim_scaling_kernel = 1024;

  constexpr unsigned int flat_fft_size = fft_size_x * fft_size_y;

  const unsigned int input_length = bs * flat_fft_size;
  const unsigned int cuda_blocks =
      (input_length + block_dim_scaling_kernel - 1) / block_dim_scaling_kernel;
     
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
  CUFFT_CHECK_AND_EXIT(
      cufftPlanMany(&plan_forward, n, dims, inembed, istride, idist,
                    onembed, ostride, odist, CUFFT_C2C, bs));
  CUFFT_CHECK_AND_EXIT(
      cufftPlanMany(&plan_inverse, n, dims, onembed, ostride, odist,
                    inembed, istride, idist, CUFFT_C2C, bs));
  CUFFT_CHECK_AND_EXIT(cufftSetStream(plan_forward, stream));
  CUFFT_CHECK_AND_EXIT(cufftSetStream(plan_inverse, stream));

  // Correctness run
  CUFFT_CHECK_AND_EXIT(
      cufftExecC2C(plan_forward, cufft_data, cufft_data, CUFFT_FORWARD));
  scaling_kernel<flat_fft_size>
      <<<cuda_blocks, block_dim_scaling_kernel, 0, stream>>>(cufft_data,
                                                             input_length, 1);
  CUFFT_CHECK_AND_EXIT(
      cufftExecC2C(plan_inverse, cufft_data, cufft_data, CUFFT_INVERSE));
  CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());

  // Copy single FFT result to host
  std::vector<T> output_host(input_length,
                             {std::numeric_limits<float>::quiet_NaN(),
                              std::numeric_limits<float>::quiet_NaN()});
  CUDA_CHECK_AND_EXIT(cudaMemcpy(output_host.data(), cufft_data,
                                 input_length * sizeof(complex_type),
                                 cudaMemcpyDeviceToHost));
  CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());

  // Performance measurements
  auto time = example::measure_execution_ms(
      [&](cudaStream_t /* stream */) {
        CUFFT_CHECK_AND_EXIT(
            cufftExecC2C(plan_forward, cufft_data, cufft_data, CUFFT_FORWARD));
        scaling_kernel<flat_fft_size>
            <<<cuda_blocks, block_dim_scaling_kernel, 0, stream>>>(
                cufft_data, input_length, 1);
        CUFFT_CHECK_AND_EXIT(
            cufftExecC2C(plan_inverse, cufft_data, cufft_data, CUFFT_INVERSE));
      },
      warm_up_runs, performance_runs, stream);

  CUFFT_CHECK_AND_EXIT(cufftDestroy(plan_forward));
  CUFFT_CHECK_AND_EXIT(cufftDestroy(plan_inverse));

  return example::fft_results<T>{output_host, (time / performance_runs)};
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
  // using fft_base =
  //     decltype(Block() + Type <
  //              fft_type::c2c() + Precision<precision_type>() + SM<Arch>());
  // using fft_y_base =
  //     decltype(fft_base() + Size<fft_size_y>() + ElementsPerThread<ept_y>() +
  //              FFTsPerBlock<fpb_y>());
  // using fft_x_base =
  //     decltype(fft_base() + Size<fft_size_x>() + ElementsPerThread<ept_x>() +
  //              FFTsPerBlock<fpb_x>());

  // using fft_y_fwd =
  //     decltype(fft_y_base() + Direction<fft_direction::forward>());
  // using fft_y_inv =
  //     decltype(fft_y_base() + Direction<fft_direction::inverse>());
  // using fft_x_fwd =
  //     decltype(fft_x_base() + Direction<fft_direction::forward>());
  // using fft_x_inv =
  //     decltype(fft_x_base() + Direction<fft_direction::inverse>());

  // static constexpr unsigned int elements_per_thread_y =
  //     use_suggested ? fft_y_fwd::elements_per_thread : ept_y;
  // static constexpr unsigned int ffts_per_block_y =
  //     use_suggested ? fft_y_fwd::suggested_ffts_per_block : fpb_y;
  // static constexpr unsigned int elements_per_thread_x =
  //     use_suggested ? fft_x_fwd::elements_per_thread : ept_x;
  // static constexpr unsigned int ffts_per_block_x =
  //     use_suggested ? fft_x_fwd::suggested_ffts_per_block : fpb_x;

  // using FFTY =
  //     decltype(fft_y_fwd() + ElementsPerThread<elements_per_thread_y>() +
  //              FFTsPerBlock<ffts_per_block_y>());
  // using IFFTY =
  //     decltype(fft_y_inv() + ElementsPerThread<elements_per_thread_y>() +
  //              FFTsPerBlock<ffts_per_block_y>());
  // using FFTX =
  //     decltype(fft_x_fwd() + ElementsPerThread<elements_per_thread_x>() +
  //              FFTsPerBlock<ffts_per_block_x>());
  // using IFFTX =
  //     decltype(fft_x_inv() + ElementsPerThread<elements_per_thread_x>() +
  //              FFTsPerBlock<ffts_per_block_x>());

  std::cout << "[C2C FFT 2D Convolution]" << std::endl;
  std::cout << "FFT size: (" << fft_size_x << ", " << fft_size_y << ")\n";
  std::cout << "Batches: " << batches << "\n";
  // std::cout << "cuFFTDx configuration: " << "\n";
  // std::cout << " - Y-dim: " << "\n";
  // std::cout << "   - Elements per thread: " << elements_per_thread_y << "\n";
  // std::cout << "   - FFTs per block: " << ffts_per_block_y << "\n";
  // std::cout << " - X-dim: " << "\n";
  // std::cout << "   - Elements per thread: " << elements_per_thread_x << "\n";
  // std::cout << "   - FFTs per block: " << ffts_per_block_x << "\n";

  // Host data
  static constexpr size_t flat_fft_size = fft_size_x * fft_size_y;
  static constexpr size_t input_size = batches * flat_fft_size;
  static constexpr size_t input_size_bytes =
      input_size * sizeof(complex_type);
  std::vector<complex_type> host_data(input_size);
  for (size_t i = 0; i < input_size; i++) {
    host_data[i] = complex_type{float(i), -float(i)};
  }

#ifdef CUFFTDX_EXAMPLE_DETAIL_DEBUG_CONV_2D
  std::cout << "input: [1st FFT]\n";
  for (size_t i = 0; i < input_size; i++) {
    std::cout << host_data[i].x << " " << host_data[i].y << std::endl;
  }
#endif

  // Copy to device buffer
  complex_type *data;
  CUDA_CHECK_AND_EXIT(cudaMalloc(&data, input_size_bytes));
  CUDA_CHECK_AND_EXIT(cudaMemcpy(data, host_data.data(), input_size_bytes,
                                 cudaMemcpyHostToDevice));

  cudaStream_t stream;
  CUDA_CHECK_AND_EXIT(cudaStreamCreate(&stream));

  // cuFFTDx convolution 2D

  // cuFFT convolution 2D as reference
  auto cufft_results =
      cufft_conv_2d<fft_size_x, fft_size_y>(data, batches, stream);

  // Free CUDA stream and CUDA buffers
  CUDA_CHECK_AND_EXIT(cudaStreamDestroy(stream));
  CUDA_CHECK_AND_EXIT(cudaFree(data));

#ifdef CUFFTDX_EXAMPLE_DETAIL_DEBUG_CONV_2D
  std::cout << "output of cuFFT: [1st FFT]\n";
  for (size_t i = 0; i < input_size; i++) {
    std::cout << cufft_results.output[i].x << " " << cufft_results.output[i].y
              << std::endl;
  }
#endif

  // Check correctness
  bool success = true;

  // Print performance results
  if (success) {
    std::cout << "===================================================";
    std::cout << "\nPerformance results:\n";

    std::cout << " FFT size: (" << fft_size_x << ", " << fft_size_y << ")\n";

    std::cout << std::setw(28) << " cuFFT: " << cufft_results.avg_time_in_ms
              << " [ms]\n";
    std::cout << "\nSuccess (validation)" << std::endl;
  } else {
    std::cout << "\nFailure (validation)" << std::endl;
    std::exit(1);
  }
}

template <unsigned int Arch> struct conv_2d_functor {
  void operator()() { return conv_2d<Arch>(); }
};

int main(int, char **) { return example::sm_runner<conv_2d_functor>(); }