#ifndef CUFFTDX_EXAMPLE_2D_io_strided_conv_smem_HPP
#define CUFFTDX_EXAMPLE_2D_io_strided_conv_smem_HPP

#include "../common/common.hpp"
#include "../common/block_io.hpp"
#include "index_mapper.hpp"

namespace example {

template<dimension Dim, bool Front, int Batches,
         class FFTX_, class IFFTX_, class FFTY_, class IFFTY_>
class io_strided_conv2d_smem {
    static constexpr bool is_c2r_conv =
        cufftdx::type_of<FFTY_>::value == cufftdx::fft_type::c2r &&
        cufftdx::type_of<IFFTY_>::value == cufftdx::fft_type::r2c;

    static constexpr bool is_r2c_conv =
        cufftdx::type_of<FFTY_>::value == cufftdx::fft_type::r2c &&
        cufftdx::type_of<IFFTY_>::value == cufftdx::fft_type::c2r;

    static constexpr bool is_c2c_conv = !is_c2r_conv && !is_r2c_conv;

    using FFTX = std::conditional_t<Front, FFTX_, IFFTX_>;
    using FFTY = std::conditional_t<Front, FFTY_, IFFTY_>;

    using value_type = typename FFTX::value_type;
    static_assert(std::is_same_v<value_type, typename FFTY::value_type>);

    static constexpr unsigned int fft_size_x = cufftdx::size_of<FFTX>::value;
    static constexpr unsigned int fft_size_y = cufftdx::size_of<FFTY>::value;
    static constexpr unsigned int y_dim = is_c2r_conv ? FFTY_::input_length : FFTY_::output_length;

    static constexpr unsigned int flat_batch_size = fft_size_x * y_dim;

    static constexpr auto x_fpb = FFTX::ffts_per_block;
    static constexpr auto x_pad = (example::warp_size + (x_fpb - 1)) / x_fpb;

    using global_layout_x = index_mapper<
        int_pair<fft_size_x, y_dim>,
        int_pair<y_dim, 1>,
        int_pair<Batches, flat_batch_size>
    >;

    using shared_layout_x = index_mapper<
        int_pair<fft_size_x, 1>,
        int_pair<x_fpb, fft_size_x + x_pad>
    >;

    static constexpr int x_pad_bytes =
        fft_size_x * x_fpb * sizeof(value_type) + x_fpb * x_pad * sizeof(value_type);
    static constexpr int x_shared_bytes = std::max<int>(FFTX::shared_memory_size, x_pad_bytes);

    // gmem -> smem for strided dim (complex path)
    template<class FFT, int Subbatches, class GlobalLayout, class SharedLayout>
    __device__ __forceinline__ void load_gmem_to_smem(const value_type* gmem, value_type* smem) const {
        GlobalLayout global_layout;
        SharedLayout shared_layout;

        constexpr auto fpb = FFT::ffts_per_block;
        const auto this_block_fpb = (blockIdx.x == Subbatches / fpb) ? (Subbatches % fpb) : fpb;
        static constexpr auto fft_size = FFT::input_length;

        const int tid = threadIdx.x + threadIdx.y * blockDim.x;
        const int rev_elem_start = tid / this_block_fpb;
        const int rev_batch_id   = tid % this_block_fpb;

        using input_t = typename FFT::input_type;
        auto input_smem = reinterpret_cast<input_t*>(smem);
        auto input_gmem = reinterpret_cast<const input_t*>(gmem);

#pragma unroll
        for (int i = 0; i < FFT::input_ept; ++i) {
            const auto rev_elem_id         = rev_elem_start + i * FFT::stride;
            const auto global_rev_batch_id = rev_batch_id + blockIdx.x * fpb;
            if (!FFT::requires_workspace || (rev_elem_id < fft_size)) {
                input_smem[ shared_layout(rev_elem_id, rev_batch_id) ] =
                    input_gmem[ global_layout(rev_elem_id, global_rev_batch_id, blockIdx.y) ];
            }
        }
    }

    // smem -> gmem for strided dim (complex path)
    template<class FFT, int Subbatches, class SharedLayout, class GlobalLayout>
    __device__ __forceinline__ void store_smem_to_gmem(const value_type* smem, value_type* gmem) const {
        GlobalLayout global_layout;
        SharedLayout shared_layout;

        constexpr auto fpb = FFT::ffts_per_block;
        const auto this_block_fpb = (blockIdx.x == Subbatches / fpb) ? (Subbatches % fpb) : fpb;
        static constexpr auto fft_size = FFT::output_length;

        const int tid = threadIdx.x + threadIdx.y * blockDim.x;
        const int rev_elem_start = tid / this_block_fpb;
        const int rev_batch_id   = tid % this_block_fpb;

        using output_t = typename FFT::output_type;
        auto output_gmem = reinterpret_cast<output_t*>(gmem);
        auto output_smem = reinterpret_cast<const output_t*>(smem);

#pragma unroll
        for (int i = 0; i < FFT::output_ept; ++i) {
            const auto rev_elem_id         = rev_elem_start + i * FFT::stride;
            const auto global_rev_batch_id = rev_batch_id + blockIdx.x * fpb;
            if (!FFT::requires_workspace || (rev_elem_id < fft_size)) {
                output_gmem[ global_layout(rev_elem_id, global_rev_batch_id, blockIdx.y) ] =
                    output_smem[ shared_layout(rev_elem_id, rev_batch_id) ];
            }
        }
    }

    // smem -> rmem
    template<class FFT, class SharedLayout, class Op>
    __device__ __forceinline__ void load_smem_to_rmem(const value_type* smem, value_type* rmem) const {
        SharedLayout shared_layout;
        Op op;

        static constexpr auto fft_size = FFT::input_length;
        using input_t = typename FFT::input_type;
        auto input_rmem = reinterpret_cast<input_t*>(rmem);
        auto input_smem = reinterpret_cast<const input_t*>(smem);

#pragma unroll
        for (int i = 0; i < FFT::input_ept; ++i) {
            const auto elem_id  = threadIdx.x + i * FFT::stride;
            const auto batch_id = threadIdx.y;
            if (!FFT::requires_workspace || (elem_id < fft_size)) {
                input_rmem[i] = op( input_smem[ shared_layout(elem_id, batch_id) ] );
            }
        }
    }

    // rmem -> smem
    template<class FFT, class SharedLayout, class Op>
    __device__ __forceinline__ void store_rmem_to_smem(const value_type* rmem, value_type* smem) const {
        SharedLayout shared_layout;
        Op op;

        static constexpr auto fft_size = FFT::output_length;
        using output_t = typename FFT::output_type;
        auto output_smem = reinterpret_cast<output_t*>(smem);
        auto output_rmem = reinterpret_cast<const output_t*>(rmem);

#pragma unroll
        for (int i = 0; i < FFT::output_ept; ++i) {
            const auto elem_id  = threadIdx.x + i * FFT::stride;
            const auto batch_id = threadIdx.y;
            if (!FFT::requires_workspace || (elem_id < fft_size)) {
                output_smem[ shared_layout(elem_id, batch_id) ] = op( output_rmem[i] );
            }
        }
    }

public:
    static constexpr __device__ __host__ __forceinline__
    size_t get_shared_bytes() {
        if (Dim == dimension::x) return x_shared_bytes;
        else return FFTY::shared_memory_size;
    }

    // gmem -> rmem
    template<typename GmemType, typename SmemType, typename RmemType, class LoadOp = example::identity>
    __device__ __forceinline__ void load_gmem_to_rmem(const GmemType* gmem, SmemType* smem, RmemType* rmem, LoadOp op = {}) const {
        if constexpr (Dim == dimension::x) {
            constexpr int x_batches = y_dim;
            load_gmem_to_smem<FFTX, x_batches, global_layout_x, shared_layout_x>(gmem, smem);
            __syncthreads();
            load_smem_to_rmem<FFTX, shared_layout_x, LoadOp>(smem, rmem);
        } else { // contiguous Y with real transforms
            constexpr auto block_offset = (Front && is_r2c_conv)
                                            ? (fft_size_x * FFTY::input_length)
                                            : flat_batch_size;
            using input_t = typename FFTY::input_type;
            auto gmem_input = reinterpret_cast<const input_t*>(gmem);
            example::io<FFTY>::load(gmem_input + blockIdx.y * block_offset, rmem, threadIdx.y, op);
        }
    }

    // rmem -> gmem
    template<typename RmemType, typename SmemType, typename GmemType, class StoreOp = example::identity>
    __device__ __forceinline__ void store_rmem_to_gmem(const RmemType* rmem, SmemType* smem, GmemType* gmem, StoreOp op = {}) const {
        if constexpr (Dim == dimension::x) {
            constexpr int x_batches = y_dim;
            store_rmem_to_smem<FFTX, shared_layout_x, StoreOp>(rmem, smem);
            __syncthreads();
            store_smem_to_gmem<FFTX, x_batches, shared_layout_x, global_layout_x>(smem, gmem);
        } else { // contiguous Y with real transforms
            constexpr auto block_offset = (!Front && is_r2c_conv)
                                            ? (fft_size_x * FFTY::output_length)
                                            : flat_batch_size;
            using output_t = typename FFTY::output_type;
            auto gmem_output = reinterpret_cast<output_t*>(gmem);
            example::io<FFTY>::store(rmem, gmem_output + blockIdx.y * block_offset, threadIdx.y, op);
        }
    }
};

// Alias for compatibility with the 3D naming convention
template<dimension Dim, bool Front, int Batches, class FFTX_, class IFFTX_, class FFTY_, class IFFTY_>
using io_strided_conv_smem_2d = io_strided_conv2d_smem<Dim, Front, Batches, FFTX_, IFFTX_, FFTY_, IFFTY_>;

} // namespace example

#endif // CUFFTDX_EXAMPLE_2D_io_strided_conv_smem_HPP
