#include <interfaces.hpp>

#include <cuda/runtime_api.hpp>

namespace my::cuda
{

namespace
{
    
template <typename T>
__global__ void multiply_kernel(const T* __restrict__ a,
                                const T* __restrict__ b,
                                T* __restrict__ c,
                                int size)
{
    // Calculate global thread ID
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Boundary check
    if (tid < size)
        c[tid] = a[tid] * b[tid];
}

}  // namespace

void multiply(const double* a,
              const double* b,
              double* c,
              int size,
              int blocks_per_grid,
              int threads_per_block)
{
    ::cuda::launch(multiply_kernel<double>,
                   ::cuda::launch_configuration_t(blocks_per_grid, threads_per_block),
                   a, b, c, size);
}

namespace
{

// Utility class used to avoid linker errors with extern
// unsized shared memory arrays with templated type
template <class T>
struct SharedMemory
{
    __device__ inline operator T*()
    {
        extern __shared__ int __smem[];
        return reinterpret_cast<T*>(__smem);
    }

    __device__ inline operator const T*() const
    {
        extern __shared__ int __smem[];
        return reinterpret_cast<T*>(__smem);
    }
};

// Specialize for double to avoid unaligned memory
// access compile errors
template <>
struct SharedMemory<double>
{
    __device__ inline operator double*()
    {
        extern __shared__ double __smem_d[];
        return reinterpret_cast<double*>(__smem_d);
    }

    __device__ inline operator const double*() const
    {
        extern __shared__ double __smem_d[];
        return reinterpret_cast<double*>(__smem_d);
    }
};

namespace cg = cooperative_groups;

template <typename T>
__global__ void reduce_kernel(const T* __restrict__ g_idata,
                              T* __restrict__ g_odata,
                              unsigned n)
{
    // Handle to thread block group
    cg::thread_block cta = cg::this_thread_block();
    T* sdata = SharedMemory<T>();

    // perform first level of reduction,
    // reading from global memory, writing to shared memory
    unsigned tid = threadIdx.x;
    unsigned i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

    T my_sum = (i < n) ? g_idata[i] : 0;

    if (i + blockDim.x < n)
    {
        my_sum += g_idata[i + blockDim.x];
    }

    sdata[tid] = my_sum;
    cg::sync(cta);

    // do reduction in shared mem
    for (unsigned s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s)
        {
            sdata[tid] = my_sum = my_sum + sdata[tid + s];
        }
        cg::sync(cta);
    }

    // write result for this block to global mem
    if (tid == 0)
    {
        g_odata[blockIdx.x] = my_sum;
    }
}

}  // namespace

void reduce(const double* idata,
            double* odata,
            int size,
            int blocks_per_grid,
            int threads_per_block)
{
    dim3 dim_block(threads_per_block, 1, 1);
    dim3 dim_grid(blocks_per_grid, 1, 1);

    // when there is only one warp per block, we need to allocate two warps
    // worth of shared memory so that we don't index shared memory out of bounds
    int smem_size = (threads_per_block <= 32)
                     ? 2 * threads_per_block * sizeof(double)
                     : threads_per_block * sizeof(double);

    ::cuda::launch(reduce_kernel<double>,
                   ::cuda::launch_configuration_t(dim_grid, dim_block, smem_size),
                   idata, odata, size);
}

}  // namespace my::cuda
