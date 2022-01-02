#ifndef PARALLEL_COMPUTING_CUDA_DOT_PRODUCT_INTERFACES_HPP_
#define PARALLEL_COMPUTING_CUDA_DOT_PRODUCT_INTERFACES_HPP_

namespace my::cuda
{

void multiply(const double* a,
              const double* b,
              double* c,
              int size,
              int blocks_per_grid,
              int threads_per_block);

void reduce(const double* idata,
            double* odata,
            int size,
            int blocks_per_grid,
            int threads_per_block);

}  // namespace my::cuda

#endif  // PARALLEL_COMPUTING_CUDA_DOT_PRODUCT_INTERFACES_HPP_
