#include <benchmark.hpp>
#include <interfaces.hpp>

#include <cuda/runtime_api.hpp>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <random>
#include <string_view>

namespace
{

static constexpr int MAX_BLOCK_DIM_SIZE = 65535;
static constexpr int MAX_THREADS_COUNT = 256;
static constexpr std::size_t ITERATIONS_COUNT = 10'000;

template <typename T>
T cpu_dot_product(const T* a, const T* b, int size) noexcept
{
    T ans = 0;
    for (int i = 0; i < size; ++i)
    {
        ans += a[i] * b[i];
    }
    return ans;
}

class DeviceSingleton
{
public:
    static DeviceSingleton& get_instance()
    {
        static DeviceSingleton instance;
        return instance;
    }

    DeviceSingleton(const DeviceSingleton&) = delete;
    DeviceSingleton& operator=(const DeviceSingleton&) = delete;
    DeviceSingleton(DeviceSingleton&&) = delete;
    DeviceSingleton& operator=(DeviceSingleton&&) = delete;

    const cuda::device_t& device() const { return m_device; }
    cuda::device_t& device() { return m_device; }
    const cuda::device::properties_t& properties() const { return m_properties; }

private:
    DeviceSingleton()
        : m_device(cuda::device::current::get())
        , m_properties(m_device.properties())
    {
    }

    cuda::device_t m_device;
    cuda::device::properties_t m_properties;
};

struct CudaParams
{
    int blocks_count;
    int threads_count;
};

// Compute the number of threads and blocks to use for reduction kernel.
// We set threads / block to the minimum of max_threads and n/2.
CudaParams get_cuda_params(int elements_count)
{
    auto next_pow2 = [](unsigned x) -> unsigned
    {
        --x;
        x |= x >> 1;
        x |= x >> 2;
        x |= x >> 4;
        x |= x >> 8;
        x |= x >> 16;
        return ++x;
    };

    int threads_count = (elements_count < MAX_THREADS_COUNT * 2) ? next_pow2((elements_count + 1) / 2) : MAX_THREADS_COUNT;
    int blocks_count = (elements_count + (threads_count * 2 - 1)) / (threads_count * 2);

    // get device capability, to avoid block/grid size exceed the upper bound
    const auto& prop = DeviceSingleton::get_instance().properties();

    if (threads_count * blocks_count > prop.maxGridSize[0] * prop.max_threads_per_block())
    {
        throw std::runtime_error("n is too large, please choose a smaller number!");
    }

    if (blocks_count > prop.maxGridSize[0])
    {
#ifndef NDEBUG
        std::cout << "Grid size " << blocks_count << " exceeds the device capability "
                  << prop.maxGridSize[0] << ", set block size as " << threads_count * 2
                  << " (original " << threads_count << ")\n";
#endif
        blocks_count /= 2;
        threads_count *= 2;
    }

    return CudaParams{ .blocks_count = blocks_count, .threads_count = threads_count };
}

template <class T>
T run_reduce(const T* device_idata, T* device_odata, T* host_odata, int elements_count)
{
    auto [blocks_count, threads_count] = get_cuda_params(elements_count);
    if (blocks_count > MAX_BLOCK_DIM_SIZE)
    {
        throw std::runtime_error("Too big num blocks!");
    }

    T gpu_result = 0;

    auto& device = DeviceSingleton::get_instance().device();
    auto device_intermediate_sums = cuda::memory::device::make_unique<T[]>(device, blocks_count);

    cudaDeviceSynchronize();

    my::cuda::reduce(device_idata, device_odata, elements_count, blocks_count, threads_count);

    cuda::memory::copy(host_odata, device_odata, blocks_count * sizeof(T));
    for (int i = 0; i < blocks_count; ++i)
    {
        gpu_result += host_odata[i];
    }

    cudaDeviceSynchronize();

    return gpu_result;
}

template <typename T>
T dot_product_gpu_device(const T* device_a, const T* device_b, int elements_count)
{
    T dot_product;

    auto& device = DeviceSingleton::get_instance().device();
    auto device_c = cuda::memory::device::make_unique<T[]>(device, elements_count);

    // multiplication
    {
        int threads_per_block = 256;
        int blocks_per_grid = (elements_count + threads_per_block - 1) / threads_per_block;
        my::cuda::multiply(device_a, device_b, device_c.get(), elements_count, blocks_per_grid, threads_per_block);
    }

    // reduction
    {
        int odata_size = std::min(elements_count / MAX_THREADS_COUNT, MAX_BLOCK_DIM_SIZE);
        auto host_odata = cuda::memory::host::make_unique<T[]>(odata_size);
        auto device_odata = cuda::memory::device::make_unique<T[]>(device, odata_size);
        dot_product = run_reduce(device_c.get(), device_odata.get(), host_odata.get(), elements_count);
    }
    return dot_product;
}

template <typename T>
class MemoryResource
{
public:
    MemoryResource(int elements_count)
    {
        auto& device = DeviceSingleton::get_instance().device();
        m_device_a = cuda::memory::device::make_unique<T[]>(device, elements_count);
        m_device_b = cuda::memory::device::make_unique<T[]>(device, elements_count);
        m_device_c = cuda::memory::device::make_unique<T[]>(device, elements_count);
        int odata_size = std::min(elements_count / MAX_THREADS_COUNT, MAX_BLOCK_DIM_SIZE);
        m_device_odata = cuda::memory::device::make_unique<T[]>(device, odata_size);
        m_host_odata = cuda::memory::host::make_unique<T[]>(odata_size);
        auto [blocks_count, threads_count] = get_cuda_params(elements_count);
        m_device_intermediate_sums = cuda::memory::device::make_unique<T[]>(device, blocks_count);
    }

    MemoryResource(const MemoryResource&) = delete;
    MemoryResource& operator=(const MemoryResource&) = delete;
    MemoryResource(MemoryResource&&) = default;
    MemoryResource& operator=(MemoryResource&&) = default;

    T* device_a() { return m_device_a.get(); }
    T* device_b() { return m_device_b.get(); }
    T* device_c() { return m_device_c.get(); }
    T* device_odata() { return m_device_odata.get(); }
    T* host_odata() { return m_host_odata.get(); }
    T* device_intermediate_sums() { return m_device_intermediate_sums.get(); }

private:
    cuda::memory::device::unique_ptr<T[]> m_device_a,
                                          m_device_b,
                                          m_device_c,
                                          m_device_odata,
                                          m_device_intermediate_sums;

    cuda::memory::host::unique_ptr<T[]> m_host_odata;
};

template <typename T>
T dot_product_gpu_device_prealloc(MemoryResource<T>& memory_resource, int elements_count)
{
    T dot_product;

    const T* device_a = memory_resource.device_a();
    const T* device_b = memory_resource.device_b();
    T* device_c = memory_resource.device_c();

    // multiplication
    {
        int threads_per_block = 256;
        int blocks_per_grid = (elements_count + threads_per_block - 1) / threads_per_block;
        my::cuda::multiply(device_a, device_b, device_c, elements_count, blocks_per_grid, threads_per_block);
    }

    // reduction
    {
        T* host_odata = memory_resource.host_odata();
        T* device_odata = memory_resource.device_odata();
        dot_product = run_reduce(device_c, device_odata, host_odata, elements_count);
    }
    return dot_product;
}

template <typename T>
T dot_product_gpu_host(const T* host_a, const T* host_b, int elements_count)
{
    auto& device = DeviceSingleton::get_instance().device();
    auto device_a = cuda::memory::device::make_unique<T[]>(device, elements_count);
    auto device_b = cuda::memory::device::make_unique<T[]>(device, elements_count);

    std::size_t bytes_count = elements_count * sizeof(T);
    cuda::memory::copy(device_a.get(), host_a, bytes_count);
    cuda::memory::copy(device_b.get(), host_b, bytes_count);

    return dot_product_gpu_device(device_a.get(), device_b.get(), elements_count);
}

}  // namespace

int main() try
{
    int elements_count = 1 << 23;

    auto host_a = cuda::memory::host::make_unique<double[]>(elements_count);
    auto host_b = cuda::memory::host::make_unique<double[]>(elements_count);
    auto host_c = cuda::memory::host::make_unique<double[]>(elements_count);

    auto generator = []()
    {
        static std::mt19937 prng(std::random_device{}());
        static std::uniform_real_distribution<double> dist(-1000, 1000);
        return dist(prng);
    };
    std::generate(host_a.get(), host_a.get() + elements_count, generator);
    std::generate(host_b.get(), host_b.get() + elements_count, generator);

    {
        auto dot_product_cpu_wrapper = [&host_a, &host_b, elements_count]()
        {
            return cpu_dot_product(host_a.get(), host_b.get(), elements_count);
        };
        double dot_product_cpu_result = my::benchmark_function(dot_product_cpu_wrapper, ITERATIONS_COUNT);
        my::print_result("     CPU time: ", dot_product_cpu_result);
    }

    {
        auto dot_product_gpu_host_wrapper = [&host_a, &host_b, elements_count]()
        {
            return dot_product_gpu_host(host_a.get(), host_b.get(), elements_count);
        };
        double dot_product_gpu_host_result = my::benchmark_function(dot_product_gpu_host_wrapper, ITERATIONS_COUNT);
        my::print_result("GPU time  (h): ", dot_product_gpu_host_result);
    }

    {
        auto& device = DeviceSingleton::get_instance().device();
        auto device_a = cuda::memory::device::make_unique<double[]>(device, elements_count);
        auto device_b = cuda::memory::device::make_unique<double[]>(device, elements_count);

        std::size_t bytes_count = elements_count * sizeof(double);
        cuda::memory::copy(device_a.get(), host_a.get(), bytes_count);
        cuda::memory::copy(device_b.get(), host_b.get(), bytes_count);

        auto dot_product_gpu_device_wrapper = [&device_a, &device_b, elements_count]()
        {
            return dot_product_gpu_device(device_a.get(), device_b.get(), elements_count);
        };
        double dot_product_gpu_device_result = my::benchmark_function(dot_product_gpu_device_wrapper, ITERATIONS_COUNT);
        my::print_result("GPU time (d1): ", dot_product_gpu_device_result);
    }

    {
        MemoryResource<double> memory_resource(elements_count);

        std::size_t bytes_count = elements_count * sizeof(double);
        cuda::memory::copy(memory_resource.device_a(), host_a.get(), bytes_count);
        cuda::memory::copy(memory_resource.device_b(), host_b.get(), bytes_count);

        auto dot_product_gpu_device_prealloc_wrapper = [&memory_resource, elements_count]()
        {
            return dot_product_gpu_device_prealloc(memory_resource, elements_count);
        };
        double dot_product_gpu_device_prealloc_result = my::benchmark_function(dot_product_gpu_device_prealloc_wrapper, ITERATIONS_COUNT);
        my::print_result("GPU time (d2): ", dot_product_gpu_device_prealloc_result);
    }

    return EXIT_SUCCESS;
}
catch (const std::exception& e)
{
    std::cerr << "Exception caught: " << e.what() << std::endl;
    return EXIT_FAILURE;
}
catch (...)
{
    std::cerr << "An unknown exception caught" << std::endl;
    return EXIT_FAILURE;
}
