#ifndef MPI_TOOLS_HPP_
#define MPI_TOOLS_HPP_

#include <chrono>
#include <cstdint>
#include <cstdlib>

namespace my
{

template <typename T>
inline __attribute__((always_inline)) void do_not_optimize(T& value)
{
#if defined(__clang__)
    asm volatile("" : "+r,m"(value) : : "memory");
#else
    asm volatile("" : "+m,r"(value) : : "memory");
#endif
}

class Timer
{
public:
    Timer(double& nanoseconds, std::size_t iterations_count = 1)
        : m_nanoseconds(nanoseconds)
        , m_iterations_count(iterations_count)
    {
        m_time_before = std::chrono::high_resolution_clock::now();
    }

    ~Timer()
    {
        try
        {
            using namespace std::chrono;
            high_resolution_clock::time_point time_after = high_resolution_clock::now();
            m_nanoseconds = duration_cast<nanoseconds>(time_after - m_time_before).count() / static_cast<double>(m_iterations_count);
        }
        catch (...)
        {
            std::exit(EXIT_FAILURE);
        }
    }

private:
    double& m_nanoseconds;
    std::size_t m_iterations_count;
    std::chrono::high_resolution_clock::time_point m_time_before;
};

}  // namespace my

#endif  // MPI_TOOLS_HPP_
