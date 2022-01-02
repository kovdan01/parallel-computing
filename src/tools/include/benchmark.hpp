#ifndef PARALLEL_COMPUTING_TOOLS_BENCHMARK_HPP_
#define PARALLEL_COMPUTING_TOOLS_BENCHMARK_HPP_

#include <chrono>
// NOTE: gcc 8.2.0 on cHARISMa does not support concepts
// Instead, use horrible std::enable_if
// #include <concepts>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string_view>
#include <type_traits>

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

inline __attribute__((always_inline)) std::uint64_t ticks()
{
    std::uint64_t tsc;
    asm volatile("mfence; "         // memory barrier
                 "rdtsc; "          // read of tsc
                 "shl $32,%%rdx; "  // shift higher 32 bits stored in rdx up
                 "or %%rdx,%%rax"   // and or onto rax
                 : "=a"(tsc)        // output to tsc
                 :
                 : "%rcx", "%rdx", "memory");
    return tsc;
}

struct TicksAndNanoseconds
{
    double ticks;
    double nanoseconds;
};

class NanosecondsTimer
{
    public:
    NanosecondsTimer(double& result, std::size_t iterations_count = 1)
        : m_result(result)
        , m_iterations_count(iterations_count)
    {
        m_time_before = std::chrono::high_resolution_clock::now();
    }

    ~NanosecondsTimer()
    {
        try
        {
            using namespace std::chrono;
            high_resolution_clock::time_point time_after = high_resolution_clock::now();
            m_result = duration_cast<nanoseconds>(time_after - m_time_before).count() / static_cast<double>(m_iterations_count);
        }
        catch (...)
        {
            std::exit(EXIT_FAILURE);
        }
    }

private:
    double& m_result;
    std::size_t m_iterations_count;
    std::chrono::high_resolution_clock::time_point m_time_before;
};

class TicksTimer
{
public:
    TicksTimer(double& result, std::size_t iterations_count = 1)
        : m_result(result)
        , m_iterations_count(iterations_count)
    {
        m_ticks_before = ticks();
    }

    ~TicksTimer()
    {
        try
        {
            std::uint64_t ticks_after = ticks();
            m_result = (ticks_after - m_ticks_before) / static_cast<double>(m_iterations_count);
        }
        catch (...)
        {
            std::exit(EXIT_FAILURE);
        }
    }

private:
    double& m_result;
    std::size_t m_iterations_count;
    std::uint64_t m_ticks_before;
};

class Timer : public NanosecondsTimer, public TicksTimer
{
public:
    Timer(TicksAndNanoseconds& result, std::size_t iterations_count = 1)
        : NanosecondsTimer(result.nanoseconds, iterations_count)
        , TicksTimer(result.ticks, iterations_count)
    {
    }
};

// NOTE: gcc 8.2.0 on cHARISMa does not support concepts
// Instead, use horrible std::enable_if

// template <typename Function>
// concept ReturnsVoid = std::same_as<std::invoke_result_t<Function>, void>;

// template <typename Function>
// concept DoesNotReturnVoid = !ReturnsVoid<Function>;

template <typename Function>
// requires DoesNotReturnVoid<Function>
std::enable_if_t<!std::is_same_v<std::invoke_result_t<Function>, void>, double>
benchmark_function(Function f, std::size_t iterations_count = 1)
{
    double nanoseconds;
    {
        NanosecondsTimer timer(nanoseconds, iterations_count);
        for (std::size_t i = 0; i < iterations_count; ++i)
        {
            auto result = f();
            do_not_optimize(result);
        }
    }
    return nanoseconds;
}

template <typename Function>
// requires ReturnsVoid<Function>
std::enable_if_t<std::is_same_v<std::invoke_result_t<Function>, void>, double>
benchmark_function(Function f, std::size_t iterations_count = 1)
{
    double nanoseconds;
    {
        NanosecondsTimer timer(nanoseconds, iterations_count);
        for (std::size_t i = 0; i < iterations_count; ++i)
        {
            f();
        }
    }
    return nanoseconds;
}

inline void print_result(std::string_view label, double result)
{
    std::cout << label;
    std::cout << std::fixed;
    std::cout << std::setprecision(2) << std::setfill(' ') << std::setw(15) << result << std::endl;
    std::cout << std::scientific;
}

}  // namespace my

#endif  // PARALLEL_COMPUTING_TOOLS_BENCHMARK_HPP_
