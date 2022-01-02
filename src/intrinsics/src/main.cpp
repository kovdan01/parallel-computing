#include <benchmark.hpp>

#include <immintrin.h>

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <string_view>

namespace
{

static_assert (sizeof(long long) == sizeof(std::int64_t), "long long and std::int64_t do not match");
static_assert (sizeof(__m256i) % sizeof(std::int64_t) == 0, "Size of __m256i is not multiple of size of std::int64_t");
static constexpr std::size_t SCALARS_IN_VECTOR = sizeof(__m256i) / sizeof(std::int64_t);

static constexpr std::size_t ITERATIONS_COUNT = 10'000'000'000;

my::TicksAndNanoseconds independent_scalar_operation(std::int64_t n1, std::int64_t n2)
{
    asm volatile("# independent_scalar_operation enter");
    my::TicksAndNanoseconds result;
    {
        my::Timer timer(result, ITERATIONS_COUNT);
        for (std::size_t i = 0; i < ITERATIONS_COUNT; ++i)
        {
            std::int64_t n3 = n1 / n2;
            my::do_not_optimize(n3);
        }
    }
    asm volatile("# independent_scalar_operation exit");
    return result;
}

my::TicksAndNanoseconds dependent_scalar_operation(std::int64_t n1, std::int64_t n2)
{
    asm volatile("# dependent_scalar_operation enter");
    my::TicksAndNanoseconds result;
    {
        my::Timer timer(result, ITERATIONS_COUNT);
        for (std::size_t i = 0; i < ITERATIONS_COUNT; ++i)
        {
            n1 /= n2;
        }
        my::do_not_optimize(n1);
    }
    asm volatile("# dependent_scalar_operation exit");
    return result;
}

void put_scalar_into_vector(std::int64_t scalar, __m256i& vector)
{
    std::int64_t array[SCALARS_IN_VECTOR];
    for (std::size_t i = 0; i < SCALARS_IN_VECTOR; ++i)
        array[i] = scalar;
    std::memcpy(&vector, array, sizeof(vector));
}

my::TicksAndNanoseconds independent_vector_operation(std::int64_t n1, std::int64_t n2)
{
    asm volatile("# independent_vector_operation enter\n");

    __m256i v1, v2;
    put_scalar_into_vector(n1, v1);
    put_scalar_into_vector(n2, v2);

    static constexpr std::size_t vector_iterations_count = ITERATIONS_COUNT / SCALARS_IN_VECTOR;
    my::TicksAndNanoseconds result;
    {
        my::Timer timer(result, ITERATIONS_COUNT);
        for (std::size_t i = 0; i < vector_iterations_count; ++i)
        {
            __m256i v3 = _mm256_div_epi64(v1, v2);
            my::do_not_optimize(v3);
        }
    }
    asm volatile("# independent_vector_operation exit");
    return result;
}

my::TicksAndNanoseconds dependent_vector_operation(std::int64_t n1, std::int64_t n2)
{
    asm volatile("# dependent_vector_operation enter\n");

    __m256i v1, v2;
    put_scalar_into_vector(n1, v1);
    put_scalar_into_vector(n2, v2);

    static constexpr std::size_t vector_iterations_count = ITERATIONS_COUNT / SCALARS_IN_VECTOR;
    my::TicksAndNanoseconds result;
    {
        my::Timer timer(result, ITERATIONS_COUNT);
        for (std::size_t i = 0; i < vector_iterations_count; ++i)
        {
            v1 = _mm256_div_epi64(v1, v2);
        }
        my::do_not_optimize(v1);
    }
    asm volatile("# dependent_vector_operation exit");
    return result;
}

struct Params
{
    std::int64_t n1;
    std::int64_t n2;
};

Params parse_cmd_line(int argc, char* argv[])
{
    if (argc != 3)
    {
        throw std::invalid_argument("Error: you must pass 2 integers as CLI-arguments");
    }

    auto int_from_string = [](const std::string& s) -> std::int64_t
    {
        try
        {
            std::size_t pos;
            std::int64_t n = std::stoll(s, &pos);
            if (pos != s.size())
            {
                throw std::invalid_argument("stoll");
            }
            return n;
        }
        catch (const std::invalid_argument& e)
        {
            throw std::invalid_argument("Error: cannot convert \"" + s + "\" to std::int64_t: \"" + e.what() + "\"");
        }
        catch (const std::out_of_range& e)
        {
            throw std::out_of_range("Error: value \"" + s + "\" is out of range of type std::int64_t: \"" + e.what() + "\"");
        }
    };

    return Params{ .n1 = int_from_string(std::string(argv[1])),
                   .n2 = int_from_string(std::string(argv[2])) };
}

void print_table_row(std::string_view label, my::TicksAndNanoseconds result)
{
    std::cout << "| " << label << " | "
              << std::setw(12) << result.ticks << " | "
              << std::setw(9)  << result.nanoseconds << " |" << std::endl
              << "+--------------------+--------------+-----------+" << std::endl;
}

}  // namespace

int main(int argc, char* argv[]) try
{
    Params params = parse_cmd_line(argc, argv);

    my::TicksAndNanoseconds independent_scalar_result = independent_scalar_operation(params.n1, params.n2);
    my::TicksAndNanoseconds dependent_scalar_result = dependent_scalar_operation(params.n1, params.n2);
    my::TicksAndNanoseconds independent_vector_result = independent_vector_operation(params.n1, params.n2);
    my::TicksAndNanoseconds dependent_vector_result = dependent_vector_operation(params.n1, params.n2);

    std::cout << "+--------------------+--------------+-----------+" << std::endl
              << "|      operation     | ticks / iter | ns / iter |" << std::endl
              << "+--------------------+--------------+-----------+" << std::endl;
    print_table_row("independent scalar", independent_scalar_result);
    print_table_row(" dependent scalar ", dependent_scalar_result);
    print_table_row("independent vector", independent_vector_result);
    print_table_row(" dependent vector ", dependent_vector_result);

    return EXIT_SUCCESS;
}
catch (const std::exception& e)
{
    std::cerr << e.what() << "\nAborting\n";
    return EXIT_FAILURE;
}
