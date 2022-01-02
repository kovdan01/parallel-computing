#include <benchmark.hpp>
#include <mpi.hpp>

#include <boost/container/vector.hpp>
#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>
#include <stdexcept>

namespace
{

static constexpr std::size_t ITERATIONS_COUNT = 100;

namespace bc = boost::container;

struct Data
{
    bc::vector<double> a;
    bc::vector<double> b;
};

Data generate_data(std::size_t size)
{
    Data data;
    if (my::mpi::is_current_process_root())
    {
        data.a.resize(size, bc::default_init_t{});
        data.b.resize(size, bc::default_init_t{});

        auto generator = []()
        {
            static std::mt19937 prng(std::random_device{}());
            static std::uniform_real_distribution<double> dist(-1000, 1000);
            return dist(prng);
        };

        std::generate(data.a.data(), data.a.data() + size, generator);
        std::generate(data.b.data(), data.b.data() + size, generator);
    }
    return data;
}

double dot_product_regular(const double* a, const double* b, std::size_t size)
{
    double dot_product = 0;
    for (std::size_t i = 0; i < size; ++i)
    {
        dot_product += a[i] * b[i];
    }
    return dot_product;
}

double dot_product_mpi(const double* a, const double* b, std::size_t size)
{
    const auto& mpi_params = my::mpi::Params::get_instance();

    auto get_offset = [&mpi_params](std::size_t index, std::size_t size) -> std::size_t
    {
        std::size_t offset;
        std::size_t quotient = size / mpi_params.process_count();
        std::size_t remainder = size % mpi_params.process_count();

        if (index <= remainder)
        {
            offset = (quotient + 1) * index;
        }
        else
        {
            offset = (quotient + 1) * remainder;
            offset += quotient * (index - remainder);
        }

        return offset;
    };

    auto get_count = [&mpi_params](std::size_t index, std::size_t size) -> std::size_t
    {
        std::size_t count;
        std::size_t quotient  = size / mpi_params.process_count();
        std::size_t remainder = size % mpi_params.process_count();

        count = quotient + (index < remainder ? 1 : 0);

        return count;
    };

    bc::vector<int> counts, offsets;

    if (my::mpi::is_current_process_root())
    {
        counts.resize (mpi_params.process_count(), bc::default_init_t{});
        offsets.resize(mpi_params.process_count(), bc::default_init_t{});

        for (std::size_t i = 0; i < mpi_params.process_count(); ++i)
        {
            offsets[i] = static_cast<int>(get_offset(i, size));
            counts[i]  = static_cast<int>(get_count(i, size));
        }
    }

    std::size_t size_part = get_count(mpi_params.process_id(), size);
    bc::vector<double> a_part(size_part, bc::default_init_t{});
    bc::vector<double> b_part(size_part, bc::default_init_t{});

    my::mpi::scatterv(a, counts.data(), offsets.data(), MPI_DOUBLE, a_part.data(), static_cast<int>(size_part), MPI_DOUBLE);
    my::mpi::scatterv(b, counts.data(), offsets.data(), MPI_DOUBLE, b_part.data(), static_cast<int>(size_part), MPI_DOUBLE);

    double dot_product_part = 0;
    for (std::size_t i = 0; i < size_part; ++i)
    {
        dot_product_part += a_part[i] * b_part[i];
    }

    double dot_product = 0;
    my::mpi::reduce(&dot_product_part, &dot_product, 1, MPI_DOUBLE, MPI_SUM);

    return dot_product;
}

void benchmark(const double* a, const double* b, std::size_t size)
{
    if (my::mpi::is_current_process_root())
    {
        auto dot_product_regular_wrapper = [a, b, size]()
        {
            return dot_product_regular(a, b, size);
        };
        double dot_product_regular_result = my::benchmark_function(dot_product_regular_wrapper, ITERATIONS_COUNT);
        my::print_result("Regular time: ", dot_product_regular_result);
    }

    double dot_product_mpi_result;
    {
        auto dot_product_mpi_wrapper = [a, b, size]()
        {
            return dot_product_mpi(a, b, size);
        };
        dot_product_mpi_result = my::benchmark_function(dot_product_mpi_wrapper, ITERATIONS_COUNT);
    }
    if (my::mpi::is_current_process_root())
    {
        my::print_result("    MPI time: ", dot_product_mpi_result);
    }
}

void test(const double* a, const double* b, std::size_t size)
{
    auto are_doubles_equal = [](double a, double b) -> bool
    {
        static constexpr double accuracy = 1e-3;
        return std::abs(a - b) < accuracy;
    };

    double result_mpi = dot_product_mpi(a, b, size);

    if (my::mpi::is_current_process_root())
    {
        double result_regular = dot_product_regular(a, b, size);
        if (!are_doubles_equal(result_regular, result_mpi))
        {
            throw std::runtime_error("Test failed!");
        }
        std::cout << "Test passed" << std::endl;
    }
}

}  // namespace

int main(int argc, char* argv[]) try
{
    my::mpi::Control mpi_control(argc, argv);

    const auto& mpi_params = my::mpi::Params::get_instance();
    static constexpr int size = 1 << 23;

    if (size < mpi_params.process_count())
    {
        throw std::runtime_error("Vector length is less than processor count, please decrease number of processors.");
    }

    auto [a, b] = generate_data(size);

    bool do_test = false;

    if (do_test)
        test(a.data(), b.data(), size);
    else
        benchmark(a.data(), b.data(), size);

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
