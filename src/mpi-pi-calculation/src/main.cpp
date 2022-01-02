#include <benchmark.hpp>
#include <mpi.hpp>

#include <gmpxx.h>
#include <mpi.h>

#include <cassert>
#include <cmath>
#include <iostream>
#include <stdexcept>

namespace
{

static constexpr std::size_t ITERATIONS_COUNT = 100;
static constexpr mp_bitcnt_t PRECISION = (1 << 7);

mpf_class pi_calculation_regular(std::size_t summand_count)
{
    mpf_class pi(0.0, PRECISION);
    mpf_class temp(0.0, PRECISION);
    for (std::size_t i = 0; i < summand_count; ++i)
    {
        temp = (i % 2 == 0 ? 1.0 : -1.0);
        temp /= 2 * i + 1;
        pi += temp;
    }
    return 4 * pi;
}

mpf_class pi_calculation_mpi(std::size_t summand_count)
{
    const auto& mpi_params = my::mpi::Params::get_instance();
    std::size_t process_id = mpi_params.process_id();
    std::size_t process_count = mpi_params.process_count();

    static_assert(sizeof(std::size_t) == sizeof(unsigned long long),
                  "Assume that we can use MPI_UNSIGNED_LONG_LONG to mark std::size_t variables");

    my::mpi::bcast(&summand_count, 1, MPI_UNSIGNED_LONG_LONG);

    mpf_class pi_part(0.0, PRECISION);

    mpf_class temp(0.0, PRECISION);
    for (std::size_t i = process_id; i < summand_count; i += process_count)
    {
        temp = (i % 2 == 0 ? 1.0 : -1.0);
        temp /= 2 * i + 1;
        pi_part += temp;
    }

    mpf_class pi(0.0, PRECISION);

    static_assert(sizeof(unsigned long) == sizeof(mp_limb_t),
                  "Assume that we can use MPI_UNSIGNED_LONG to mark mp_limb_t variables");
    static_assert(sizeof(long) == sizeof(mp_exp_t),
                  "Assume that we can use MPI_LONG_INT to mark mp_exp_t variables");

    my::mpi::barrier();

    // TODO: create a user-defined type for GMP float
    if (my::mpi::is_current_process_root())
    {
        pi = std::move(pi_part);
        for (std::size_t rank = 1; rank < process_count; ++rank)
        {
            mpf_t another_pi_part;

            my::mpi::recv(&another_pi_part->_mp_prec, 1, MPI_INT,      rank);
            my::mpi::recv(&another_pi_part->_mp_size, 1, MPI_INT,      rank);
            my::mpi::recv(&another_pi_part->_mp_exp,  1, MPI_LONG_INT, rank);

            auto mp_d = std::make_unique<mp_limb_t[]>(another_pi_part->_mp_prec + 1);
            another_pi_part->_mp_d = mp_d.get();
            my::mpi::recv(another_pi_part->_mp_d,
                          std::abs(another_pi_part->_mp_size),
                          MPI_UNSIGNED_LONG,
                          rank);

            mpf_add(pi.get_mpf_t(), pi.get_mpf_t(), another_pi_part);
        }
    }
    else
    {
        mpf_ptr pi_part_raw = pi_part.get_mpf_t();

        my::mpi::send(&pi_part_raw->_mp_prec, 1,                               MPI_INT,           my::mpi::ROOT_ID);
        my::mpi::send(&pi_part_raw->_mp_size, 1,                               MPI_INT,           my::mpi::ROOT_ID);
        my::mpi::send(&pi_part_raw->_mp_exp,  1,                               MPI_LONG_INT,      my::mpi::ROOT_ID);
        my::mpi::send(pi_part_raw->_mp_d,     std::abs(pi_part_raw->_mp_size), MPI_UNSIGNED_LONG, my::mpi::ROOT_ID);
    }

    return 4 * pi;
}

void benchmark()
{
    const auto& mpi_params = my::mpi::Params::get_instance();
    std::size_t summand_count = std::size_t{1} << 26;

    if (summand_count < mpi_params.process_count())
    {
        throw std::runtime_error("Summand count is less than processor count, please decrease number of processors.");
    }

    if (my::mpi::is_current_process_root())
    {
        auto pi_calculation_regular_wrapper = [summand_count]()
        {
            return pi_calculation_regular(summand_count);
        };
        double pi_calculation_regular_result = my::benchmark_function(pi_calculation_regular_wrapper, ITERATIONS_COUNT);
        my::print_result("Regular time: ", pi_calculation_regular_result);
    }

    double pi_calculation_mpi_result;
    {
        auto pi_calculation_mpi_wrapper = [summand_count]()
        {
            return pi_calculation_mpi(summand_count);
        };
        pi_calculation_mpi_result = my::benchmark_function(pi_calculation_mpi_wrapper, ITERATIONS_COUNT);
    }
    if (my::mpi::is_current_process_root())
    {
        my::print_result("    MPI time: ", pi_calculation_mpi_result);
    }
}

void calculate()
{
    const auto& mpi_params = my::mpi::Params::get_instance();
    std::size_t summand_count = std::size_t{1} << 45;

    if (summand_count < mpi_params.process_count())
    {
        throw std::runtime_error("Summand count is less than processor count, please decrease number of processors.");
    }

    mpf_class pi_mpi = pi_calculation_mpi(summand_count);

    if (my::mpi::is_current_process_root())
    {
        mp_exp_t exp;
        std::string pi_string = pi_mpi.get_str(exp);
        assert(exp == 1);
        std::cout << std::string_view(pi_string.data(), 1);
        std::cout << '.';
        std::cout << std::string_view(pi_string.data() + 1, pi_string.size()) << std::endl;
    }
}

}  // namespace

int main(int argc, char* argv[]) try
{
    my::mpi::Control mpi_control(argc, argv);

    bool do_benchmark = true;

    if (do_benchmark)
        benchmark();
    else
        calculate();

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
