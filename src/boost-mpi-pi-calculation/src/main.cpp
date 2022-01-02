#include <benchmark.hpp>

#include <boost/mpi.hpp>
#include <gmpxx.h>

#include <cassert>
#include <cmath>
#include <iostream>
#include <stdexcept>

namespace mpi = boost::mpi;

namespace
{

static constexpr std::size_t ITERATIONS_COUNT = 100;
static constexpr mp_bitcnt_t PRECISION = (1 << 7);
static constexpr int ROOT_ID = 0;
static constexpr int TAG = 0;

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

mpf_class pi_calculation_mpi(std::size_t summand_count, const mpi::communicator& world)
{
    std::size_t process_id = world.rank();
    std::size_t process_count = world.size();

    mpi::broadcast(world, summand_count, ROOT_ID);

    mpf_class pi_part(0.0, PRECISION);

    mpf_class temp(0.0, PRECISION);
    for (std::size_t i = process_id; i < summand_count; i += process_count)
    {
        temp = (i % 2 == 0 ? 1.0 : -1.0);
        temp /= 2 * i + 1;
        pi_part += temp;
    }

    mpf_class pi(0.0, PRECISION);

    world.barrier();

    // TODO: create a user-defined type for GMP float
    if (process_id == ROOT_ID)
    {
        pi = std::move(pi_part);
        for (std::size_t rank = 1; rank < process_count; ++rank)
        {
            mpf_t another_pi_part;

            world.recv(rank, TAG, another_pi_part->_mp_prec);
            world.recv(rank, TAG, another_pi_part->_mp_size);
            world.recv(rank, TAG, another_pi_part->_mp_exp);

            auto mp_d = std::make_unique<mp_limb_t[]>(another_pi_part->_mp_prec + 1);
            another_pi_part->_mp_d = mp_d.get();
            world.recv(rank, TAG, another_pi_part->_mp_d, std::abs(another_pi_part->_mp_size));

            mpf_add(pi.get_mpf_t(), pi.get_mpf_t(), another_pi_part);
        }
    }
    else
    {
        mpf_ptr pi_part_raw = pi_part.get_mpf_t();

        world.send(ROOT_ID, TAG, pi_part_raw->_mp_prec);
        world.send(ROOT_ID, TAG, pi_part_raw->_mp_size);
        world.send(ROOT_ID, TAG, pi_part_raw->_mp_exp);
        world.send(ROOT_ID, TAG, pi_part_raw->_mp_d, std::abs(pi_part_raw->_mp_size));
    }

    return 4 * pi;
}

void benchmark(const mpi::communicator& world)
{
    std::size_t summand_count = std::size_t{1} << 26;

    if (summand_count < static_cast<std::size_t>(world.size()))
    {
        throw std::runtime_error("Summand count is less than processor count, please decrease number of processors.");
    }

    if (world.rank() == ROOT_ID)
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
        auto pi_calculation_mpi_wrapper = [summand_count, &world]()
        {
            return pi_calculation_mpi(summand_count, world);
        };
        pi_calculation_mpi_result = my::benchmark_function(pi_calculation_mpi_wrapper, ITERATIONS_COUNT);
    }
    if (world.rank() == ROOT_ID)
    {
        my::print_result("    MPI time: ", pi_calculation_mpi_result);
    }
}

void calculate(const mpi::communicator& world)
{
    std::size_t summand_count = std::size_t{1} << 42;

    if (summand_count < static_cast<std::size_t>(world.size()))
    {
        throw std::runtime_error("Summand count is less than processor count, please decrease number of processors.");
    }

    mpf_class pi_mpi = pi_calculation_mpi(summand_count, world);

    if (world.rank() == ROOT_ID)
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
    mpi::environment env(argc, argv);
    mpi::communicator world;

    bool do_benchmark = true;

    if (do_benchmark)
        benchmark(world);
    else
        calculate(world);

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
