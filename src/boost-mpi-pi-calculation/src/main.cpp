#include <benchmark.hpp>
#include <pi_helpers.hpp>

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
static constexpr int ROOT_ID = 0;
static constexpr int TAG = 0;

void send_mpf_pi_part(const mpf_class& pi_part, const mpi::communicator& world)
{
    mpf_srcptr pi_part_raw = pi_part.get_mpf_t();

    world.send(ROOT_ID, TAG, pi_part_raw->_mp_prec);
    world.send(ROOT_ID, TAG, pi_part_raw->_mp_size);
    world.send(ROOT_ID, TAG, pi_part_raw->_mp_exp);
    world.send(ROOT_ID, TAG, pi_part_raw->_mp_d, std::abs(pi_part_raw->_mp_size));
}

void recv_and_add_mpf_pi_part(mpf_class& pi, int rank, const mpi::communicator& world)
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

void pi_sum_reduce(const mpf_class& pi_part, mpf_class& pi, const mpi::communicator& world)
{
    // TODO: create a user-defined type for GMP float
    if (world.rank() == ROOT_ID)
    {
        pi = std::move(pi_part);
        std::size_t process_count = world.size();
        for (std::size_t rank = 1; rank < process_count; ++rank)
        {
            recv_and_add_mpf_pi_part(pi, rank, world);
        }
    }
    else
    {
        send_mpf_pi_part(pi_part, world);
    }
}

mpf_class pi_leibniz_mpi(std::size_t summand_count, mp_bitcnt_t precision,
                         const mpi::communicator& world)
{
    mpi::broadcast(world, summand_count, ROOT_ID);

    mpf_class pi_part = my::pi::pi_part_leibniz_mpi(summand_count, precision, world.rank(), world.size());
    world.barrier();

    mpf_class pi(0.0, precision);
    pi_sum_reduce(pi_part, pi, world);

    return 4 * pi;
}

mpf_class pi_bellard_mpi(std::size_t summand_count, mp_bitcnt_t precision,
                         const mpi::communicator& world)
{
    mpi::broadcast(world, summand_count, ROOT_ID);

    mpf_class pi_part = my::pi::pi_part_bellard_mpi(summand_count, precision, world.rank(), world.size());
    world.barrier();

    mpf_class pi(0.0, precision);
    pi_sum_reduce(pi_part, pi, world);

    pi /= (1 << 6);
    return pi;
}

template <typename RegularPiCalculationFunction,
          typename MPIPiCalculationFunction>
void benchmark(std::size_t summand_count,
               mp_bitcnt_t precision,
               const mpi::communicator& world,
               RegularPiCalculationFunction pi_regular,
               MPIPiCalculationFunction pi_mpi)
{
    if (summand_count < static_cast<std::size_t>(world.size()))
    {
        throw std::runtime_error("Summand count is less than processor count, please decrease number of processors.");
    }

    if (world.rank() == ROOT_ID)
    {
        auto pi_regular_wrapper = [summand_count, precision, pi_regular]()
        {
            return pi_regular(summand_count, precision);
        };
        double pi_regular_result = my::benchmark_function(pi_regular_wrapper, ITERATIONS_COUNT);
        my::print_result("Regular time: ", pi_regular_result);
    }

    double pi_mpi_result;
    {
        auto pi_mpi_wrapper = [summand_count, precision, pi_mpi, world]()
        {
            return pi_mpi(summand_count, precision, world);
        };
        pi_mpi_result = my::benchmark_function(pi_mpi_wrapper, ITERATIONS_COUNT);
    }
    if (world.rank() == ROOT_ID)
    {
        my::print_result("    MPI time: ", pi_mpi_result);
    }
}

template <typename MPIPiCalculationFunction>
void calculate(std::size_t summand_count,
               mp_bitcnt_t precision,
               const mpi::communicator& world,
               MPIPiCalculationFunction pi_mpi)
{
    if (summand_count < static_cast<std::size_t>(world.size()))
    {
        throw std::runtime_error("Summand count is less than processor count, please decrease number of processors.");
    }

    mpf_class pi_mpi_result = pi_mpi(summand_count, precision, world);

    if (world.rank() == ROOT_ID)
    {
        mp_exp_t exp;
        std::string pi_string = pi_mpi_result.get_str(exp);
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

    struct AlgorithmInfo
    {
        const std::function<mpf_class(std::size_t summand_count, mp_bitcnt_t precision)> pi_regular;
        const std::function<mpf_class(std::size_t summand_count, mp_bitcnt_t precision, const mpi::communicator& world)> pi_mpi;
        my::pi::AlgorithmParams params;
    };

    std::unordered_map<my::pi::AlgorithmType, AlgorithmInfo>
    algorithm_info_map =
    {
        {
            my::pi::AlgorithmType::BELLARD,
            {
                .pi_regular = my::pi::pi_bellard_regular,
                .pi_mpi = pi_bellard_mpi,
                .params =
                {
                    .precision = (1 << 22),
                    .benchmark_summand_count = std::size_t{1} << 8,
                    .calculation_summand_count = std::size_t{1} << 27
                }
            }
        },
        {
            my::pi::AlgorithmType::LEIBNIZ,
            {
                .pi_regular = my::pi::pi_leibniz_regular,
                .pi_mpi = pi_leibniz_mpi,
                .params =
                {
                    .precision = (1 << 22),
                    .benchmark_summand_count = std::size_t{1} << 26,
                    .calculation_summand_count = std::size_t{1} << 45
                }
            }
        },
    };

    bool do_benchmark = true;
    auto algorithm = my::pi::AlgorithmType::BELLARD;

    const AlgorithmInfo& algorithm_info = algorithm_info_map.at(algorithm);

    if (do_benchmark)
    {
        benchmark(algorithm_info.params.benchmark_summand_count,
                  algorithm_info.params.precision,
                  world,
                  algorithm_info.pi_regular,
                  algorithm_info.pi_mpi);
    }
    else
    {
        calculate(algorithm_info.params.calculation_summand_count,
                  algorithm_info.params.precision,
                  world,
                  algorithm_info.pi_mpi);
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
