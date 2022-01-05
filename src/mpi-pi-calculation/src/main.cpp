#include <benchmark.hpp>
#include <mpi.hpp>
#include <pi_helpers.hpp>

#include <gmpxx.h>
#include <mpi.h>

#include <cassert>
#include <cmath>
#include <iostream>
#include <functional>
#include <stdexcept>
#include <unordered_map>

namespace
{

static_assert(sizeof(std::size_t) == sizeof(unsigned long long),
              "Assume that we can use MPI_UNSIGNED_LONG_LONG to mark std::size_t variables");
static_assert(sizeof(unsigned long) == sizeof(mp_limb_t),
              "Assume that we can use MPI_UNSIGNED_LONG to mark mp_limb_t variables");
static_assert(sizeof(long) == sizeof(mp_exp_t),
              "Assume that we can use MPI_LONG_INT to mark mp_exp_t variables");

void send_mpf_pi_part(const mpf_class& pi_part)
{
    mpf_srcptr pi_part_raw = pi_part.get_mpf_t();

    my::mpi::send(&pi_part_raw->_mp_prec, 1,                               MPI_INT,           my::mpi::ROOT_ID);
    my::mpi::send(&pi_part_raw->_mp_size, 1,                               MPI_INT,           my::mpi::ROOT_ID);
    my::mpi::send(&pi_part_raw->_mp_exp,  1,                               MPI_LONG_INT,      my::mpi::ROOT_ID);
    my::mpi::send(pi_part_raw->_mp_d,     std::abs(pi_part_raw->_mp_size), MPI_UNSIGNED_LONG, my::mpi::ROOT_ID);
}

void recv_and_add_mpf_pi_part(mpf_class& pi, int rank)
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

void pi_sum_reduce(const mpf_class& pi_part, mpf_class& pi)
{
    // TODO: create a user-defined type for GMP float
    if (my::mpi::is_current_process_root())
    {
        pi = std::move(pi_part);
        std::size_t process_count = my::mpi::Params::get_instance().process_count();
        for (std::size_t rank = 1; rank < process_count; ++rank)
        {
            recv_and_add_mpf_pi_part(pi, rank);
        }
    }
    else
    {
        send_mpf_pi_part(pi_part);
    }
}

mpf_class pi_leibniz_mpi(std::size_t summand_count, mp_bitcnt_t precision)
{
    const auto& mpi_params = my::mpi::Params::get_instance();
    std::size_t process_id = mpi_params.process_id();
    std::size_t process_count = mpi_params.process_count();

    my::mpi::bcast(&summand_count, 1, MPI_UNSIGNED_LONG_LONG);

    mpf_class pi_part = my::pi::pi_part_leibniz_mpi(summand_count, precision, process_id, process_count);
    my::mpi::barrier();

    mpf_class pi(0.0, precision);
    pi_sum_reduce(pi_part, pi);

    return 4 * pi;
}

mpf_class pi_bellard_mpi(std::size_t summand_count, mp_bitcnt_t precision)
{
    const auto& mpi_params = my::mpi::Params::get_instance();
    std::size_t process_id = mpi_params.process_id();
    std::size_t process_count = mpi_params.process_count();

    my::mpi::bcast(&summand_count, 1, MPI_UNSIGNED_LONG_LONG);

    mpf_class pi_part = my::pi::pi_part_bellard_mpi(summand_count, precision, process_id, process_count);
    my::mpi::barrier();

    mpf_class pi(0.0, precision);
    pi_sum_reduce(pi_part, pi);

    pi /= (1 << 6);
    return pi;
}

template <typename RegularPiCalculationFunction,
          typename MPIPiCalculationFunction>
void benchmark(std::size_t summand_count,
               mp_bitcnt_t precision,
               RegularPiCalculationFunction pi_regular,
               MPIPiCalculationFunction pi_mpi)
{
    static constexpr std::size_t ITERATIONS_COUNT = 100;
    const auto& mpi_params = my::mpi::Params::get_instance();

    if (summand_count < mpi_params.process_count())
    {
        throw std::runtime_error("Summand count is less than processor count, please decrease number of processors.");
    }

    if (my::mpi::is_current_process_root())
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
        auto pi_mpi_wrapper = [summand_count, precision, pi_mpi]()
        {
            return pi_mpi(summand_count, precision);
        };
        pi_mpi_result = my::benchmark_function(pi_mpi_wrapper, ITERATIONS_COUNT);
    }
    if (my::mpi::is_current_process_root())
    {
        my::print_result("    MPI time: ", pi_mpi_result);
    }
}

template <typename MPIPiCalculationFunction>
void calculate(std::size_t summand_count,
               mp_bitcnt_t precision,
               MPIPiCalculationFunction pi_mpi)
{
    const auto& mpi_params = my::mpi::Params::get_instance();

    if (summand_count < mpi_params.process_count())
    {
        throw std::runtime_error("Summand count is less than processor count, please decrease number of processors.");
    }

    mpf_class pi_mpi_result = pi_mpi(summand_count, precision);

    if (my::mpi::is_current_process_root())
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
    my::mpi::Control mpi_control(argc, argv);

    struct AlgorithmInfo
    {
        const std::function<mpf_class(std::size_t summand_count, mp_bitcnt_t precision)> pi_regular;
        const std::function<mpf_class(std::size_t summand_count, mp_bitcnt_t precision)> pi_mpi;
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
                    .precision = (1 << 7),
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
                  algorithm_info.pi_regular,
                  algorithm_info.pi_mpi);
    }
    else
    {
        calculate(algorithm_info.params.calculation_summand_count,
                  algorithm_info.params.precision,
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
