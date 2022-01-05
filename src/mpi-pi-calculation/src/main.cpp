#include <benchmark.hpp>
#include <mpi.hpp>

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

static constexpr std::size_t ITERATIONS_COUNT = 100;

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

mpf_class pi_leibniz_regular(std::size_t summand_count, mp_bitcnt_t precision)
{
    mpf_class pi(0.0, precision);
    mpf_class temp(0.0, precision);
    for (std::size_t i = 0; i < summand_count; ++i)
    {
        temp = (i % 2 == 0 ? 1.0 : -1.0);
        temp /= 2 * i + 1;
        pi += temp;
    }
    return 4 * pi;
}

mpf_class pi_bellard_regular(std::size_t summand_count, mp_bitcnt_t precision)
{
    mpf_class pi(0.0, precision);
    mpf_class multiplier(1.0, precision);
    static const mpf_class multiplier_change = mpf_class(-1.0, precision) / (1 << 10);
    mpf_class part1(0.0, precision);
    mpf_class part1_denominator(1.0, precision);
    mpf_class part2(0.0, precision);
    mpf_class part2_denominator(3.0, precision);
    mpf_class part3(0.0, precision);
    mpf_class part3_denominator(1.0, precision);
    mpf_class part4(0.0, precision);
    mpf_class part4_denominator(3.0, precision);
    mpf_class part5(0.0, precision);
    mpf_class part5_denominator(5.0, precision);
    mpf_class part6(0.0, precision);
    mpf_class part6_denominator(7.0, precision);
    mpf_class part7(0.0, precision);
    mpf_class part7_denominator(9.0, precision);
    mpf_class temp(0.0, precision);
    for (std::size_t i = 0; i < summand_count; ++i)
    {
        part1 = -(1 << 5);
        part1 /= part1_denominator;
        part2 = -(1 << 0);
        part2 /= part2_denominator;
        part3 = +(1 << 8);
        part3 /= part3_denominator;
        part4 = -(1 << 6);
        part4 /= part4_denominator;
        part5 = -(1 << 2);
        part5 /= part5_denominator;
        part6 = -(1 << 2);
        part6 /= part6_denominator;
        part7 = +(1 << 0);
        part7 /= part7_denominator;

        temp  = part1;
        temp += part2;
        temp += part3;
        temp += part4;
        temp += part5;
        temp += part6;
        temp += part7;
        temp *= multiplier;

        pi += temp;

        multiplier *= multiplier_change;
        part1_denominator += 4;
        part2_denominator += 4;
        part3_denominator += 10;
        part4_denominator += 10;
        part5_denominator += 10;
        part6_denominator += 10;
        part7_denominator += 10;
    }
    pi /= (1 << 6);
    return pi;
}

mpf_class pi_leibniz_mpi(std::size_t summand_count, mp_bitcnt_t precision)
{
    const auto& mpi_params = my::mpi::Params::get_instance();
    std::size_t process_id = mpi_params.process_id();
    std::size_t process_count = mpi_params.process_count();

    my::mpi::bcast(&summand_count, 1, MPI_UNSIGNED_LONG_LONG);

    mpf_class pi_part(0.0, precision);

    mpf_class temp(0.0, precision);
    for (std::size_t i = process_id; i < summand_count; i += process_count)
    {
        temp = (i % 2 == 0 ? 1.0 : -1.0);
        temp /= 2 * i + 1;
        pi_part += temp;
    }

    mpf_class pi(0.0, precision);

    my::mpi::barrier();

    pi_sum_reduce(pi_part, pi);

    return 4 * pi;
}

mpf_class pi_bellard_mpi(std::size_t summand_count, mp_bitcnt_t precision)
{
    const auto& mpi_params = my::mpi::Params::get_instance();
    std::size_t process_id = mpi_params.process_id();
    std::size_t process_count = mpi_params.process_count();

    my::mpi::bcast(&summand_count, 1, MPI_UNSIGNED_LONG_LONG);

    mpf_class pi_part(0.0, precision);

    mpf_class multiplier(1.0, precision);
    static mpf_class multiplier_change = mpf_class(-1.0, precision) / (1 << 10);
    mpf_pow_ui(multiplier.get_mpf_t(), multiplier_change.get_mpf_t(), process_id);
    mpf_pow_ui(multiplier_change.get_mpf_t(), multiplier_change.get_mpf_t(), process_count);
    mpf_class part1(0.0, precision);
    mpf_class part1_denominator( 4 * process_id + 1, precision);
    mpf_class part2(0.0, precision);
    mpf_class part2_denominator( 4 * process_id + 3, precision);
    mpf_class part3(0.0, precision);
    mpf_class part3_denominator(10 * process_id + 1, precision);
    mpf_class part4(0.0, precision);
    mpf_class part4_denominator(10 * process_id + 3, precision);
    mpf_class part5(0.0, precision);
    mpf_class part5_denominator(10 * process_id + 5, precision);
    mpf_class part6(0.0, precision);
    mpf_class part6_denominator(10 * process_id + 7, precision);
    mpf_class part7(0.0, precision);
    mpf_class part7_denominator(10 * process_id + 9, precision);
    mpf_class temp(0.0, precision);
    for (std::size_t i = process_id; i < summand_count; i += process_count)
    {
        part1 = -(1 << 5);
        part1 /= part1_denominator;
        part2 = -(1 << 0);
        part2 /= part2_denominator;
        part3 = +(1 << 8);
        part3 /= part3_denominator;
        part4 = -(1 << 6);
        part4 /= part4_denominator;
        part5 = -(1 << 2);
        part5 /= part5_denominator;
        part6 = -(1 << 2);
        part6 /= part6_denominator;
        part7 = +(1 << 0);
        part7 /= part7_denominator;

        temp  = part1;
        temp += part2;
        temp += part3;
        temp += part4;
        temp += part5;
        temp += part6;
        temp += part7;
        temp *= multiplier;

        pi_part += temp;

        multiplier *= multiplier_change;
        part1_denominator +=  4 * process_count;
        part2_denominator +=  4 * process_count;
        part3_denominator += 10 * process_count;
        part4_denominator += 10 * process_count;
        part5_denominator += 10 * process_count;
        part6_denominator += 10 * process_count;
        part7_denominator += 10 * process_count;
    }

    mpf_class pi(0.0, precision);

    my::mpi::barrier();

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

enum class PiCalculationAlgorithm
{
    BELLARD,
    LEIBNIZ,
};

struct PiCalculationAlgorithmDetails
{
    const std::function<mpf_class(std::size_t summand_count, mp_bitcnt_t precision)> pi_regular;
    const std::function<mpf_class(std::size_t summand_count, mp_bitcnt_t precision)> pi_mpi;
    const mp_bitcnt_t precision;
    const std::size_t benchmark_summand_count;
    const std::size_t calculation_summand_count;
};

}  // namespace

int main(int argc, char* argv[]) try
{
    my::mpi::Control mpi_control(argc, argv);

    std::unordered_map<PiCalculationAlgorithm, PiCalculationAlgorithmDetails>
    algorithm_details_map =
    {
        { PiCalculationAlgorithm::BELLARD, { .pi_regular = pi_bellard_regular,
                                             .pi_mpi = pi_bellard_mpi,
                                             .precision = (1 << 22),
                                             .benchmark_summand_count = std::size_t{1} << 8,
                                             .calculation_summand_count = std::size_t{1} << 27} },
        { PiCalculationAlgorithm::LEIBNIZ, { .pi_regular = pi_leibniz_regular,
                                             .pi_mpi = pi_leibniz_mpi,
                                             .precision = (1 << 7),
                                             .benchmark_summand_count = std::size_t{1} << 26,
                                             .calculation_summand_count = std::size_t{1} << 45} },
    };

    bool do_benchmark = true;
    auto algorithm = PiCalculationAlgorithm::BELLARD;

    const PiCalculationAlgorithmDetails& algorithm_details = algorithm_details_map.at(algorithm);

    if (do_benchmark)
    {
        benchmark(algorithm_details.benchmark_summand_count,
                  algorithm_details.precision,
                  algorithm_details.pi_regular,
                  algorithm_details.pi_mpi);
    }
    else
    {
        calculate(algorithm_details.calculation_summand_count,
                  algorithm_details.precision,
                  algorithm_details.pi_mpi);
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
