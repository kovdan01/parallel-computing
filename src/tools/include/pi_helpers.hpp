#ifndef PARALLEL_COMPUTING_TOOLS_PI_HELPERS_HPP_
#define PARALLEL_COMPUTING_TOOLS_PI_HELPERS_HPP_

#include <my/pi_helpers/export.h>

#include <gmpxx.h>

#include <cstddef>

namespace my::pi
{

MY_PI_HELPERS_EXPORT mpf_class pi_leibniz_regular(std::size_t summand_count, mp_bitcnt_t precision);

MY_PI_HELPERS_EXPORT mpf_class pi_bellard_regular(std::size_t summand_count, mp_bitcnt_t precision);

MY_PI_HELPERS_EXPORT mpf_class pi_part_leibniz_mpi(std::size_t summand_count, mp_bitcnt_t precision,
                                                   std::size_t process_id, std::size_t process_count);

MY_PI_HELPERS_EXPORT mpf_class pi_part_bellard_mpi(std::size_t summand_count, mp_bitcnt_t precision,
                                                   std::size_t process_id, std::size_t process_count);

enum class AlgorithmType
{
    BELLARD,
    LEIBNIZ,
};

struct AlgorithmParams
{
    const mp_bitcnt_t precision;
    const std::size_t benchmark_summand_count;
    const std::size_t calculation_summand_count;
};

}  // namespace my::pi

#endif  // PARALLEL_COMPUTING_TOOLS_PI_HELPERS_HPP_
