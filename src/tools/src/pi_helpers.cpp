#include <pi_helpers.hpp>

#include <gmpxx.h>

#include <cstddef>

namespace my::pi
{

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

mpf_class pi_part_leibniz_mpi(std::size_t summand_count, mp_bitcnt_t precision,
                              std::size_t process_id, std::size_t process_count)
{
    mpf_class pi_part(0.0, precision);
    mpf_class temp(0.0, precision);
    for (std::size_t i = process_id; i < summand_count; i += process_count)
    {
        temp = (i % 2 == 0 ? 1.0 : -1.0);
        temp /= 2 * i + 1;
        pi_part += temp;
    }
    return pi_part;
}

mpf_class pi_part_bellard_mpi(std::size_t summand_count, mp_bitcnt_t precision,
                              std::size_t process_id, std::size_t process_count)
{
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
    return pi_part;
}

}  // namespace my::pi
