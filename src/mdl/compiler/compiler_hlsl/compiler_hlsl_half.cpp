/******************************************************************************
 * Copyright (c) 2018-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *****************************************************************************/

#include "pch.h"

#include "compiler_hlsl_half.h"

namespace mi {
namespace mdl {
namespace hlsl {

/// Convert an IEEE 754 single float to an IEEE 754-2008 half float.
///
/// \param f  bitwise representation of the single float value
uint16_t bit_single_to_half(uint32_t f)
{
    uint32_t s = (f >> 16) & 0x8000;
    int32_t  e = ((f >> 23) & 0x000000FF) - 127;
    uint32_t m = f & 0x007FFFFF;

    if (e < -15 - 10) {
        // a normal or a denormal: due to limited exponent will be flushed to zero
        return s;
    }
    if (e <= -15) {
        // a normal that will be converted into a denormal
        m |= 0x00800000;  // make implicit 1 of mantissa explicit

        int32_t  t = -e - 1;                  // determine how many bits we have to throw away

        // is value equally far away from two nearest floating-point numbers?
        if ((m & ((1u << t) - 1)) == 1u << (t - 1))
            m += m & (1u << t);  // round to even
        else
            m += 1u << (t - 1);  // round normally

        return s | (m >> t);
    }
    if (e == 128) {
        if (m == 0) {
            // f is an infinity
            return s | (0x1F << 10);
        } else {
            // f is sNAN or qNAN
            uint32_t qNAN = m & (1u << 22);
            if (qNAN != 0) {
                // preserve the lowest bits
                m &= 0x3FF;
                return s | (0x1F << 10) | (1u << 9) | m;
            } else {
                // an sNAN, preserve the lowest if possible
                m &= 0x3FF;
                if (m == 0)
                    m = 1;
                return s | (0x1F << 10) | m;
            }
        }
    }

    // supported range

    // is value equally far away from two nearest floating-point numbers?
    if ((m & ((1u << 13) - 1)) == 1u << 12)
        m += m & (1u << 13);  // round to even
    else
        m += 1u << 12;        // round normally

    if (m == 0x00800000) {
        // overflow
        ++e;
        m = 0;
    }

    if (e > 15) {
        // too big, convert to infinity
        return s | (0x1F << 10);
    }

    return s | ((e + 15) << 10) | (m >> 13);
}

/// Convert an IEEE 754-2008 half float to an IEEE 754 single float.
///
/// \param f  bitwise representation of the half float value
uint32_t bit_half_to_single(uint16_t h)
{
    uint32_t s = (h & 0x8000) << 16;
    int32_t  e = ((h >> 10) & 0x0000001F) - 15;
    uint32_t m = h & 0x000003FF;

    if (e == 16) {
        if (m == 0) {
            // h is an infinity
            return s | (0xFF << 23);
        } else {
            // h is sNAN or qNAN
            uint32_t qNAN = m & (1u << 9);
            m &= ~(1u << 9);
            // preserve the lowest bits
            return s | (0xFF << 23) | (qNAN << 13) | m;
        }
    }

    if (e == -15) {
        // a denormal or zero, which use a bias of 14 for the exponent
        e = -14;

        // normalize by shifting the mantissa left, until it overflows or we're out of bits
        uint32_t i = 0;
        for (; i < 10; ++i) {
            --e;
            m <<= 1u;
            if ((m & (1u << 10)) != 0) {
                break;
            }
        }

        if (m == 0)
            return s;

        m &= ~(1u << 10);

        // fall through, now it is normalized
    }

    // a normal
    return s | ((e + 127) << 23) | (m << 13);
}


#ifdef TEST_HALF_CONVERSION

static float int_as_float(uint32_t v)
{
    union
    {
        uint32_t bit;
        float    value;
    } temp;

    temp.bit = v;
    return temp.value;
}

static uint32_t float_as_int(float v)
{
    union
    {
        uint32_t bit;
        float    value;
    } temp;

    temp.value = v;
    return temp.bit;
}

static bool is_nan(float v)
{
    uint32_t bit_v = float_as_int(v);

    // exponent == 0xff && mantissa != 0
    return (bit_v & 0x7f800000) == 0x7f800000 && (bit_v & 0x007fffff) != 0;
}

static int test_rounding(
    uint32_t f_bit,
    uint16_t expected_h_down,
    uint16_t expected_h_middle,
    uint16_t expected_h_up,
    bool denormalized)
{
    int num_failed = 0;
    for (uint32_t low_mantissa = 0; low_mantissa < 0x2000; ++low_mantissa) {
        uint32_t one_plus_low_bit = f_bit + low_mantissa * (denormalized ? 2 : 1);
        uint16_t h_bit = bit_single_to_half(one_plus_low_bit);
        uint16_t expected_h_bit;
        if (low_mantissa < 0x1000)
            expected_h_bit = expected_h_down;
        else if (low_mantissa == 0x1000)
            expected_h_bit = expected_h_middle;
        else
            expected_h_bit = expected_h_up;
        if (h_bit != expected_h_bit) {
            printf("bad rounding for %.8X -> %.4X != %.4X\n",
                one_plus_low_bit, h_bit, expected_h_bit);
            ++num_failed;
        }
    }
    return num_failed;
}

int test_hlsl_half()
{
    int num_failed = 0;

    // for all half numbers:
    //   convert half number to float
    //     convert back -> should be same
    //     compare with previous float number, should be bigger, unless NaN (unordered)

    float prev_s_val = 0;
    for (uint32_t h_bit_32 = 0; h_bit_32 < 0x10000; ++h_bit_32) {
        uint16_t h_bit = uint16_t(h_bit_32);

        uint32_t s_bit = bit_half_to_single(h_bit);
        float s_val = int_as_float(s_bit);
        uint16_t h_bit_back = bit_single_to_half(s_bit);

        if (h_bit != h_bit_back) {
            printf("%.4X -single-> %.8X (%f) -half-> %.4X not same value again!\n",
                h_bit_32, s_bit, s_val, uint32_t(h_bit_back));
            ++num_failed;
        }

        // For positive values: prev_s_val < s_val
        // For negative values: prev_s_val > s_val
        bool is_neg = (h_bit & 0x8000) != 0;
        if (h_bit_32 > 0 && ((!is_neg && !(prev_s_val < s_val))
                || (is_neg && !(prev_s_val > s_val)))) {
            // NaNs are unordered, so ensure we don't compare NaNs
            if (!is_nan(prev_s_val) && !is_nan(s_val)) {
                printf("%.4X (%f) is not < %.4X (%f)!\n",
                    h_bit_32 - 1, prev_s_val, h_bit_32, s_val);
                ++num_failed;
            }
        }

        prev_s_val = s_val;
    }

    float res_1_3 = int_as_float(bit_half_to_single(0x3555));
    if (res_1_3 != 0.333251953125) {
        printf("1/3 test mismatch: 0x3555 -> %f\n", res_1_3);
        ++num_failed;
    }

    float res_smallest_after_1 = int_as_float(bit_half_to_single(0x3c01));
    if (res_smallest_after_1 != 1.0009765625) {
        printf("smallest after one test mismatch: 0x3c01 -> %f\n", res_smallest_after_1);
        ++num_failed;
    }

    // test rounding:
    //  - for non-representable mantissa < "0.5" -> round down
    //  - for non-representable mantissa == "0.5" -> round to even
    //  - for non-representable mantissa >= "0.5" -> round up
    //
    // the mask for the non-representable mantissa part is: 0x00001fff

    // test: 1 + non-representable mantissa
    //   round to even -> rounds down in this case (last binary digit is 0)
    num_failed += test_rounding(0x3f800000, 0x3c00, 0x3c00, 0x3c01, false);

    // test: 1.000...01 + non-representable mantissa
    //   round to even -> rounds up in this case (last binary digit it 1)
    num_failed += test_rounding(0x3f802000, 0x3c01, 0x3c02, 0x3c02, false);

    // test: 2^-14 - 2^-24 + non-representable mantissa  (maximum denormal case)
    //   round to even -> rounds up in this case (last binary digit is 1)
    //   overflows into normalized value
    num_failed += test_rounding(0x387FC000, 0x03ff, 0x0400, 0x0400, true);

    return num_failed;
}

#endif  // TEST_HALF_CONVERSIONS

}  // hlsl
}  // mdl
}  // mi
