/******************************************************************************
 * Copyright (c) 2019-2020, NVIDIA CORPORATION. All rights reserved.
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
/** \file
 ** \brief
 **/

#pragma once


#include <climits>
#include <limits>

#ifndef __CUDACC__
#include <base/system/main/platform.h>
#include <base/system/main/i_assert.h>
#include <base/system/stlext/i_stlext_binary_cast.h>
#include <algorithm>
#include <cmath>
#endif


namespace MI {
namespace IMAGE {

// usually not needed extra-safe dequantization that forces to use a division instead of a multiplication with (1./(N = (1 << bits) - 1))
//  tested with 8,16 and up to 30 bits
//  the test did dequantize an unsigned (X bits) to float, then back to unsigned via quantize_unsigned (below) with both variants
//  the intermediate float values then obviously differed a bit, but the resulting unsigned from the back and forth conversion was always the same!
/*template <unsigned char bits> // number of bits to map to
MI_HOST_DEVICE_INLINE float dequantize_unsigned(const unsigned int i)
{
#ifndef __CUDACC__
    using std::min;
#endif
    enum { N = (1u << bits) - 1 };
    return min(precise_divide((float)i, (float)N), 1.f); //!! test: optimize div or does this break precision?
}*/

template <unsigned char bits> // number of bits to map to
MI_HOST_DEVICE_INLINE unsigned quantize_unsigned(const float x)
{
#ifndef __CUDACC__
    MI_ASSERT(!std::isnan(x));
    //MI_ASSERT(std::isfinite(x)); // is handled in the second code variant below
    MI_ASSERT(x >= 0.f);
    using std::min;
    static_assert(bits < 31, "bits must be smaller 31");
#endif
    enum { N = (1u << bits) - 1, Np1 = (1u << bits) };
    //return min((unsigned)(x * (float)Np1),(unsigned)N); // does not handle large values, as these trigger undefined behavior (on x86: 0)
#ifdef __CUDACC__
    return (unsigned)(fminf(x, uint_as_float(0x3f800000u-1)) * (float)Np1);
#else
    return (unsigned)(min(x, MI::STLEXT::binary_cast<float>(0x3f800000u-1)) * (float)Np1);
#endif
}


template <typename T>
MI_HOST_DEVICE_INLINE T quantize_unsigned(const float x)
{
#ifndef __CUDACC__
    static_assert(std::numeric_limits<T>::is_integer && !std::numeric_limits<T>::is_signed,"can only quantize to unsigned integer types");
#endif
    return T(quantize_unsigned<sizeof(T)*CHAR_BIT>(x));
}


template <unsigned char bits> // number of bits to map to, including negative numbers, so mapping just to unsigned: bits+1 (0..255 -> bits = 9)
MI_HOST_DEVICE_INLINE int quantize_signed(const float x)
{
#ifndef __CUDACC__
    MI_ASSERT(!std::isnan(x));
    MI_ASSERT(std::isfinite(x));
    using std::max;
    using std::min;
    static_assert(bits < 32, "bits must be smaller 32"); //!! 33?
#endif
    enum { N = (1u << (bits - 1)) - 1 };
    const float sign = (x >= 0.f) ? 0.5f : -0.5f;
    return (int)(min(max(x * (float)((double)N + 0.5) + sign, -(float)N), (float)N));
}


template <typename T>
MI_HOST_DEVICE_INLINE T quantize_signed(const float x)
{
#ifndef __CUDACC__
    static_assert(std::numeric_limits<T>::is_integer && std::numeric_limits<T>::is_signed,"can only quantize to signed integer types");
#endif
    return T(quantize_signed<sizeof(T)*CHAR_BIT>(x));
}


namespace DETAIL {

template <bool is_signed>
struct Quantize
{
    template <typename T>
    MI_HOST_DEVICE_INLINE static T quantize(const float x) { return quantize_signed<T>(x); }
};

template <>
struct Quantize<false>
{
    template <typename T>
    MI_HOST_DEVICE_INLINE static T quantize(const float x) { return quantize_unsigned<T>(x); }
};

}


template <typename T>
MI_HOST_DEVICE_INLINE T quantize(const float x)
{
    return DETAIL::Quantize<std::numeric_limits<T>::is_signed>::template quantize<T>(x);
}

}}


