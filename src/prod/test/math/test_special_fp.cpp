/******************************************************************************
 * Copyright (c) 2008-2025, NVIDIA CORPORATION. All rights reserved.
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
 ******************************************************************************/

// Test floating point functions to identify special IEEE 754 values
// such as "not a number" (NaN), infinity, etc.

#include "pch.h"
#include <base/system/test/i_test_auto_case.h>
#include <mi/math/function.h>
#include <limits> // for inf, NaN

using namespace MI;

// test function to abstract the floating point type
template <typename T>
void test_func()
{
    // check some "normal" float values
    T x = T(1.00123);
    for (int i = 0; i < 15; ++i, x *= x)
    {
        MI_CHECK(!mi::math::isnan(x));
        MI_CHECK(!mi::math::isinfinite(x));
        MI_CHECK(mi::math::isfinite(x));
        MI_CHECK(!mi::math::isnan(-x));
        MI_CHECK(!mi::math::isinfinite(-x));
        MI_CHECK(mi::math::isfinite(-x));
    }

    // check zero values
    MI_CHECK(!mi::math::isnan(T(0.)));
    MI_CHECK(!mi::math::isinfinite(T(0.)));
    MI_CHECK(mi::math::isfinite(T(0.)));
    MI_CHECK(!mi::math::isnan(T(-0.)));
    MI_CHECK(!mi::math::isinfinite(T(-0.)));
    MI_CHECK(mi::math::isfinite(T(-0.)));

    // check min / max values
    MI_CHECK(!mi::math::isnan(std::numeric_limits<T>::min()));
    MI_CHECK(!mi::math::isinfinite(std::numeric_limits<T>::min()));
    MI_CHECK(mi::math::isfinite(std::numeric_limits<T>::min()));
    MI_CHECK(!mi::math::isnan(std::numeric_limits<T>::max()));
    MI_CHECK(!mi::math::isinfinite(std::numeric_limits<T>::max()));
    MI_CHECK(mi::math::isfinite(std::numeric_limits<T>::max()));

    // check denormalized values
    MI_CHECK(!mi::math::isnan(std::numeric_limits<T>::denorm_min()));
    MI_CHECK(!mi::math::isinfinite(std::numeric_limits<T>::denorm_min()));
    MI_CHECK(mi::math::isfinite(std::numeric_limits<T>::denorm_min()));
    MI_CHECK(!mi::math::isnan(-std::numeric_limits<T>::denorm_min()));
    MI_CHECK(!mi::math::isinfinite(-std::numeric_limits<T>::denorm_min()));
    MI_CHECK(mi::math::isfinite(-std::numeric_limits<T>::denorm_min()));

    // check infinity
    MI_CHECK(std::numeric_limits<T>::has_infinity);
    MI_CHECK(!mi::math::isnan(std::numeric_limits<T>::infinity()));
    MI_CHECK(mi::math::isinfinite(std::numeric_limits<T>::infinity()));
    MI_CHECK(!mi::math::isfinite(std::numeric_limits<T>::infinity()));
    MI_CHECK(!mi::math::isnan(-std::numeric_limits<T>::infinity()));
    MI_CHECK(mi::math::isinfinite(-std::numeric_limits<T>::infinity()));
    MI_CHECK(!mi::math::isfinite(-std::numeric_limits<T>::infinity()));

    // check NaN
    MI_CHECK(std::numeric_limits<T>::has_quiet_NaN);
    MI_CHECK(mi::math::isnan(std::numeric_limits<T>::quiet_NaN()));
    MI_CHECK(!mi::math::isinfinite(std::numeric_limits<T>::quiet_NaN()));
    MI_CHECK(!mi::math::isfinite(std::numeric_limits<T>::quiet_NaN()));
}

MI_TEST_AUTO_FUNCTION( test_special_fp )
{
    // run tests for float and double
    test_func<float>();
    test_func<double>();
}

