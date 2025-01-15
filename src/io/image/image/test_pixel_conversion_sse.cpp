/******************************************************************************
 * Copyright (c) 2017-2025, NVIDIA CORPORATION. All rights reserved.
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

#define MI_TEST_AUTO_SUITE_NAME "Regression Test Suite for io/image/image"
#define MI_TEST_IMPLEMENT_TEST_MAIN_INSTEAD_OF_MAIN

#include <base/system/test/i_test_auto_driver.h>
#include <base/system/test/i_test_auto_case.h>

#include "i_image.h"
#include "i_image_pixel_conversion.h"
#include "image_canvas_impl.h"
#include "image_tile_impl.h"

#include <random>

#include <mi/base/handle.h>
#include <mi/math/color.h>

#include <base/system/main/access_module.h>
#include <base/hal/time/time_stopwatch.h>
#include <base/lib/log/i_log_module.h>
#include <base/lib/mem/mem.h>

using namespace MI;

const mi::Uint32 W = 4096;
const mi::Uint32 H = 2160;
const mi::Uint32 K = W * H;

SYSTEM::Access_module<IMAGE::Image_module> g_image_module;

mi::Size g_failures = 0;

mi::Float32 rand_float32()
{
    static std::mt19937 prng;
    return static_cast<float>(static_cast<double>(prng()) / prng.max());
}

template <IMAGE::Pixel_type Source, IMAGE::Pixel_type Dest>
void test_conversion()
{
    const char* source_type = IMAGE::convert_pixel_type_enum_to_string( Source);
    const char* dest_type   = IMAGE::convert_pixel_type_enum_to_string( Dest);
    std::cout << "testing conversion from " << source_type << " to " << dest_type << std::endl;

    mi::Size source_cpp = IMAGE::Pixel_type_traits<Source>::s_components_per_pixel;
    mi::Size dest_cpp   = IMAGE::Pixel_type_traits<Dest>::s_components_per_pixel;
    mi::Size source_bpp = IMAGE::get_bytes_per_pixel( Source);
    mi::Size dest_bpp   = IMAGE::get_bytes_per_pixel( Dest);

    // generate K source pixels in PT_COLOR format
    std::vector<mi::math::Color> c_in( K);
    for( size_t k = 0; k < K; ++k) {
        c_in[k] = mi::math::Color( rand_float32(), rand_float32(), rand_float32(), rand_float32());
        if( source_cpp == 1 || dest_cpp == 1)
            c_in[k].g = c_in[k].b = c_in[k].r;
        if(    !IMAGE::Pixel_type_traits<Source>::s_has_alpha
            || !IMAGE::Pixel_type_traits<Dest>::s_has_alpha)
            c_in[k].a = 1.0f;
    }

    // buffers for intermediate and final result
    using Source_base_type = typename IMAGE::Pixel_type_traits<Source>::Base_type;
    using Dest_base_type   = typename IMAGE::Pixel_type_traits<Dest>::Base_type;
    std::vector<Source_base_type> source( 5*K);
    std::vector<Dest_base_type> dest( 5*K);
    std::vector<mi::math::Color> c_out( K);

    // convert from PT_COLOR to Source to Target to PT_COLOR
    IMAGE::Pixel_converter<IMAGE::PT_COLOR, Source>::convert( &c_in[0].r, &source[0], K);
    TIME::Stopwatch stopwatch;
    stopwatch.start();
    IMAGE::Pixel_converter<Source, Dest>::convert(
        &source[0], &dest[0], W, H, W*source_bpp, W*dest_bpp);
    stopwatch.stop();
    std::cout << "conversion of " << W << "x" << H << " pixels: "
              << 1000.0 * stopwatch.elapsed() << " milliseconds" << std::endl;
    IMAGE::Pixel_converter<Dest, IMAGE::PT_COLOR>::convert( &dest[0], &c_out[0].r, K);

    // compare results
    for( size_t k = 0; k < K; ++k) {
        mi::math::Vector<mi::Float32,4> diff( c_out[k] - c_in[k]);
        if( source_cpp == 2 || dest_cpp == 2)
            diff[2] = 0.0f;
        if( length( diff) > 0.0104) {
            g_failures++;
            std::cout << "(";
            for( mi::Size j = 0; j < 4; ++j)
                std::cout << (j>0 ? ", " : "") << c_in[k][j];
            std::cout << ") => (";
            for( mi::Size j = 0; j < source_cpp; ++j)
                std::cout << (j>0 ? ", " : "") << (float) source[k*source_cpp+j];
            std::cout << ") => (";
            for( mi::Size j = 0; j < dest_cpp; ++j)
                std::cout << (j>0 ? ", " : "") << (float) dest[k*dest_cpp+j];
            std::cout << ") => (";
            for( mi::Size j = 0; j < 4; ++j)
                std::cout << (j>0 ? ", " : "") << c_out[k][j];
            std::cout << "), |diff| = " << length( diff) << std::endl;
        }
    }
}

MI_TEST_AUTO_FUNCTION( test_pixel_conversion )
{
    SYSTEM::Access_module<MEM::Mem_module> mem_module( false);
    SYSTEM::Access_module<LOG::Log_module> log_module( false);

    g_image_module.set();

    using namespace MI::IMAGE;

    test_conversion<PT_RGB_FP,PT_RGB >();
    test_conversion<PT_COLOR, PT_RGBA>();

    MI_CHECK_EQUAL( g_failures, 0);
}

MI_TEST_MAIN_CALLING_TEST_MAIN();
