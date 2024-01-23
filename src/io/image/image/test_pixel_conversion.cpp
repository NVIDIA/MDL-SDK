/******************************************************************************
 * Copyright (c) 2011-2024, NVIDIA CORPORATION. All rights reserved.
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
#include <base/lib/mem/mem.h>
#include <base/lib/log/i_log_module.h>

using namespace MI;
using namespace MI::IMAGE;

const mi::Uint32 N = 100000;

Image_module* g_image_module = nullptr;


class Module_holder
{
public:
    Module_holder()
      : m_mem_module( false),
        m_log_module( false),
        m_image_module( false)
    {
        g_image_module = m_image_module.operator->();
    }

    ~Module_holder()
    {
        g_image_module = nullptr;
    }

private:
    Module_holder( const Module_holder&) = delete;
    Module_holder& operator=( const Module_holder&) = delete;

    SYSTEM::Access_module<MEM::Mem_module> m_mem_module;
    SYSTEM::Access_module<LOG::Log_module> m_log_module;
    SYSTEM::Access_module<IMAGE::Image_module> m_image_module;
};


mi::Float32 rand_float32()
{
    static std::mt19937 prng;
    return static_cast<float>(static_cast<double>(prng()) / prng.max());
}


template <IMAGE::Pixel_type Type>
void test_set_get()
{
    const char* pixel_type = IMAGE::convert_pixel_type_enum_to_string( Type);
    std::cout << "testing set/get for " << pixel_type << std::endl;

    // buffers for intermediate result
    mi::base::Handle<mi::neuraylib::ITile> tile( create_tile( Type, N, 1));

    // test N randomly generated pixels
    for( mi::Uint32 i = 0; i < N; ++i) {

        // generate the source pixel in PT_COLOR format
        mi::math::Color c_in( rand_float32(), rand_float32(), rand_float32(), rand_float32());
        int cpp = IMAGE::Pixel_type_traits<Type>::s_components_per_pixel;
        if( strcmp( pixel_type, "Sint32") != 0 && strcmp( pixel_type, "Float32<4>") != 0) {
            if( cpp == 1)
                c_in.g = c_in.b = c_in.r;
            if( cpp == 2)
                c_in.b = 0.0f;
            if( !IMAGE::Pixel_type_traits<Type>::s_has_alpha)
                c_in.a = 1.0f;
        }

        // buffers for final result
        mi::math::Color c_out;

        // convert from PT_COLOR to Source to Target to PT_COLOR
        tile->set_pixel( i, 0, &c_in.r);
        tile->get_pixel( i, 0, &c_out.r);

        // compare results
        mi::math::Vector<mi::Float32,4> diff( c_out - c_in);
        const float allowed_diff = 0.0102f;
        if( length( diff) > allowed_diff) {
            std::cout << "(";
            for( mi::Size j = 0; j < 4; ++j)
                std::cout << (j>0 ? ", " : "") << c_in[j];
            std::cout << ") => (";
            for( mi::Size j = 0; j < 4; ++j)
                std::cout << (j>0 ? ", " : "") << c_out[j];
            std::cout << "), |diff| = " << length( diff) << std::endl;
        }
        MI_CHECK_LESS_OR_EQUAL(length(diff),allowed_diff);
    }
}

template <IMAGE::Pixel_type Source, IMAGE::Pixel_type Dest>
void test_conversion()
{
    const char* source_type = IMAGE::convert_pixel_type_enum_to_string( Source);
    const char* dest_type   = IMAGE::convert_pixel_type_enum_to_string( Dest);
    std::cout << "testing conversion from " << source_type << " to " << dest_type << std::endl;

    mi::Size source_cpp = IMAGE::Pixel_type_traits<Source>::s_components_per_pixel;
    mi::Size dest_cpp   = IMAGE::Pixel_type_traits<Dest>::s_components_per_pixel;

    // test roughly N randomly generated pixels, use array of K pixels to test SSE code paths
    const size_t K = 31;
    for( mi::Size i = 0; i < N/K; ++i) {

        // generate K source pixels in PT_COLOR format
        mi::math::Color c_in[K];
        for( size_t k = 0; k < K; ++k) {
            c_in[k] = mi::math::Color(
                rand_float32(), rand_float32(), rand_float32(), rand_float32());
            if( source_cpp == 1 || dest_cpp == 1)
                c_in[k].g = c_in[k].b = c_in[k].r;
            if(    !IMAGE::Pixel_type_traits<Source>::s_has_alpha
                || !IMAGE::Pixel_type_traits<Dest>::s_has_alpha)
                c_in[k].a = 1.0f;
        }

        // buffers for intermediate and final result
        typedef typename IMAGE::Pixel_type_traits<Source>::Base_type Source_base_type;
        typedef typename IMAGE::Pixel_type_traits<Dest>::Base_type   Dest_base_type;
        Source_base_type source[5*K];
        Dest_base_type dest[5*K];
        mi::math::Color c_out[K];

        // convert from PT_COLOR to Source to Target to PT_COLOR
        IMAGE::Pixel_converter<IMAGE::PT_COLOR, Source>::convert( &c_in[0].r, &source[0], K);
        IMAGE::Pixel_converter<Source, Dest>::convert( &source[0], &dest[0], K);
        IMAGE::Pixel_converter<Dest, IMAGE::PT_COLOR>::convert( &dest[0], &c_out[0].r, K);

        // compare results
        for( size_t k = 0; k < K; ++k) {
            mi::math::Vector<mi::Float32,4> diff( c_out[k] - c_in[k]);
            if( source_cpp == 2 || dest_cpp == 2)
                diff[2] = 0.0f;
            const float allowed_diff = 0.014f; // larger than above due to SINT8 -> RGBE only
            if( length( diff) > allowed_diff) {
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
            MI_CHECK_LESS_OR_EQUAL(length(diff),allowed_diff);
        }
    }
}

template <IMAGE::Pixel_type Type>
void test_copy()
{
    const char* pixel_type = IMAGE::convert_pixel_type_enum_to_string( Type);
    std::cout << "testing copy for " << pixel_type << std::endl;

    // test N randomly generated pixels
    for( mi::Uint32 i = 0; i < N; ++i) {

        // generate the source pixel in PT_COLOR format
        mi::math::Color c_in( rand_float32(), rand_float32(), rand_float32(), rand_float32());
        mi::Size cpp = IMAGE::Pixel_type_traits<Type>::s_components_per_pixel;
        if( strcmp( pixel_type, "Sint32") != 0 && strcmp( pixel_type, "Float32<4>") != 0) {
            if( cpp == 1)
                c_in.g = c_in.b = c_in.r;
            if( cpp == 2)
                c_in.b = 0.0f;
            if( !IMAGE::Pixel_type_traits<Type>::s_has_alpha)
                c_in.a = 1.0f;
        }

        // buffers for intermediate and final result
        typedef typename IMAGE::Pixel_type_traits<Type>::Base_type Base_type;
        Base_type source[5];
        Base_type dest[5];
        mi::math::Color c_out;

        // convert from PT_COLOR to Source to Target to PT_COLOR
        IMAGE::Pixel_converter<IMAGE::PT_COLOR, Type>::convert( &c_in.r, &source[0]);
        IMAGE::Pixel_copier<Type>::copy( &source[0], &dest[0]);
        IMAGE::Pixel_converter<Type, IMAGE::PT_COLOR>::convert( &dest[0], &c_out.r);

        // compare results
        mi::math::Vector<mi::Float32,4> diff( c_out - c_in);
        const float allowed_diff = 0.0102f;
        if( length( diff) > allowed_diff) {
            std::cout << "(";
            for( mi::Size j = 0; j < 4; ++j)
                std::cout << (j>0 ? ", " : "") << c_in[j];
            std::cout << ") => (";
            for( mi::Size j = 0; j < cpp; ++j)
                std::cout << (j>0 ? ", " : "") << (float) source[j];
            std::cout << ") => (";
            for( mi::Size j = 0; j < cpp; ++j)
                std::cout << (j>0 ? ", " : "") << (float) dest[j];
            std::cout << ") => (";
            for( mi::Size j = 0; j < 4; ++j)
                std::cout << (j>0 ? ", " : "") << c_out[j];
            std::cout << "), |diff| = " << length( diff) << std::endl;
        }
        MI_CHECK_LESS_OR_EQUAL(length(diff),allowed_diff);
    }
}

template <IMAGE::Pixel_type Type>
void test_gamma()
{
    const char* pixel_type = IMAGE::convert_pixel_type_enum_to_string( Type);
    std::cout << "testing gamma for " << pixel_type << std::endl;

    // generate the source pixels in PT_COLOR format
    mi::base::Handle<mi::neuraylib::ICanvas> input(
        new IMAGE::Canvas_impl( IMAGE::PT_COLOR, N, 1, 1, false, 1.0f));
    MI_CHECK_EQUAL( input->get_gamma(), 1.0f);

    mi::base::Handle<mi::neuraylib::ITile> input_tile( input->get_tile());
    for( mi::Uint32 i = 0; i < N; ++i) {
        mi::math::Color c_in( rand_float32(), rand_float32(), rand_float32(), rand_float32());
        int cpp = IMAGE::Pixel_type_traits<Type>::s_components_per_pixel;
        if( strcmp( pixel_type, "Sint32") != 0 && strcmp( pixel_type, "Float32<4>") != 0) {
            if( cpp == 1)
                c_in.g = c_in.b = c_in.r;
            if( cpp == 2)
                c_in.b = 0.0f;
            if( !IMAGE::Pixel_type_traits<Type>::s_has_alpha)
                c_in.a = 1.0f;
        }
        input_tile->set_pixel( i, 0, &c_in.r);
    }

    // generate the source pixels in Type format
    mi::base::Handle<mi::neuraylib::ICanvas> tmp1(
        g_image_module->convert_canvas( input.get(), Type));
    MI_CHECK_EQUAL( tmp1->get_gamma(), 1.0f);

    // encode for gamma 2.2 (LDR)
    mi::base::Handle<mi::neuraylib::ICanvas> tmp2( g_image_module->copy_canvas( tmp1.get()));
    MI_CHECK_EQUAL( tmp2->get_gamma(), 1.0f);
    g_image_module->adjust_gamma( tmp2.get(), 2.2f);
    MI_CHECK_EQUAL( tmp2->get_gamma(), 2.2f);

    // encode for gamma 1.0 (HDR, linear)
    mi::base::Handle<mi::neuraylib::ICanvas> tmp3( g_image_module->copy_canvas( tmp2.get()));
    MI_CHECK_EQUAL( tmp3->get_gamma(), 2.2f);
    g_image_module->adjust_gamma( tmp3.get(), 1.0f);
    MI_CHECK_EQUAL( tmp3->get_gamma(), 1.0f);

    // convert the dest pixel in PT_COLOR format
    mi::base::Handle<mi::neuraylib::ICanvas> output(
        g_image_module->convert_canvas( tmp3.get(), IMAGE::PT_COLOR));
    MI_CHECK_EQUAL( output->get_gamma(), 1.0f);

    // compare with source pixels
    mi::base::Handle<mi::neuraylib::ITile> tmp1_tile( tmp1->get_tile());
    mi::base::Handle<mi::neuraylib::ITile> tmp2_tile( tmp2->get_tile());
    mi::base::Handle<mi::neuraylib::ITile> tmp3_tile( tmp3->get_tile());
    mi::base::Handle<mi::neuraylib::ITile> output_tile( output->get_tile());
    for( mi::Uint32 i = 0; i < N; ++i) {
        mi::math::Color c_in, c_tmp1, c_tmp2, c_tmp3, c_out;
        input_tile ->get_pixel( i, 0, &c_in.r);
        tmp1_tile  ->get_pixel( i, 0, &c_tmp1.r);
        tmp2_tile  ->get_pixel( i, 0, &c_tmp2.r);
        tmp3_tile  ->get_pixel( i, 0, &c_tmp3.r);
        output_tile->get_pixel( i, 0, &c_out.r);
        mi::math::Vector<mi::Float32,4> change( c_tmp2 - c_in);
        mi::Float32 min_change = std::min( std::min( change[0], change[1]), change[2]);
        mi::math::Vector<mi::Float32,4> diff( c_out - c_in);
        const float allowed_diff = 0.0449f;
        const float allowed_min_diff = -0.004f;
        if( length( diff) > allowed_diff || min_change <= allowed_min_diff) {
            std::cout << "(";
            for( mi::Size j = 0; j < 4; ++j)
                std::cout << (j>0 ? ", " : "") << c_in[j];
            std::cout << ") => (";
            for( mi::Size j = 0; j < 4; ++j)
                std::cout << (j>0 ? ", " : "") << c_tmp1[j];
            std::cout << ") => (";
            for( mi::Size j = 0; j < 4; ++j)
                std::cout << (j>0 ? ", " : "") << c_tmp2[j];
            std::cout << ") => (";
            for( mi::Size j = 0; j < 4; ++j)
                std::cout << (j>0 ? ", " : "") << c_tmp3[j];
            std::cout << ") => (";
            for( mi::Size j = 0; j < 4; ++j)
                std::cout << (j>0 ? ", " : "") << c_out[j];
            std::cout << "), |diff| = " << length( diff)
                      << ", min change = " << min_change << std::endl;
        }
        MI_CHECK_LESS_OR_EQUAL(length(diff),allowed_diff);
        MI_CHECK_GREATER(min_change,allowed_min_diff);
    }
}


MI_TEST_AUTO_FUNCTION( test_pixel_conversion_set_get )
{
    Module_holder mods;

    test_set_get<PT_SINT8    >();
    test_set_get<PT_SINT32   >();
    test_set_get<PT_FLOAT32  >();
    test_set_get<PT_FLOAT32_2>();
    test_set_get<PT_FLOAT32_3>();
    test_set_get<PT_FLOAT32_4>();
    test_set_get<PT_RGB      >();
    test_set_get<PT_RGBA     >();
    test_set_get<PT_RGBE     >();
    test_set_get<PT_RGBEA    >();
    test_set_get<PT_RGB_16   >();
    test_set_get<PT_RGBA_16  >();
    test_set_get<PT_RGB_FP   >();
    test_set_get<PT_COLOR    >();
}


MI_TEST_AUTO_FUNCTION( test_pixel_conversion )
{
    Module_holder mods;

    // Note that PT_SINT32, PT_FLOAT32_3, and PT_FLOAT_32_4 are not tested here because there is
    // no template specialization (the free convert() methods map these pixel types to PT_RGBA,
    // PT_RGB_FP, and PT_COLOR, respectively).

    test_conversion<PT_SINT8,       PT_SINT8    >();
    test_conversion<PT_SINT8,       PT_FLOAT32  >();
    test_conversion<PT_SINT8,       PT_FLOAT32_2>();
    test_conversion<PT_SINT8,       PT_RGB      >();
    test_conversion<PT_SINT8,       PT_RGBA     >();
    test_conversion<PT_SINT8,       PT_RGBE     >();
    test_conversion<PT_SINT8,       PT_RGBEA    >();
    test_conversion<PT_SINT8,       PT_RGB_16   >();
    test_conversion<PT_SINT8,       PT_RGBA_16  >();
    test_conversion<PT_SINT8,       PT_RGB_FP   >();
    test_conversion<PT_SINT8,       PT_COLOR    >();

    test_conversion<PT_FLOAT32,     PT_SINT8    >();
    test_conversion<PT_FLOAT32,     PT_FLOAT32  >();
    test_conversion<PT_FLOAT32,     PT_FLOAT32_2>();
    test_conversion<PT_FLOAT32,     PT_RGB      >();
    test_conversion<PT_FLOAT32,     PT_RGBA     >();
    test_conversion<PT_FLOAT32,     PT_RGBE     >();
    test_conversion<PT_FLOAT32,     PT_RGBEA    >();
    test_conversion<PT_FLOAT32,     PT_RGB_16   >();
    test_conversion<PT_FLOAT32,     PT_RGBA_16  >();
    test_conversion<PT_FLOAT32,     PT_RGB_FP   >();
    test_conversion<PT_FLOAT32,     PT_COLOR    >();

    test_conversion<PT_FLOAT32_2,   PT_SINT8    >();
    test_conversion<PT_FLOAT32_2,   PT_FLOAT32  >();
    test_conversion<PT_FLOAT32_2,   PT_FLOAT32_2>();
    test_conversion<PT_FLOAT32_2,   PT_RGB      >();
    test_conversion<PT_FLOAT32_2,   PT_RGBA     >();
    test_conversion<PT_FLOAT32_2,   PT_RGBE     >();
    test_conversion<PT_FLOAT32_2,   PT_RGBEA    >();
    test_conversion<PT_FLOAT32_2,   PT_RGB_16   >();
    test_conversion<PT_FLOAT32_2,   PT_RGBA_16  >();
    test_conversion<PT_FLOAT32_2,   PT_RGB_FP   >();
    test_conversion<PT_FLOAT32_2,   PT_COLOR    >();

    test_conversion<PT_RGB,       PT_SINT8    >();
    test_conversion<PT_RGB,       PT_FLOAT32  >();
    test_conversion<PT_RGB,       PT_FLOAT32_2>();
    test_conversion<PT_RGB,       PT_RGB      >();
    test_conversion<PT_RGB,       PT_RGBA     >();
    test_conversion<PT_RGB,       PT_RGBE     >();
    test_conversion<PT_RGB,       PT_RGBEA    >();
    test_conversion<PT_RGB,       PT_RGB_16   >();
    test_conversion<PT_RGB,       PT_RGBA_16  >();
    test_conversion<PT_RGB,       PT_RGB_FP   >();
    test_conversion<PT_RGB,       PT_COLOR    >();

    test_conversion<PT_RGBA,      PT_SINT8    >();
    test_conversion<PT_RGBA,      PT_FLOAT32  >();
    test_conversion<PT_RGBA,      PT_FLOAT32_2>();
    test_conversion<PT_RGBA,      PT_RGB      >();
    test_conversion<PT_RGBA,      PT_RGBA     >();
    test_conversion<PT_RGBA,      PT_RGBE     >();
    test_conversion<PT_RGBA,      PT_RGBEA    >();
    test_conversion<PT_RGBA,      PT_RGB_16   >();
    test_conversion<PT_RGBA,      PT_RGBA_16  >();
    test_conversion<PT_RGBA,      PT_RGB_FP   >();
    test_conversion<PT_RGBA,      PT_COLOR    >();

    test_conversion<PT_RGBE,      PT_SINT8    >();
    test_conversion<PT_RGBE,      PT_FLOAT32  >();
    test_conversion<PT_RGBE,      PT_FLOAT32_2>();
    test_conversion<PT_RGBE,      PT_RGB      >();
    test_conversion<PT_RGBE,      PT_RGBA     >();
    test_conversion<PT_RGBE,      PT_RGBE     >();
    test_conversion<PT_RGBE,      PT_RGBEA    >();
    test_conversion<PT_RGBE,      PT_RGB_16   >();
    test_conversion<PT_RGBE,      PT_RGBA_16  >();
    test_conversion<PT_RGBE,      PT_RGB_FP   >();
    test_conversion<PT_RGBE,      PT_COLOR    >();

    test_conversion<PT_RGBEA,     PT_SINT8    >();
    test_conversion<PT_RGBEA,     PT_FLOAT32  >();
    test_conversion<PT_RGBEA,     PT_FLOAT32_2>();
    test_conversion<PT_RGBEA,     PT_RGB      >();
    test_conversion<PT_RGBEA,     PT_RGBA     >();
    test_conversion<PT_RGBEA,     PT_RGBE     >();
    test_conversion<PT_RGBEA,     PT_RGBEA    >();
    test_conversion<PT_RGBEA,     PT_RGB_16   >();
    test_conversion<PT_RGBEA,     PT_RGBA_16  >();
    test_conversion<PT_RGBEA,     PT_RGB_FP   >();
    test_conversion<PT_RGBEA,     PT_COLOR    >();

    test_conversion<PT_RGB_16,    PT_SINT8    >();
    test_conversion<PT_RGB_16,    PT_FLOAT32  >();
    test_conversion<PT_RGB_16,    PT_FLOAT32_2>();
    test_conversion<PT_RGB_16,    PT_RGB      >();
    test_conversion<PT_RGB_16,    PT_RGBA     >();
    test_conversion<PT_RGB_16,    PT_RGBE     >();
    test_conversion<PT_RGB_16,    PT_RGBEA    >();
    test_conversion<PT_RGB_16,    PT_RGB_16   >();
    test_conversion<PT_RGB_16,    PT_RGBA_16  >();
    test_conversion<PT_RGB_16,    PT_RGB_FP   >();
    test_conversion<PT_RGB_16,    PT_COLOR    >();

    test_conversion<PT_RGBA_16,   PT_SINT8    >();
    test_conversion<PT_RGBA_16,   PT_FLOAT32  >();
    test_conversion<PT_RGBA_16,   PT_FLOAT32_2>();
    test_conversion<PT_RGBA_16,   PT_RGB      >();
    test_conversion<PT_RGBA_16,   PT_RGBA     >();
    test_conversion<PT_RGBA_16,   PT_RGBE     >();
    test_conversion<PT_RGBA_16,   PT_RGBEA    >();
    test_conversion<PT_RGBA_16,   PT_RGB_16   >();
    test_conversion<PT_RGBA_16,   PT_RGBA_16  >();
    test_conversion<PT_RGBA_16,   PT_RGB_FP   >();
    test_conversion<PT_RGBA_16,   PT_COLOR    >();

    test_conversion<PT_RGB_FP,    PT_SINT8    >();
    test_conversion<PT_RGB_FP,    PT_FLOAT32  >();
    test_conversion<PT_RGB_FP,    PT_FLOAT32_2>();
    test_conversion<PT_RGB_FP,    PT_RGB      >();
    test_conversion<PT_RGB_FP,    PT_RGBA     >();
    test_conversion<PT_RGB_FP,    PT_RGBE     >();
    test_conversion<PT_RGB_FP,    PT_RGBEA    >();
    test_conversion<PT_RGB_FP,    PT_RGB_16   >();
    test_conversion<PT_RGB_FP,    PT_RGBA_16  >();
    test_conversion<PT_RGB_FP,    PT_RGB_FP   >();
    test_conversion<PT_RGB_FP,    PT_COLOR    >();

    test_conversion<PT_COLOR,     PT_SINT8    >();
    test_conversion<PT_COLOR,     PT_FLOAT32  >();
    test_conversion<PT_COLOR,     PT_FLOAT32_2>();
    test_conversion<PT_COLOR,     PT_RGB      >();
    test_conversion<PT_COLOR,     PT_RGBA     >();
    test_conversion<PT_COLOR,     PT_RGBE     >();
    test_conversion<PT_COLOR,     PT_RGBEA    >();
    test_conversion<PT_COLOR,     PT_RGB_16   >();
    test_conversion<PT_COLOR,     PT_RGBA_16  >();
    test_conversion<PT_COLOR,     PT_RGB_FP   >();
    test_conversion<PT_COLOR,     PT_COLOR    >();
}


MI_TEST_AUTO_FUNCTION( test_pixel_conversion_copy )
{
    Module_holder mods;

    test_copy<PT_SINT8    >();
    test_copy<PT_FLOAT32  >();
    test_copy<PT_RGB      >();
    test_copy<PT_RGBA     >();
    test_copy<PT_RGBE     >();
    test_copy<PT_RGBEA    >();
    test_copy<PT_RGB_16   >();
    test_copy<PT_RGBA_16  >();
    test_copy<PT_RGB_FP   >();
    test_copy<PT_COLOR    >();
}


MI_TEST_AUTO_FUNCTION( test_pixel_conversion_gamma )
{
    Module_holder mods;

    //test_gamma<PT_SINT8    >(); // does not make sense as this converts to float, does the gamma,
    //test_gamma<PT_SINT32   >(); // then converts back to INTX which defeats the whole purpose of
                                  // the gamma operation
    test_gamma<PT_FLOAT32  >();
    test_gamma<PT_FLOAT32_2>();
    test_gamma<PT_FLOAT32_3>();
    test_gamma<PT_FLOAT32_4>();
    //test_gamma<PT_RGB      >(); // dto.
    //test_gamma<PT_RGBA     >(); // dto.
    test_gamma<PT_RGBE     >();
    test_gamma<PT_RGBEA    >();
    //test_gamma<PT_RGB_16   >(); // dto.
    //test_gamma<PT_RGBA_16  >(); // dto.
    test_gamma<PT_RGB_FP   >();
    test_gamma<PT_COLOR    >();
}
