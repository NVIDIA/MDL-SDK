/******************************************************************************
 * Copyright (c) 2019-2024, NVIDIA CORPORATION. All rights reserved.
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


#include "pch.h"
#include "i_image_pixel_conversion.h"

#define MI_TEST_AUTO_SUITE_NAME "color quantization test"

#include <base/system/test/i_test_auto_driver.h>

#include <vector>


namespace MI {
namespace IMAGE {

MI_TEST_AUTO_FUNCTION(test_quantization)
{
    // ensure that quantization preserves original values
    for (unsigned i=0; i<256; ++i) {
        MI_CHECK_EQUAL(i,quantize<unsigned char>(i/255.f));
    }
}

MI_TEST_AUTO_FUNCTION(test_sse_quantization3)
{
    // ensure that quantization preserves original values
    const unsigned components = 3;
    const unsigned count = 1000;
    std::vector<float> source(components*count);
    for (std::size_t i=0; i<source.size(); ++i) {
        source[i] = (i%256)/255.f;
    }

    std::vector<unsigned char> dest(source.size());
    Pixel_converter<PT_RGB_FP,PT_RGB>::convert(source.data(),dest.data(),count);

    for (std::size_t i=0; i<dest.size(); ++i) {
        MI_CHECK_EQUAL(i%256,(unsigned)dest[i]);
    }
}

MI_TEST_AUTO_FUNCTION(test_sse_quantization4)
{
    // ensure that quantization preserves original values
    const unsigned components = 4;
    const unsigned count = 1000;
    std::vector<float> source(components*count);
    for (std::size_t i=0; i<source.size(); ++i) {
        source[i] = (i%256)/255.f;
    }

    std::vector<unsigned char> dest(source.size());
    Pixel_converter<PT_COLOR,PT_RGBA>::convert(source.data(),dest.data(),count);

    for (std::size_t i=0; i<dest.size(); ++i) {
        MI_CHECK_EQUAL(i%256,(unsigned)dest[i]);
    }
}


}}
