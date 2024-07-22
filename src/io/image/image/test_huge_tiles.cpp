/******************************************************************************
 * Copyright (c) 2016-2024, NVIDIA CORPORATION. All rights reserved.
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

#include <mi/base/handle.h>
#include <mi/neuraylib/itile.h>

#include <base/system/main/access_module.h>
#include <base/lib/mem/mem.h>
#include <base/lib/log/i_log_module.h>

using namespace MI;

void fill_data( void* data, size_t n)
{
   ASSERT( M_IMAGE, n % sizeof( size_t) == 0);

   n /= sizeof( size_t);
   auto* p = static_cast<size_t*>( data);

   for( size_t i = 0; i < n; ++i)
       p[i] = i;
}

void verify_data( void* data, size_t n)
{
   ASSERT( M_IMAGE, n % sizeof( size_t) == 0);

   n /= sizeof( size_t);
   auto* p = static_cast<size_t*>( data);

   for( size_t i = 0; i < n; ++i)
       MI_CHECK_EQUAL( p[i], i);
}

MI_TEST_AUTO_FUNCTION( test_huge_tiles )
{
    SYSTEM::Access_module<MEM::Mem_module> mem_module( false);
    SYSTEM::Access_module<LOG::Log_module> log_module( false);

    SYSTEM::Access_module<IMAGE::Image_module> image_module( false);

    mi::Uint32 width  = 1u << 17;
    mi::Uint32 height = 1u << 16;
    IMAGE::Pixel_type pt = IMAGE::PT_SINT8;
    mi::Uint32 bpp = IMAGE::get_bytes_per_pixel( pt);
    mi::Size size = static_cast<mi::Size>( width) * static_cast<mi::Size>( height) * bpp;
    ASSERT( M_IMAGE, size >  1ull << 32); // 4 GB
    ASSERT( M_IMAGE, size == 1ull << 33); // 8 GB

    mi::base::Handle<mi::neuraylib::ITile> tile( image_module->create_tile( pt, width, height));
    MI_CHECK( tile);

    void* data = tile->get_data();
    fill_data( data, size);
    verify_data( data, size);

    // Other tests (copy, serialization, pixel access, pixel type conversion) could be added here,
    // but they need huge amounts of memory.
}

MI_TEST_MAIN_CALLING_TEST_MAIN();
