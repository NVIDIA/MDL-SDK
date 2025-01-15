/******************************************************************************
 * Copyright (c) 2006-2025, NVIDIA CORPORATION. All rights reserved.
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

 /// \file
 /// \brief Simple unit tests of Compressed_index's members.

#ifndef BASE_LIB_CONT_TEST_COMPRESSED_INDEX_H
#define BASE_LIB_CONT_TEST_COMPRESSED_INDEX_H

#include "i_cont_compressed_index.h"

#include <base/lib/log/i_log_logger.h>
#include <base/system/main/i_module_id.h>
#include <vector>

using namespace MI::CONT;
using MI::LOG::Mod_log;

//
// forward declarations of functions
//
bool test_compressed_index();
void test_compressed_push_back();

// The overall Compressed_index test function.
bool test_compressed_index()
{
    test_compressed_push_back();

    return true;
}

// Test the insertion of single items.
void test_compressed_push_back()
{
    Compressed_index<int> array( -7, 10000 );
    MI_REQUIRE(array.empty());
    MI_REQUIRE_EQUAL(array.size(), 0);

    array.push_back(-3);
    MI_REQUIRE(!array.empty());
    MI_REQUIRE_EQUAL(array.size(), 1);
    MI_REQUIRE_EQUAL(array[0], -3);
    array.push_back(2);
    MI_REQUIRE(!array.empty());
    MI_REQUIRE_EQUAL(array.size(), 2);
    MI_REQUIRE_EQUAL(array[1], 2);
    array.push_back(3);
    MI_REQUIRE(!array.empty());
    MI_REQUIRE_EQUAL(array.size(), 3);
    MI_REQUIRE_EQUAL(array[2], 3);
    array.push_back(4);
    MI_REQUIRE(!array.empty());
    MI_REQUIRE_EQUAL(array.size(), 4);
    MI_REQUIRE_EQUAL(array[3], 4);

    array.push_back(5);
    MI_REQUIRE(!array.empty());
    MI_REQUIRE_EQUAL(array.size(), 5);
    MI_REQUIRE_EQUAL(array[4], 5);
    array.push_back(5);
    MI_REQUIRE(!array.empty());
    MI_REQUIRE_EQUAL(array.size(), 6);
    MI_REQUIRE_EQUAL(array[5], 5);
    array.push_back(5);
    MI_REQUIRE(!array.empty());
    MI_REQUIRE_EQUAL(array.size(), 7);
    MI_REQUIRE_EQUAL(array[5], 5);

    for (int i = 7; i < 10000; i++)
    {
        array.push_back(i);
        MI_REQUIRE_EQUAL(array.size(), size_t(i+1));
        MI_REQUIRE_EQUAL(array[i], i);
    }

    array = Compressed_index<int>();
    for (int i = 1000; i < 5000; i++)
        array.push_back((i * 33555 + 21211) % 65536);

    std::vector<int> array_copy( array.size() );
    for (size_t i = 0; i < array.size(); i++)
        array_copy[i] = array[i];

    array.pack();
    array.shrink();

    for (size_t i = 0; i < array.size(); i++)
        MI_REQUIRE_EQUAL(array[i], array_copy[i] );

    array = Compressed_index<int>(array_copy.begin(), array_copy.end());

    for (size_t i = 0; i < array.size(); i++)
        MI_REQUIRE_EQUAL(array[i], array_copy[i] );

    array = Compressed_index<int>(array.get_min(), array.get_max(),
        array_copy.begin(), array_copy.end());

    for (size_t i = 0; i < array.size(); i++)
        MI_REQUIRE_EQUAL(array[i], array_copy[i] );

    array = Compressed_index<int>(0, 10, size_t(10), 9);

    MI_REQUIRE_EQUAL(array.size(), 10 );
    for (size_t i = 0; i < array.size(); i++)
        MI_REQUIRE_EQUAL(array[i], 9 );
}

#endif
