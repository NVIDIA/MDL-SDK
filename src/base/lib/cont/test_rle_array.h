/******************************************************************************
 * Copyright (c) 2004-2024, NVIDIA CORPORATION. All rights reserved.
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
 /// \brief Test dynamic arrays.

#ifndef BASE_LIB_CONT_TEST_ARRAY_H
#define BASE_LIB_CONT_TEST_ARRAY_H

#include "i_cont_rle_array.h"

#include <base/lib/log/i_log_logger.h>
#include <base/system/main/i_module_id.h>

using namespace MI::CONT;
using MI::LOG::Mod_log;

//
// Pretty-print test results
//
namespace std {

template <class T, class CONT, class IND>
inline std::ostream & operator<< (std::ostream & os, Rle_iterator<T, CONT, IND> const & i)
{
    return os << "[Rle_iterator]";
}

template <class T, class CONT, class IND>
inline std::ostream & operator<< (std::ostream & os, Rle_chunk_iterator<T, CONT, IND> const & i)
{
    return os << "[Rle_chunk_iterator]";
}

}

//
// forward declarations of functions
//
bool test_rle_array();
void test_push_back();
Rle_array<int> test_push_back_n();
void test_iterators(
    const Rle_array<int>& array);       // array to operate on
void test_chunk_iterators(
    const Rle_array<int>& array);       // array to operate on
void test_accessors(
    const Rle_array<int>& array);       // array to operate on
void test_removal(
    Rle_array<int>& array);             // array to operate on

// The overall Rle_array test function.
bool test_rle_array()
{
#ifdef MI_TEST_VERBOSE
    mod_log->progress(M_CONT, Mod_log::C_MEMORY, 100, "-- Insertion --");
#endif
    test_push_back();
    Rle_array<int> array = test_push_back_n();
#ifdef MI_TEST_VERBOSE
    mod_log->progress(M_CONT, Mod_log::C_MEMORY, 100, "-- Iterators --");
#endif
    test_iterators(array);
#ifdef MI_TEST_VERBOSE
    mod_log->progress(M_CONT, Mod_log::C_MEMORY, 100, "-- Chunk Iterators --");
#endif
    test_chunk_iterators(array);
#ifdef MI_TEST_VERBOSE
    mod_log->progress(M_CONT, Mod_log::C_MEMORY, 100, "-- Accessors --");
#endif
    test_accessors(array);
#ifdef MI_TEST_VERBOSE
    mod_log->progress(M_CONT, Mod_log::C_MEMORY, 100, "-- Removals --");
#endif
    test_removal(array);

    return true;
}

// Test the insertion of single items.
void test_push_back()
{
    Rle_array<int> array;
    MI_REQUIRE(array.empty());
    MI_REQUIRE_EQUAL(array.size(), 0);
    MI_REQUIRE_EQUAL(array.get_index_size(), 0);

    array.push_back(1);
    MI_REQUIRE(!array.empty());
    MI_REQUIRE_EQUAL(array.size(), 1);
    MI_REQUIRE_EQUAL(array.get_index_size(), 1);
    array.push_back(1);
    MI_REQUIRE(!array.empty());
    MI_REQUIRE_EQUAL(array.size(), 2);
    MI_REQUIRE_EQUAL(array.get_index_size(), 1);
    array.push_back(1);
    MI_REQUIRE(!array.empty());
    MI_REQUIRE_EQUAL(array.size(), 3);
    MI_REQUIRE_EQUAL(array.get_index_size(), 1);
    array.push_back(1);
    MI_REQUIRE(!array.empty());
    MI_REQUIRE_EQUAL(array.size(), 4);
    MI_REQUIRE_EQUAL(array.get_index_size(), 1);

    array.push_back(3);
    MI_REQUIRE(!array.empty());
    MI_REQUIRE_EQUAL(array.size(), 5);
    MI_REQUIRE_EQUAL(array.get_index_size(), 2);
    array.push_back(3);
    MI_REQUIRE(!array.empty());
    MI_REQUIRE_EQUAL(array.size(), 6);
    MI_REQUIRE_EQUAL(array.get_index_size(), 2);
    array.push_back(3);
    MI_REQUIRE(!array.empty());
    MI_REQUIRE_EQUAL(array.size(), 7);
    MI_REQUIRE_EQUAL(array.get_index_size(), 2);

    array.push_back(0);
    MI_REQUIRE(!array.empty());
    MI_REQUIRE_EQUAL(array.size(), 8);
    MI_REQUIRE_EQUAL(array.get_index_size(), 3);

    array.push_back(1);
    MI_REQUIRE(!array.empty());
    MI_REQUIRE_EQUAL(array.size(), 9);
    MI_REQUIRE_EQUAL(array.get_index_size(), 4);

    array.push_back(2);
    MI_REQUIRE(!array.empty());
    MI_REQUIRE_EQUAL(array.size(), 10);
    MI_REQUIRE_EQUAL(array.get_index_size(), 5);
    array.push_back(2);
    MI_REQUIRE(!array.empty());
    MI_REQUIRE_EQUAL(array.size(), 11);
    MI_REQUIRE_EQUAL(array.get_index_size(), 5);
}

// Test the insertion of multiple items.
Rle_array<int> test_push_back_n()
{
    Rle_array<int> array;
    MI_REQUIRE(array.empty());
    MI_REQUIRE_EQUAL(array.size(), 0);
    MI_REQUIRE_EQUAL(array.get_index_size(), 0);

    array.push_back(1, 4);
    MI_REQUIRE(!array.empty());
    MI_REQUIRE_EQUAL(array.size(), 4);
    MI_REQUIRE_EQUAL(array.get_index_size(), 1);

    array.push_back(3);
    MI_REQUIRE(!array.empty());
    MI_REQUIRE_EQUAL(array.size(), 5);
    MI_REQUIRE_EQUAL(array.get_index_size(), 2);
    array.push_back(3, 2);
    MI_REQUIRE(!array.empty());
    MI_REQUIRE_EQUAL(array.size(), 7);
    MI_REQUIRE_EQUAL(array.get_index_size(), 2);

    array.push_back(0);
    MI_REQUIRE(!array.empty());
    MI_REQUIRE_EQUAL(array.size(), 8);
    MI_REQUIRE_EQUAL(array.get_index_size(), 3);

    array.push_back(1);
    MI_REQUIRE(!array.empty());
    MI_REQUIRE_EQUAL(array.size(), 9);
    MI_REQUIRE_EQUAL(array.get_index_size(), 4);

    array.push_back(2, 2);
    MI_REQUIRE(!array.empty());
    MI_REQUIRE_EQUAL(array.size(), 11);
    MI_REQUIRE_EQUAL(array.get_index_size(), 5);

    return array;
}

// Test the iterators.
void test_iterators(
    const Rle_array<int>& array)        //  array to iterate on
{
    Rle_iterator<int> iter = array.begin();
    MI_REQUIRE_EQUAL(*iter, 1);

    for (int i=0; i<4; ++i, ++iter)
        MI_REQUIRE_EQUAL(*iter, 1);
    for (int i=0; i<3; ++i, ++iter)
        MI_REQUIRE_EQUAL(*iter, 3);
    MI_REQUIRE_EQUAL(*iter, 0);
    ++iter;
    MI_REQUIRE_EQUAL(*iter, 1);
    ++iter;
    MI_REQUIRE_EQUAL(*iter, 2);
    ++iter;
    MI_REQUIRE_EQUAL(*iter, 2);

    MI_REQUIRE(iter != array.end());
    ++iter;
    MI_REQUIRE_EQUAL(iter, array.end());

    // test assignment
    Rle_iterator<int> tmp_01;
    tmp_01 = iter;
    MI_REQUIRE_EQUAL(tmp_01, array.end());

    Rle_iterator<int> tmp_02(iter);
    MI_REQUIRE_EQUAL(tmp_02, array.end());
}

// Test the iterators.
void test_chunk_iterators(
    const Rle_array<int>& array)        // array to operate on
{
    Rle_chunk_iterator<int> iter = array.begin_chunk();

    MI_REQUIRE_EQUAL(iter->count(), 4);
    MI_REQUIRE_EQUAL((*iter).count(), 4);
    MI_REQUIRE_EQUAL(iter->data(), 1);

    ++iter;
    MI_REQUIRE(iter != array.end_chunk());
    MI_REQUIRE_EQUAL(iter.count(), 3);
    MI_REQUIRE_EQUAL((*iter).count(), 3);
    MI_REQUIRE_EQUAL(iter->data(), 3);

    ++iter;
    MI_REQUIRE(iter != array.end_chunk());
    MI_REQUIRE_EQUAL(iter->count(), 1);
    MI_REQUIRE_EQUAL(iter->data(), 0);

    ++iter;
    MI_REQUIRE(iter != array.end_chunk());
    MI_REQUIRE_EQUAL(iter->count(), 1);
    MI_REQUIRE_EQUAL(iter->data(), 1);

    ++iter;
    MI_REQUIRE(iter != array.end_chunk());
    MI_REQUIRE_EQUAL(iter->count(), 2);
    MI_REQUIRE_EQUAL(iter->data(), 2);

    ++iter;
    MI_REQUIRE_EQUAL(iter, array.end_chunk());
}


// Test the accessors.
void test_accessors(
    const Rle_array<int>& array)        // array to operate on
{
    size_t i = 0;
    for (; i<4; ++i)
        MI_REQUIRE_EQUAL(array[i], 1);
    for (; i<7; ++i)
        MI_REQUIRE_EQUAL(array[i], 3);
    MI_REQUIRE_EQUAL(array[i++], 0);
    MI_REQUIRE_EQUAL(array[i++], 1);
    MI_REQUIRE_EQUAL(array[i++], 2);
    MI_REQUIRE_EQUAL(array[i++], 2);

    MI_REQUIRE_EQUAL(i, array.size());
}

// Test the data removal.
void test_removal(
    Rle_array<int>& array)              // array to operate on
{
    MI_REQUIRE(!array.empty());
    MI_REQUIRE_EQUAL(array.size(), 11);
    MI_REQUIRE_EQUAL(array.get_index_size(), 5);

    array.pop_back();
    MI_REQUIRE(!array.empty());
    MI_REQUIRE_EQUAL(array.size(), 10);
    MI_REQUIRE_EQUAL(array.get_index_size(), 5);

    array.pop_back();
    MI_REQUIRE(!array.empty());
    MI_REQUIRE_EQUAL(array.size(), 9);
    MI_REQUIRE_EQUAL(array.get_index_size(), 4);

    array.pop_back();
    MI_REQUIRE(!array.empty());
    MI_REQUIRE_EQUAL(array.size(), 8);
    MI_REQUIRE_EQUAL(array.get_index_size(), 3);

    array.pop_back();
    MI_REQUIRE(!array.empty());
    MI_REQUIRE_EQUAL(array.size(), 7);
    MI_REQUIRE_EQUAL(array.get_index_size(), 2);

    array.pop_back();
    MI_REQUIRE(!array.empty());
    MI_REQUIRE_EQUAL(array.size(), 6);
    MI_REQUIRE_EQUAL(array.get_index_size(), 2);
    array.pop_back();
    MI_REQUIRE(!array.empty());
    MI_REQUIRE_EQUAL(array.size(), 5);
    MI_REQUIRE_EQUAL(array.get_index_size(), 2);
    array.pop_back();
    MI_REQUIRE(!array.empty());
    MI_REQUIRE_EQUAL(array.size(), 4);
    MI_REQUIRE_EQUAL(array.get_index_size(), 1);

    array.pop_back();
    MI_REQUIRE(!array.empty());
    MI_REQUIRE_EQUAL(array.size(), 3);
    MI_REQUIRE_EQUAL(array.get_index_size(), 1);
    array.pop_back();
    MI_REQUIRE(!array.empty());
    MI_REQUIRE_EQUAL(array.size(), 2);
    MI_REQUIRE_EQUAL(array.get_index_size(), 1);
    array.pop_back();
    MI_REQUIRE(!array.empty());
    MI_REQUIRE_EQUAL(array.size(), 1);
    MI_REQUIRE_EQUAL(array.get_index_size(), 1);
    array.pop_back();
    MI_REQUIRE(array.empty());
    MI_REQUIRE_EQUAL(array.size(), 0);
    MI_REQUIRE_EQUAL(array.get_index_size(), 0);
}

#endif
