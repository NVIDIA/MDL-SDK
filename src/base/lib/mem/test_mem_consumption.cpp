/******************************************************************************
 * Copyright (c) 2011-2025, NVIDIA CORPORATION. All rights reserved.
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

#include "pch.h"

#define MI_TEST_AUTO_SUITE_NAME "MI::MEM Regression Test Suite"

#include <base/system/test/i_test_auto_driver.h>

#include <mi/base/config.h>
#include <base/lib/mem/i_mem_consumption.h>
#include <unordered_map>

size_t round_up_to_8( size_t n)
{
    return (n + 7) / 8 * 8;
}

MI_TEST_AUTO_FUNCTION( test_mem_consumption)
{
#if defined(MI_PLATFORM_LINUX) || (defined (MI_PLATFORM_MACOSX) && defined(__GLIBCXX__))
    size_t size_node_base = sizeof(std::_Rb_tree_node_base);

    std::string string0( 0, 'x');
#if defined(MI_PLATFORM_LINUX)
    MI_CHECK_EQUAL( dynamic_memory_consumption( string0), 0);
    std::string string1( 1, 'x');
    MI_CHECK_EQUAL( dynamic_memory_consumption( string1), 0);
    std::string string15( 15, 'x');
    MI_CHECK_EQUAL( dynamic_memory_consumption( string15), 0);
    std::string string16( 16, 'x');
    MI_CHECK_EQUAL( dynamic_memory_consumption( string16), 16 + 1);
#else
    MI_CHECK_EQUAL( dynamic_memory_consumption( string0), 0 + 1);
    std::string string1( 1, 'x');
    MI_CHECK_EQUAL( dynamic_memory_consumption( string1), 1 + 1);
    std::string string15( 15, 'x');
    MI_CHECK_EQUAL( dynamic_memory_consumption( string15), 15 + 1);
    std::string string16( 16, 'x');
    MI_CHECK_EQUAL( dynamic_memory_consumption( string16), 16 + 1);
#endif

    std::pair<int,int> pair_int = std::make_pair( 42, 43);
    MI_CHECK_EQUAL( dynamic_memory_consumption( pair_int), 2*sizeof(int));

    std::vector<int> vector_int;
    vector_int.push_back( 42);
    MI_CHECK_EQUAL( dynamic_memory_consumption( vector_int), sizeof(int));
    vector_int.reserve( 10);
    MI_CHECK_EQUAL( dynamic_memory_consumption( vector_int), 10*sizeof(int));

    std::map<int,int> map_int;
    map_int[42] = 42;
    MI_CHECK_EQUAL( dynamic_memory_consumption( map_int), size_node_base + sizeof(std::pair<int,int>));
    map_int[43] = 43;
    MI_CHECK_EQUAL( dynamic_memory_consumption( map_int), 2*size_node_base + 2*sizeof(std::pair<int,int>));

    std::multimap<int,int> multimap_int;
    multimap_int.insert( std::make_pair( 42, 42));
    MI_CHECK_EQUAL( dynamic_memory_consumption( multimap_int), size_node_base + sizeof(std::pair<int,int>));
    multimap_int.insert( std::make_pair( 42, -42));
    MI_CHECK_EQUAL( dynamic_memory_consumption( multimap_int), 2*size_node_base + 2*sizeof(std::pair<int,int>));

    std::set<int> set_int;
    set_int.insert( 42);
    MI_CHECK_EQUAL( dynamic_memory_consumption( set_int), size_node_base + round_up_to_8(sizeof(int)));
    set_int.insert( 43);
    MI_CHECK_EQUAL( dynamic_memory_consumption( set_int), 2*size_node_base + 2*round_up_to_8(sizeof(int)));

    // TODO This is not correct.
    std::unordered_map<int,int> unordered_map_int;
    unordered_map_int[42] = 42;
#if defined(MI_PLATFORM_LINUX)
    MI_CHECK_EQUAL( dynamic_memory_consumption( unordered_map_int), 56);
    unordered_map_int[43] = 43;
    MI_CHECK_EQUAL( dynamic_memory_consumption( unordered_map_int), 56);
    for( int i = 0; i < 1000; ++i)
        unordered_map_int[i] = i;
    MI_CHECK_EQUAL( dynamic_memory_consumption( unordered_map_int), 56);
#else
    MI_CHECK_EQUAL( dynamic_memory_consumption( unordered_map_int), 64);
    unordered_map_int[43] = 43;
    MI_CHECK_EQUAL( dynamic_memory_consumption( unordered_map_int), 64);
#endif

    std::vector<std::string> vector_string;
    vector_string.push_back( string0);
#if defined(MI_PLATFORM_LINUX)
    MI_CHECK_EQUAL( dynamic_memory_consumption( vector_string), 1*sizeof(std::string));
    vector_string.push_back( string16);
    MI_CHECK_EQUAL( dynamic_memory_consumption( vector_string), 2*sizeof(std::string) + 16 + 1);
#else
    MI_CHECK_EQUAL( dynamic_memory_consumption( vector_string), 1*sizeof(std::string) + 0 + 1);
    vector_string.push_back( string16);
    MI_CHECK_EQUAL( dynamic_memory_consumption( vector_string), 2*sizeof(std::string) + 0 + 1 + 16 + 1);
#endif

#else
    // not implemented
#endif
}
