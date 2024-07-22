/******************************************************************************
 * Copyright (c) 2010-2024, NVIDIA CORPORATION. All rights reserved.
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
/// \brief regression test public/mi/base components
///
/// See \ref mi_base_iinterface
///

#include "pch.h"
#include <base/system/test/i_test_auto_case.h>

#include <mi/base/default_allocator.h>
#include <mi/base/std_allocator.h>

#include <list>
#include <vector>
#include <map>

MI_TEST_AUTO_FUNCTION( test_allocator )
{
    mi::base::IAllocator* alloc = mi::base::Default_allocator::get_instance();
    MI_CHECK_EQUAL( 1, alloc->retain());
    MI_CHECK_EQUAL( 1, alloc->release());
    void* p = alloc->malloc( 1024);
    MI_CHECK( p);
    alloc->free( p);
}

MI_TEST_AUTO_FUNCTION( test_std_allocator )
{
    mi::base::IAllocator* alloc = mi::base::Default_allocator::get_instance();

    using Std_int_alloc = mi::base::Std_allocator<int>;
    Std_int_alloc std_int_alloc( alloc);

    std::list< int, Std_int_alloc> ls( std_int_alloc);
    ls.push_back( 5);
    MI_CHECK_EQUAL( 5, ls.front());

    std::vector< int, Std_int_alloc> vs( std_int_alloc);
    vs.push_back( 5);
    MI_CHECK_EQUAL( 5, vs.front());

    using Std_pair_int_int_alloc = mi::base::Std_allocator<std::pair<const int, int>>;
    Std_pair_int_int_alloc std_pair_int_int_alloc( alloc);

    std::map< int, int, std::less<>, Std_pair_int_int_alloc> ms( std_pair_int_int_alloc);
    ms[42] = 5;
    MI_CHECK_EQUAL( 5, ms[42]);
}
