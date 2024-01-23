/******************************************************************************
 * Copyright (c) 2007-2024, NVIDIA CORPORATION. All rights reserved.
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
#include "i_test_auto_case.h"
#include <vector>

MI_TEST_AUTO_FUNCTION( check_two_equal_collections )
{
    std::vector<int>          vec1( 100000u );
    std::generate(vec1.begin(), vec1.end(), ::rand);
    std::vector<int> const    vec2( vec1.begin(), vec1.end() );
    MI_CHECK_EQUAL_COLLECTIONS(vec1.begin(), vec1.end(), vec2.begin(), vec2.end());
}


template <class Container>
class two_equal_collections_test : public MI::TEST::Named_test_case
{
    Container vec1, vec2;

public:
    two_equal_collections_test(size_t collection_size = 100000u)
        : MI::TEST::Named_test_case( std::string("check the equal collection macro")
                                   + " (using " + MI::TEST::show(collection_size) + " elements)"
                                   )
        , vec1(collection_size), vec2(collection_size)
    {
        std::generate(vec1.begin(), vec1.end(), ::rand);
        std::copy(vec1.begin(), vec1.end(), vec2.begin());
    }

    void run()
    {
        MI_CHECK_EQUAL_COLLECTIONS(vec1.begin(), vec1.end(), vec2.begin(), vec2.end());
    }
};

MI_TEST_AUTO_CASE( new two_equal_collections_test< std::vector<int> >(1000u) );
MI_TEST_AUTO_CASE( new two_equal_collections_test< std::vector<int> >(2000u) );
