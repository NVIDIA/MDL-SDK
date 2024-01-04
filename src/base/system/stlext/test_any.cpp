/***************************************************************************************************
 * Copyright (c) 2008-2023, NVIDIA CORPORATION. All rights reserved.
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
 **************************************************************************************************/
/// \file
/// \brief Unit tests of the Any class.

#include "pch.h"

#include "i_stlext_any.h"

#include <list>
#include <string>

#include <base/system/test/i_test_auto_case.h>

MI_TEST_AUTO_FUNCTION( test_any )
{
    typedef std::list<MI::STLEXT::Any> Many;
    Many myList;
    // fill the list
    myList.push_back(12);
    myList.push_back(std::string("some_string"));
    MI::STLEXT::Any aFloat = 14.5f;
    myList.push_back(aFloat);
    myList.push_back(MI::STLEXT::Any());

    // iterate through it and check
    std::list<MI::STLEXT::Any>::const_iterator it=myList.begin();
    MI_CHECK(it->type() == typeid(int));
    MI_CHECK_EQUAL(*MI::STLEXT::any_cast<int>(&(*it)), 12);
    // check that an incorrect type access fails
    MI_CHECK_EQUAL(MI::STLEXT::any_cast<bool>(&(*it)), 0);
    ++it;
    MI_CHECK(it->type() == typeid(std::string));
    MI_CHECK_EQUAL((MI::STLEXT::any_cast<std::string>(&(*it)))->c_str(), std::string("some_string"));
    ++it;
    MI_CHECK(it->type() == typeid(float));
    MI_CHECK(*MI::STLEXT::any_cast<float>(&(*it)) > 14.5-0.0001
        && *MI::STLEXT::any_cast<float>(&(*it)) < 14.5+0.0001);
    ++it;
    MI_CHECK(it->type() == typeid(void));
    MI_CHECK(it->empty());
}
