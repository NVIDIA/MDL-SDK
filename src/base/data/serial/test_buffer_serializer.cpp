/***************************************************************************************************
 * Copyright (c) 2008-2024, NVIDIA CORPORATION. All rights reserved.
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
/// \brief Test serialization.

#include "pch.h"

#include "test_function.h"

#include <base/system/test/i_test_auto_case.h>

#include "serial.h"

#include "i_serial_buffer_serializer.h"

#include "test_serializable.h"

using namespace MI;
using namespace MI::SERIAL;

/// test for Test_type_0
static void test_buffer_serializer_0()
{
    // setting up input and output
    Test_type_0 src;
    src.m_int = 99;


#ifdef MI_TEST_VERBOSE
    std::cout << "serialize data only test for Test_type0.";
#endif
    {
        Buffer_serializer ser;
        // src --(serialize)--> serializer (buffer)
        src.serialize(&ser);

        Test_type_0 dst;
        Buffer_deserializer deser;
        // This is a bit non symmetric. It is not like as the following.
        //
        // src.serialize  (&ser);    // correct
        // dst.deserialize(&deser);  // not in this way
        //
        // But as the following
        // deserializer --(deserialize)--> dest (from buffer)
        deser.deserialize(&dst, ser.get_buffer(), ser.get_buffer_size());

        // check result for equality
        MI_REQUIRE_EQUAL(src, dst);
    }
#ifdef MI_TEST_VERBOSE
    std::cout << "... passed" << std::endl;
#endif

#ifdef MI_TEST_VERBOSE
    std::cout << "serialize by datamanager for Test_type0.";
#endif
    {
        Buffer_serializer ser;
        // serializer --> serialize the src to serializer buffer
        ser.serialize(&src);

        Buffer_deserializer deser(g_deserialization_manager);
        size_t const serialized_buf_size = ser.get_buffer_size();
        auto data = ser.takeover_buffer();
        deser.reset(data.data(), serialized_buf_size);

        // The following seems symmetric and compiled,
        //   deser.deserialize(&dst);
        // but this is Deserializer_impl::deserialize(bool)!
        // a pointer is implicitly converted to bool.

        Test_type_0 dst;
        Serializable* p_newed_dst = deser.deserialize();
        MI_CHECK(p_newed_dst->get_class_id() == ID_TEST_TYPE_0_CLASS_ID);
        // dst = *(dynamic_cast<Test_type_0 *>(p_newed_dst)); // this is correct.
        dst = *(static_cast<Test_type_0 *>(p_newed_dst)); // default copy constructor
        delete p_newed_dst;

        // check result for equality
        MI_REQUIRE_EQUAL(src, dst);
    }
#ifdef MI_TEST_VERBOSE
    std::cout << "... passed" << std::endl;
#endif
}


/// test for Test_type_1
static void test_buffer_serializer_1()
{
    // setting up input and output
    Test_type_1 src;
    src.m_int = 99;
    for (size_t i=0; i<9; ++i){
        src.m_array.push_back(i*10);
    }

#ifdef MI_TEST_VERBOSE
    std::cout << "serialize data only test for Test_type1.";
#endif
    {
        Buffer_serializer ser;
        // src --(serialize)--> serializer (buffer)
        src.serialize(&ser);

        Test_type_1 dst;
        Buffer_deserializer deser;
        // This is a bit non symmetric. It is not like as the following.
        //
        // src.serialize  (&ser);    // correct
        // dst.deserialize(&deser);  // not in this way
        //
        // But as the following
        // deserializer --(deserialize)--> dest (from buffer)
        deser.deserialize(&dst, ser.get_buffer(), ser.get_buffer_size());

        // check result for equality
        MI_REQUIRE_EQUAL(src, dst);
    }
#ifdef MI_TEST_VERBOSE
    std::cout << "... passed" << std::endl;
#endif

#ifdef MI_TEST_VERBOSE
    std::cout << "serialize by datamanager for Test_type1.";
#endif
    {
        Buffer_serializer ser;
        // serializer --> serialize the src to serializer buffer
        ser.serialize(&src);

        Buffer_deserializer deser(g_deserialization_manager);
        size_t const serialized_buf_size = ser.get_buffer_size();
        deser.reset(ser.get_buffer(), serialized_buf_size);

        Test_type_1 dst;
        Serializable* p_newed_dst = deser.deserialize();
        MI_CHECK(p_newed_dst->get_class_id() == ID_TEST_TYPE_1_CLASS_ID);
        // dst = *(dynamic_cast<Test_type_0 *>(p_newed_dst)); // this is correct.
        dst = *(static_cast<Test_type_1 *>(p_newed_dst)); // default copy constructor
        delete p_newed_dst;

        // check result for equality
        MI_REQUIRE_EQUAL(src, dst);
    }
#ifdef MI_TEST_VERBOSE
    std::cout << "... passed" << std::endl;
#endif
}

/// the test
void test_buffer_serializer()
{
#ifdef MI_TEST_VERBOSE
    std::cout << "--- test_buffer_serializer_0 ---" << std::endl;
#endif
    test_buffer_serializer_0();
#ifdef MI_TEST_VERBOSE
    std::cout << "--- test_buffer_serializer_1 ---" << std::endl;
#endif
    test_buffer_serializer_1();
}
