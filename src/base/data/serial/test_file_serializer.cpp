/***************************************************************************************************
 * Copyright (c) 2008-2025, NVIDIA CORPORATION. All rights reserved.
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
/// \brief Test serialization to/from a file.

#include "pch.h"

#include <base/system/test/i_test_auto_case.h>
#include <base/hal/disk/disk.h>

#include "serial.h"
#include "i_serial_file_serializer.h"

#include "test_serializable.h"
#include "test_function.h"


using namespace MI;
using namespace MI::SERIAL;

/// test file serializer with Serializable_type
/// Serializable_type should be a serializable class.
template < class Serializable_type >
void test_file_serializer_template(Serializable_type const & serialize_src)
{
    //------------------------------------------------------------
    // test serializable object serializer
    //------------------------------------------------------------
#ifdef MI_TEST_VERBOSE
    std::cout << "test serializable object serializer ["
                << Serializable_type::get_classname()
                << "] to a file."   << std::endl;
#endif
    std::string const serialized_fname0 = "test_serial_0.data";
    {
        DISK::File serialized_file;
        serialized_file.open(serialized_fname0.c_str(), DISK::IFile::M_READWRITE);

        File_serializer fserializer;
        fserializer.set_output_file(&serialized_file);
        // this is only write the data, no Class ID
        serialize_src.serialize(&fserializer);
    }

    // deserialize data only
    Serializable_type dst0;
    {
        DISK::File serialized_file;
        serialized_file.open(serialized_fname0.c_str(), DISK::IFile::M_READ);
        File_deserializer fdeserializer(g_deserialization_manager);
        fdeserializer.set_input_file(&serialized_file);

        dst0.deserialize(&fdeserializer);
    }
    // check result for equality
    MI_REQUIRE_EQUAL(serialize_src, dst0);
#ifdef MI_TEST_VERBOSE
    std::cout << "serialize/deserialize data only Test_type_0 passed." << std::endl;
#endif


    //------------------------------------------------------------
    // test file serializer
    //------------------------------------------------------------
#ifdef MI_TEST_VERBOSE
    std::cout << "test serializer with serializable ["
                << Serializable_type::get_classname() << "] to file." << std::endl;
#endif
    std::string const serialized_fname1 = "test_serial_1.data";
    Serializable_type dst1;
    // serialize with pointer and class id
    {
        DISK::File serialized_file;
        serialized_file.open(serialized_fname1.c_str(), DISK::IFile::M_READWRITE);

        File_serializer fserializer;
        fserializer.set_output_file(&serialized_file);
        // this add the pointer to the top for check double serialization
        fserializer.serialize(&serialize_src);
    }

    // deserialize with pointer and class id
    {
        DISK::File serialized_file;
        serialized_file.open(serialized_fname1.c_str(), DISK::IFile::M_READ);
        File_deserializer fdeserializer(g_deserialization_manager);
        fdeserializer.set_input_file(&serialized_file);

        Serializable * p_new_dst = fdeserializer.deserialize_file();
        MI_CHECK(p_new_dst->get_class_id() == dst1.get_class_id());
        dst1 = *(static_cast< Serializable_type *>(p_new_dst));
        delete p_new_dst;
    }
#ifdef MI_TEST_VERBOSE
    std::cout << "dst1 = " << dst1 << std::endl;
#endif

    MI_REQUIRE_EQUAL(serialize_src, dst1);
#ifdef MI_TEST_VERBOSE
    std::cout << "serialize/deserialize all Test_type_0 passed." << std::endl;
#endif

    // remove temporarily files
    //     DISK::file_remove(serialized_fname0.c_str());
    //     DISK::file_remove(serialized_fname1.c_str());
}


/// test file serializer/deserializer
void test_file_serializer()
{
#ifdef MI_TEST_VERBOSE
    std::cout << "--- test_file_serializer Test_type_0 ---" << std::endl;
#endif
    {
        Test_type_0 src_0;
        src_0.m_int = 99;
        test_file_serializer_template< Test_type_0 >(src_0);
    }

#ifdef MI_TEST_VERBOSE
    std::cout << "--- test_file_serializer  Test_type_1 ---" << std::endl;
#endif
    {
        Test_type_1 src_1;
        src_1.m_int = 99;
        for (size_t i=0; i<9; ++i){
            src_1.m_array.push_back(i*10);
        }
        test_file_serializer_template< Test_type_1 >(src_1);
    }
}
