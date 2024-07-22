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
/// \brief some serializable classes for test.

#ifndef BASE_DATA_SERIAL_TEST_SERIALIZABLE_H
#define BASE_DATA_SERIAL_TEST_SERIALIZABLE_H

#include "serial.h"

#include <iterator>
#include <sstream>

/// class ID: for Test_type_0
static MI::SERIAL::Class_id const ID_TEST_TYPE_0_CLASS_ID = 2;
/// class ID: for Test_type_1
static MI::SERIAL::Class_id const ID_TEST_TYPE_1_CLASS_ID = 3;
/// Class ID: for Test_type_2
static MI::SERIAL::Class_id const ID_TEST_TYPE_2_CLASS_ID = 4;
/// Class ID: for Test_type_3
static MI::SERIAL::Class_id const ID_TEST_TYPE_3_CLASS_ID = 5;
/// Class ID: for Test_type_4
static MI::SERIAL::Class_id const ID_TEST_TYPE_4_CLASS_ID = 6;

/// Test_type_0 serializer/deserializer test, simplest. only one int
/// member.  class where new members can easily be added to.
class Test_type_0 : public MI::SERIAL::Serializable
{
public:
    /// constructor
    Test_type_0() : m_int(7)
    {
        // empty
    }
    /// destructor
    virtual ~Test_type_0()
    {
        // empty
    }
    /// get class name for test
    static std::string get_classname() {
        return std::string("Test_type_0");
    }

public:
    // implement serializable
    /// get class id
    MI::SERIAL::Class_id get_class_id() const
    {
        return ID_TEST_TYPE_0_CLASS_ID;
    }
    /// serialize method
    const MI::SERIAL::Serializable* serialize(
        MI::SERIAL::Serializer* serializer) const; // serialize to this serializer
    /// deserialize method
    MI::SERIAL::Serializable* deserialize(
        MI::SERIAL::Deserializer* deserializer); // deserialize from here
public:
    // Deliberately left public st the test can configure this class easily.
    /// int value
    MI::Sint32 m_int;
};

/// Test_type_1 serializer/deserializer test, a bit complex. one int
/// and vector<int>.  where new members can easily be added to.
class Test_type_1 : public MI::SERIAL::Serializable
{
public:
    Test_type_1()
    {
        // empty
    }
    virtual ~Test_type_1()
    {
        // empty
    }
    /// get class name for test
    static std::string get_classname() {
        return std::string("Test_type_1");
    }

public:
    // implement serializable
    /// get class id
    MI::SERIAL::Class_id get_class_id() const
    {
        return ID_TEST_TYPE_1_CLASS_ID;
    }
    /// serialize method
    const MI::SERIAL::Serializable* serialize(
        MI::SERIAL::Serializer* serializer) const; // serialize to this serializer
    /// deserialize method
    MI::SERIAL::Serializable* deserialize(
        MI::SERIAL::Deserializer* deserializer); // deserialize from here
public:
    // Deliberately left public st the test can configure this class easily.
    /// int value
    int m_int;
    /// int array
    std::vector<int> m_array;
    /// string
    std::string m_string;
};

/// Test_type_2 serializer/deserializer test. Based on Test_type_0, with an
/// (optional) extension marker added.
class Test_type_2 : public MI::SERIAL::Serializable
{
public:
    /// constructor
    Test_type_2(bool use_ext = true)
        : m_int(8),
          m_ext(10),
          m_use_ext(use_ext),
          m_ext_found(false)
    {}

    /// destructor
    virtual ~Test_type_2()
    {}

    /// get class name for test
    static std::string get_classname() {
        return std::string("Test_type_2");
    }

public:
    // implement serializable
    /// get class id
    MI::SERIAL::Class_id get_class_id() const
    {
        return ID_TEST_TYPE_2_CLASS_ID;
    }
    /// serialize method
    const MI::SERIAL::Serializable* serialize(
    MI::SERIAL::Serializer* serializer) const; // serialize to this serializer
    /// deserialize method
    MI::SERIAL::Serializable* deserialize(
    MI::SERIAL::Deserializer* deserializer); // deserialize from here
public:
    // Deliberately left public st the test can configure this class easily.
    MI::Sint32 m_int;
    MI::Sint32 m_ext;
    bool m_use_ext;
    bool m_ext_found;
};

/// Test_type_3 embeds Test_type_2.
class Test_type_3 : public MI::SERIAL::Serializable
{
public:
    /// constructor
    Test_type_3(Test_type_2 *tt2 = NULL)
        : m_ptr(tt2)
    {}

    /// destructor
    virtual ~Test_type_3()
    {}

    /// get class name for test
    static std::string get_classname() {
        return std::string("Test_type_3");
    }

public:
    // implement serializable
    /// get class id
    MI::SERIAL::Class_id get_class_id() const
    {
        return ID_TEST_TYPE_3_CLASS_ID;
    }
    /// serialize method
    const MI::SERIAL::Serializable* serialize(
    MI::SERIAL::Serializer* serializer) const; // serialize to this serializer
    /// deserialize method
    MI::SERIAL::Serializable* deserialize(
    MI::SERIAL::Deserializer* deserializer); // deserialize from here

public:
    MI::Sint32 m_data;
    Test_type_2 *m_ptr;
};

/// Test_type_4 recursively serializes/deserializes based the number of given
/// iterations.
class Test_type_4 : public MI::SERIAL::Serializable
{
public:
    /// constructor
    Test_type_4(mi::Sint32 iterations = 0)
        : m_iteration(iterations)
    {}

    /// destructor
    virtual ~Test_type_4()
    {}

    /// get class name for test
    static std::string get_classname() {
        return std::string("Test_type_4");
    }

public:
    // implement serializable
    /// get class id
    MI::SERIAL::Class_id get_class_id() const
    {
        return ID_TEST_TYPE_4_CLASS_ID;
    }
    /// serialize method
    const MI::SERIAL::Serializable* serialize(
    MI::SERIAL::Serializer* serializer) const; // serialize to this serializer
    /// deserialize method
    MI::SERIAL::Serializable* deserialize(
    MI::SERIAL::Deserializer* deserializer); // deserialize from here

public:
    mutable MI::Sint32 m_iteration;
};

/// Factory for the Test_type 0
extern MI::SERIAL::Serializable* test_type_0_factory();
/// Factory for the Test_type 1
extern MI::SERIAL::Serializable* test_type_1_factory();
/// Factory for the Test_type 2
extern MI::SERIAL::Serializable* test_type_2_factory();
extern MI::SERIAL::Serializable* test_type_2_no_ext_factory();
/// Factory for the Test_type 3
extern MI::SERIAL::Serializable* test_type_3_factory();
/// Factory for the Test_type 4
extern MI::SERIAL::Serializable* test_type_4_factory();

/// Comparison of Test_types_0 for equality.
extern bool operator==(
    const Test_type_0& lhs,                             // lhs
    const Test_type_0& rhs);                            // rhs

/// Comparison of Test_types_1 for equality.
extern bool operator==(
    const Test_type_1& lhs,                             // lhs
    const Test_type_1& rhs);                            // rhs

/// Comparison of Test_types_2 for equality.
extern bool operator==(
    const Test_type_2& lhs,                             // lhs
    const Test_type_2& rhs);                            // rhs

/// Comparison of Test_types_3 for equality.
extern bool operator==(
    const Test_type_3& lhs,                             // lhs
    const Test_type_3& rhs);                            // rhs

/// Pretty-print Test_type_0; needed when MI_REQUIRE_EQUAL() fails.
inline std::ostream & operator<< (std::ostream & os, Test_type_0 const & tt)
{
    os << "m_int = " << tt.m_int;
    return os;
}

/// Pretty-print Test_type; needed when MI_REQUIRE_EQUAL() fails.
inline std::ostream & operator<< (std::ostream & os, Test_type_1 const & tt)
{
    os << "m_int = " << tt.m_int << "; m_array = { ";
    std::copy(tt.m_array.begin(), tt.m_array.end(), std::ostream_iterator<int>(os, ", "));
    os << " }; m_string = \"" << tt.m_string.c_str() << "\"";
    return os;
}

/// Pretty-print Test_type_2; needed when MI_REQUIRE_EQUAL() fails.
inline std::ostream & operator<< (std::ostream & os, Test_type_2 const & tt)
{
    os << "m_int = " << tt.m_int << ", m_ext = " << tt.m_ext;
    return os;
}

/// Pretty-print Test_type_3; needed when MI_REQUIRE_EQUAL() fails.
inline std::ostream & operator<< (std::ostream & os, Test_type_3 const & tt)
{
    os << "m_data = " << tt.m_data << ", m_ptr = " << tt.m_ptr;
    if (tt.m_ptr)
        os << " [" << *tt.m_ptr << "]";
    return os;
}

/// Pretty-print Test_type_4; needed when MI_REQUIRE_EQUAL() fails.
inline std::ostream & operator<< (std::ostream & os, Test_type_4 const & tt)
{
    os << "m_iteration = " << tt.m_iteration;
    return os;
}

#endif // #ifndef BASE_DATA_SERIAL_TEST_SERIALIZABLE_H
