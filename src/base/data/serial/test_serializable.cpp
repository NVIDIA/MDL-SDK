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
/// \brief some serializable classes for test.

#include "pch.h"

#include "test_serializable.h"

using namespace MI::SERIAL;

//----------------------------------------------------------------------
// Factory for the Test_type_0
MI::SERIAL::Serializable* test_type_0_factory()
{
    return new Test_type_0;
}

//----------------------------------------------------------------------
// Factory for the Test_type_1
MI::SERIAL::Serializable* test_type_1_factory()
{
    return new Test_type_1;
}

//----------------------------------------------------------------------
// Factory for the Test_type_2
MI::SERIAL::Serializable* test_type_2_factory()
{
    return new Test_type_2;
}

MI::SERIAL::Serializable* test_type_2_no_ext_factory()
{
    return new Test_type_2(false);
}

//----------------------------------------------------------------------
// Factory for the Test_type_3
MI::SERIAL::Serializable* test_type_3_factory()
{
    return new Test_type_3;
}
//----------------------------------------------------------------------
// Factory for the Test_type_4
MI::SERIAL::Serializable* test_type_4_factory()
{
    return new Test_type_4;
}

//----------------------------------------------------------------------
// Comparison of Test_type_0 for equality.
bool operator==(
    const Test_type_0 & lhs,                            // lhs
    const Test_type_0 & rhs)                            // rhs
{
    return lhs.m_int == rhs.m_int;
}

//----------------------------------------------------------------------
// Comparison of Test_type_1 for equality.
bool operator==(
    const Test_type_1 & lhs,                            // lhs
    const Test_type_1 & rhs)                            // rhs
{
    if (lhs.m_int != rhs.m_int)
        return false;
    if (lhs.m_array.size() != rhs.m_array.size())
        return false;
    std::vector<int>::const_iterator it_lhs, it_rhs;
    std::vector<int>::const_iterator end_lhs=lhs.m_array.end(), end_rhs=rhs.m_array.end();
    for (
        it_lhs=lhs.m_array.begin(),it_rhs=rhs.m_array.begin();
        (it_lhs != end_lhs) && (it_rhs != end_rhs);
        ++it_lhs, ++it_rhs)
    {
        if (*it_lhs != *it_rhs)
            return false;
    }
    if (lhs.m_string != rhs.m_string)
        return false;

    return true;
}

//----------------------------------------------------------------------
// Comparison of Test_type_2 for equality.
bool operator==(
    const Test_type_2 & lhs,                            // lhs
    const Test_type_2 & rhs)                            // rhs
{
    // Compare m_int and m_ext (if both objects have the extension set)
    return (lhs.m_int == rhs.m_int) &&
        (!(lhs.m_use_ext && rhs.m_use_ext) || (lhs.m_ext == rhs.m_ext));
}

//----------------------------------------------------------------------
// Comparison of Test_type_3 for equality.
bool operator==(const Test_type_3& lhs, const Test_type_3& rhs)
{
    bool data_ok = lhs.m_data && rhs.m_data;
    bool internal_ok = (lhs.m_ptr && rhs.m_ptr && *lhs.m_ptr == *rhs.m_ptr) || (!lhs.m_ptr && !rhs.m_ptr);
    return data_ok && internal_ok;
}

//----------------------------------------------------------------------
// Serialization.
const Serializable* Test_type_0::serialize(
    Serializer* serializer) const
{
    serializer->write(m_int);
    return this + 1;
}

//----------------------------------------------------------------------
// Deserialization.
Serializable* Test_type_0::deserialize(
    Deserializer* deserializer)                         // deserialize from here
{
    deserializer->read(&m_int);
    return this + 1;
}

//----------------------------------------------------------------------
// Serialization.
const Serializable* Test_type_1::serialize(
    Serializer* serializer) const
{
    serializer->write(m_int);
    MI::SERIAL::write(serializer, m_array);
    MI::SERIAL::write(serializer, m_string.c_str());
    return this + 1;
}

//----------------------------------------------------------------------
// Deserialization.
Serializable* Test_type_1::deserialize(
    Deserializer* deserializer)                         // deserialize from here
{
    deserializer->read(&m_int);
    MI::SERIAL::read(deserializer, &m_array);
    MI::SERIAL::read(deserializer, &m_string);
    return this + 1;
}

//----------------------------------------------------------------------
// Serialization.
const Serializable* Test_type_2::serialize(
    Serializer* serializer) const
{
    serializer->write(m_int);
    if (m_use_ext)
    {
        serializer->start_extension();
        serializer->write(m_ext);
    }
    return this + 1;
}

//----------------------------------------------------------------------
// Deserialization.
Serializable* Test_type_2::deserialize(
    Deserializer* deserializer)                         // deserialize from here
{
    deserializer->read(&m_int);
    if (m_use_ext && deserializer->check_extension())
    {
        m_ext_found = true;
        deserializer->read(&m_ext);
    }

    return this + 1;
}

//----------------------------------------------------------------------
// Serialization.
const Serializable* Test_type_3::serialize(
    Serializer* serializer) const
{
    serializer->write(m_data);
    serializer->serialize(m_ptr);
    return this + 1;
}

//----------------------------------------------------------------------
// Deserialization.
Serializable* Test_type_3::deserialize(
    Deserializer* deserializer)                         // deserialize from here
{
    deserializer->read(&m_data);
    m_ptr = dynamic_cast<Test_type_2*>(deserializer->deserialize());
    return this + 1;
}

//----------------------------------------------------------------------
// Serialization.
const Serializable* Test_type_4::serialize(Serializer* serializer) const
{
    serializer->write(m_iteration--);
    if (m_iteration > 0)
        serializer->serialize(this);

    return this+1;
}

//----------------------------------------------------------------------
// Deserialization.
Serializable* Test_type_4::deserialize(Deserializer* deserializer)
{
    deserializer->read(&m_iteration);
    if (m_iteration > 1)
        deserializer->deserialize();
    return this;
}
