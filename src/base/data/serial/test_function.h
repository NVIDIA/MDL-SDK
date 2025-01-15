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
/// \brief Test functions list

#ifndef BASE_DATA_SERIAL_TEST_FUNCTION_H
#define BASE_DATA_SERIAL_TEST_FUNCTION_H

#include <boost/core/noncopyable.hpp>

#include <base/system/main/access_module.h>
#include <base/lib/log/i_log_module.h>

namespace MI::SERIAL { class Deserialization_manager; }

class Deserialization_manager_holder : public boost::noncopyable
{
public:
    Deserialization_manager_holder();
    ~Deserialization_manager_holder();
    MI::SERIAL::Deserialization_manager* get() { return m_deserialization_manager; }
private:
    MI::SYSTEM::Access_module<MI::LOG::Log_module> m_log_module;
    MI::SERIAL::Deserialization_manager* m_deserialization_manager;
};

extern MI::SERIAL::Deserialization_manager* g_deserialization_manager;

// test_buffer_serializer.cpp
extern void test_buffer_serializer();

// test_file_serializer.cpp
extern void test_file_serializer();

#endif // #ifndef BASE_DATA_SERIAL_TEST_FUNCTION_H

