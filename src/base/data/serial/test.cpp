/***************************************************************************************************
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
 **************************************************************************************************/
/// \file
/// \brief Test serialization.

#include "pch.h"


#define MI_TEST_AUTO_SUITE_NAME "base/data/serial Test Suite"

#include <base/system/test/i_test_auto_driver.h>

#include "test_serializable.h"
#include "test_function.h"

using namespace MI;

MI_TEST_AUTO_FUNCTION( test_verify_serializer_module )
{
    Deserialization_manager_holder deserialization_manager_holder;

    {
        deserialization_manager_holder.get()->register_class(
            ID_TEST_TYPE_0_CLASS_ID, test_type_0_factory);
        deserialization_manager_holder.get()->register_class(
            ID_TEST_TYPE_1_CLASS_ID, test_type_1_factory);

        test_buffer_serializer();
        test_file_serializer();
    }
}

Deserialization_manager_holder::Deserialization_manager_holder()
{
    m_log_module.set();
    m_deserialization_manager = SERIAL::Deserialization_manager::create();
    g_deserialization_manager = m_deserialization_manager;
}

Deserialization_manager_holder::~Deserialization_manager_holder()
{
    g_deserialization_manager = nullptr;
    SERIAL::Deserialization_manager::release(m_deserialization_manager);
    m_log_module.reset();
}

MI::SERIAL::Deserialization_manager* g_deserialization_manager;
