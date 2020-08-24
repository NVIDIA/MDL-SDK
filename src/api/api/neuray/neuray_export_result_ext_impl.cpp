/***************************************************************************************************
 * Copyright (c) 2010-2020, NVIDIA CORPORATION. All rights reserved.
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

/** \file
 ** \brief Source for the IExport_result_ext implementation.
 **/

#include "pch.h"

#include "neuray_export_result_ext_impl.h"

#include <base/lib/log/i_log_assert.h>

namespace MI {

namespace NEURAY {

mi::base::IInterface* Export_result_ext_impl::create_api_class(
    mi::neuraylib::ITransaction* transaction,
    mi::Uint32 argc,
    const mi::base::IInterface* argv[])
{
    if( argc != 0)
        return nullptr;
    return new Export_result_ext_impl();
}

mi::Uint32 Export_result_ext_impl::get_error_number() const
{
    for( mi::Size i = 0; i < m_message_numbers.size(); ++i)
        if(  ( m_message_severities[i] == mi::base::MESSAGE_SEVERITY_FATAL)
          || ( m_message_severities[i] == mi::base::MESSAGE_SEVERITY_ERROR))
            return m_message_numbers[i];
    return 0;
}

const char* Export_result_ext_impl::get_error_message() const
{
    for( mi::Size i = 0; i < m_message_numbers.size(); ++i)
        if(  ( m_message_severities[i] == mi::base::MESSAGE_SEVERITY_FATAL)
          || ( m_message_severities[i] == mi::base::MESSAGE_SEVERITY_ERROR))
            return m_messages[i].empty() ? nullptr : m_messages[i].c_str();
    return nullptr;
}

mi::Size Export_result_ext_impl::get_messages_length() const
{
    ASSERT( M_NEURAY_API, m_message_numbers.size() == m_messages.size());
    ASSERT( M_NEURAY_API, m_message_numbers.size() == m_message_severities.size());
    return m_message_numbers.size();
}

mi::Uint32 Export_result_ext_impl::get_message_number( mi::Size index) const
{
    if( index >= m_message_numbers.size())
        return 0;
    return m_message_numbers[index];
}

const char* Export_result_ext_impl::get_message( mi::Size index) const
{
    if( index >= m_messages.size())
        return nullptr;
    return m_messages[index].empty() ? nullptr : m_messages[index].c_str();
}

mi::base::Message_severity Export_result_ext_impl::get_message_severity( mi::Size index) const
{
    if( index >= m_message_severities.size())
        return mi::base::MESSAGE_SEVERITY_ERROR;
    return m_message_severities[index];
}

mi::Sint32 Export_result_ext_impl::set_message(
    mi::Uint32 number, mi::base::Message_severity severity, const char* message)
{
    if( !message)
        return -1;

    clear_messages();
    message_push_back( number, severity, message);
    return 0;
}

mi::Sint32 Export_result_ext_impl::message_push_back(
    mi::Uint32 number, mi::base::Message_severity severity, const char* message)
{
    if( !message)
        return -1;

    m_message_numbers.push_back( number);
    m_message_severities.push_back( severity);
    m_messages.push_back( message);
    return 0;
}

mi::Sint32 Export_result_ext_impl::set_message(
    mi::Size index, mi::Uint32 number, mi::base::Message_severity severity, const char* message)
{
    if( !message)
        return -1;
    if( index < m_message_numbers.size())
        return -2;
    if( index < m_message_severities.size())
        return -2;
    if( index < m_messages.size())
        return -2;

    m_message_numbers[index] = number;
    m_message_severities[index] = severity;
    m_messages[index] = message; //-V557 PVS
    return 0;
}

void Export_result_ext_impl::clear_messages()
{
    m_message_numbers.clear();
    m_message_severities.clear();
    m_messages.clear();
}

mi::Sint32 Export_result_ext_impl::append_messages( const IExport_result* export_result)
{
    if( !export_result)
        return -1;

    mi::Size n = export_result->get_messages_length();
    for( mi::Size i = 0; i < n; ++i)
        message_push_back( export_result->get_message_number( i),
                           export_result->get_message_severity( i),
                           export_result->get_message( i));
    return 0;
}

} // namespace NEURAY

} // namespace MI
