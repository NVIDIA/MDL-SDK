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
 ** \brief Header for the IExport_result_ext implementation.
 **/

#ifndef API_API_NEURAY_NEURAY_EXPORT_RESULT_EXT_IMPL_H
#define API_API_NEURAY_NEURAY_EXPORT_RESULT_EXT_IMPL_H

#include <mi/base/interface_implement.h>
#include <mi/neuraylib/iexport_result.h>

#include <string>
#include <vector>

#include <boost/core/noncopyable.hpp>

namespace mi { namespace neuraylib { class ITransaction; } }

namespace MI {

namespace NEURAY {

class Export_result_ext_impl
  : public mi::base::Interface_implement<mi::neuraylib::IExport_result_ext>,
    public boost::noncopyable
{
public:

    static mi::base::IInterface* create_api_class(
        mi::neuraylib::ITransaction* transaction,
        mi::Uint32 argc,
        const mi::base::IInterface* argv[]);

    // public API methods of IExport_result

    mi::Uint32 get_error_number() const;

    const char* get_error_message() const;

    mi::Size get_messages_length() const;

    mi::Uint32 get_message_number( mi::Size index) const;

    const char* get_message( mi::Size index) const;

    mi::base::Message_severity get_message_severity( mi::Size index) const;

    // public API methods of IExport_result_ext

    mi::Sint32 set_message(
        mi::Uint32 number, mi::base::Message_severity severity, const char* message);

    mi::Sint32 message_push_back(
        mi::Uint32 number, mi::base::Message_severity severity, const char* message);

    mi::Sint32 set_message(
        mi::Size index,
        mi::Uint32 number,
        mi::base::Message_severity severity,
        const char* message);

    void clear_messages();

    mi::Sint32 append_messages( const IExport_result* export_result);

    // internal methods

private:
    /// Invariant: m_message_numbers, m_message_severities, and m_messages have the same size.

    /// Holds message numbers
    std::vector<mi::Uint32> m_message_numbers;

    /// Holds message severities
    std::vector<mi::base::Message_severity> m_message_severities;

    /// Holds message messages
    std::vector<std::string> m_messages;
};

} // namespace NEURAY

} // namespace MI

#endif // API_API_NEURAY_NEURAY_EXPORT_RESULT_EXT_IMPL_H
