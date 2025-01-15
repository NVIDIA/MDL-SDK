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

/** \file
 ** \brief Header for the ILogging_configuration implementation.
 **/

#ifndef API_API_NEURAY_NEURAY_LOGGING_CONFIGURATION_IMPL_H
#define API_API_NEURAY_NEURAY_LOGGING_CONFIGURATION_IMPL_H

#include <mi/base/handle.h>
#include <mi/base/interface_implement.h>
#include <mi/base/lock.h>
#include <mi/neuraylib/ilogging_configuration.h>

#include <boost/core/noncopyable.hpp>
#include <base/system/main/access_module.h>

namespace mi { namespace base { class ILogger; } }

namespace MI {

namespace LOG { class Log_module; }
namespace MEM { class Mem_module; }

namespace NEURAY {

class Neuray_impl;
class Receiving_logger;

class Logging_configuration_impl
  : public mi::base::Interface_implement<mi::neuraylib::ILogging_configuration>,
    public boost::noncopyable
{
public:
    /// Construction of Logging_configuration_impl
    ///
    /// \param neuray_impl      The neuray instance which contains this Logging_configuration_impl
    Logging_configuration_impl( Neuray_impl* neuray_impl);

    /// Destructor of Logging_configuration_impl
    ~Logging_configuration_impl();

    // public API methods

    void set_receiving_logger( mi::base::ILogger* logger);

    mi::base::ILogger* get_receiving_logger() const;

    mi::base::ILogger* get_forwarding_logger() const;

    mi::Sint32 set_log_level( mi::base::Message_severity level);

    mi::base::Message_severity get_log_level() const;

    mi::Sint32 set_log_level_by_category( const char* category, mi::base::Message_severity level);

    mi::base::Message_severity get_log_level_by_category( const char* category) const;

    void set_log_prefix( mi::Uint32 prefix);

    mi::Uint32 get_log_prefix() const;

    mi::Sint32 set_log_priority( mi::Sint32 priority);

    mi::Sint32 get_log_priority() const;

    mi::Sint32 set_log_locally( bool value);

    bool get_log_locally() const;

    // internal methods

private:

    /// Pointer to Neuray_impl.
    Neuray_impl* m_neuray_impl;

    /// The LOG module.
    SYSTEM::Access_module<LOG::Log_module> m_log_module;

    /// The MEM module.
    ///
    /// Not really used here, just to make sure it remains initialized.
    SYSTEM::Access_module<MEM::Mem_module> m_mem_module;

    /// The lock for m_forwarding_logger, m_receiving_logger, and m_internal_receiving_logger.
    mutable mi::base::Lock m_lock;

    /// The (external) forwarding logger. Needs to be protected by m_lock.
    mi::base::Handle<mi::base::ILogger> m_forwarding_logger;

    /// The (external) receiving logger. Needs to be protected by m_lock.
    mi::base::Handle<mi::base::ILogger> m_receiving_logger;

    /// The (internal) receiving logger (Log_target adaptor). Needs to be protected by m_lock.
    Receiving_logger* m_internal_receiving_logger;
};

} // namespace NEURAY

} // namespace MI

#endif // API_API_NEURAY_NEURAY_LOGGING_CONFIGURATION_IMPL_H
