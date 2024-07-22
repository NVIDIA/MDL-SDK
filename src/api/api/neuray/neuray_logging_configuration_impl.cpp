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

/** \file
 ** \brief Source for the ILogging_configuration implementation.
 **/

#include "pch.h"

#include "neuray_logging_configuration_impl.h"


#include <cstring>
#include <new>

#include <mi/base/ilogger.h>

#include <base/lib/log/i_log_assert.h>
#include <base/lib/log/i_log_module.h>
#include <base/lib/log/i_log_target.h>
#include <base/lib/log/i_log_utilities.h>
#include <base/lib/mem/mem.h>

namespace MI {

namespace NEURAY {

mi::base::Handle<mi::base::ILogger> g_receiving_logger;

void fatal_memory_callback()
{
    // Reset callback to avoid an infinite loop if another allocation failure occurs inside the
    // callback.
    std::set_new_handler( nullptr);

    if( g_receiving_logger) {
        g_receiving_logger->message(
            mi::base::MESSAGE_SEVERITY_FATAL, "API:MEMORY", "Memory allocation failed.");
    }
}

/// This logger adapts mi::base::ILogger to LOG::ILog_target and is used for the receiving logger.
class Receiving_logger : public LOG::ILog_target, public boost::noncopyable
{
public:
    /// Constructor.
    Receiving_logger( mi::base::ILogger* logger);

    bool message(
        const char* module,
        LOG::Mod_log::Category category,
        LOG::Mod_log::Severity severity,
        const mi::base::Message_details& det,
        const char* prefix,
        const char* message);

private:

    /// The adapted logger.
    mi::base::Handle<mi::base::ILogger> m_logger;
};

Receiving_logger::Receiving_logger( mi::base::ILogger* logger)
  : m_logger( logger, mi::base::DUP_INTERFACE)
{
}

bool Receiving_logger::message(
    const char* module,
    LOG::Mod_log::Category category,
    LOG::Mod_log::Severity severity,
    const mi::base::Message_details& det,
    const char* prefix,
    const char* message)
{
    // Convert severity from LOG::Severity to mi::base::Message_severity
    const mi::base::Message_severity severity_enum = LOG::convert_severity( severity);

    // Convert module and category into a string separated by ":"
    std::string module_category = module;
    module_category += ':';
    module_category += LOG::convert_category_to_string( category);

    // Convert prefix and message into a string separated by " "
    std::string full_message = prefix;
    full_message += message;

    m_logger->message( severity_enum, module_category.c_str(), det, full_message.c_str());
    return true;
}

Logging_configuration_impl::Logging_configuration_impl( Neuray_impl* neuray_impl)
  : m_neuray_impl( neuray_impl),
    m_internal_receiving_logger( nullptr)
{
    std::set_new_handler( fatal_memory_callback);

    m_mem_module.set();
    m_log_module.set();
    m_log_module->set_prefix( LOG::ILogger::P_SEVERITY);

    m_forwarding_logger = new LOG::Forwarding_logger;
}

Logging_configuration_impl::~Logging_configuration_impl()
{
    set_receiving_logger( nullptr);
    m_forwarding_logger = nullptr;

    m_log_module.reset();
    m_mem_module.reset();

    m_neuray_impl = nullptr;

    std::set_new_handler( nullptr);
}

void Logging_configuration_impl::set_receiving_logger( mi::base::ILogger* logger)
{
    mi::base::Lock::Block block( &m_lock);

    if( m_receiving_logger) {
        ASSERT( M_NEURAY_API, m_internal_receiving_logger);
        m_log_module->remove_log_target( m_internal_receiving_logger);
        delete m_internal_receiving_logger;
        m_internal_receiving_logger = nullptr;
        m_receiving_logger = nullptr;
        g_receiving_logger = nullptr;
    }

    ASSERT( M_NEURAY_API, !m_receiving_logger);
    ASSERT( M_NEURAY_API, !m_internal_receiving_logger);

    if( logger) {
        m_receiving_logger = make_handle_dup( logger);
        g_receiving_logger = make_handle_dup( logger);
        m_internal_receiving_logger = new Receiving_logger( logger);
        m_log_module->add_log_target_front( m_internal_receiving_logger);
    }

    m_log_module->emit_delayed_log_messages();
}

mi::base::ILogger* Logging_configuration_impl::get_receiving_logger() const
{
    mi::base::Lock::Block block( &m_lock);

    if( !m_receiving_logger)
        return nullptr;
    m_receiving_logger->retain();
    return m_receiving_logger.get();
}

mi::base::ILogger* Logging_configuration_impl::get_forwarding_logger() const
{
    mi::base::Lock::Block block( &m_lock);

    if( !m_forwarding_logger)
        return nullptr;
    m_forwarding_logger->retain();
    return m_forwarding_logger.get();
}

mi::Sint32 Logging_configuration_impl::set_log_level( mi::base::Message_severity level)
{
    int severity = LOG::convert_severity( level);
    if( severity == -1)
        return -1;

    m_log_module->set_severity_limit( severity);
    return 0;
}

mi::base::Message_severity Logging_configuration_impl::get_log_level() const
{
    int severity = m_log_module->get_severity_limit();
    mi::base::Message_severity level = LOG::convert_severity( severity);
    return level;
}

mi::Sint32 Logging_configuration_impl::set_log_level_by_category(
    const char* category,
    mi::base::Message_severity level)
{
    LOG::Mod_log::Category category_enum = LOG::Mod_log::C_MAIN; // avoid warning
    if( strcmp( category, "ALL") == 0)
        category_enum = LOG::Mod_log::C_ALL;
    else if( !LOG::convert_string_to_category( category, category_enum))
        return -1;

    int severity = LOG::convert_severity( level);
    if( severity == -1)
        return -1;

    m_log_module->set_severity_by_category( category_enum, severity);
    return 0;
}

mi::base::Message_severity Logging_configuration_impl::get_log_level_by_category(
    const char* category) const
{
    LOG::Mod_log::Category category_enum = LOG::Mod_log::C_MAIN; // avoid warning
    if( !LOG::convert_string_to_category( category, category_enum))
        return static_cast<mi::base::Message_severity>( -1);

    return LOG::convert_severity( m_log_module->get_severity_by_category( category_enum));
}

void Logging_configuration_impl::set_log_prefix( mi::Uint32 prefix)
{
    mi::Uint32 internal_prefix = 0;
    using namespace mi::neuraylib;
    if( prefix & LOG_PREFIX_TIME         ) internal_prefix |= LOG::ILogger::P_TIME;
    if( prefix & LOG_PREFIX_TIME_SECONDS ) internal_prefix |= LOG::ILogger::P_TIME_SECONDS;
    if( prefix & LOG_PREFIX_HOST_NAME    ) internal_prefix |= LOG::ILogger::P_HOST_NAME;
    if( prefix & LOG_PREFIX_HOST_THREAD  ) internal_prefix |= LOG::ILogger::P_HOST_THREAD;
    if( prefix & LOG_PREFIX_CUDA_DEVICE  ) internal_prefix |= LOG::ILogger::P_CUDA_DEVICE;
    if( prefix & LOG_PREFIX_TAGS         ) internal_prefix |= LOG::ILogger::P_TAGS;
    if( prefix & LOG_PREFIX_MODULE       ) internal_prefix |= LOG::ILogger::P_MODULE;
    if( prefix & LOG_PREFIX_CATEGORY     ) internal_prefix |= LOG::ILogger::P_CATEGORY;
    if( prefix & LOG_PREFIX_SEVERITY     ) internal_prefix |= LOG::ILogger::P_SEVERITY;

    m_log_module->set_prefix( internal_prefix);
}

mi::Uint32 Logging_configuration_impl::get_log_prefix() const
{
    unsigned int internal_prefix = m_log_module->get_prefix();

    mi::Uint32 prefix = 0;
    using namespace mi::neuraylib;
    if( internal_prefix & LOG::ILogger::P_TIME        ) prefix |= LOG_PREFIX_TIME;
    if( internal_prefix & LOG::ILogger::P_TIME_SECONDS) prefix |= LOG_PREFIX_TIME_SECONDS;
    if( internal_prefix & LOG::ILogger::P_HOST_THREAD ) prefix |= LOG_PREFIX_HOST_THREAD;
    if( internal_prefix & LOG::ILogger::P_CUDA_DEVICE ) prefix |= LOG_PREFIX_CUDA_DEVICE;
    if( internal_prefix & LOG::ILogger::P_TAGS        ) prefix |= LOG_PREFIX_TAGS;
    if( internal_prefix & LOG::ILogger::P_HOST_NAME   ) prefix |= LOG_PREFIX_HOST_NAME;
    if( internal_prefix & LOG::ILogger::P_MODULE      ) prefix |= LOG_PREFIX_MODULE;
    if( internal_prefix & LOG::ILogger::P_CATEGORY    ) prefix |= LOG_PREFIX_CATEGORY;
    if( internal_prefix & LOG::ILogger::P_SEVERITY    ) prefix |= LOG_PREFIX_SEVERITY;

    return prefix;
}

mi::Sint32 Logging_configuration_impl::set_log_priority( mi::Sint32 priority)
{
    return -1;
}

mi::Sint32 Logging_configuration_impl::get_log_priority() const
{
    return 0;
}

mi::Sint32 Logging_configuration_impl::set_log_locally( bool value)
{
    return -1;
}

bool Logging_configuration_impl::get_log_locally() const
{
    return false;
}

} // namespace NEURAY

} // namespace MI

