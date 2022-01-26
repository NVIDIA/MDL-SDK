/******************************************************************************
 * Copyright (c) 2010-2022, NVIDIA CORPORATION. All rights reserved.
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
 *****************************************************************************/

/// \file
/// \brief      STL ostream interface for neuray's logging facilities

#ifndef BASE_LIB_LOG_I_LOG_STREAM_H
#define BASE_LIB_LOG_I_LOG_STREAM_H

#include "i_log_logger.h"

#include <base/system/main/i_module_id.h>

namespace MI {

namespace LOG {

namespace MESSAGE {

namespace DETAIL {

class Base : public mi::base::Log_stream
{
public:
    Base(
            SYSTEM::Module_id mid,
            LOG::ILogger::Category cid,
            mi::base::Message_severity sev,
            const mi::base::Message_details& det)
    : Log_stream(nullptr,"",sev,det)
    , m_module{mid}
    , m_category{cid}
    {}

    ~Base()
    {
        flush();
    }

private:
    SYSTEM::Module_id           m_module;
    LOG::ILogger::Category      m_category;

    void message(
            mi::base::Message_severity level,
            const char*,
            const mi::base::Message_details& details,
            const char* message) const override
    {
        switch (level) {
            case mi::base::MESSAGE_SEVERITY_FATAL:
                return LOG::mod_log->fatal(m_module,m_category,details,"%s",message);
            case mi::base::MESSAGE_SEVERITY_ERROR:
                return LOG::mod_log->error(m_module,m_category,details,"%s",message);
            case mi::base::MESSAGE_SEVERITY_WARNING:
                return LOG::mod_log->warning(m_module,m_category,details,"%s",message);
            case mi::base::MESSAGE_SEVERITY_INFO:
                return LOG::mod_log->info(m_module,m_category,details,"%s",message);
            case mi::base::MESSAGE_SEVERITY_VERBOSE:
                return LOG::mod_log->vstat(m_module,m_category,details,"%s",message);
            case mi::base::MESSAGE_SEVERITY_DEBUG:
            default:
                return LOG::mod_log->debug(m_module,m_category,details,"%s",message);
        }
    }
};


template <mi::base::Message_severity S>
struct Base_t : public Base
{
    using Stream = Base_t<S>;

    Base_t(
            SYSTEM::Module_id mid,
            LOG::ILogger::Category cid,
            const mi::base::Message_details& det={})
    : Base(mid,cid,S,det)
    {}
};

}

// mapping as per log_utilities.cpp
using Fatal = DETAIL::Base_t<mi::base::MESSAGE_SEVERITY_FATAL>;
using Error = DETAIL::Base_t<mi::base::MESSAGE_SEVERITY_ERROR>;
using Warning = DETAIL::Base_t<mi::base::MESSAGE_SEVERITY_WARNING>;
using Info = DETAIL::Base_t<mi::base::MESSAGE_SEVERITY_INFO>;
using Progress = DETAIL::Base_t<mi::base::MESSAGE_SEVERITY_INFO>;
using Debug = DETAIL::Base_t<mi::base::MESSAGE_SEVERITY_DEBUG>;
using Stat = DETAIL::Base_t<mi::base::MESSAGE_SEVERITY_DEBUG>;
using Vstat = DETAIL::Base_t<mi::base::MESSAGE_SEVERITY_DEBUG>;
using Vdebug = DETAIL::Base_t<mi::base::MESSAGE_SEVERITY_DEBUG>;


} // namespace MESSAGE

} // namespace LOG

} // namespace MI

#endif // BASE_LIB_LOG_I_LOG_STREAM_H
