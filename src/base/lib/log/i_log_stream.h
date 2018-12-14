/******************************************************************************
 * Copyright (c) 2010-2018, NVIDIA CORPORATION. All rights reserved.
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

#include <sstream>
#include <base/system/main/i_module_id.h>

namespace MI {

namespace LOG {

namespace MESSAGE {


class Base : public std::ostringstream
{
public:
    typedef MI::SYSTEM::Module_id       Module_id;
    typedef MI::LOG::ILogger::Category  Category_id;

protected:
    Base(Module_id mid, Category_id cid) : m_mid(mid), m_cid(cid) { }

    const Module_id   m_mid;
    const Category_id m_cid;
};


struct Fatal : public Base
{
    Fatal(Module_id mid, Category_id cid) : Base(mid, cid) { }
    ~Fatal()
    {
        const std::string& s = str();
        if (!s.empty())
            mod_log->fatal(m_mid, m_cid, 0, "%s", s.c_str());
    }
};


struct Error : public Base
{
    Error(Module_id mid, Category_id cid) : Base(mid, cid) { }
    ~Error()
    {
        const std::string& s = str();
        if (!s.empty())
            mod_log->error(m_mid, m_cid, 0, "%s", s.c_str());
    }
};


struct Warning : public Base
{
    Warning(Module_id mid, Category_id cid) : Base(mid, cid) { }
    ~Warning()
    {
        const std::string& s = str();
        if (!s.empty())
            mod_log->warning(m_mid, m_cid, 0, "%s", s.c_str());
    }
};


struct Stat : public Base
{
    Stat(Module_id mid, Category_id cid) : Base(mid, cid) { }
    ~Stat()
    {
        const std::string& s = str();
        if (!s.empty())
            mod_log->stat(m_mid, m_cid, "%s", s.c_str());
    }
};


struct Vstat : public Base
{
    Vstat(Module_id mid, Category_id cid) : Base(mid, cid) { }
    ~Vstat()
    {
        const std::string& s = str();
        if (!s.empty())
            mod_log->vstat(m_mid, m_cid, "%s", s.c_str());
    }
};


struct Progress : public Base
{
    Progress(Module_id mid, Category_id cid) : Base(mid, cid) { }
    ~Progress()
    {
        const std::string& s = str();
        if (!s.empty())
            mod_log->progress(m_mid, m_cid, 0, "%s", s.c_str());
    }
};


struct Info : public Base
{
    Info(Module_id mid, Category_id cid) : Base(mid, cid) { }
    ~Info()
    {
        const std::string& s = str();
        if (!s.empty())
            mod_log->info(m_mid, m_cid, "%s", s.c_str());
    }
};


struct Debug : public Base
{
    Debug(Module_id mid, Category_id cid) : Base(mid, cid) { }
    ~Debug()
    {
        const std::string& s = str();
        if (!s.empty())
            mod_log->debug(m_mid, m_cid, "%s", s.c_str());
    }
};


struct Vdebug : public Base
{
    Vdebug(Module_id mid, Category_id cid) : Base(mid, cid) { }
    ~Vdebug()
    {
        const std::string& s = str();
        if (!s.empty())
            mod_log->vdebug(m_mid, m_cid, "%s", s.c_str());
    }
};


} // namespace MESSAGE

} // namespace LOG

} // namespace MI

#endif // BASE_LIB_LOG_I_LOG_STREAM_H
