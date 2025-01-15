/******************************************************************************
 * Copyright (c) 2007-2025, NVIDIA CORPORATION. All rights reserved.
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
/// \brief regression test infrastructure

#ifndef BASE_SYSTEM_TEST_LOGGER_H
#define BASE_SYSTEM_TEST_LOGGER_H

#include "i_test_case.h"
#include <base/lib/log/i_log_logger.h>
#include <cstdio>
#include <cstdarg>
#include <sstream>

namespace MI { namespace TEST {

struct Mod_log : public LOG::ILogger
{
    typedef MI::SYSTEM::Module_id       Module_id;
    typedef MI::LOG::Mod_log::Category  Category;
    typedef MI::LOG::Mod_log::Prefix    Prefix;

    void fatal(Module_id mid, Category cid, int errcode, char const * fmt, ...)
    {
        va_list args;
        va_start(args, fmt);
        message(mid, cid, errcode, 0, fmt, args);
        va_end(args);
    }
    void error(Module_id mid, Category cid, int errcode, char const * fmt, ...)
    {
        va_list args;
        va_start(args, fmt);
        message(mid, cid, errcode, 1, fmt, args);
        va_end(args);
    }
    void warning(Module_id mid, Category cid, int errcode, char const * fmt, ...)
    {
        va_list args;
        va_start(args, fmt);
        message(mid, cid, errcode, 2, fmt, args);
        va_end(args);
    }
    void stat(Module_id mid, Category cid, char const * fmt, ...)
    {
        va_list args;
        va_start(args, fmt);
        message(mid, cid, 0, 3, fmt, args);
        va_end(args);
    }
    void vstat(Module_id mid, Category cid, char const * fmt, ...)
    {
        va_list args;
        va_start(args, fmt);
        message(mid, cid, 0, 4, fmt, args);
        va_end(args);
    }
    void progress(Module_id mid, Category cid, int errcode, char const * fmt, ...)
    {
        va_list args;
        va_start(args, fmt);
        message(mid, cid, errcode, 5, fmt, args);
        va_end(args);
    }
    void info(Module_id mid, Category cid, char const * fmt, ...)
    {
        va_list args;
        va_start(args, fmt);
        message(mid, cid, 0, 6, fmt, args);
        va_end(args);
    }
    void debug(Module_id mid, Category cid, char const * fmt, ...)
    {
        va_list args;
        va_start(args, fmt);
        message(mid, cid, 0, 7, fmt, args);
        va_end(args);
    }
    void vdebug(Module_id mid, Category cid, char const * fmt, ...)
    {
        va_list args;
        va_start(args, fmt);
        message(mid, cid, 0, 8, fmt, args);
        va_end(args);
    }
    void assertfailed(Module_id mid, char const * str, char const * file, int line)
    {
        char buf[1024];
        snprintf(buf, sizeof(buf), "%s:%d", file, line);
        MI::TEST::Test_suite_failure err(str, "", buf);
        throw err;
    }
    void message(Module_id mid, Category cid, int errcode, int severity, const char * fmt, va_list ap)
    {
        char buf[4096];
        vsnprintf(buf, sizeof(buf), fmt, ap);
        if (severity <= 1)          // abort on fatal() and error()
        {
            std::ostringstream os;
            os << "module "    << mid
               << ":category " << cid
               << ":errcode "  << errcode
               << ":severity " << severity;
            Test_suite_failure err(buf, "", os.str());
            throw err;
        }
#ifndef NDEBUG
        else
            printf("%s\n", buf);
#endif
    }
};

}} // MI::TEST

#endif // BASE_SYSTEM_TEST_LOGGER_H
