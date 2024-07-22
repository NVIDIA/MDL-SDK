/******************************************************************************
 * Copyright (c) 2004-2024, NVIDIA CORPORATION. All rights reserved.
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
 /// \brief Provides utilities for timing.

#ifndef BASE_LIB_CONT_TEST_H
#define BASE_LIB_CONT_TEST_H

#include <base/hal/time/i_time.h>
#include <base/lib/log/i_log_logger.h>
#include <base/system/main/i_module_id.h>
#include <base/system/test/i_test_case.h>

using MI::LOG::Mod_log;
using MI::LOG::mod_log;
using namespace MI::SYSTEM;

class Test {
public:

    Test() {
        // empty
    }
    virtual ~Test(){}

    virtual bool test() = 0;

    void start_time() {
        m_watch.reset();
        m_watch.start();
    }

    void stop_time() {
        m_watch.stop();
    }

    void print_time() {
        MI::LOG::mod_log->debug(M_CONT, MI::LOG::Mod_log::C_MEMORY,
                                "elapsed time %g\n", m_watch.elapsed());
    }

private:
    MI::TIME::Stopwatch m_watch;
};

#endif
