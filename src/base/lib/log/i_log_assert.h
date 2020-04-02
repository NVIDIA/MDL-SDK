/******************************************************************************
 * Copyright (c) 2008-2020, NVIDIA CORPORATION. All rights reserved.
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
/// \brief Macros for assertions.

#ifndef BASE_LIB_LOG_I_LOG_ASSERT_H
#define BASE_LIB_LOG_I_LOG_ASSERT_H

#include <base/system/main/i_module_id.h>
#include <base/system/main/i_assert.h>

namespace MI {

namespace LOG {

/// The ASSERT() macro calls this function to report an assertion failure.
///
/// The function will usually not return. If mod_log is initialized, the message will be forwarded
/// to mod_log::assertfailed(). Otherwise, this function prints a message to the standard error
/// stream and aborts the process.
void report_assertion_failure(
    SYSTEM::Module_id module,
    const char* expression,
    const char* file,
    unsigned int line);

/// Define ASSERT and DEBUG_ASSERT.
///
/// ASSERT is defined debug builds or if ENABLE_ASSERT is defined. DEBUG_ASSERT is only defined in
/// debug builds, which is useful to assert in code that is only compiled in debug versions.
#undef ASSERT
#if defined(DEBUG) || defined(ENABLE_ASSERT)
#define ASSERT(M, EXP)                                                  \
    do {                                                                \
     if (!(EXP))                                                        \
       ::MI::LOG::report_assertion_failure(M,#EXP,__FILE__,__LINE__);   \
    } while(0)
#else
#define ASSERT(M, X) ((void)0)
#endif

#undef DEBUG_ASSERT
#if defined(DEBUG)
#define DEBUG_ASSERT(M, EXP)                                            \
    do {                                                                \
     if (!(EXP))                                                        \
       ::MI::LOG::report_assertion_failure(M,#EXP,__FILE__,__LINE__);   \
    } while(0)
#else
#define DEBUG_ASSERT(M, X) ((void)0)
#endif

} // namespace LOG

} // namespace MI

#endif // BASE_LIB_LOG_I_LOG_ASSERT_H
