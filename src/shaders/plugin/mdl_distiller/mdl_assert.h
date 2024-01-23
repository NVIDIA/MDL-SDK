/******************************************************************************
 * Copyright (c) 2017-2024, NVIDIA CORPORATION. All rights reserved.
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

#pragma once

#include <iostream>

// remap internal ASSERT to our asset here, needs to come before and 
// other header gets to include the internal base/lib/lig/i_log_assert.h file
#define BASE_LIB_LOG_I_LOG_ASSERT_H
#define ASSERT(m,x) mdl_assert(x)
#define DEBUG_ASSERT(m,x) mdl_assert(x)

/// If possible, lets the asserts support function names in their message.
#if defined(__FUNCSIG__)
#  define MDL_ASSERT_FUNCTION __FUNCSIG__
#elif defined( __cplusplus) && defined(__GNUC__) && defined(__GNUC_MINOR__) \
        && ((__GNUC__ << 16) + __GNUC_MINOR__ >= (2 << 16) + 6)
#  define MDL_ASSERT_FUNCTION    __PRETTY_FUNCTION__
#else
#  if defined __STDC_VERSION__ && __STDC_VERSION__ >= 199901L
#    define MDL_ASSERT_FUNCTION    __func__
#  else
#    define MDL_ASSERT_FUNCTION    ("unknown")
#  endif
#endif

#ifdef NDEBUG
#define mdl_assert(expr) (static_cast<void>(0)) // valid but void null stmt
#define mdl_assert_msg(expr, msg) (static_cast<void>(0)) // valid but void null stmt
#else
#define mdl_assert(expr)  \
    (void)((expr) || (mdl_assert_fct( __FILE__, __LINE__, MDL_ASSERT_FUNCTION, #expr),0))
#define mdl_assert_msg(expr, msg) \
    (void)((expr) || (mdl_assert_fct( __FILE__, __LINE__, MDL_ASSERT_FUNCTION, msg),0))
#endif // NDEBUG

void mdl_assert_fct (const char *file, int line, const char* fct, const char *msg);
