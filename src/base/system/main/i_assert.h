/******************************************************************************
 * Copyright (c) 2012-2020, NVIDIA CORPORATION. All rights reserved.
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
/// \brief light-weight assertions
///
/// This file provides light-weight assertions, bypassing the log, that can
/// be enabled in release mode.

#ifndef BASE_SYSTEM_MAIN_I_ASSERT_H
#define BASE_SYSTEM_MAIN_I_ASSERT_H

#include <cstdio>
#include <cstdlib>


//
// Define assert which can be enabled in release builds as well as debug.
// Also ensure that ENABLE_ASSERT is defined in DEBUG builds.
//
#undef MI_ASSERT
#if !defined(DEBUG) && !defined(ENABLE_ASSERT)
#  define MI_ASSERT(X)       ((void)0)
#else                                   // DEBUG or ENABLE_ASSERT defined.
#  ifndef ENABLE_ASSERT                 // Always define ENABLE_ASSERT.
#    define ENABLE_ASSERT
#  endif
#  define MI_ASSERT(EXP) \
    do { \
        if (!(EXP)) { \
            ::fprintf(stderr,"assertion failed in %s %d: \"%s\"\n",__FILE__,__LINE__,#EXP); \
            ::abort(); \
        } \
    } while(0)
#endif

#endif //BASE_SYSTEM_MAIN_I_ASSERT_H
