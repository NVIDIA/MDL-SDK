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

#ifndef MDL_COMPILERCORE_ASSERT_H
#define MDL_COMPILERCORE_ASSERT_H 1

#include <mi/mdl/mdl_assert.h>

namespace mi {
namespace mdl {

extern IAsserter *iasserter;

/// The MDL_ASSERT() macro calls this function to report an assertion failure.
/// The function will usually not return.
/// \param exp   the expression that failed
/// \param file  name of the file containing the assertion
/// \param line  line number of assertion in file
void report_assertion_failure(
    char const   *exp,
    char const   *file,
    unsigned int line);

}  // mdl
}  // mi

//
// Define MDL_ASSERT and DEBUG_MDL_ASSERT. MDL_ASSERT is always available in debug builds
// and also in release builds if ENABLE_ASSERT is defined. DEBUG_MDL_ASSERT is
// *only* available in debug builds, which is useful to assert in code that
// is only compiled in debug versions.
//
#undef DEBUG_MDL_ASSERT
#undef MDL_ASSERT
#if !defined(DEBUG) && !defined(ENABLE_ASSERT)
#  define MDL_ASSERT(X)       ((void)0)
#  define DEBUG_MDL_ASSERT(X) ((void)0)
#else                                   // DEBUG or ENABLE_ASSERT defined.
#  ifndef ENABLE_ASSERT                 // Always define ENABLE_ASSERT.
#    define ENABLE_ASSERT
#  endif
#  define MDL_ASSERT(EXP)                                                   \
    do {                                                                \
     if (!(EXP))                                                        \
       ::mi::mdl::report_assertion_failure(#EXP,__FILE__,__LINE__);     \
    } while(0)
#  ifdef DEBUG
#    define DEBUG_MDL_ASSERT(EXP)                                           \
      do {                                                              \
       if (!(EXP))                                                      \
         ::mi::mdl::report_assertion_failure(#EXP,__FILE__,__LINE__);   \
      } while(0)
#  else
#    define DEBUG_MDL_ASSERT(EXP) ((void)0)
#  endif
#endif

#endif
