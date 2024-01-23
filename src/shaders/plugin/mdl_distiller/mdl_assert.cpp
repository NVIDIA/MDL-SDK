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

#include "mdl_assert.h"

#include <stdio.h>
#include <stdlib.h>

// Reporting function for failed mdl_assert().
void mdl_assert_fct( const char *file, int line, const char* fct, const char *msg) {
    fprintf( stderr, 
             "ASSERT: MDL Distiller Plugin: failed in %s %u, function %s: \"%s\".\n",
             file, line, fct, msg);
    fflush(stderr);
    abort();
}

namespace mi {
namespace mdl {

/// Reporting function for failed MDL_ASSERT() macro in mdl/ tree. calls this function to report an assertion failure.
/// The function will usually not return.
/// \param exp   the expression that failed
/// \param file  name of the file containing the assertion
/// \param line  line number of assertion in file
void report_assertion_failure(
    char const   *exp,
    char const   *file,
    unsigned int line) 
{
    fprintf(stderr, "ASSERT: MDL Distiller Plugin: failed in %s %u: \"%s\"\n", file, line, exp);
    fflush(stderr);
    abort();
}


}  // mdl
}  // mi
