/***************************************************************************************************
 * Copyright (c) 2012-2025, NVIDIA CORPORATION. All rights reserved.
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

/// \file
/// \brief Factory for the MDL Core library.

#include "pch.h"

#include <mdl/compiler/compilercore/compilercore_mdl.h>
#include <mdl/compiler/compilercore/compilercore_assert.h>
#include <mdl/compiler/compilercore/compilercore_fatal.h>
#include <base/system/version/i_version.h>

// This ensures that the version cookies (==@@==) in version.cpp are not optimized out.
static char const* s_pver = MI::VERSION::get_platform_version();

namespace {

/// An allocator based on malloc.
class Allocator_malloc : public mi::base::Interface_implement<mi::base::IAllocator>
{
public:
    void *malloc(mi::Size size)     { return ::malloc(size); }
    void free(void *memory)         { ::free(memory); }
};

} // anonymous

extern "C" MI_DLL_EXPORT
mi::mdl::IMDL *mi_mdl_factory(bool mat_ior_is_varying, mi::mdl::IAllocator *alloc)
{
    if (alloc == NULL) {
        mi::mdl::IAllocator* allocator = new Allocator_malloc;
        mi::mdl::IMDL *res = mi::mdl::initialize(mat_ior_is_varying, allocator);
        allocator->release();
        return res;
    } else {
        return mi::mdl::initialize(mat_ior_is_varying, alloc);
    }
}

// report assertion failure
extern "C"
void mi_report_assertion_failure(
    const char   *exp,      // the expression that failed
    const char   *file,     // file containing the assertion
    unsigned int line)      // line number of assertion in file
{
    mi::mdl::report_assertion_failure(exp, file, line);
}
