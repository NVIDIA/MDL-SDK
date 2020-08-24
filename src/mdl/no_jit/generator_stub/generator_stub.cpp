/******************************************************************************
 * Copyright (c) 2013-2020, NVIDIA CORPORATION. All rights reserved.
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

#include "pch.h"

#include <mi/mdl/mdl_generated_executable.h>

#include <base/system/main/i_assert.h>
#include <mdl/compiler/compilercore/compilercore_allocator.h>

namespace mi {
namespace mdl {

class ICode_generator;
class MDL;
class Jitted_code;

// Default implementation if NO JIT code generator is available.
ICode_generator *create_code_generator_jit(IAllocator *alloc, MDL *mdl)
{
    return NULL;
}

Jitted_code *create_jitted_code_singleton(IAllocator *alloc)
{
    return NULL;
}

void terminate_jitted_code_singleton(Jitted_code *jitted_code)
{
}

class Generated_code_value_layout
    : public Allocator_interface_implement<IGenerated_code_value_layout>
{
public:
    char const* get_layout_data(size_t& size) const;
    void set_layout_data(char const* /*data*/, size_t& /*size*/);
    bool get_strings_mapped_to_ids() const;
    void set_strings_mapped_to_ids(bool /*value*/);
};

char const* Generated_code_value_layout::get_layout_data(size_t& size) const
{
    MI_ASSERT(!"not implemented");
    return NULL;
}

void Generated_code_value_layout::set_layout_data(char const* /*data*/, size_t& /*size*/)
{
    MI_ASSERT(!"not implemented");
}

bool Generated_code_value_layout::get_strings_mapped_to_ids() const
{
    MI_ASSERT(!"not implemented");
    return false;
}

void Generated_code_value_layout::set_strings_mapped_to_ids(bool /*value*/)
{
    MI_ASSERT(!"not implemented");
}

} // mdl
} // mi
