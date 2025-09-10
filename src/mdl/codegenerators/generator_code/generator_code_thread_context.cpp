/******************************************************************************
 * Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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

#include <cstring>

#include <mi/mdl/mdl_values.h>
#include <mi/mdl/mdl_symbols.h>
#include <mi/mdl/mdl_generated_code.h>

#include "mdl/compiler/compilercore/compilercore_assert.h"
#include "mdl/compiler/compilercore/compilercore_tools.h"
#include "generator_code_thread_context.h"

namespace mi {
namespace mdl {

// Constructor.
Code_generator_thread_context::Code_generator_thread_context(
    IAllocator         *alloc,
    Options_impl const *options)
: Base(alloc)
, m_msg_list(alloc, /*owner_fname=*/"")
, m_options(alloc, *options)
{
}

/// Access code generator messages of last operation.
Messages_impl const &Code_generator_thread_context::access_messages() const
{
    return m_msg_list;
}

// Access code generator messages of last operation.
Messages_impl &Code_generator_thread_context::access_messages()
{
    return m_msg_list;
}

// Access code generator options for the invocation.
Options_impl const &Code_generator_thread_context::access_options() const
{
    return m_options;
}

// Access code generator options for the invocation.
Options_impl &Code_generator_thread_context::access_options()
{
    return m_options;
}

} // mdl
} // mi
