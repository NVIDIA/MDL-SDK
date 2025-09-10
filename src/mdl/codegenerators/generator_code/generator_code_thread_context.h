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

#ifndef MDL_GENERATOR_CODE_THREAD_CONTEXT_H
#define MDL_GENERATOR_CODE_THREAD_CONTEXT_H 1

#include <mi/mdl/mdl_code_generators.h>

#include "mdl/compiler/compilercore/compilercore_cc_conf.h"
#include "mdl/compiler/compilercore/compilercore_allocator.h"
#include "mdl/compiler/compilercore/compilercore_messages.h"
#include "mdl/compiler/compilercore/compilercore_options.h"

namespace mi {
namespace mdl {

/// Implementation of the ICode_genenator_thread_context interface.
class Code_generator_thread_context :
    public Allocator_interface_implement<ICode_generator_thread_context>
{
    typedef Allocator_interface_implement<ICode_generator_thread_context> Base;
    friend class Allocator_builder;
public:
    /// Access code generator messages of last operation.
    Messages_impl const &access_messages() const MDL_FINAL;

    /// Access code generator messages of last operation.
    Messages_impl &access_messages() MDL_FINAL;

    /// Access code generator options for the invocation.
    ///
    /// \note Options set in the thread context will overwrite options set on the backend
    ///       directly but are not persistent, i.e. only valid during the time this thread
    ///       context is in use.
    ///
    Options_impl const &access_options() const MDL_FINAL;

    /// Access code generator options for the invocation.
    ///
    /// \note Options set in the thread context will overwrite options set on the backend
    ///       directly but are not persistent, i.e. only valid during the time this thread
    ///       context is in use.
    ///
    Options_impl &access_options() MDL_FINAL;

public:
    /// Clear the compiler messages.
    void clear_messages() { m_msg_list.clear(); }

private:
    /// Constructor.
    ///
    /// \param alloc     the allocator
    /// \param options   the compiler options to inherit from
    explicit Code_generator_thread_context(
        IAllocator *alloc,
        Options_impl const *options);

private:
    /// Messages.
    Messages_impl m_msg_list;

    /// Options.
    Options_impl m_options;
};

}  // mdl
}  // mi

#endif // MDL_GENERATOR_CODE_TREAD_CONTEXT_H
