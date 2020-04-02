/******************************************************************************
 * Copyright (c) 2014-2020, NVIDIA CORPORATION. All rights reserved.
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
/// \file mi/mdl/mdl_thread_context.h
/// \brief Interfaces to handle thread contextes in MDL Core
#ifndef MDL_THREAD_CONTEXT_H
#define MDL_THREAD_CONTEXT_H 1

#include <mi/base/iinterface.h>
#include <mi/base/interface_declare.h>

namespace mi {
namespace mdl {

class Messages;
class Options;

/// An interface for handling different thread contexts inside the MDL compiler.
///
/// When the compiler is used from different threads, every thread should have its own
/// context. If no context is provided, the compiler automatically creates one.
class IThread_context : public
    mi::base::Interface_declare<0x98779789,0x92bc,0x4530,0x8b,0xfd,0xc3,0xb2,0xfc,0x1e,0xb7,0x8d,
    mi::base::IInterface>
{
public:
    /// Access compiler messages of last operation that used this context.
    ///
    /// \note When compiling a module, error message are also copied
    ///       into the IModule if one was created.
    virtual Messages const &access_messages() const = 0;

    /// Access compiler options for the next invocation.
    ///
    /// Get access to the MDL compiler options, \see mdl_compiler_options.
    ///
    /// \note Options set in the thread context will overwrite options set on the compiler
    ///       directly but are not persistent, i.e. only valid during the time this thread
    ///       context is in use.
    ///
    virtual Options const &access_options() const = 0;

    /// Access compiler options for the next invocation.
    ///
    /// Get access to the MDL compiler options, \see mdl_compiler_options.
    ///
    /// \note Options set in the thread context will overwrite options set on the compiler
    ///       directly but are not persistent, i.e. only valid during the time this thread
    ///       context is in use.
    ///
    virtual Options &access_options() = 0;
};

}  // mdl
}  // mi

#endif
