/******************************************************************************
 * Copyright (c) 2018-2020, NVIDIA CORPORATION. All rights reserved.
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

#ifndef MDL_COMPILER_HLSL_COMPILER_H
#define MDL_COMPILER_HLSL_COMPILER_H 1

#include <mi/base/iinterface.h>
#include <mi/base/interface_declare.h>
#include <mi/base/interface_implement.h>

#include "mdl/compiler/compilercore/compilercore_allocator.h"

#include "compiler_hlsl_cc_conf.h"
#include "compiler_hlsl_compilation_unit.h"
#include "compiler_hlsl_printers.h"

namespace mi {
namespace mdl {

class IInput_stream;
class IOutput_stream;

namespace hlsl {

/// Interface of the HLSL compiler.
class ICompiler : public
    mi::base::Interface_declare<0xbcd33b4a,0x7ddc,0x4da1,0xb2,0x53,0x71,0x2e,0x10,0x3e,0x9b,0x9b,
    mi::base::IInterface>
{
    friend class mi::mdl::Allocator_builder;
public:
    /// Creates a new empty compilation unit.
    ///
    /// \param fname  a file name for this new compilation unit
    virtual ICompilation_unit *create_unit(
        char const    *fname) = 0;

    /// Predefined streams kinds.
    enum Std_stream {
        OS_STDOUT,  ///< Mapped to OS specific standard out.
        OS_STDERR,  ///< Mapped to OS specific standard error.
        OS_STDDBG,  ///< Mapped to OS specific standard debug output.
    };

    /// Create an IOutput_stream standard stream.
    ///
    /// \param kind  a standard stream kind
    ///
    /// \return an IOutput_stream stream
    virtual mdl::IOutput_stream *create_std_stream(Std_stream kind) const = 0;

    /// Create a printer.
    ///
    /// \param stream  an output stream the new printer will print to
    ///
    /// Pass an IOutput_stream_colored for colored output.
    virtual IPrinter *create_printer(mdl::IOutput_stream *stream) const = 0;
};

/// Implementation of the HLSL compiler.
class Compiler : public Allocator_interface_implement<ICompiler>
{
    typedef Allocator_interface_implement<ICompiler> Base;
    friend class mi::mdl::Allocator_builder;
public:
    /// Creates a new empty compilation unit.
    ///
    /// \param fname  a file name for this new compilation unit
    Compilation_unit *create_unit(
        const char    *fname) HLSL_FINAL;

    /// Create an IOutput_stream standard stream.
    ///
    /// \param kind  a standard stream kind
    ///
    /// \return an IOutput_stream stream
    mdl::IOutput_stream *create_std_stream(Std_stream kind) const HLSL_FINAL;

    /// Create a printer.
    ///
    /// \param stream  an output stream the new printer will print to
    ///
    /// Pass an IOutput_stream_colored for colored output.
    Printer *create_printer(mdl::IOutput_stream *stream) const HLSL_FINAL;

private:
    /// Get the allocator of this compiler.
    IAllocator *get_allocator() const { return m_builder.get_allocator(); }

private:
    /// Constructor.
    explicit Compiler(IAllocator *alloc);

    /// Destructor.
    ~Compiler();

    // non copyable
    Compiler(Compiler const &) MDL_DELETED_FUNCTION;
    Compiler &operator=(Compiler const &) MDL_DELETED_FUNCTION;

private:
    /// The builder for all created interfaces.
    mutable Allocator_builder m_builder;

    typedef list<string>::Type String_list;

    /// The list of all include paths.
    String_list m_include_paths;
};

/// Initializes the HLSL compiler and obtains its interface.
///
/// \param allocator  the allocator to be used
ICompiler *initialize(IAllocator *allocator);

}  // hlsl
}  // mdl
}  // mi

#endif
