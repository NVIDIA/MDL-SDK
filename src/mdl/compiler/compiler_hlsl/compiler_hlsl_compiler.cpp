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

#include "pch.h"

#include <cstdio>

#include <mi/mdl/mdl_streams.h>

#include "mdl/compiler/compilercore/compilercore_malloc_allocator.h"
#include "mdl/compiler/compilercore/compilercore_debug_tools.h"
#include "mdl/compiler/compilercore/compilercore_streams.h"

#include "compiler_hlsl_compilation_unit.h"
#include "compiler_hlsl_printers.h"
#include "compiler_hlsl_messages.h"
#include "compiler_hlsl_compiler.h"

namespace mi {
namespace mdl {
namespace hlsl {

// -------------------------------- Compiler --------------------------------

// Constructor.
Compiler::Compiler(IAllocator *alloc)
: Base(alloc)
, m_builder(alloc)
, m_include_paths(alloc)
{
}

// Destructor.
Compiler::~Compiler()
{
}

// Creates a new empty compilation unit.
Compilation_unit *Compiler::create_unit(
    const char   *fname)
{
    return m_builder.create<Compilation_unit>(get_allocator(), fname);
}

// Create an IOutput_stream standard stream.
mdl::IOutput_stream *Compiler::create_std_stream(Std_stream kind) const
{
    switch (kind) {
    case OS_STDOUT:
        return m_builder.create<File_Output_stream>(get_allocator(), stdout, false);
    case OS_STDERR:
        return m_builder.create<File_Output_stream>(get_allocator(), stderr, false);
    case OS_STDDBG:
        return m_builder.create<Debug_Output_stream>(get_allocator());
    }
    return NULL;
}

// Create a printer.
Printer *Compiler::create_printer(mdl::IOutput_stream *stream) const
{
    return m_builder.create<Printer>(get_allocator(), stream);
}


namespace {

#ifdef DEBUG

mi::mdl::dbg::DebugMallocAllocator dbgMallocAlloc;

#endif  // DEBUG

static Compiler *create_hlsl_compiler(IAllocator *alloc)
{
    Allocator_builder builder(alloc);

    mi::mdl::hlsl::Compiler *p = builder.create<mi::mdl::hlsl::Compiler>(alloc);
    return p;
}

}  // anonymous

// Initializes the HLSL compiler and obtains its interface.
ICompiler *initialize(IAllocator *allocator)
{
    if (allocator != NULL)
        return create_hlsl_compiler(allocator);

    mi::base::Handle<mi::base::IAllocator> alloc(
#ifdef DEBUG
        // does not work with neuray's own allocator, so we use the debug allocator
        // only if MDL uses its own allocation
        &dbgMallocAlloc
#else
        mi::mdl::MallocAllocator::create_instance()
#endif
        );

    return create_hlsl_compiler(alloc.get());
}

}  // hlsl
}  // mdl
}  // mi
