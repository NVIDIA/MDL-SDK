/******************************************************************************
 * Copyright (c) 2015-2025, NVIDIA CORPORATION. All rights reserved.
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

#include "compiler_glsl_version.h"
#include "compiler_glsl_compilation_unit.h"
#include "compiler_glsl_printers.h"
#include "compiler_glsl_messages.h"
#include "compiler_glsl_compiler.h"

namespace mi {
namespace mdl {
namespace glsl {

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

// Compile the given stream.
Compilation_unit const *Compiler::compile(
    GLSL_language lang,
    IInput_stream *s)
{
    // not implemented
    return NULL;
}

// Compile a file.
Compilation_unit const *Compiler::compile(
    GLSL_language lang,
    char const *file_name)
{
    // not implemented
    return NULL;
}

// Preprocess the given stream.
Preprocessor_result const *Compiler::preprocess(
    GLSL_language  lang,
    IInput_stream *in_stream,
    IOutput_stream *out_stream)
{
    // not implemented
    return NULL;
}

// Preprocess a file.
Preprocessor_result const *Compiler::preprocess(
    GLSL_language lang,
    char const    *in_file_name,
    char const    *out_file_name)
{
    // not implemented
    return NULL;
}

// Creates a new empty compilation unit.
Compilation_unit *Compiler::create_unit(
    GLSL_language lang,
    const char   *fname)
{
    return m_builder.create<Compilation_unit>(get_allocator(), lang, fname);
}

// Add an include path.
void Compiler::add_include_path(
    char const *path)
{
    m_include_paths.push_back(string(path, get_allocator()));
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

static Compiler *create_glsl_compiler(IAllocator *alloc)
{
    Allocator_builder builder(alloc);

    mi::mdl::glsl::Compiler *p = builder.create<mi::mdl::glsl::Compiler>(alloc);
    return p;
}

}  // anonymous

// Initializes the GLSL compiler and obtains its interface.
ICompiler *initialize(IAllocator *allocator)
{
    if (allocator != NULL)
        return create_glsl_compiler(allocator);

    mi::base::Handle<mi::base::IAllocator> alloc(
#ifdef DEBUG
        // does not work with neuray's own allocator, so we use the debug allocator
        // only if MDL uses its own allocation
        &dbgMallocAlloc
#else
        mi::mdl::MallocAllocator::create_instance()
#endif
        );

    return create_glsl_compiler(alloc.get());
}

}  // glsl
}  // mdl
}  // mi

