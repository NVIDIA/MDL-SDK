/******************************************************************************
 * Copyright (c) 2011-2020, NVIDIA CORPORATION. All rights reserved.
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
/// \file mi/mdl/mdl.h
/// \brief MDL Core main header, includes all other headers and declares the main factory function.
#ifndef MDL_H
#define MDL_H 1

#include <mi/base/version.h>
#include <mi/base/types.h>
#include <mi/base/iinterface.h>
#include <mi/base/interface_declare.h>

#include <mi/mdl/mdl_annotations.h>
#include <mi/mdl/mdl_archiver.h>
#include <mi/mdl/mdl_assert.h>
#include <mi/mdl/mdl_code_generators.h>
#include <mi/mdl/mdl_comparator.h>
#include <mi/mdl/mdl_declarations.h>
#include <mi/mdl/mdl_definitions.h>
#include <mi/mdl/mdl_entity_resolver.h>
#include <mi/mdl/mdl_expressions.h>
#include <mi/mdl/mdl_fatal.h>
#include <mi/mdl/mdl_generated_code.h>
#include <mi/mdl/mdl_generated_dag.h>
#include <mi/mdl/mdl_generated_executable.h>
#include <mi/mdl/mdl_iowned.h>
#include <mi/mdl/mdl_mdl.h>
#include <mi/mdl/mdl_messages.h>
#include <mi/mdl/mdl_modules.h>
#include <mi/mdl/mdl_names.h>
#include <mi/mdl/mdl_options.h>
#include <mi/mdl/mdl_positions.h>
#include <mi/mdl/mdl_printers.h>
#include <mi/mdl/mdl_serializer.h>
#include <mi/mdl/mdl_statements.h>
#include <mi/mdl/mdl_stdlib_types.h>
#include <mi/mdl/mdl_streams.h>
#include <mi/mdl/mdl_symbols.h>
#include <mi/mdl/mdl_thread_context.h>
#include <mi/mdl/mdl_types.h>
#include <mi/mdl/mdl_values.h>

#if MI_BASE_VERSION_MAJOR != 1
#error "MI_BASE_VERSION_MAJOR is not equal 1, but 1 is required for MDL headers."
#endif

#ifdef MI_PLATFORM_WINDOWS
#define DLL_EXPORT __declspec(dllexport)
#else
#define DLL_EXPORT
#endif

//-----------------------------------------------------------------------------
// Initialize MDL Core
//

extern "C" {

/// Initializes the MDL Core library and obtains the primary MDL interface.
/// This function is the entry point to using the MDL Core API.
///
/// \param alloc  If non-NULL, an allocator interface that will be used for all
///               memory allocations in this compiler.
///               If NULL, a malloc-based allocator will be used.
///
/// \returns    A pointer to the primary MDL interface.
DLL_EXPORT mi::mdl::IMDL *mi_mdl_factory(mi::base::IAllocator *alloc);

} // extern "C"

namespace mi {
/// Common namespace for MDL Core APIs of NVIDIA Advanced Rendering Center GmbH.
namespace mdl {
}  // mdl
}  // mi

#endif // MDL_H

