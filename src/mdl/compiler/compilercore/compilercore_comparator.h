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

#ifndef MDL_COMPILERCORE_COMPARATOR_H
#define MDL_COMPILERCORE_COMPARATOR_H 1

#include <mi/base/handle.h>
#include <mi/mdl/mdl_comparator.h>

#include "compilercore_allocator.h"
#include "compilercore_mdl.h"
#include "compilercore_messages.h"
#include "compilercore_options.h"

namespace mi {
namespace mdl {

class Definition;
class Error_params;
class Err_location;
class Module;
class Thread_context;

/// Implementation of the IMDL_comparator interface.
class MDL_comparator : public Allocator_interface_implement<IMDL_comparator>
{
    typedef Allocator_interface_implement<IMDL_comparator> Base;
    friend class Allocator_builder;

public:
    static char const MESSAGE_CLASS = 'V';

public:
    /// Load an "original" module with a given name.
    ///
    /// \param context      if non-NULL, the thread context for this operation
    /// \param module_name  the absolute module name
    ///
    /// \returns            an interface to the loaded module
    ///
    /// This is basically a thin wrapper around IMDL::load_module(). It loads a MDL module
    /// from the current search path.
    ///
    /// Node that in contrast to IMDL::load_module() no code cache can be provided. This module
    /// is always loaded from the file system.
    IModule const *load_module(
        IThread_context *context,
        char const      *module_name) MDL_FINAL;

    /// Load a "replacement" module with a given name.
    ///
    /// \param context      if non-NULL, the thread context for this operation
    /// \param module_name  the absolute module name
    /// \param file_name    a file location were the replacement module is found
    ///
    /// \returns            an interface to the loaded module
    ///
    /// This methods loads a replacement module, i.e. instead of loading the modules from
    /// the provided search path, it is loaded from the given location. Note that only
    /// the module itself is replaced, any other imports/resources are loaded from the search path.
    /// If necessary, the search path must be adopted.
    ///
    /// Node that in contrast to IMDL::load_module() no code cache can be provided. This module
    /// is always loaded from the file system.
    IModule const *load_replacement_module(
        IThread_context *context,
        char const      *module_name,
        char const      *file_name) MDL_FINAL;

    /// Compare two modules and evaluate, if modB can be used as a replacement for modA.
    ///
    /// \param ctx   a non-NULL thread context
    /// \param modA  the original module
    /// \param modB  the replacement module
    ///
    /// The compare result messages will be written to the thread context.
    /// Any errors there means that modules cannot be replaced.
    void compare_modules(
        IThread_context *ctx,
        IModule const   *modA,
        IModule const   *modB) MDL_FINAL;

    /// Compare two archives and evaluate, if archivB can be used as a replacement for archiveA.
    ///
    /// \param ctx      a non-NULL thread context
    /// \param archivA  the full path to the original archive A
    /// \param archivB  the full path to replacement archive B
    ///
    /// The compare result messages will be written to the thread context.
    /// Any errors there means that archives cannot be replaced.
    void compare_archives(
        IThread_context *ctx,
        char const      *archivA,
        char const      *archivB) MDL_FINAL;

    /// Install a MDL search path helper for all replacement modules/archives.
    ///
    /// \param search_path  the new search path helper to install, takes ownership
    ///
    /// The new search path helper will be released at this interface
    /// life time. Any previously set helper will be released now.
    void install_replacement_search_path(IMDL_search_path *search_path) MDL_FINAL;

    // Access options.
    Options &access_options() MDL_FINAL;

    /// Set an event callback.
    ///
    /// \param cb  the event interface
    void set_event_cb(IMDL_comparator_event *cb) MDL_FINAL;

protected:
    /// Constructor.
    ///
    /// \param alloc     the allocator
    /// \param compiler  the MDL compiler
    MDL_comparator(
        IAllocator *alloc,
        MDL        *compiler);

private:
    /// The MDL compiler.
    mi::base::Handle<mi::mdl::MDL> m_compiler;

    /// The replacement search path if any.
    mi::base::Handle<IMDL_search_path> m_repl_sp;

    /// Archiver options.
    Options_impl m_options;

    /// The event callback if any.
    IMDL_comparator_event *m_cb;
};

/// Compares two modules for equality.
///
/// The comparison is performed on a syntactical level, e.g., the order of unrelated
/// declarations matters (this is important later for function/material/annotation indices in
/// the DAG representation, which are also used by the DB elements). The comparison might be a
/// bit stricter than what is actually required, but we prefer to err on the safe side.
///
/// All positions are ignored. Imported modules are assumed to be equal.
bool equal( const mi::mdl::IModule* module_a, const mi::mdl::IModule* module_b);
    
}  // mdl
}  // mi

#endif
