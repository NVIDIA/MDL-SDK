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
/// \file mi/mdl/mdl_comparator.h
/// \brief Interfaces for comparing MDL modules and archives
#ifndef MDL_MDL_COMPARATOR_H
#define MDL_MDL_COMPARATOR_H 1

#include <mi/base/iinterface.h>
#include <mi/base/interface_declare.h>
#include <mi/mdl/mdl_iowned.h>

namespace mi {
namespace mdl {

class IMDL_search_path;
class IModule;
class IThread_context;
class Options;

/// A simple event report callback interface for the comparator tool.
///
/// \note This interface must be implemented by the user application
class IMDL_comparator_event {
public:
    enum Event {
        EV_COMPARING_MODULE,   ///< Comparing two modules.
        EV_COMPARING_EXPORT,   ///< Comparing two exports.
    };

    /// Called when an event is fired.
    ///
    /// \param ev    the event
    /// \param name  if non-NULL, an additional name
    virtual void fire_event(
        Event      ev,
        char const *name) = 0;

    /// Called to report a percentage.
    ///
    /// \param curr    the index of the currently processed element
    /// \param count   number of elements to process
    virtual void percentage(
        size_t curr,
        size_t count) = 0;
};

/// This interface gives access to the MDL comparator tool.
class IMDL_comparator : public
    mi::base::Interface_declare<0x485e0ab3,0x6d3c,0x4561,0x9a,0x2b,0x20,0xee,0x0a,0x3a,0x88,0x6b,
    mi::base::IInterface>
{
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
    virtual IModule const *load_module(
        IThread_context *context,
        char const      *module_name) = 0;

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
    virtual IModule const *load_replacement_module(
        IThread_context *context,
        char const      *module_name,
        char const      *file_name) = 0;

    /// Compare two modules and evaluate, if modB can be used as a replacement for modA.
    ///
    /// \param ctx   a non-NULL thread context
    /// \param modA  the original module
    /// \param modB  the replacement module
    ///
    /// The compare result messages will be written to the thread context.
    /// Any errors there means that modules cannot be replaced.
    virtual void compare_modules(
        IThread_context *ctx,
        IModule const   *modA,
        IModule const   *modB) = 0;

    /// Compare two archives and evaluate, if archivB can be used as a replacement for archiveA.
    ///
    /// \param ctx      a non-NULL thread context
    /// \param archivA  the full path to the original archive A
    /// \param archivB  the full path to replacement archive B
    ///
    /// The compare result messages will be written to the thread context.
    /// Any errors there means that archives cannot be replaced.
    virtual void compare_archives(
        IThread_context *ctx,
        char const      *archivA,
        char const      *archivB) = 0;

    /// Install a MDL search path helper for all replacement modules/archives.
    ///
    /// \param search_path  the new search path helper to install, takes ownership
    ///
    /// The new search path helper will be released at this interface
    /// life time. Any previously set helper will be released now.
    virtual void install_replacement_search_path(IMDL_search_path *search_path) = 0;

    /// Access options.
    virtual Options &access_options() = 0;

    /// Set an event callback.
    ///
    /// \param cb  the event interface
    virtual void set_event_cb(IMDL_comparator_event *cb) = 0;
};

}  // mdl
}  // mi

#endif // MDL_MDL_COMPARATOR_H
