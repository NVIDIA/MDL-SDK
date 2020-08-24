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

#ifndef MDL_COMPILERCORE_THREAD_CONTEXT_H
#define MDL_COMPILERCORE_THREAD_CONTEXT_H 1

#include <mi/mdl/mdl_thread_context.h>
#include "compilercore_allocator.h"
#include "compilercore_messages.h"
#include "compilercore_options.h"

namespace mi {
namespace mdl {

// forward
class IModule;
class IValue;

/// Interface for handling resource restrictions.
class IResource_restriction_handler {
public:
    /// Possible return values for a resource restriction.
    enum Resource_restriction {
        RR_OK = 0,          ///< Valid resource.
        RR_NOT_EXISTANT,    ///< The resource does not exists.
        RR_OUTSIDE_ARCHIVE, ///< The resource is outside the current archive.
    };

    /// Process a referenced resource.
    ///
    /// \param owner    the owner module of the resource
    /// \param res      the URL of a resource
    virtual Resource_restriction process(
        IModule const *owner,
        char const    *res) = 0;
};

/// Implementation of the IThread_context interface.
class Thread_context : public Allocator_interface_implement<IThread_context>
{
    typedef Allocator_interface_implement<IThread_context> Base;
    friend class Allocator_builder;
public:
    /// Access compiler messages of last operation.
    Messages const &access_messages() const MDL_FINAL;

    /// Access compiler options for the invocation.
    ///
    /// Get access to the MDL compiler options. \see mdl_compiler_options
    ///
    /// \note Options set in the thread context will overwrite options set on the compiler
    ///       directly but are not persistent, i.e. only valid during the time this thread
    ///       context is in use.
    ///
    Options_impl const &access_options() const MDL_FINAL;

    /// Access compiler options for the invocation.
    ///
    /// Get access to the MDL compiler options. \see mdl_compiler_options
    ///
    /// \note Options set in the thread context will overwrite options set on the compiler
    ///       directly but are not persistent, i.e. only valid during the time this thread
    ///       context is in use.
    ///
    Options_impl &access_options() MDL_FINAL;

public:
    enum String_buffer_id {
        SBI_MANGLE_BUFFER,
        SBI_DAG_MANGLE_BUFFER
    };

    /// Get a string buffer.
    ///
    /// \param id  the string buffer id
    char const *get_string_buffer(String_buffer_id id) const;

    /// Set a string buffer.
    ///
    /// \param id  the string buffer id
    /// \param s   the content
    ///
    /// \returns the context of the buffer AFTER s was written (i.ew. a copy of s)
    char const *set_string_buffer(String_buffer_id id, char const *s);

    /// Access messages.
    Messages_impl &access_messages_impl();

    /// Clear the compiler messages.
    void clear_messages() { m_msg_list.clear(); }

    /// Get the resource restriction handler if any.
    IResource_restriction_handler *get_resource_restriction_handler() const {
        return m_rrh;
    }

    /// Set the resource restriction handler.
    ///
    /// \param rrh  the resource restriction handler
    void set_resource_restriction_handler(IResource_restriction_handler *rrh) {
        m_rrh = rrh;
    }

    /// Get the front path.
    char const *get_front_path() const {
        return m_front_path.empty() ? NULL : m_front_path.c_str();
    }

    /// Set the front path.
    ///
    /// \param front_path   the new front path
    ///
    /// If non-NULL, this path is added silently in front of all other search paths,
    /// so the entity resolver will first look for an entity here
    void set_front_path(char const *front_path) {
        m_front_path = front_path == NULL ? "" : front_path;
    }

    /// Set the module replacement path.
    ///
    /// \param module_name  an absolute module name
    /// \param file_name    the file name that should be used to resolve the module
    void set_module_replacement_path(
        char const *module_name,
        char const *file_name);

    /// Get the replacement module name if any.
    char const *get_replacement_module_name() const {
        return m_repl_module_name.empty() ? NULL : m_repl_module_name.c_str();
    }

    /// Get the replacement file name if any.
    char const *get_replacement_file_name() const {
        return m_repl_file_name.empty() ? NULL : m_repl_file_name.c_str();
    }

    /// Disable all warnings.
    void disable_all_warnings() { m_all_warnings_are_off = true; }

    /// Return true if all warnings are disabled.
    bool all_warnings_are_off() const { return m_all_warnings_are_off; }

private:
    /// Constructor.
    ///
    /// \param alloc     the allocator
    /// \param options   the compiler options to inherit from
    explicit Thread_context(
        IAllocator         *alloc,
        Options_impl const *options);

private:
    // strings buffers.
    string m_mangle_buffer;
    string m_dag_buffer;

    /// Messages.
    Messages_impl m_msg_list;

    /// Options.
    Options_impl m_options;

    /// The resource restriction handler if any.
    IResource_restriction_handler *m_rrh;

    /// Front path.
    string m_front_path;

    /// Module replacement: absolute module name
    string m_repl_module_name;

    /// Module replacement: file name
    string m_repl_file_name;

    /// If true, disable all warnings.
    bool m_all_warnings_are_off;
};

}  // mdl
}  // mi

#endif
