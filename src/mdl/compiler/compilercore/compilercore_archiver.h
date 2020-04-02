/******************************************************************************
 * Copyright (c) 2016-2020, NVIDIA CORPORATION. All rights reserved.
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

#ifndef MDL_COMPILERCORE_ARCHIVER_H
#define MDL_COMPILERCORE_ARCHIVER_H 1

#include <mi/base/handle.h>
#include <mi/mdl/mdl_archiver.h>

#include "compilercore_allocator.h"
#include "compilercore_mdl.h"
#include "compilercore_messages.h"
#include "compilercore_options.h"
#include "compilercore_manifest.h"
#include "compilercore_zip_utils.h"

namespace mi {
namespace mdl {

class IMDL_resource_reader;

class MDL_zip_container_archive : public MDL_zip_container
{
    friend class Allocator_builder;

public:
    /// Open a container file.
    ///
    /// \param[in]  alloc                   the allocator
    /// \param[in]  path                    the UTF8 encoded archive path
    /// \param[out] err                     error code
    /// \param[in]  with_manifest  load and check the manifest
    static MDL_zip_container_archive *open(
        IAllocator                     *alloc,
        char const                     *path,
        MDL_zip_container_error_code  &err,
        bool                            with_manifest = true);

    /// Destructor
    virtual ~MDL_zip_container_archive();

    /// Get the manifest of this archive.
    /// If the manifest was not loaded with open, already, it will be loaded now.
    Manifest const *get_manifest();

private:
    /// Constructor.
    explicit MDL_zip_container_archive(
        IAllocator  *alloc,
        char const  *path,
        zip_t       *za,
        bool         with_manifest);

    /// Get the manifest.
    Manifest *parse_manifest();

    /// The manifest of this archive.
    mi::base::Handle<Manifest const> m_manifest;
};



/// Implementation of the IArchiv Interface.
class Archive : public Allocator_interface_implement<IArchive>
{
    typedef Allocator_interface_implement<IArchive> Base;
    friend class Allocator_builder;
public:
    /// Get the archive name.
    char const *get_archive_name() const MDL_FINAL;

    /// Get the MANIFEST of an archive.
    Manifest const *get_manifest() const MDL_FINAL;

    /// Set the archive name.
    void set_archive_name(char const *name);

    /// Get the MANIFEST of an archive.
    Manifest *get_manifest();

protected:
    /// Constructor.
    ///
    /// \param MDL  the MDL compiler
    explicit Archive(
        MDL *compiler);

private:
    /// The name of the archive.
    string m_archive_name;

    /// The Manifest.
    mi::base::Handle<Manifest> m_manifest;
};

/// Implementation of the IArchive_tool interface.
class Archive_tool : public Allocator_interface_implement<IArchive_tool>
{
    typedef Allocator_interface_implement<IArchive_tool> Base;
    friend class Allocator_builder;

public:
    static char const MESSAGE_CLASS = 'A';

public:
    /// Create a new archive.
    ///
    /// \param root_path     The path to the root of all files to store into the archive.
    /// \param root_package  The absolute package name of the root package.
    /// \param dest_path     The path where the created archive should be stored.
    ///
    /// \returns true on success, false of error
    IArchive const *create_archive(
        char const            *root_path,
        char const            *root_package,
        char const            *dest_path,
        Key_value_entry const manifest_entries[],
        size_t                me_cnt) MDL_FINAL;

    /// Create an archive MANIFEST template.
    ///
    /// \param root_path     The path to the root of all files to store into the archive.
    /// \param root_package  The absolute package name of the root package.
    ///
    /// \returns true on success, false of error
    IArchive_manifest const *create_manifest_template(
        char const *root_path,
        char const *root_package) MDL_FINAL;

    /// Extract an archive to the file system.
    ///
    /// \param archive_path  The archive to extract.
    /// \param dest_path     The path where the extracted files should be stored.
    void extract_archive(
        char const *archive_path,
        char const *dest_path) MDL_FINAL;

    /// Get the MANIFEST from an archive to the file system.
    ///
    /// \param archive_path  The archive to extract.
    Manifest const *get_manifest(
        char const *archive_path) MDL_FINAL;

    /// Get the MANIFEST content from an archive on the file system.
    ///
    /// \param archive_path  The archive to extract.
    IInput_stream *get_manifest_content(
        char const *archive_path) MDL_FINAL;

    /// Get the content from any file out of an archive on the file system.
    ///
    /// \param archive_path  The archive to extract.
    /// \param file_name     The file name of the file inside the archive.
    ///
    /// \return The file content as in input stream of NULL if the file does not exists in
    ///         the archive.
    ///
    /// \note does not allow to read the MANIFEST content, use get_manifest_content()
    ///       for that
    IInput_stream *get_file_content(
        char const *archive_path,
        char const *file_name) MDL_FINAL;

    /// Access archiver messages of last archive operation.
    Messages const &access_messages() const MDL_FINAL;

    /// Access options.
    ///
    /// Get access to the MDL archiver options.
    ///
    Options &access_options() MDL_FINAL;

    /// Set an event callback.
    ///
    /// \param cb  the event interface
    void set_event_cb(IArchive_tool_event *cb) MDL_FINAL;

public:
    /// Creates a new error.
    ///
    /// \param code    the error code
    /// \param params  additional parameters
    void error(int code, Error_params const &params);

    /// Creates a new warning.
    ///
    /// \param code    the error code
    /// \param params  additional parameters
    void warning(int code, Error_params const &params);

    /// Adds a new note to the previous message.
    ///
    /// \param code    the error code
    /// \param params  additional parameters
    void add_note(int code, Error_params const &params);

    /// Copy all messages from the given message list.
    void copy_messages(Messages const &src) {
        m_msg_list.copy_messages(src);
    }

    /// Get the MANIFEST from resource reader.
    ///
    /// \param archive_path  The archive to extract.
    static Manifest *parse_manifest(
        IAllocator           *alloc,
        IMDL_resource_reader *reader);

protected:
    /// Constructor.
    ///
    /// \param alloc     the allocator
    /// \param compiler  the MDL compiler
    Archive_tool(
        IAllocator *alloc,
        MDL        *compiler);

private:
    /// The MDL compiler.
    mi::base::Handle<mi::mdl::MDL> m_compiler;

    /// The message list.
    Messages_impl m_msg_list;

    /// Archiver options.
    Options_impl m_options;

    /// The event callback if any.
    IArchive_tool_event *m_cb;

    /// Last message index.
    int m_last_msg_idx;
};

}  // mdl
}  // mi

#endif
