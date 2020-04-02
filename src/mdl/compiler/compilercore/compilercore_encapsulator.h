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

#ifndef MDL_COMPILERCORE_ENCAPSULATOR_H
#define MDL_COMPILERCORE_ENCAPSULATOR_H 1

#include <mi/base/handle.h>
#include <mi/mdl/mdl_encapsulator.h>

#include "compilercore_allocator.h"
#include "compilercore_errors.h"
#include "compilercore_mdl.h"
#include "compilercore_messages.h"
#include "compilercore_options.h"
#include "compilercore_zip_utils.h"

namespace mi {
namespace mdl {

class IModule;
class IMDL_resource_reader;

class MDL_zip_container_mdle : public MDL_zip_container
{
    friend class Allocator_builder;

public:
    /// Open a container file.
    ///
    /// \param[in]  alloc                   the allocator
    /// \param[in]  path                    the UTF8 encoded archive path
    /// \param[out] err                     error code
    static MDL_zip_container_mdle *open(
        IAllocator                     *alloc,
        char const                     *path,
        MDL_zip_container_error_code  &err);

    /// Destructor
    virtual ~MDL_zip_container_mdle();

    /// Get the stored top level MD5 hash that allows to compare MDLE files without iterating
    /// over the entire content.
    ///
    /// \param[out] hash                    the computed hash
    ///
    /// \return true on success, false otherwise
    bool get_hash(unsigned char hash[16]) const;

private:
    /// Constructor.
    explicit MDL_zip_container_mdle(
        IAllocator  *alloc,
        char const  *path,
        zip_t       *za);
};

/// Implementation of the IEncapsulate_tool interface.
class Encapsulate_tool : public Allocator_interface_implement<IEncapsulate_tool>
{
    typedef Allocator_interface_implement<IEncapsulate_tool> Base;
    friend class Allocator_builder;

public:
    static char const MESSAGE_CLASS = 'E';

public:
    /// Create a new encapsulated mdl file from a given module.
    ///
    /// \param module           Module that should be written to an MDLe file.
    /// \param mdle_name        MDLe file name without path and without extension.
    /// \param dest_path        The path where the created MDLe file should be stored.
    /// \param desc             Contains information about resource resolving and additional
    ///                         user specified data.
    ///
    /// \returns true on success, false of error
    virtual bool create_encapsulated_module(
        IModule const                   *module,
        char const                      *mdle_name,
        char const                      *dest_path,
        Mdle_export_description const   &desc) MDL_FINAL;

    /// Get the content from any file out of an MDLe on the file system.
    ///
    /// \param mdle_path    The MDLe file that contains the requested file.
    /// \param file_name    The file name of the file inside the MDLe.
    ///
    /// \return The file content as in input stream of NULL if the file does not exists in
    ///         the MDLe.
    IInput_stream *get_file_content(
        char const *mdle_path,
        char const *file_name) MDL_FINAL;

    /// Checks the MD5 hashes of all files in the MDLE to identify changes from outside
    ///
    /// \param mdle_path    The MDLe file that contains the requested file.
    ///
    /// \return true when the actual content matches the stored hash.
    bool check_integrity(
        char const *mdle_path) MDL_FINAL;

    /// For each resource, an MD5 hash is stored in the extra fields of the archive files.
    /// This method checks if the content matches the hash to identify changes from outside.
    ///
    /// \param mdle_path        The MDLe file that contains the requested file.
    /// \param file_name        The file name of the file inside the MDLe.
    /// \param[out] out_hash    Will contain the MD5 hash of 'file_name' in case of success.
    ///
    /// \return true when the actual content matches the stored hash.
    bool check_file_content_integrity(
        char const *mdle_path,
        char const *file_name,
        unsigned char out_hash[16]) MDL_FINAL;

    /// Opens an MDLE zip container for access to its file list and the MD5 hashes.
    /// \param mdle_path    The MDLE file to open.
    ///
    /// \return a pointer to the opened file or NULL in case of failure.
    MDL_zip_container_mdle* open_encapsulated_module(char const *mdle_path) MDL_FINAL;

    /// For each resource, an MD5 hash is stored in the extra fields of the archive files.
    /// This method reads this hash for a selected file.
    ///
    /// \param mdle             The opened MDLE that contains the requested file.
    /// \param file_name        The name of the file inside the MDLE.
    /// \param[out] out_hash    The output buffer that will contain the hash in case of success.
    ///
    /// \return true when the operation was successful
    bool get_file_content_hash( 
        MDL_zip_container_mdle* mdle,
        char const *file_name,
        unsigned char out_hash[16]) MDL_FINAL;

    /// Access messages of the last operation.
    Messages const &access_messages() const MDL_FINAL;

    /// Access options.
    ///
    /// Get access to the options.
    ///
    Options &access_options() MDL_FINAL;

protected:
    /// Constructor.
    ///
    /// \param alloc     the allocator
    /// \param compiler  the MDL compiler
    Encapsulate_tool(
        IAllocator *alloc,
        MDL        *compiler);

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

private:

    /// Adds a file uncompressed to the given ZIP archive.
    zip_int64_t add_file_uncompressed(
        zip_t *za, 
        char const *name, 
        zip_source_t *src);

    /// Adds a file uncompressed to the given ZIP archive.
    bool add_file_uncompressed(
        zip_t *za, 
        mi::mdl::IMDL_resource_reader* reader,
        char const *target_name,
        char const *mdle_name,
        vector<Resource_zip_source*>::Type &add_sources,
        unsigned char hash[16]);

    IMDL_resource_reader *get_content_buffer(
        char const *archive_name,
        char const *file_name);

    // This method checks if the content matches the hash to identify changes from outside.
    bool check_file_content_integrity(
        MDL_zip_container_mdle* mdle,
        char const *file_name,
        unsigned char out_hash[16]);


    // Translate zip errors.
    void translate_zip_error(char const *mdle_name, int zip_error);

    // Translate container errors.
    void translate_container_error(
        MDL_zip_container_error_code   err,
        char const                     *archive_name,
        char const                     *file_name);

    // Translate zip errors.
    void translate_zip_error(char const *mdle_name, zip_t *za);

    // Translate zip errors.
    void translate_zip_error(char const *mdle_name, zip_error_t const &ze);

    // Translate zip errors.
    void translate_zip_error(char const *mdle_name, zip_file_t *src);

    /// The compiler.
    MDL *m_compiler;

    /// The message list.
    Messages_impl m_msg_list;

    /// Last message index.
    int m_last_msg_idx;

    /// Options.
    Options_impl m_options;
};

}  // mdl
}  // mi

#endif
