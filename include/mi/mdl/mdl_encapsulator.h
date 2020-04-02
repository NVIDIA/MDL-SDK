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
/// \file mi/mdl/mdl_encapsulator.h
/// \brief Interfaces for handling encapsulated MDL files.
#ifndef MDL_ENCAPSULATOR_H
#define MDL_ENCAPSULATOR_H 1

#include <mi/base/iinterface.h>
#include <mi/base/interface_declare.h>

namespace mi {
namespace mdl {

class Messages;
class IMDL_exporter_resource_callback;
class IMDL_resource_reader;
class IModule;
class IInput_stream;
class ISemantic_version;
class Options;
class MDL_zip_container_mdle;

/// Interface to that map that relates the MDLE resource paths to the original data.
class IEncapsulate_tool_resource_collector
{
public:

    /// Number of resources files to encapsulate.
    virtual size_t get_resource_count() const = 0;

    /// Get the resource path that should be used in the MDLE main module.
    virtual char const *get_mlde_resource_path(
        size_t index) const = 0;

    /// Get a stream reader interface that gives access to the requested resource data.
    virtual mi::mdl::IMDL_resource_reader *get_resource_reader(
        size_t index) const = 0;

    // Get a stream reader interface that gives access to the requested addition data file.
    virtual mi::mdl::IMDL_resource_reader *get_additional_data_reader(
        char const* path) = 0;
};

/// This is the interface to the encapsulated MDL tool.
class IEncapsulate_tool : public
    mi::base::Interface_declare<0xf5c07301,0x2bca,0x48ad,0xaf,0xa6,0x8a,0xd1,0x8e,0xdb,0x1d,0x2a,
    mi::base::IInterface>
{
public:
 
    // Contains information about resource resolving and additional and user specified data.
    struct Mdle_export_description {
        IMDL_exporter_resource_callback             *resource_callback;
        IEncapsulate_tool_resource_collector        *resource_collector;
        size_t                                       additional_file_count;
        char const *const                           *additional_file_source_paths;
        char const *const                           *additional_file_target_paths;
        char const                                  *authoring_tool_name_and_version;
    };

    /// The name of the option to overwrite existing MDLE files.
    #define MDL_ENCAPS_OPTION_OVERWRITE "overwrite"

    /// Create a new encapsulated mdl file from a given module.
    ///
    /// \param module           Module that should be written to an MDLE file.
    /// \param mdle_name        MDLE file name without path and without extension.
    /// \param dest_path        The path where the created MDLE file should be stored.
    /// \param desc             Contains information about resource resolving and additional
    ///                         as well as user specified data.
    ///
    /// \returns true on success, false of error
    virtual bool create_encapsulated_module(
        IModule const                   *module,
        char const                      *mdle_name,
        char const                      *dest_path,
        Mdle_export_description const   &desc) = 0;

    /// Get the content from any file out of an MDLE on the file system.
    ///
    /// \param mdle_path    The MDLE file that contains the requested file.
    /// \param file_name    The file name of the file inside the MDLE.
    ///
    /// \return The file content as in input stream of NULL if the file does not exists in
    ///         the MDLE.
    virtual IInput_stream *get_file_content(
        char const *mdle_path,
        char const *file_name) = 0;

    /// Checks the MD5 hashes of all files in the MDLE to identify changes from outside
    ///
    /// \param mdle_path    The MDLE file that contains the requested file.
    ///
    /// \return true when the actual content matches the stored hash.
    virtual bool check_integrity(
        char const *mdle_path) = 0;

    /// For each resource, an MD5 hash is stored in the extra fields of the archive files.
    /// This method checks if the content matches the hash to identify changes from outside.
    ///
    /// \param mdle_path        The MDLE file that contains the requested file.
    /// \param file_name        The name of the file inside the MDLE.
    /// \param[out] out_hash    Will contain the MD5 hash of 'file_name' in case of success.
    ///
    /// \return true when the actual content matches the stored hash.
    virtual bool check_file_content_integrity(
        char const *mdle_path,
        char const *file_name,
        unsigned char out_hash[16]) = 0;

    /// Opens an MDLE zip container for access to its file list and MD5 hashes.
    /// \param mdle_path    The MDLE file to open.
    ///
    /// \return a pointer to the opened file or NULL in case of failure.
    virtual MDL_zip_container_mdle* open_encapsulated_module(char const *mdle_path) = 0;

    /// For each resource, an MD5 hash is stored in the extra fields of the archive files.
    /// This method reads this hash for a selected file.
    ///
    /// \param mdle             The opened MDLE that contains the requested file.
    /// \param file_name        The name of the file inside the MDLE.
    /// \param[out] out_hash    The output buffer that will contain the hash in case of success.
    ///
    /// \return true when the operation was successful
    virtual bool get_file_content_hash(
        MDL_zip_container_mdle* mdle,
        char const *file_name,
        unsigned char out_hash[16]) = 0;

    /// Access messages of last operation.
    virtual Messages const &access_messages() const = 0;

    /// Access options.
    ///
    /// Get access to the options.
    ///
    virtual Options &access_options() = 0;
};

} // mdl
} // mi

#endif // MDL_ENCAPSULATOR_H
