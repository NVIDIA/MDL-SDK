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
/// \file mi/mdl/mdl_archiver.h
/// \brief Interfaces for handling MDL archives
#ifndef MDL_ARCHIEVER_H
#define MDL_ARCHIEVER_H 1

#include <mi/base/iinterface.h>
#include <mi/base/interface_declare.h>
#include <mi/mdl/mdl_mdl.h>
#include <mi/mdl/mdl_iowned.h>

namespace mi {
namespace mdl {

class Options;
class Messages;
class ISemantic_version;

/// A simple event report callback interface for the archiver tool.
///
/// \note This interface must be implemented by the user application
class IArchive_tool_event {
public:
    enum Event {
        EV_COMPILING,   ///< A module is compiled.
        EV_DISCOVERED,  ///< A file was discovered and will be added.
        EV_IGNORED,     ///< A File was discovered but will be ignored.
        EV_COMPRESSING, ///< A file is compressed and stored into the archive.
        EV_STORING,     ///< A file is stored into the archive.
        EV_EXTRACTED,   ///< A file was extracted from an archive.
    };

    /// Called when an event is fired.
    ///
    /// \param ev    the event
    /// \param name  if non-NULL, an additional name
    virtual void fire_event(Event ev, char const *name) = 0;
};

/// This Interface describes an export inside the archive manifest.
class IArchive_manifest_export : public Interface_owned
{
public:
    /// Get the (unqualified) export name.
    virtual char const *get_export_name() const = 0;

    /// Get the next export of the same kind or NULL if this was the last one
    /// from the given module.
    virtual IArchive_manifest_export const *get_next() const = 0;
};

/// This Interface describes a value inside the archive manifest.
class IArchive_manifest_value : public Interface_owned
{
public:
    /// Get the value.
    virtual char const *get_value() const = 0;

    /// Get the next value of the same key or NULL if this was the last one
    /// from the given key.
    virtual IArchive_manifest_value const *get_next() const = 0;
};

/// This Interface describes a dependency inside the archive manifest.
class IArchive_manifest_dependency : public Interface_owned
{
public:
    /// Get the archive name this archive depends on.
    virtual char const *get_archive_name() const = 0;

    /// Get the semantic version this archive depends on.
    virtual ISemantic_version const *get_version() const = 0;

    /// Get the next value of the same key or NULL if this was the last one
    /// from the given key.
    virtual IArchive_manifest_dependency const *get_next() const = 0;
};

/// An interface describing an archive MANIFEST.
class IArchive_manifest : public
    mi::base::Interface_declare<0xbbf91310,0x81b2,0x4099,0x94,0xd8,0x25,0x80,0x9f,0xf8,0xfa,0xa6,
    mi::base::IInterface>
{
public:
    /// Export kinds.
    enum Export_kind {
        EK_FUNCTION,        ///< MDL functions.
        EK_MATERIAL,        ///< MDL materials.
        EK_STRUCT,          ///< MDL structs.
        EK_ENUM,            ///< MDL enums.
        EK_CONST,           ///< MDL constants.
        EK_ANNOTATION,      ///< MDL annotations.
        EK_LAST = EK_ANNOTATION
    };

    /// Predefined keys inside an archive MANIFEST.
    ///
    /// Mandatory keys exists always inside a MANIFEST.
    enum Predefined_key {
        PK_MDL              =  0,  ///< Mandatory, MDL version of the archive.
        PK_VERSION          =  1,  ///< Mandatory, semantic archive version.
        PK_DEPENDENCY       =  2,  ///< Mandatory, dependencies of other archives.
        PK_MODULE           =  3,  ///< Mandatory, list of contained modules.
        PK_EX_FUNCTION      =  4,  ///< Mandatory, exported functions.
        PK_EX_MATERIAL      =  5,  ///< Mandatory, exported materials.
        PK_EX_STRUCT        =  6,  ///< Mandatory, exported struct types.
        PK_EX_ENUM          =  7,  ///< Mandatory, exported enum types.
        PK_EX_CONST         =  8,  ///< Mandatory, exported constants.
        PK_EX_ANNOTATION    =  9,  ///< Mandatory, exported annotations.
        PK_AUTHOR           = 10,  ///< Optional, list of authors.
        PK_CONTRIBUTOR      = 11,  ///< Optional, list of contributors.
        PK_COPYRIGHT_NOTICE = 12,  ///< Optional, copyright information of the archive.
        PK_DESCRIPTION      = 13,  ///< Optional, description of the archive.
        PK_CREATED          = 14,  ///< Optional, creation time.
        PK_MODIFIED         = 15,  ///< Optional, modification time.

        PK_FIRST_USER_ID    = 16,  ///< Id of the first user supplied key
    };

    /// Get the name of the archive this manifest belongs too.
    virtual char const *get_archive_name() const = 0;

    /// Get the MDL version of the archive.
    ///
    /// \note This is the highest version over all MDL modules inside the archive
    //        and hence the version an MDL compiler must support to handle this archive.
    virtual IMDL::MDL_version get_mdl_version() const = 0;

    /// Get the semantic version of the archive.
    virtual ISemantic_version const *get_sema_version() const = 0;

    /// Get the first dependency if any.
    virtual IArchive_manifest_dependency const *get_first_dependency() const = 0;

    /// Get the authors list of the archive if any.
    virtual IArchive_manifest_value const *get_opt_author() const = 0;

    /// Get the contributors list of the archive if any.
    virtual IArchive_manifest_value const *get_opt_contributor() const = 0;

    /// Get the copyright notice of the archive if any.
    virtual char const *get_opt_copyrigth_notice() const = 0;

    /// Get the description of the archive if any.
    virtual char const *get_opt_description() const = 0;

    /// Get the created date of the archive if any.
    virtual char const *get_opt_created() const = 0;

    /// Get the modified date of the archive if any.
    virtual char const *get_opt_modified() const = 0;

    /// Get the number of modules inside the archive.
    virtual size_t get_module_count() const = 0;

    /// Get the i'th module name inside the archive.
    ///
    /// \param i  the module index
    ///
    /// \return the absolute module name without the leading '::'
    virtual char const *get_module_name(size_t i) const = 0;

    /// Get the first export of the given kind from the given module.
    ///
    /// \param i     the module index
    /// \param kind  the export kind
    ///
    /// \return the first export entry of the given kind or NULL
    virtual IArchive_manifest_export const *get_first_export(
        size_t      i,
        Export_kind kind) const = 0;

    /// Get the number of all (predefined and user supplied) keys.
    virtual size_t get_key_count() const = 0;

    /// Get the i'th key.
    ///
    /// \param i  the key index
    ///
    /// \return the key
    ///
    /// \note keys with index < PK_FIRST_USER_ID are the predefined keys
    virtual char const *get_key(size_t i) const = 0;

    /// Get the first value from the given key index.
    ///
    /// \param i     the key index
    ///
    /// \return the first value entry or NULL if this key has no value set
    ///
    /// \note if i < PK_FIRST_USER_ID, this is a predefined key. Use the specific
    ///       access functions, get_first_value() will return NULL here
    virtual IArchive_manifest_value const *get_first_value(size_t i) const = 0;

    /// Possible erro codes.
    enum Error_code {
        ERR_OK,                ///< Value set successfully.
        ERR_NULL_ARG,          ///< An argument is NULL.
        ERR_TIME_FORMAT,       ///< The value must be in time format.
        ERR_VERSION_FORMAT,    ///< The value must be in semantic version format.
        ERR_FORBIDDEN,         ///< This key cannot be changed.
        ERR_SINGLE,            ///< This key can only have a single value.
    };

    /// Add a key, value pair. Works for predefined and user keys.
    ///
    /// \param key    the key to set
    /// \param value  the value
    virtual Error_code add_key_value(
        char const *key,
        char const *value) = 0;
};

/// An interface describing an MDL archive.
class IArchive : public
    mi::base::Interface_declare<0x58121074,0x1741,0x42d0,0xb0,0x72,0x17,0x45,0x0b,0x61,0xe0,0xa3,
    mi::base::IInterface>
{
public:
    /// Get the archive name.
    virtual char const *get_archive_name() const = 0;

    /// Get the MANIFEST of an archive.
    virtual IArchive_manifest const *get_manifest() const = 0;
};

/// This is the intercafe to the MDL archive tool, handling various MDL archive tasks.
class IArchive_tool : public
    mi::base::Interface_declare<0x664b8161,0x009b,0x42cf,0xa5,0x6a,0xec,0x60,0x92,0xf8,0xf3,0x28,
    mi::base::IInterface>
{
public:
    /// The name of the option to include only referenced resources.
    #define MDL_ARC_OPTION_ONLY_REFERENCED "only_referenced"

    /// The name of the option to overwrite existing archives.
    #define MDL_ARC_OPTION_OVERWRITE "overwrite"

    /// The name of the option to ignore extra files in the source directory.
    #define MDL_ARC_OPTION_IGNORE_EXTRA_FILES "ignore_extra_files"

    /// The name of the option to specify resource suffixes that should be compressed.
    #define MDL_ARC_OPTION_COMPRESS_SUFFIXES "compress_suffixes"

    /// An entry into the manifest.
    struct Key_value_entry {
        char const *key;
        char const *value;
    };

public:
    /// Create a new MDL archive.
    ///
    /// \param root_path         The path to the root of all files to store into the archive.
    /// \param root_package      The absolute package name of the root package.
    /// \param dest_path         The path where the created archive should be stored.
    /// \param manifest_entries  Extra manifest entries.
    /// \param me_cnt            Number of extra manifest entries.
    ///
    /// \returns true on success, false of error
    virtual IArchive const *create_archive(
        char const            *root_path,
        char const            *root_package,
        char const            *dest_path,
        Key_value_entry const manifest_entries[],
        size_t                me_cnt) = 0;

    /// Create an archive MANIFEST template.
    ///
    /// \param root_path     The path to the root of all files to store into the archive.
    /// \param root_package  The absolute package name of the root package.
    ///
    /// \returns true on success, false of error
    virtual IArchive_manifest const *create_manifest_template(
        char const *root_path,
        char const *root_package) = 0;

    /// Extract an archive to the file system.
    ///
    /// \param archive_path  The archive to extract.
    /// \param dest_path     The path where the extracted files should be stored.
    virtual void extract_archive(
        char const *archive_path,
        char const *dest_path) = 0;

    /// Get the MANIFEST from an archive to the file system.
    ///
    /// \param archive_path  The archive to extract.
    virtual IArchive_manifest const *get_manifest(
        char const *archive_path) = 0;

    /// Get the MANIFEST content from an archive on the file system.
    ///
    /// \param archive_path  The archive to extract.
    virtual IInput_stream *get_manifest_content(
        char const *archive_path) = 0;

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
    virtual IInput_stream *get_file_content(
        char const *archive_path,
        char const *file_name) = 0;

    /// Access archiver messages of last archive operation.
    virtual Messages const &access_messages() const = 0;

    /// Access options.
    ///
    /// Get access to the MDL archiver options.
    ///
    virtual Options &access_options() = 0;

    /// Set an event callback.
    ///
    /// \param cb  the event interface
    virtual void set_event_cb(IArchive_tool_event *cb) = 0;
};

} // mdl
} // mi

/*!
 * \page mdl_archive_tool Options for the MDL archive tool
 *
 * You can configure the MDL archive tool by setting options on the #mi::mdl::Options object
 * returned by #mi::mdl::IArchive_tool::access_options().
 *
 * \section mdl_archive_tool_options MDL archive_tool_options
 *
 *   \anchor mdl_archive_tool_option_only_referenced
 *   - <b>only_referenced:</b> If set to "true", the archive tool will only insert resources
 *     into an archive that are at least referenced by one MDL module inside of the archive.
 *     Default: "false"
 *
 *   \anchor mdl_archive_tool_option_overwrite
 *   - <b>overwrite:</b> If set to "true", the archive tool will overwrite any existing MDL
 *     archive if a new one is created, else the creation will be aborted.
 *     Default: "false"
 *
 *   \anchor mdl_archive_tool_option_ignore_extra_files
 *   - <b>ignore_extra_files:</b> If set to "true", the archive tool will ignore any extra
 *     files that occurs in places above the specified archive root, otherwise the creation
 *     will be aborted-
 *     Default: "true"
 *
 *   \anchor mdl_archive_tool_option_compress_suffixes
 *   - <b>compress_suffixes:</b> This option allows to specify a comma separated list of
 *     file extentions that should be stored uncompressed inside an MDL archive.
 *     Default: ".ies,.mbsdf,.txt,.html"
 */
#endif // MDL_ARCHIEVER_H
