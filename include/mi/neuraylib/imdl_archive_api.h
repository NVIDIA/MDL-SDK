/***************************************************************************************************
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
 **************************************************************************************************/
/// \file
/// \brief      API component that gives access to MDL archive functionality

#ifndef MI_NEURAYLIB_IMDL_ARCHIVE_API_H
#define MI_NEURAYLIB_IMDL_ARCHIVE_API_H

#include <mi/base/interface_declare.h>

namespace mi {

class IArray;

namespace neuraylib {

class IManifest;
class IReader;

/** \addtogroup mi_neuray_mdl_types
@{
*/

/// This API component provides functionality related to MDL archives.
class IMdl_archive_api : public
    mi::base::Interface_declare<0x4b41b483,0xdb0b,0x4658,0xaf,0x65,0x64,0xb1,0xd6,0x9d,0x26,0xb0>
{
public:
    /// Creates an MDL archive given a directory.
    ///
    /// \param directory         The contents of this directory will be packed into the archive.
    ///                          Logically, the directory needs to be on the same level as a
    ///                          directory of the search path, i.e., packages in the MDL archive
    ///                          name are represented as empty sub-directories of the given
    ///                          directory.
    /// \param archive           The filename of the MDL archive to be created.
    /// \param manifest_fields   A static or dynamic array of structs of type \c "Manifest_field"
    ///                          which holds fields with optional or user-defined keys to be added
    ///                          to the manifest. The struct has two members, \c "key" and
    ///                          \c "value", both of type \c "String". \c NULL is treated like an
    ///                          empty array.
    /// \return
    ///                          -  0: Success.
    ///                          - -1: Invalid parameters (\c NULL pointer).
    ///                          - -2: \p archive does not end in \c ".mdr".
    ///                          - -3: An array element of \p manifest_fields or a struct member
    ///                                of one of the array elements has an incorrect type.
    ///                          - -4: Failed to create the archive.
    ///                          - -5: Archive creation is not supported with an external entity
    ///                                resolver (see
    ///                                #mi::neuraylib::IMdl_configuration::set_entity_resolver()).
    virtual Sint32 create_archive(
        const char* directory, const char* archive, const IArray* manifest_fields) = 0;

    /// Unpacks an MDL archive into a given directory.
    ///
    /// \param archive     The filename of the MDL archive to be extracted.
    /// \param directory   The directory into which the contents of the MDL archive will be
    ///                    extracted.
    /// \return
    ///                    -  0: Success.
    ///                    - -1: Invalid parameters (\c NULL pointer).
    ///                    - -2: Failure.
    virtual Sint32 extract_archive( const char* archive, const char* directory) = 0;

    /// Returns the manifest for an MDL archive, or \c NULL in case of failure.
    virtual const IManifest* get_manifest( const char* archive) = 0;

    /// Returns an arbitrary file from an MDL archive.
    ///
    /// \note This method can not be used to obtain the manifest, use #get_manifest() instead.
    ///
    /// \note Although the returned reader supports random access this operation is slow in
    ///       compressed files. If you need a lot of fast random access operations it might be
    ///       beneficial to buffer the entire file content.
    ///
    /// \param archive     The filename of the MDL archive.
    /// \param filename    The name of the file inside the MDL archive.
    /// \return            A reader to the file, or \c NULL in case of failures.
    virtual IReader* get_file( const char* archive, const char* filename) = 0;

    /// Returns an arbitrary file from an MDL archive.
    ///
    /// \note This method can not be used to obtain the manifest, use #get_manifest() instead.
    ///
    /// \note Although the returned reader supports random access this operation is slow in
    ///       compressed files. If you need a lot of fast random access operations it might be
    ///       beneficial to buffer the entire file content.
    ///
    /// \param filename    The name of the archive followed by a colon followed by the name of
    ///                    the file inside the archive
    ///                    (e. g. my_archive.mdr:my_package/my_file.mdl).
    /// \return            A reader to the file, or \c NULL in case of failures.
    virtual IReader* get_file(const char* filename) = 0;

    /// Sets the file types to be compressed in archives.
    ///
    /// \param extensions    A comma-separated list of file name extensions. Files with a matching
    ///                      extension will be compressed when archives are created. Independent of
    ///                      this setting here, \c .mdl files will always be compressed, and
    ///                      textures as defined in [\ref MDLLS] will never be compressed.
    virtual Sint32 set_extensions_for_compression( const char* extensions) = 0;

    /// Returns the file types to be compressed in archives.
    virtual const char* get_extensions_for_compression() const = 0;
};

/// Represents the manifest in an MDL archive.
class IManifest : public
    mi::base::Interface_declare<0x9849828e,0xc383,0x4b6b,0x9f,0x49,0xdf,0xf0,0x1f,0xc7,0xe8,0xd7>
{
public:
    /// Returns the number of fields.
    virtual Size get_number_of_fields() const = 0;

    /// Returns the key of the \p index -th field, or \c NULL if \p index is out of bounds.
    virtual const char* get_key( Size index) const = 0;

    /// Returns the value of the \p index -th field, or \c NULL if \p index is out of bounds.
    virtual const char* get_value( Size index) const = 0;

    /// Returns the number of fields with the given key.
    virtual Size get_number_of_fields( const char* key) const = 0;

    /// Returns the value of the \p index -th field with the given key, or \c NULL if \p index is
    /// out of bounds.
    virtual const char* get_value( const char* key, Size index) const = 0;
};

/*@}*/ // end group mi_neuray_mdl_types

} // namespace neuraylib

} // namespace mi

#endif // MI_NEURAYLIB_IMDL_ARCHIVE_API_H
