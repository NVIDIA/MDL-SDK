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
/// \file mi/mdl/mdl_entity_resolver.h
/// \brief Interfaces for resolving and accessing MDL entities
#ifndef MDL_MDL_ENTITY_RESOLVER_H
#define MDL_MDL_ENTITY_RESOLVER_H 1

#include <mi/base/types.h>
#include <mi/base/iinterface.h>
#include <mi/base/interface_declare.h>
#include <mi/mdl/mdl_messages.h>

namespace mi {
namespace mdl {

// forward
class IInput_stream;
class IThread_context;

// Supported UDIM modes for texture resources.
enum UDIM_mode {
    NO_UDIM = 0,
    UM_MARI,   // "<UDIM>"     UDIM (Mari), expands to four digits calculated as 1000+(u+1+v*10)
    UM_ZBRUSH, // "<UVTILE0>"  0-based (Zbrush), expands to "_u0_v0" for the first tile
    UM_MUDBOX, // "<UVTILE1>"  1-based (Mudbox), expands to "_u1_v1" for the first tile
};

/// A resource reader.
///
/// This interface is returned by the entity resolver and allows read-access to resources files
/// (including from MDL archives).
class IMDL_resource_reader : public
    mi::base::Interface_declare<0xa994e230,0x401a,0x4608,0x88,0xa5,0x39,0x02,0x39,0x29,0xf3,0x92,
    mi::base::IInterface>
{
public:
    /// Seek positions.
    enum Position {
        MDL_SEEK_SET = 0,   ///< Beginning of resource.
        MDL_SEEK_CUR = 1,   ///< Current position inside the resource.
        MDL_SEEK_END = 2,   ///< End of resource.
    };

public:
    /// Read a memory block from the resource.
    ///
    /// \param ptr   Pointer to a block of memory with a size of \p size bytes
    /// \param size  Number of bytes to read
    ///
    /// \returns    The total number of bytes successfully read.
    virtual Uint64 read(void *ptr, Uint64 size) = 0;

    /// Get the current offset position in the resource stream.
    virtual Uint64 tell() = 0;

    /// Reposition stream position indicator.
    ///
    /// \param offset  Number of bytes to offset from origin
    /// \param origin  Position used as reference for the offset
    ///
    /// \return true on success
    virtual bool seek(Sint64 offset, Position origin) = 0;

    /// Get the UTF8 encoded file name of the resource on which this reader operates.
    ///
    /// \returns    The name of the resource or NULL.
    ///
    /// \note If this resource is inside an MDL archive, the returned name
    ///       uses the format 'MDL_ARCHIVE_FILENAME:RESOURCE_FILENAME'.
    virtual char const *get_filename() const = 0;

    /// Get the UTF8 encoded absolute MDL resource name on which this reader operates.
    ///
    /// \returns    The absolute MDL URL of the resource or NULL.
    virtual char const *get_mdl_url() const = 0;

    /// Returns the associated hash of this resource.
    ///
    /// \param[out] hash  get the hash value (16 bytes)
    ///
    /// \return true if this resource has an associated hash value, false otherwise
    virtual bool get_resource_hash(unsigned char hash[16]) = 0;
};

/// An interface describing an module import result.
class IMDL_import_result : public
    mi::base::Interface_declare<0xb7b3de9d,0xa9ce,0x4e19,0x93,0x87,0x46,0x13,0x69,0xdb,0xf0,0xef,
    mi::base::IInterface>
{
public:
    /// Return the absolute MDL name of the found entity, or NULL, if the entity could not
    /// be resolved.
    virtual char const *get_absolute_name() const = 0;

    /// Return the OS-dependent file name of the found entity, or NULL, if the entity could not
    /// be resolved.
    virtual char const *get_file_name() const = 0;

    /// Return an input stream to the given entity if found, NULL otherwise.
    virtual IInput_stream *open(IThread_context *ctx) const = 0;
};

/// An interface describing an ordered set of resolved resources.
///
/// While most resources in MDL can be mapped to exactly one entity, some resources
/// like UDIM textures are mapped to a set of entities.
class IMDL_resource_set : public
    mi::base::Interface_declare<0x1f09d3cc,0xd9b8,0x4ab4,0x8c,0xb7,0x75,0x2f,0x03,0x45,0x3a,0x97,
    mi::base::IInterface>
{
public:
    /// Get the MDL URL mask of the ordered set.
    ///
    /// \returns the MDL URL mask of the set
    virtual char const *get_mdl_url_mask() const = 0;

    /// Get the file name mask of the ordered set.
    ///
    /// \returns the file name mask of the set
    ///
    /// \note If this resource is inside an MDL archive, the returned name
    ///       uses the format 'MDL_ARCHIVE_FILENAME:RESOURCE_FILENAME'.
    virtual char const *get_filename_mask() const = 0;

    /// Get the number of resolved entities.
    virtual size_t get_count() const = 0;

    /// Get the i'th MDL URL of the ordered set.
    ///
    /// \param i  the index
    ///
    /// \returns the i'th MDL URL of the set or NULL if the index is out of range.
    virtual char const *get_mdl_url(size_t i) const = 0;

    /// Get the i'th file name of the ordered set.
    ///
    /// \param i  the index
    ///
    /// \returns the i'th file name of the set or NULL if the index is out of range.
    ///
    /// \note If this resource is inside an MDL archive, the returned name
    ///       uses the format 'MDL_ARCHIVE_FILENAME:RESOURCE_FILENAME'.
    virtual char const *get_filename(size_t i) const = 0;

    /// If the ordered set represents an UDIM mapping, returns it, otherwise NULL.
    ///
    /// \param[in]  i  the index
    /// \param[out] u  the u coordinate
    /// \param[out] v  the v coordinate
    ///
    /// \returns true if a mapping is available, false otherwise
    virtual bool get_udim_mapping(size_t i, int &u, int &v) const = 0;

    /// Opens a reader for the i'th entry.
    ///
    /// \param i  the index
    ///
    /// \returns an reader for the i'th entry of the set or NULL if the index is out of range.
    virtual IMDL_resource_reader *open_reader(size_t i) const = 0;

    /// Get the UDIM mode for this set.
    virtual UDIM_mode get_udim_mode() const = 0;

    /// Get the resource hash value for the i'th file in the set if any.
    ///
    /// \param[in]  i     the index
    /// \param[out] hash  the hash value if exists
    ///
    /// \return true if this entry has a hash, false otherwise
    virtual bool get_resource_hash(
        size_t i,
        unsigned char hash[16]) const = 0;
};

/// An interface for resolving MDL entities.
///
/// This interface offers functionality to resolve MDL URLs to file names or
/// archive locations and retrieve input streams to the data.
/// It is used by the MDL core compiler to open imports as well as to open
/// MDL resource files.
///
/// The entity resolver is created by the IMDL::create_entity_resolver() method.
/// It gets the search path IMDL_search_path from the compiler at this moment.
/// Note that the search path of an entity resolver cannot be changed and is not
/// changed when the search path of the compiler is changed, create a new resolver in that case.
class IEntity_resolver : public
    mi::base::Interface_declare<0x02107ce1,0x00ab,0x4ced,0x97,0x3d,0x1f,0x59,0xaf,0x1c,0xb8,0x8f,
    mi::base::IInterface>
{
public:
    /// Resolve a resource file name.
    ///
    /// If \p owner_name and \p owner_file_path are not provided, no relative paths can be
    /// resolved, i.e. \p file_path must be an absolute location.
    ///
    /// \param file_path         The MDL file path of the resource to resolve. In addition, for
    ///                          resources from MDLE files, it is also possible to provide the
    ///                          absolute OS file system path to the MDLE file (with slashes instead
    ///                          of backslashes on Windows), followed by a colon, followed by the
    ///                          relative path inside the MDLE container.
    /// \param owner_file_path   if non-NULL, the file path of the owner
    /// \param owner_name        if non-NULL, the absolute name of the owner
    /// \param pos               if non-NULL, the position of the import statement for error
    ///                          messages
    ///
    /// \return the set of resolved resources or NULL if this name could not be resolved,
    ///         \see IMDL_resource_set for a description
    virtual IMDL_resource_set *resolve_resource_file_name(
        char const     *file_path,
        char const     *owner_file_path,
        char const     *owner_name,
        Position const *pos) = 0;

    /// Resolve a module name.
    ///
    /// \param module_name       the (weak-)relative or absolute MDL module name to resolve
    /// \param owner_file_path   if non-NULL, the file path of the owner
    /// \param owner_name        if non-NULL, the absolute name of the owner
    /// \param pos               if non-NULL, the position of the import statement for error
    ///                          messages
    ///
    /// \return the absolute module name or NULL if this name could not be resolved
    virtual IMDL_import_result *resolve_module(
        char const     *module_name,
        char const     *owner_file_path,
        char const     *owner_name,
        Position const *pos) = 0;

    /// Access messages of last resolver operation.
    virtual Messages const &access_messages() const = 0;
};

} // mdl
} // mi

#endif // MDL_MDL_ENTITY_RESOLVER_H
