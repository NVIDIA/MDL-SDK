/***************************************************************************************************
 * Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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
/// \brief Interfaces for resolving and accessing MDL entities.

#ifndef MI_NEURAYLIB_IMDL_ENTITY_RESOLVER_H
#define MI_NEURAYLIB_IMDL_ENTITY_RESOLVER_H

#include <mi/base/interface_declare.h>

namespace mi {

namespace neuraylib {

class IReader;

/** \addtogroup mi_neuray_mdl_types
@{
*/

/// An interface describing an ordered set of resolved resources.
///
/// While most resources in MDL can be mapped to exactly one entity, some resources
/// like UDIM textures are mapped to a set of entities.
class IMdl_resource_set : public
    base::Interface_declare<0x3b2638fd, 0x42f6, 0x4edc, 0xb7, 0x2f, 0x34, 0xab, 0xb, 0xc5, 0x20, 0xdb>
{
public:
    /// Get the number of resolved entities.
    virtual Size get_count() const = 0;

    /// Get the i'th MDL URL of the ordered set.
    ///
    /// \param i  the index
    ///
    /// \returns the i'th MDL URL of the set or NULL if the index is out of range.
    virtual const char *get_mdl_url(Size i) const = 0;

    /// Get the i'th file name of the ordered set.
    ///
    /// \param i  the index
    ///
    /// \returns the i'th file name of the set or NULL if the index is out of range.
    ///
    /// \note If this resource is inside an MDL archive, the returned name
    ///       uses the format 'MDL_ARCHIVE_FILENAME:RESOURCE_FILENAME'.
    virtual const char *get_filename(Size i) const = 0;

    /// If the ordered set represents an UDIM mapping, returns it, otherwise NULL.
    ///
    /// \param[in]  i  the index
    /// \param[out] u  the u coordinate
    /// \param[out] v  the v coordinate
    ///
    /// \returns true if a mapping is available, false otherwise
    virtual bool get_udim_mapping(Size i, Sint32 &u, Sint32 &v) const = 0;

    /// Opens a reader for the i'th entry.
    ///
    /// \param i  the index
    ///
    /// \returns an reader for the i'th entry of the set or NULL if the index is out of range.
    virtual IReader *open_reader(Size i) const = 0;
};

/// An interface for resolving MDL entities.
///
/// This interface offers functionality to resolve MDL URLs to file names or
/// archive locations and retrieve input streams to the data.
///
/// The entity resolver gets the search path from the compiler at the time of creation.
/// Note that the search path of an entity resolver cannot be changed and is not
/// changed when the search path of the compiler is changed, create a new resolver in that case.
class IMdl_entity_resolver : public
    mi::base::Interface_declare<0x2bfa7315,0x58d9,0x4a3c,0x8e,0x19,0x23,0x93,0x1f,0x63,0x3,0xd5>
{
public:

    /// Resolve a resource file name.
    ///
    /// If \p owner_name and \p owner_file_path are not provided, no relative paths can be
    /// resolved, i.e. \p file_path must be an absolute location.
    ///
    /// \param file_path         the MDL file path to resolve
    /// \param owner_file_path   if non-NULL, the file path of the owner
    /// \param owner_name        if non-NULL, the absolute name of the owner
    ///
    /// \return the set of resolved resources or NULL if this name could not be resolved,
    ///         \see IMdl_resource_set for a description
    virtual IMdl_resource_set *resolve_resource_file_name(
        const char     *file_path,
        const char     *owner_file_path,
        const char     *owner_name) = 0;

    /// Opens a resource.
    ///
    /// If \p owner_name and \p owner_file_path are not provided, no relative paths can be
    /// resolved, i.e. \p file_path must be an absolute location.
    ///
    /// \param file_path         the MDL file path to resolve
    /// \param owner_file_path   if non-NULL, the file path of the owner
    /// \param owner_name        if non-NULL, the absolute name of the owner
    ///
    /// \return a resource reader for the requested resource or NULL if it could not be resolved
    virtual IReader *open_resource(
        const char     *file_path,
        const char     *owner_file_path,
        const char     *owner_name) = 0;

    /// Resolve a module name.
    ///
    /// \param name  an MDL module name
    ///
    /// \return the absolute module name or NULL if this name could not be resolved
    virtual const char *resolve_module_name(
        const char *name) = 0;
};

/*@}*/ // end group mi_neuray_mdl_types

} // namespace neuraylib

} // namespace mi

#endif // MI_NEURAYLIB_IMDL_ENTITY_RESOLVER_H
