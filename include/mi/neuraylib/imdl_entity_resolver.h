/***************************************************************************************************
 * Copyright (c) 2019-2022, NVIDIA CORPORATION. All rights reserved.
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

class IMdl_execution_context;
class IMdl_resolved_module;
class IMdl_resolved_resource;
class IReader;

/** \addtogroup mi_neuray_mdl_sdk_misc
@{
*/

/// The entity resolver is used to resolve MDL modules and resources in such modules.
///
/// This interface is used in two ways: (a) to make the resolver of the MDL compiler available to
/// users, and (b) to control how the MDL compiler resolves modules and resources.
///
/// \see #mi::neuraylib::IMdl_configuration::get_entity_resolver(),
///      #mi::neuraylib::IMdl_configuration::set_entity_resolver()
class IMdl_entity_resolver : public
    mi::base::Interface_declare<0xfe6e553a,0x6a1f,0x4300,0xb5,0x46,0x96,0xc8,0xee,0x12,0xcf,0x95>
{
public:
    /// Resolves a module name.
    ///
    /// If \p owner_name and \p owner_file_path are not provided, no relative module names can be
    /// resolved.
    ///
    /// \param module_name       The relative or absolute MDL module name to resolve.
    /// \param owner_file_path   The optional file path of the owner (or \c NULL if not available).
    /// \param owner_name        The absolute name of the owner (or \c NULL if not available).
    /// \param pos_line          The line of the corresponding source code location (or 0 if not
    ///                          available).
    /// \param pos_column        The column of the corresponding source code location (or 0 if not
    ///                          available).
    /// \param context           The execution context which can be used to retrieve messages.
    /// \return                  A description of the resolved module, or \c NULL in case of
    ///                          errors.
    virtual IMdl_resolved_module* resolve_module(
        const char* module_name,
        const char* owner_file_path,
        const char* owner_name,
        Sint32 pos_line,
        Sint32 pos_column,
        IMdl_execution_context* context = 0) = 0;

    /// Resolves a resource file path.
    ///
    /// If \p owner_name and \p owner_file_path are not provided, no relative paths can be resolved.
    /// The method can also be used to resolve files in an MDLE, e.g, to get a resource set for
    /// embedded UDIM textures.
    ///
    /// \param file_path         The MDL file path of the resource to resolve. In addition, for
    ///                          resources from MDLE files, it is also possible to provide the
    ///                          absolute OS file system path to the MDLE file (with slashes instead
    ///                          of backslashes on Windows), followed by a colon, followed by the
    ///                          relative path inside the MDLE container.
    /// \param owner_file_path   The optional file path of the owner (or \c NULL if not available).
    /// \param owner_name        The absolute name of the owner (or \c NULL if not available).
    /// \param pos_line          The line of the corresponding source code location (or 0 if not
    ///                          available).
    /// \param pos_column        The column of the corresponding source code location (or 0 if not
    ///                          available).
    /// \param context           The execution context which can be used to retrieve messages.
    /// \return                  A description of the resolved resource, or \c NULL in case of
    ///                          errors.
    virtual IMdl_resolved_resource* resolve_resource(
        const char* file_path,
        const char* owner_file_path,
        const char* owner_name,
        Sint32 pos_line,
        Sint32 pos_column,
        IMdl_execution_context* context = 0) = 0;
};

/// Supported uvtile modes for resources.
///
/// For light profiles and BSDF measurements only #mi::neuraylib::UVTILE_MODE_NONE is valid.
///
/// \see #mi::neuraylib::IImage::reset_file() for details about the different modes.
enum Uvtile_mode {
    UVTILE_MODE_NONE         = 0, ///< No uvtile mode.
    UVTILE_MODE_UDIM         = 1, ///< The UDIM uvtile mode.
    UVTILE_MODE_UVTILE0      = 2, ///< The UVTILE0 uvtile mode.
    UVTILE_MODE_UVTILE1      = 3, ///< The UVTILE1 uvtile mode.
    UVTILE_MODE_FORCE_32_BIT = 0xffffffffU
};

/// Describes a resolved module (or a failed attempt).
class IMdl_resolved_module : public
    mi::base::Interface_declare<0xd725c3bb,0xd34d,0x4a1a,0x93,0x5d,0xa3,0x96,0x53,0x9f,0xb1,0x76>
{
public:
    /// Returns the MDL name of the module.
    virtual const char* get_module_name() const = 0;

    /// Returns the absolute resolved filename of the module.
    virtual const char* get_filename() const = 0;

    /// Returns a reader for the module.
    ///
    /// The reader does \em need to support absolute access or recorded access.
    virtual IReader* create_reader() const = 0;
};

/// Describes an ordered set of resolved resources elements.
///
/// While most resources in MDL can be mapped to exactly one entity, some resources like uvtile
/// textures are mapped to a set of entities.
class IMdl_resolved_resource_element : public
    mi::base::Interface_declare<0x0c49fcd6,0xc675,0x4ca5,0xbf,0xae,0xb1,0x59,0xd9,0x75,0x5f,0xe2>
{
public:
    /// Returns the frame number of this element.
    ///
    /// \return   Always 0 for textures without frame index, and for light profiles and BSDF
    ///           measurements.
    virtual Size get_frame_number() const = 0;

    /// Returns the number of resources for this element.
    virtual Size get_count() const = 0;

    /// Returns the MDL file path of the \p i -th resource, or \c NULL if the index is out of range.
    ///
    /// \see #mi::neuraylib::IMdl_resolved_resource::get_mdl_file_path_mask()
    virtual const char* get_mdl_file_path( Size i) const = 0;

    /// Returns the absolute resolved filename of the \p i -th resource, or \c NULL if the index
    /// is out of range.
    ///
    /// \note If this resource is located inside a container (an MDL archive or MDLE), the returned
    ///       string is a concatenation of the container filename, a colon, and the container member
    ///       name.
    ///
    /// \see #mi::neuraylib::IMdl_resolved_resource::get_filename_mask()
    virtual const char* get_filename( Size i) const = 0;

    /// Returns a reader for the \p i -th resource (or \c NULL if the index is out of range).
    ///
    /// The reader needs to support absolute access.
    virtual IReader* create_reader( Size i) const = 0;

    /// Returns the resource hash value for the \p i -th resource.
    ///
    /// \return The hash value of the \p i -th resource, or a zero-initialized value if the hash
    ///         value is unknown or the index is out of range.
    virtual base::Uuid get_resource_hash( Size i) const = 0;

    /// Returns the u and v tile indices for the \p i -th resource.
    ///
    /// \param[in]  i  The index of the resource.
    /// \param[out] u  The u coordinate of the resource.
    /// \param[out] v  The v coordinate of the resource.
    /// \return        \c true if the uvtile mode is not #mi::neuraylib::UVTILE_MODE_NONE and \p i is
    ///                in range, \c false otherwise (and the output values are undefine).
    virtual bool get_uvtile_uv( Size i, Sint32& u, Sint32& v) const = 0;
};

/// Describes an ordered set of resolved resources (or a failed attempt).
///
/// While most resources in MDL can be mapped to exactly one element, some resources like animated
/// textures are mapped to a set of elements.
class IMdl_resolved_resource : public
    mi::base::Interface_declare<0x650cbe23,0xed44,0x4c2f,0x9f,0xbc,0x3b,0x64,0x4a,0x08,0x15,0xa1>
{
public:
    /// Indicates whether this resource has a sequence marker.
    ///
    /// The return value \c false implies that there is a single frame with frame number 0.
    virtual bool has_sequence_marker() const = 0;

    /// Returns the uvtile mode for this resource.
    ///
    /// The return value \c false implies that there is a single uv-tile (per frame) with u- and v-
    /// coordinates of 0.
    ///
    /// \return   Always #mi::neuraylib::UVTILE_MODE_NONE for light profiles and BSDF measurements.
    virtual Uvtile_mode get_uvtile_mode() const = 0;

    /// Returns the MDL file path mask for this resource.
    ///
    /// The MDL file path mask is identical to the MDL file path, except that it contains the uvtile
    /// marker if the uvtile mode is not #mi::neuraylib::UVTILE_MODE_NONE.
    ///
    /// \see #mi::neuraylib::IMdl_resolved_resource_element::get_mdl_file_path(),
    ///      #get_uvtile_mode()
    virtual const char* get_mdl_file_path_mask() const = 0;

    /// Returns the absolute resolved filename mask for this resource.
    ///
    /// The filename mask is identical to the filename, except that it contains the uvtile marker if
    /// the uvtile mode is not #mi::neuraylib::UVTILE_MODE_NONE.
    ///
    /// \note If this resource is located inside a container (an MDL archive or MDLE), the returned
    ///       string is a concatenation of the container filename, a colon, and the container member
    ///       name.
    ///
    /// \see #mi::neuraylib::IMdl_resolved_resource_element::get_filename(),
    ///      #get_uvtile_mode()
    virtual const char* get_filename_mask() const = 0;

    /// Returns the number of elements of the resolved resource.
    virtual Size get_count() const = 0;

    /// Returns the \p i -th element of the resolved resource, or \c NULL if \p i is out of bounds.
    ///
    /// Resource elements are sorted by increasing frame numbers.
    virtual const IMdl_resolved_resource_element* get_element( Size i) const = 0;
};

/*@}*/ // end group mi_neuray_mdl_sdk_misc

} // namespace neuraylib

} // namespace mi

#endif // MI_NEURAYLIB_IMDL_ENTITY_RESOLVER_H
