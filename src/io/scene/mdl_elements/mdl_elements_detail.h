/***************************************************************************************************
 * Copyright (c) 2012-2020, NVIDIA CORPORATION. All rights reserved.
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
/// \brief      Module-internal low-level utilities related to MDL scene
///             elements in namespace MI::MDL::DETAIL.

#ifndef IO_SCENE_MDL_ELEMENTS_MDL_ELEMENTS_DETAIL_H
#define IO_SCENE_MDL_ELEMENTS_MDL_ELEMENTS_DETAIL_H

#include "i_mdl_elements_type.h"

#include <mi/mdl/mdl_types.h>
#include <mi/mdl/mdl_values.h>
#include <mi/mdl/mdl_generated_dag.h>
#include <mi/mdl/mdl_entity_resolver.h>
#include <mi/mdl/mdl_streams.h>
#include <mi/neuraylib/ireader.h>

#include <string>
#include <boost/unordered_map.hpp>

#include <base/data/db/i_db_tag.h>
#include <io/image/image/i_image.h>
#include <io/scene/dbimage/i_dbimage.h>

namespace mi { namespace mdl {
    class IArchive_tool;
    class IMDL_resource_reader;
} }

namespace MI {

namespace DB { class Transaction; }

namespace MDL {

namespace DETAIL {

/// Searches a thumbnail image for the given mdl definition by considering both the
/// "thumbnail"-annotation and the 'module_name.definition_name.ext' convention.
///
/// \param module_filename      resolved module file name
/// \param module_name          MDL name of the module
/// \param def_simple_name      MDL simple name of the definitionA
/// \param annotations          annotations of the mdl definition
/// \param archive_tool         to resolve thumbnails in archives
/// \return the resolved file name of the thumbnail or an empty string if none could be found
std::string lookup_thumbnail(
    const std::string& module_filename,
    const std::string& module_name,
    const std::string& def_simple_name,
    const IAnnotation_block* annotations,
    mi::mdl::IArchive_tool* archive_tool);

/// Returns the DB tag corresponding to an MDL resource.
///
/// For values with known tags, the tag is returned immediately. Otherwise, the string from the
/// value is used to search the resource according to search rules. If no matching file is found, an
/// invalid tag is returned. If found, but not yet in the DB, it is loaded into the DB first.
/// Finally, its tag is returned.
///
/// This method uses filename resolving as specified in the MDL specification. The method is
/// intended to be used for MDL resources in defaults/material bodies, not for regular arguments.
/// The found resources are always shared.
///
/// \param transaction          The DB transaction to use.
/// \param value                The MDL resource to convert into a tag.
/// \param module_filename      Absolute filename of the MDL module (using OS-specific separators),
///                             or \c NULL for string-based modules.
/// \param module_name          The fully-qualified MDL module name, or \c NULL for import of
///                             resources (absolute file paths only) without module context.
/// \return                     The tag for the MDL resource (invalid in case of failures).
DB::Tag mdl_resource_to_tag(
    DB::Transaction* transaction,
    const mi::mdl::IValue_resource* value,
    const char* module_filename,
    const char* module_name);

/// Returns the DB tag corresponding to an MDL texture.
///
/// \see #mdl_resource_to_tag(). The found resources are always shared.
DB::Tag mdl_texture_to_tag(
    DB::Transaction* transaction,
    const mi::mdl::IValue_texture* value,
    const char* module_filename,
    const char* module_name);

/// Returns the DB tag corresponding to an MDL texture.
///
/// \see #mdl_resource_to_tag()
DB::Tag mdl_texture_to_tag(
    DB::Transaction* transaction,
    const char* file_path,
    const char* module_filename,
    const char* module_name,
    bool shared,
    mi::Float32 gamma);

/// Returns the DB tag corresponding to an MDL light profile.
///
/// \see #mdl_resource_to_tag(). The found resources are always shared.
DB::Tag mdl_light_profile_to_tag(
    DB::Transaction* transaction,
    const mi::mdl::IValue_light_profile* value,
    const char* module_filename,
    const char* module_name);

/// Returns the DB tag corresponding to an MDL light profile.
///
/// \see #mdl_resource_to_tag()
DB::Tag mdl_light_profile_to_tag(
    DB::Transaction* transaction,
    const char* file_path,
    const char* module_filename,
    const char* module_name,
    bool shared);

/// Returns the DB tag corresponding to an MDL BSDF measurement.
///
/// \see #mdl_resource_to_tag(). The found resources are always shared.
DB::Tag mdl_bsdf_measurement_to_tag(
    DB::Transaction* transaction,
    const mi::mdl::IValue_bsdf_measurement* value,
    const char* module_filename,
    const char* module_name);

/// Returns the DB tag corresponding to an MDL BSDF measurement.
///
/// \see #mdl_resource_to_tag()
DB::Tag mdl_bsdf_measurement_to_tag(
    DB::Transaction* transaction,
    const char* file_path,
    const char* module_filename,
    const char* module_name,
    bool shared);

/// Generates a name that is unique in the DB (at least from the given transaction's point of view).
std::string generate_unique_db_name( DB::Transaction* transaction, const char* prefix);

/// Converts mi::mdl::IType_alias modifiers into MI::MDL::IType_alias modifiers.
inline mi::Uint32 mdl_modifiers_to_int_modifiers( mi::Uint32 modifiers)
{
    ASSERT( M_SCENE, (modifiers
        & ~mi::mdl::IType_alias::MK_UNIFORM
        & ~mi::mdl::IType_alias::MK_VARYING) == 0);

    mi::Uint32 result = 0;
    if( modifiers & mi::mdl::IType_alias::MK_UNIFORM) result |= IType_alias::MK_UNIFORM;
    if( modifiers & mi::mdl::IType_alias::MK_VARYING) result |= IType_alias::MK_VARYING;
    return result;
}

/// Converts MI::MDL::IType_alias modifiers into mi::mdl::IType_alias modifiers.
inline mi::Uint32 int_modifiers_to_mdl_modifiers( mi::Uint32 modifiers)
{
    ASSERT( M_SCENE, (modifiers
        & ~IType_alias::MK_UNIFORM
        & ~IType_alias::MK_VARYING) == 0);

    mi::Uint32 result = 0;
    if( modifiers & IType_alias::MK_UNIFORM) result |= mi::mdl::IType_alias::MK_UNIFORM;
    if( modifiers & IType_alias::MK_VARYING) result |= mi::mdl::IType_alias::MK_VARYING;
    return result;
}

/// Converts predefined mi::mdl::IType_enum IDs into predefined MI::MDL::IType_enum IDs.
inline IType_enum::Predefined_id mdl_enum_id_to_int_enum_id(
    mi::mdl::IType_enum::Predefined_id enum_id)
{
    switch( enum_id) {
        case mi::mdl::IType_enum::EID_USER:           return IType_enum::EID_USER;
        case mi::mdl::IType_enum::EID_TEX_GAMMA_MODE: return IType_enum::EID_TEX_GAMMA_MODE;
        case mi::mdl::IType_enum::EID_INTENSITY_MODE: return IType_enum::EID_INTENSITY_MODE;
    }

    ASSERT( M_SCENE, false);
    return IType_enum::EID_USER;
}

/// Converts predefined MI::MDL::IType_enum IDs into predefined mi::mdl::IType_enum IDs.
inline mi::mdl::IType_enum::Predefined_id int_enum_id_to_mdl_enum_id(
    IType_enum::Predefined_id enum_id)
{
    switch( enum_id) {
        case IType_enum::EID_USER:           return mi::mdl::IType_enum::EID_USER;
        case IType_enum::EID_TEX_GAMMA_MODE: return mi::mdl::IType_enum::EID_TEX_GAMMA_MODE;
        case IType_enum::EID_INTENSITY_MODE: return mi::mdl::IType_enum::EID_INTENSITY_MODE;
        case IType_enum::EID_FORCE_32_BIT:
            ASSERT( M_SCENE, false); return mi::mdl::IType_enum::EID_USER;
    }

    ASSERT( M_SCENE, false);
    return mi::mdl::IType_enum::EID_USER;
}

/// Converts predefined mi::mdl::IType_struct IDs into predefined MI::MDL::IType_struct IDs.
inline IType_struct::Predefined_id mdl_struct_id_to_int_struct_id(
    mi::mdl::IType_struct::Predefined_id struct_id)
{
    switch( struct_id) {
        case mi::mdl::IType_struct::SID_USER:
            return IType_struct::SID_USER;
        case mi::mdl::IType_struct::SID_MATERIAL_EMISSION:
            return IType_struct::SID_MATERIAL_EMISSION;
        case mi::mdl::IType_struct::SID_MATERIAL_SURFACE:
            return IType_struct::SID_MATERIAL_SURFACE;
        case mi::mdl::IType_struct::SID_MATERIAL_VOLUME:
            return IType_struct::SID_MATERIAL_VOLUME;
        case mi::mdl::IType_struct::SID_MATERIAL_GEOMETRY:
            return IType_struct::SID_MATERIAL_GEOMETRY;
        case mi::mdl::IType_struct::SID_MATERIAL:
            return IType_struct::SID_MATERIAL;
    }

    ASSERT( M_SCENE, false);
    return IType_struct::SID_USER;
}

/// Converts predefined MI::MDL::IType_struct IDs into predefined mi::mdl::IType_struct IDs.
inline mi::mdl::IType_struct::Predefined_id int_struct_id_to_mdl_struct_id(
    IType_struct::Predefined_id struct_id)
{
    switch( struct_id) {
        case IType_struct::SID_USER:
            return mi::mdl::IType_struct::SID_USER;
        case IType_struct::SID_MATERIAL_EMISSION:
            return mi::mdl::IType_struct::SID_MATERIAL_EMISSION;
        case IType_struct::SID_MATERIAL_SURFACE:
            return mi::mdl::IType_struct::SID_MATERIAL_SURFACE;
        case IType_struct::SID_MATERIAL_VOLUME:
            return mi::mdl::IType_struct::SID_MATERIAL_VOLUME;
        case IType_struct::SID_MATERIAL_GEOMETRY:
            return mi::mdl::IType_struct::SID_MATERIAL_GEOMETRY;
        case IType_struct::SID_MATERIAL:
            return mi::mdl::IType_struct::SID_MATERIAL;
        case IType_struct::SID_FORCE_32_BIT:
            ASSERT( M_SCENE, false); return mi::mdl::IType_struct::SID_USER;
    }

    ASSERT( M_SCENE, false);
    return mi::mdl::IType_struct::SID_USER;
}

/// Converts mi::mdl::IType_texture shapes into MI::MDL::IType_texture shapes.
inline IType_texture::Shape mdl_shape_to_int_shape( mi::mdl::IType_texture::Shape shape)
{
    switch( shape) {
        case mi::mdl::IType_texture::TS_2D:        return IType_texture::TS_2D;
        case mi::mdl::IType_texture::TS_3D:        return IType_texture::TS_3D;
        case mi::mdl::IType_texture::TS_CUBE:      return IType_texture::TS_CUBE;
        case mi::mdl::IType_texture::TS_PTEX:      return IType_texture::TS_PTEX;
        case mi::mdl::IType_texture::TS_BSDF_DATA: return IType_texture::TS_BSDF_DATA;
    }

    ASSERT( M_SCENE, false);
    return IType_texture::TS_2D;
}

/// Converts MI::MDL::IType_texture shapes into mi::mdl::IType_texture shapes.
inline mi::mdl::IType_texture::Shape int_shape_to_mdl_shape( IType_texture::Shape shape)
{
    switch( shape) {
        case IType_texture::TS_2D:           return mi::mdl::IType_texture::TS_2D;
        case IType_texture::TS_3D:           return mi::mdl::IType_texture::TS_3D;
        case IType_texture::TS_CUBE:         return mi::mdl::IType_texture::TS_CUBE;
        case IType_texture::TS_PTEX:         return mi::mdl::IType_texture::TS_PTEX;
        case IType_texture::TS_BSDF_DATA:    return mi::mdl::IType_texture::TS_BSDF_DATA;
        case IType_texture::TS_FORCE_32_BIT:
            ASSERT( M_SCENE, false); return mi::mdl::IType_texture::TS_2D;
    }

    ASSERT( M_SCENE, false);
    return mi::mdl::IType_texture::TS_2D;
}

/// A hash functor for pointers.
template <typename T>
struct Hash_ptr
{
    size_t operator()(const T* p) const
    {
        size_t t = p - (T const *)0;
        return ((t) / (sizeof(size_t) * 2)) ^ (t >> 16);
    }
};

/// An equal functor for pointers.
template <typename T>
struct Equal_ptr
{
    unsigned int operator()(const T* a, const T* b) const
    {
        return a == b;
    }
};

/// Helper class for parameter type binding and checking.
class Type_binder
{
    typedef boost::unordered_map<
        const mi::mdl::IType_array*,
        const mi::mdl::IType_array*,
        const Hash_ptr<mi::mdl::IType_array>,
        const Equal_ptr<mi::mdl::IType_array>
    > Bind_type_map;

    typedef boost::unordered_map<
        const mi::mdl::IType_array_size*,
        int,
        const Hash_ptr<mi::mdl::IType_array_size>,
        const Equal_ptr<mi::mdl::IType_array_size>
    > Bind_size_map;

public:
    /// Constructor.
    Type_binder( mi::mdl::IType_factory* type_factory);

    /// Checks whether two types match taking types bound so far into account. Records new type
    /// bindings as necessary.
    ///
    /// \param parameter_type   The parameter type.
    /// \param argument_type    The argument type.
    /// \return
    ///                         -  0: The types match.
    ///                         - -1: The types do not match (excluding size of deferred size
    ///                               arrays). This case is supposed to happen only if one
    ///                               overwrites the DB element for a function call with another
    ///                               function call of different return type after setting up the
    ///                               attachment.
    ///                         - -2: The types do not match because of the length of deferred size
    ///                               arrays.
    mi::Sint32 check_and_bind_type(
        const mi::mdl::IType* parameter_type, const mi::mdl::IType* argument_type);

    /// Returns the bound type for an array type.
    ///
    /// Primarily used to get the bound type for the return type.
    ///
    /// \param a_type   An deferred size array type.
    /// \return         Returns the immediate size array type bound to \p a_type, or \c NULL if not
    ///                 bound.
    const mi::mdl::IType_array* get_bound_type( const mi::mdl::IType_array* a_type);

private:
    /// Binds the given abstract type to a concrete type.
    ///
    /// Both array types need to have the same element type.
    ///
    /// \param abs_type  The abstract type (deferred size array type).
    /// \param type      The concrete type (immediate size array type).
    void bind_param_type( const mi::mdl::IType_array* abs_type, const mi::mdl::IType_array* type);

    /// The type factory, used for creating new array types.
    mi::mdl::IType_factory* m_type_factory;

    /// Type/size bindings.
    Bind_type_map m_type_bindings;
    Bind_size_map m_size_bindings;
};

/// Adapts mi::mdl::IIput_stream to mi::neuraylib::IReader.
class Input_stream_reader_impl : public mi::base::Interface_implement<mi::neuraylib::IReader>
{
public:
    Input_stream_reader_impl( mi::mdl::IInput_stream* reader);

    // public API methods

    // Error handling not implemented.
    mi::Sint32 get_error_number() const { return 0; }

    // Error handling not implemented.
    const char* get_error_message() const { return nullptr; }

    // In this implementation, EOF is (incorrect) only reported after an attempt to read beyond the
    // end.
    bool eof() const { return m_eof; }

    mi::Sint32 get_file_descriptor() const { return -1; }

    bool supports_recorded_access() const { return false; }

    const mi::neuraylib::IStream_position* tell_position() const { return nullptr; }

    bool seek_position( const mi::neuraylib::IStream_position* stream_position) { return false; }

    bool rewind() { return false; }

    bool supports_absolute_access() const { return false; }

    mi::Sint64 tell_absolute() const { return -1; }

    bool seek_absolute( mi::Sint64 pos) { return false; }

    mi::Sint64 get_file_size() const { return 0; }

    bool seek_end() { return false; }

    mi::Sint64 read( char* buffer, mi::Sint64 size);

    bool readline( char* buffer, mi::Sint32 size);

    /// Lookahead is not supported in this implementation since the signature of lookahead()
    /// is not thread-safe.
    bool supports_lookahead() const { return false; }

    /// Lookahead is not supported in this implementation since the signature of lookahead()
    /// is not thread-safe.
    mi::Sint64 lookahead( mi::Sint64 size, const char** buffer) const { return 0; }

private:
    mi::base::Handle<mi::mdl::IInput_stream> m_stream;
    bool m_eof;
};

/// Adapts mi::mdl::IMDL_resource_reader to mi::neuraylib::IReader.
class Resource_reader_impl : public mi::base::Interface_implement<mi::neuraylib::IReader>
{
public:
    Resource_reader_impl( mi::mdl::IMDL_resource_reader* reader);

    // public API methods

    // Error handling not implemented.
    mi::Sint32 get_error_number() const { return 0; }

    // Error handling not implemented.
    const char* get_error_message() const { return nullptr; }

    bool eof() const;

    mi::Sint32 get_file_descriptor() const { return -1; }

    bool supports_recorded_access() const { return false; }

    const mi::neuraylib::IStream_position* tell_position() const { return nullptr; }

    bool seek_position( const mi::neuraylib::IStream_position* stream_position) { return false; }

    bool rewind();

    bool supports_absolute_access() const { return true; }

    mi::Sint64 tell_absolute() const;

    bool seek_absolute( mi::Sint64 pos);

    mi::Sint64 get_file_size() const;

    bool seek_end();

    mi::Sint64 read( char* buffer, mi::Sint64 size);

    bool readline( char* buffer, mi::Sint32 size);

    /// Lookahead is not supported in this implementation since the signature of lookahead()
    /// is not thread-safe.
    bool supports_lookahead() const { return false; }

    /// Lookahead is not supported in this implementation since the signature of lookahead()
    /// is not thread-safe.
    mi::Sint64 lookahead( mi::Sint64 size, const char** buffer) const { return 0; }

private:
    mi::base::Handle<mi::mdl::IMDL_resource_reader> m_reader;
};

/// Adapts mi::neuraylib::IReader to mi::mdl::Input_stream.
class Input_stream_impl : public mi::base::Interface_implement<mi::mdl::IInput_stream>
{
public:
    Input_stream_impl( mi::neuraylib::IReader* reader, const std::string& filename);

    // MDL core API methods

    int read_char();

    const char* get_filename();

private:
    mi::base::Handle<mi::neuraylib::IReader> m_reader;
    std::string m_filename;
};

/// Adapts mi::neuraylib::IReader to mi::mdl::IMdle_input_stream.
class Mdle_input_stream_impl
  : public mi::base::Interface_implement<mi::mdl::IMdle_input_stream>, Input_stream_impl
{
public:
    Mdle_input_stream_impl( mi::neuraylib::IReader* reader, const std::string& filename);

    // MDL core API methods

    int read_char();

    const char* get_filename();
};

/// Adapts mi::neuraylib::IReader to mi::mdl::IMDL_resource_reader.
class Mdl_resource_reader_impl : public mi::base::Interface_implement<mi::mdl::IMDL_resource_reader>
{
public:
    Mdl_resource_reader_impl(
        mi::neuraylib::IReader* reader,
        const std::string& file_path,
        const std::string& filename,
        const mi::base::Uuid& hash);

    // MDL core API methods

    Uint64 read( void* ptr, Uint64 size);

    Uint64 tell();

    bool seek( Sint64 offset, Position origin);

    const char* get_filename() const;

    const char* get_mdl_url() const;

    bool get_resource_hash( unsigned char hash[16]);

private:
    mi::base::Handle<mi::neuraylib::IReader> m_reader;
    std::string m_file_path;
    std::string m_filename;
    mi::base::Uuid m_hash;
};

/// Implementation of IMAGE::IMdl_container_callback.
class Mdl_container_callback : public mi::base::Interface_implement<IMAGE::IMdl_container_callback>
{
public:
    mi::neuraylib::IReader* get_reader( const char* container_filename, const char* member_filename);
};

/// The implementation of Image_set used by the MDL integration
class Mdl_image_set : public DBIMAGE::Image_set
{
public:

    Mdl_image_set( mi::mdl::IMDL_resource_set* set, const std::string& file_path);

    mi::Size get_length() const;

    bool is_uvtile() const;

    bool is_mdl_container() const;

    void get_uv_mapping( mi::Size i, mi::Sint32 &u, mi::Sint32 &v) const;

    const char* get_original_filename() const;

    const char* get_container_filename() const;

    const char* get_mdl_file_path() const;

    const char* get_resolved_filename( mi::Size i) const;

    const char* get_container_membername( mi::Size i) const;

    mi::neuraylib::IReader* open_reader( mi::Size i) const;

    mi::neuraylib::ICanvas* get_canvas( mi::Size i) const;

    const char* get_image_format() const;

private:

    mi::base::Handle<mi::mdl::IMDL_resource_set> m_resource_set;
    std::string m_mdl_file_path;
    std::string m_container_filename;
    std::string m_file_format;
    bool m_is_container;
};

} // namespace DETAIL

} // namespace MDL

} // namespace MI

#endif // IO_SCENE_MDL_ELEMENTS_MDL_ELEMENTS_DETAIL_H
