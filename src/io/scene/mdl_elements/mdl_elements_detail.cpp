/***************************************************************************************************
 * Copyright (c) 2012-2019, NVIDIA CORPORATION. All rights reserved.
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

#include "pch.h"

#include "mdl_elements_detail.h"
#include "i_mdl_elements_utilities.h"
#include "mdl_elements_utilities.h"

#include <mi/base/config.h>
#include <mi/base/atom.h>
#include <mi/mdl/mdl_archiver.h>
#include <mi/mdl/mdl_encapsulator.h>
#include <mi/mdl/mdl_entity_resolver.h>
#include <mi/mdl/mdl_mdl.h>
#include <mi/mdl/mdl_streams.h>
#include <boost/algorithm/string/replace.hpp>
#include <boost/core/ignore_unused.hpp>
#include <base/system/main/access_module.h>
#include <base/hal/disk/disk.h>
#include <base/hal/hal/i_hal_ospath.h>
#include <base/lib/log/i_log_logger.h>
#include <base/lib/path/i_path.h>
#include <base/data/db/i_db_transaction.h>
#include <mdl/integration/mdlnr/i_mdlnr.h>
#include <mdl/codegenerators/generator_dag/generator_dag_tools.h>
#include <io/scene/bsdf_measurement/i_bsdf_measurement.h>
#include <io/scene/lightprofile/i_lightprofile.h>
#include <io/scene/texture/i_texture.h>

namespace MI {

namespace MDL {

using mi::mdl::as;
using mi::mdl::cast;

namespace DETAIL {

bool is_container_member( const char* filename)
{
    return filename && (strstr(filename, ".mdr:") != 0 || strstr(filename, ".mdle:") != 0);
}


std::string get_container_filename( const char* filename)
{
    if( !filename)
        return "";

    // archive
    const char* pos = strstr( filename, ".mdr:");
    size_t offset = 4;

    // MDLe
    if ( pos == NULL) {
        pos = strstr( filename, ".mdle:");
        offset = 5;
    }

    // none
    if( !pos)
        return "";

    return std::string( filename, pos + offset);
}

std::string get_container_membername( const char* filename)
{
    if( !filename)
        return "";

    // archive
    const char* pos = strstr( filename, ".mdr:");
    size_t offset = 5;

    // MDLe
    if ( pos == NULL) {
        pos = strstr(filename, ".mdle:");
        offset = 6;
    }

    // none
    if( !pos)
        return "";

    return std::string( pos + offset);
}

namespace {

/// Converts OS-specific directory separators into slashes.
std::string convert_os_separators_to_slashes( const std::string& s)
{
#ifdef MI_PLATFORM_WINDOWS
    return boost::replace_all_copy( s, HAL::Ospath::sep(), "/");
#else
    // return boost::replace_all_copy( s, HAL::Ospath::sep(), "/");
    return s; // optimization
#endif
}

/// Calls the MDL entity resolver with the given arguments and returns the file name set, or
/// \c NULL in case of failure. The flag \c log_messages indicates whether error messages should be
/// logged.
mi::mdl::IMDL_resource_set* get_resource_set(
    const char* file_path,
    const char* module_file_system_path,
    const char* module_name,
    bool log_messages)
{
    ASSERT( M_SCENE, file_path);
    ASSERT( M_SCENE, !module_file_system_path || DISK::is_path_absolute( module_file_system_path));

    SYSTEM::Access_module<MDLC::Mdlc_module> mdlc_module( false);
    mi::base::Handle<mi::mdl::IMDL> mdl( mdlc_module->get_mdl());
    mi::base::Handle<mi::mdl::IEntity_resolver> resolver(
        mdl->create_entity_resolver( /*module_cache*/ 0));
    mi::mdl::IMDL_resource_set* res_set = resolver->resolve_resource_file_name(
        file_path, module_file_system_path, module_name, /*pos*/ 0);
    if( log_messages)
        report_messages( resolver->access_messages(), /*out_messages*/ 0);
    return res_set;
}

/// Calls the MDL entity resolver with the given arguments and returns the resulting reader, or
/// \c NULL in case of failure. The flag \c log_messages indicates whether error messages should be
/// logged.
mi::mdl::IMDL_resource_reader* get_reader(
    const char* file_path,
    const char* module_file_system_path,
    const char* module_name,
    bool log_messages)
{
    ASSERT( M_SCENE, file_path);
    ASSERT( M_SCENE, !module_file_system_path || DISK::is_path_absolute( module_file_system_path));

    SYSTEM::Access_module<MDLC::Mdlc_module> mdlc_module( false);
    mi::base::Handle<mi::mdl::IMDL> mdl( mdlc_module->get_mdl());
    mi::base::Handle<mi::mdl::IEntity_resolver> resolver(
        mdl->create_entity_resolver( /*module_cache*/ 0));
    mi::mdl::IMDL_resource_reader* reader
        = resolver->open_resource( file_path, module_file_system_path, module_name, /*pos*/ 0);
    if( log_messages)
        report_messages( resolver->access_messages(), /*out_messages*/ 0);
    return reader;
}

/// Calls the MDL entity resolver with the given arguments and returns the result of
/// mi::mdl::IMDL_resource_reader::get_filename(), or the empty string in case of failure.
std::string resolve_resource_filename(
    const char* file_path,
    const char* module_file_system_path,
    const char* module_name,
    bool log_messages = false)
{

    mi::base::Handle<mi::mdl::IMDL_resource_set> name_set( get_resource_set(
        file_path, module_file_system_path, module_name, log_messages));
    if( !name_set)
        return "";

    const char* resolved_filename = name_set->get_filename( 0);
    return std::string( resolved_filename);
}

}

std::string unresolve_resource_filename(
    const char* filename, const char* module_filename, const char* module_name)
{
    ASSERT( M_SCENE, !module_filename || DISK::is_path_absolute( module_filename));

    std::string norm_filename = HAL::Ospath::normpath_v2( filename);
    std::string shortened_filename;
    bool norm_filename_absolute = DISK::is_path_absolute( norm_filename);

    SYSTEM::Access_module<PATH::Path_module> m_path_module( false);
    const std::vector<std::string>& mdl_paths
        = m_path_module->get_search_path( PATH::MDL);
    mi::Size mdl_paths_count = mdl_paths.size();
    for( mi::Size i = 0; shortened_filename.empty() && i < mdl_paths_count; ++i) {
        std::string path = HAL::Ospath::normpath_v2( mdl_paths[i]);
        if( path == ".") {
            if( !norm_filename_absolute)
                shortened_filename = norm_filename;
        } else {
            path += HAL::Ospath::sep();
            mi::Size path_size = path.size();
            if( norm_filename.substr( 0, path_size) == path)
                shortened_filename = norm_filename.substr( path_size);
        }
    }

    // return failure if no search path is a prefix
    if( shortened_filename.empty())
        return "";

    // construct MDL file path corresponding to shortened file name
    std::string file_path = "/" + convert_os_separators_to_slashes( shortened_filename);

    // return failure if re-resolving of file path yields a different filename
    std::string resolved_filename = resolve_resource_filename(
        file_path.c_str(), module_filename, module_name);
    resolved_filename = HAL::Ospath::normpath_v2( resolved_filename);
    if( is_container_member( resolved_filename.c_str()))
        return "";
    if( resolved_filename != norm_filename)
        return "";

    // return absolute MDL file path in case of success
    ASSERT( M_SCENE, file_path[0] == '/');
    return file_path;
}

std::string unresolve_resource_filename(
    const char* container_filename,
    const char* container_membername,
    const char* module_filename,
    const char* module_name)
{
    ASSERT( M_SCENE, !module_filename || DISK::is_path_absolute( module_filename));

    std::string norm_filename = HAL::Ospath::normpath_v2(container_filename);
    std::string shortened_filename;
    bool norm_filename_absolute = DISK::is_path_absolute( norm_filename);

    SYSTEM::Access_module<PATH::Path_module> m_path_module( false);
    const std::vector<std::string>& mdl_paths
        = m_path_module->get_search_path( PATH::MDL);
    mi::Size mdl_paths_count = mdl_paths.size();
    for( mi::Size i = 0; shortened_filename.empty() && i < mdl_paths_count; ++i) {
        std::string path = HAL::Ospath::normpath_v2( mdl_paths[i]);
        if( path == ".") {
            if( !norm_filename_absolute)
                shortened_filename = norm_filename;
        } else {
            path += HAL::Ospath::sep();
            mi::Size path_size = path.size();
            if( norm_filename.substr( 0, path_size) == path)
                shortened_filename = norm_filename.substr( path_size);
        }
    }

    // return failure if no search path is a prefix
    if( shortened_filename.empty())
        return "";

    // construct MDL file path corresponding to shortened file name
    std::string file_path = "/" + convert_os_separators_to_slashes(container_membername);

    // return failure if re-resolving of file path yields a different filename
    std::string resolved_filename = resolve_resource_filename(
        file_path.c_str(), module_filename, module_name);
    resolved_filename = HAL::Ospath::normpath_v2( resolved_filename);
    if( !is_container_member( resolved_filename.c_str()))
        return "";
    resolved_filename = get_container_filename( resolved_filename.c_str());
    if( resolved_filename != norm_filename)
        return "";

    // return absolute MDL file path in case of success
    ASSERT( M_SCENE, file_path[0] == '/');
    return file_path;
}

DB::Tag mdl_resource_to_tag(
    DB::Transaction* transaction,
    const mi::mdl::IValue_resource* value,
    const char* module_filename,
    const char* module_name)
{
    switch (value->get_kind()) {
    case mi::mdl::IValue::VK_TEXTURE:
        {
            const mi::mdl::IValue_texture* texture = cast<mi::mdl::IValue_texture>(value);
            return mdl_texture_to_tag(transaction, texture, module_filename, module_name);
        }

    case mi::mdl::IValue::VK_LIGHT_PROFILE:
        {
            const mi::mdl::IValue_light_profile* lp = cast<mi::mdl::IValue_light_profile>(value);
            return mdl_light_profile_to_tag(transaction, lp, module_filename, module_name);
        }

    case mi::mdl::IValue::VK_BSDF_MEASUREMENT:
        {
            const mi::mdl::IValue_bsdf_measurement* bsdfm =
                cast<mi::mdl::IValue_bsdf_measurement>( value);
            return mdl_bsdf_measurement_to_tag( transaction, bsdfm, module_filename, module_name);
        }

    default:
        ASSERT( M_SCENE, false);
        return DB::Tag();
    }
}

DB::Tag mdl_texture_to_tag(
    DB::Transaction* transaction,
    const mi::mdl::IValue_texture* value,
    const char* module_filename,
    const char* module_name)
{
    mi::Uint32 tag_uint32 = value->get_tag_value();
    const char* file_path = value->get_string_value();

    // Check whether the tag value is set before checking whether the string value is not set.
    // Resources in compiled materials typically do not have the string value set.
    if( tag_uint32)
        return DB::Tag( tag_uint32);

    // Fail if neither tag nor string value is set.
    if( !file_path || !file_path[0])
        return DB::Tag( 0);

    mi::Float32 gamma = 0.0f;
    switch( value->get_gamma_mode()) {
        case mi::mdl::IValue_texture::gamma_default: gamma = 0.0f; break; // encode as 0.0
        case mi::mdl::IValue_texture::gamma_linear:  gamma = 1.0f; break;
        case mi::mdl::IValue_texture::gamma_srgb:    gamma = 2.2f; break;
    }

    // Convert string value into tag value.
    return mdl_texture_to_tag(
        transaction, file_path, module_filename, module_name, /*shared*/ true, gamma);
}

DB::Tag mdl_texture_to_tag(
    DB::Transaction* transaction,
    const char* file_path,
    const char* module_filename,
    const char* module_name,
    bool shared,
    mi::Float32 gamma)
{
    // we might have UDIM textures here, so load a whole set
    mi::base::Handle<mi::mdl::IMDL_resource_set> res_set( get_resource_set(
        file_path, module_filename, module_name, /*log_messages*/ true));
    if( !res_set) {
        LOG::mod_log->warning( M_SCENE, LOG::Mod_log::C_IO,
            "Failed to resolve \"%s\" in \"%s\".", file_path, module_name);
        return DB::Tag( 0);
    }

    const char* first_filename = res_set->get_filename(0);
    LOG::mod_log->debug( M_SCENE, LOG::Mod_log::C_IO,
        "Resolved \"%s\" in \"%s\" to \"%s\"%s.",
        file_path,
        module_name,
        first_filename,
        res_set->get_count() > 1 ? " ..." : "");
    ASSERT( M_SCENE, first_filename);

    DB::Tag tag;
    Mdl_image_set image_set( res_set.get(), file_path, get_container_filename( first_filename));

    tag = TEXTURE::load_mdl_texture(
        transaction, &image_set, shared, gamma);

    LOG::mod_log->debug( M_SCENE, LOG::Mod_log::C_IO,
        "... and mapped to texture \"%s\" (tag %u).",
        transaction->tag_to_name( tag), tag.get_uint());

    return tag;
}

DB::Tag mdl_light_profile_to_tag(
    DB::Transaction* transaction,
    const mi::mdl::IValue_light_profile* value,
    const char* module_filename,
    const char* module_name)
{
    mi::Uint32 tag_uint32 = value->get_tag_value();

    // Check whether the tag value is set before checking whether the string value is not set.
    // Resources in compiled materials typically do not have the string value set.
    if( tag_uint32)
        return DB::Tag( tag_uint32);

    const char* file_path = value->get_string_value();

    // Fail if neither tag nor string value is set.
    if( !file_path || !file_path[0])
        return DB::Tag( 0);

    // Convert string value into tag value.
    return mdl_light_profile_to_tag(
        transaction, file_path, module_filename, module_name, /*shared*/ true);
}

DB::Tag mdl_light_profile_to_tag(
    DB::Transaction* transaction,
    const char* file_path,
    const char* module_filename,
    const char* module_name,
    bool shared)
{
    mi::base::Handle<mi::mdl::IMDL_resource_reader> reader(
        get_reader( file_path, module_filename, module_name, /*log_messages*/ true));
    if( !reader) {
        LOG::mod_log->warning( M_SCENE, LOG::Mod_log::C_IO,
            "Failed to resolve \"%s\" in \"%s\".", file_path, module_name);
        return DB::Tag( 0);
    }

    const char* resolved_filename = reader->get_filename();
    LOG::mod_log->debug( M_SCENE, LOG::Mod_log::C_IO,
        "Resolved \"%s\" in \"%s\" to \"%s\".", file_path, module_name, resolved_filename);
    ASSERT( M_SCENE, resolved_filename);

    DB::Tag tag;
    const std::string& absolute_mdl_file_path = reader->get_mdl_url();

    if( is_container_member( resolved_filename)) {

        // Imported resource is in an container
        const std::string& container_filename = get_container_filename( resolved_filename);
        const std::string& member_filename  = get_container_membername( resolved_filename);

        File_reader_impl wrapped_reader( reader.get());
        tag = LIGHTPROFILE::load_mdl_lightprofile(
            transaction, &wrapped_reader,
            container_filename, member_filename, absolute_mdl_file_path, shared);

    } else {

        tag = LIGHTPROFILE::load_mdl_lightprofile(
            transaction, resolved_filename, absolute_mdl_file_path, shared);

    }

    LOG::mod_log->debug( M_SCENE, LOG::Mod_log::C_IO,
        "... and mapped to lightprofile \"%s\" (tag %u).",
        transaction->tag_to_name( tag), tag.get_uint());
    return tag;
}

DB::Tag mdl_bsdf_measurement_to_tag(
    DB::Transaction* transaction,
    const mi::mdl::IValue_bsdf_measurement* value,
    const char* module_filename,
    const char* module_name)
{
    mi::Uint32 tag_uint32 = value->get_tag_value();

    // Check whether the tag value is set before checking whether the string value is not set.
    // Resources in compiled materials typically do not have the string value set.
    if( tag_uint32)
        return DB::Tag( tag_uint32);

    const char* file_path = value->get_string_value();

    // Fail if neither tag nor string value is set.
    if( !file_path || !file_path[0])
        return DB::Tag( 0);

    // Convert string value into tag value.
    return mdl_bsdf_measurement_to_tag(
        transaction, file_path, module_filename, module_name, /*shared*/ true);
}

DB::Tag mdl_bsdf_measurement_to_tag(
    DB::Transaction* transaction,
    const char* file_path,
    const char* module_filename,
    const char* module_name,
    bool shared)
{
    mi::base::Handle<mi::mdl::IMDL_resource_reader> reader(
        get_reader( file_path, module_filename, module_name, /*log_messages*/ true));
    if( !reader) {
        LOG::mod_log->warning( M_SCENE, LOG::Mod_log::C_IO,
            "Failed to resolve \"%s\" in \"%s\".", file_path, module_name);
        return DB::Tag( 0);
    }

    const char* resolved_filename = reader->get_filename();
    LOG::mod_log->debug( M_SCENE, LOG::Mod_log::C_IO,
        "Resolved \"%s\" in \"%s\" to \"%s\".", file_path, module_name, resolved_filename);
    ASSERT( M_SCENE, resolved_filename);

    DB::Tag tag;
    const std::string& absolute_mdl_file_path = reader->get_mdl_url();

    if( is_container_member( resolved_filename)) {

        // Imported resource is in an container
        const std::string& container_filename = get_container_filename( resolved_filename);
        const std::string& member_filename  = get_container_membername( resolved_filename);

        File_reader_impl wrapped_reader( reader.get());
        tag = BSDFM::load_mdl_bsdf_measurement(
            transaction, &wrapped_reader,
            container_filename, member_filename, absolute_mdl_file_path, shared);

    } else {

        tag = BSDFM::load_mdl_bsdf_measurement(
            transaction, resolved_filename, absolute_mdl_file_path, shared);

    }

    LOG::mod_log->debug( M_SCENE, LOG::Mod_log::C_IO,
        "... and mapped to scene element \"%s\" (tag %u).",
        transaction->tag_to_name( tag), tag.get_uint());
    return tag;
}

static mi::base::Atom32 uniq_id_for_generate_suffix;

std::string generate_suffix()
{
    return std::to_string( Uint32( uniq_id_for_generate_suffix++));
}

std::string generate_unique_db_name( DB::Transaction* transaction, const char* prefix)
{
    std::string prefix_str = prefix ? prefix : "";
    std::string name = prefix_str + "_" + generate_suffix();
    while( transaction->name_to_tag( name.c_str()))
        name = prefix_str + "_" + generate_suffix();
    return name;
}


// *********** Type_binder *************************************************************************

Type_binder::Type_binder( mi::mdl::IType_factory* type_factory)
  : m_type_factory( type_factory)
{
}

mi::Sint32 Type_binder::check_and_bind_type(
    const mi::mdl::IType* parameter_type, const mi::mdl::IType* argument_type)
{
    argument_type = m_type_factory->import( argument_type->skip_type_alias());

    // for non-arrays, parameter and argument type have to match exactly
    const mi::mdl::IType_array* a_parameter_type = as<mi::mdl::IType_array>( parameter_type);
    const mi::mdl::IType_array* a_argument_type  = as<mi::mdl::IType_array>( argument_type);
    if( !a_parameter_type || !a_argument_type)
        return parameter_type == argument_type ? 0 : -1;

    ASSERT( M_SCENE, a_argument_type->is_immediate_sized());

    // for immediate-sized arrays, parameter and argument types have to match exactly
    if( a_parameter_type->is_immediate_sized())
        return parameter_type == argument_type ? 0 : -1;

    // for deferred-sized arrays element types have to match exactly
    const mi::mdl::IType* e_argument_type  = a_argument_type->get_element_type();
    const mi::mdl::IType* e_parameter_type = a_parameter_type->get_element_type();
    if( e_argument_type != e_parameter_type)
        return -1;

    // if the parameter type is unbound, bind it
    const mi::mdl::IType_array* bound_parameter_type = get_bound_type( a_parameter_type);
    if( !bound_parameter_type) {
        bind_param_type( a_parameter_type, a_argument_type);
        return 0;
    }

    // the parameter type is bound, compare bounded type against argument type
    ASSERT( M_SCENE, bound_parameter_type->is_immediate_sized());
    const mi::mdl::IType* e_bound_parameter_type = bound_parameter_type->get_element_type();
    ASSERT( M_SCENE, e_bound_parameter_type == e_parameter_type);
    boost::ignore_unused( e_bound_parameter_type);
    return bound_parameter_type == a_argument_type ? 0 : -2;
}

const mi::mdl::IType_array* Type_binder::get_bound_type( const mi::mdl::IType_array* a_type)
{
    // check if the type is bound
    Bind_type_map::const_iterator it = m_type_bindings.find( a_type);
    if( it != m_type_bindings.end())
        return it->second;

    // check if the size is bound
    const mi::mdl::IType_array_size* a_size = a_type->get_deferred_size();
    Bind_size_map::const_iterator size_it = m_size_bindings.find( a_size);
    if( size_it == m_size_bindings.end())
        return 0;

    // bind the type
    const mi::mdl::IType* e_type = a_type->get_element_type();
    const mi::mdl::IType_array* bound_type
        = as<mi::mdl::IType_array>( m_type_factory->create_array( e_type, size_it->second));
    m_type_bindings[a_type] = bound_type;

    return bound_type;
}

void Type_binder::bind_param_type(
    const mi::mdl::IType_array* abs_type, const mi::mdl::IType_array* type)
{
    ASSERT( M_SCENE, !abs_type->is_immediate_sized() && type->is_immediate_sized());
    ASSERT( M_SCENE, abs_type->get_element_type() == type->get_element_type());
    const mi::mdl::IType_array_size* abs_size = abs_type->get_deferred_size();
    int size = type->get_size();
    m_size_bindings[abs_size] = size;
    m_type_bindings[abs_type] = type;
}


File_reader_impl::File_reader_impl( mi::mdl::IMDL_resource_reader* reader)
  : m_reader( reader, mi::base::DUP_INTERFACE)
{
}

mi::Sint32 File_reader_impl::get_error_number() const
{
    ASSERT( M_SCENE, false);
    return 0;
}

const char* File_reader_impl::get_error_message() const
{
    ASSERT( M_SCENE, false);
    return 0;
}

bool File_reader_impl::eof() const
{
    return tell_absolute() == get_file_size();
}

bool File_reader_impl::rewind()
{
    return m_reader->seek( 0, mi::mdl::IMDL_resource_reader::MDL_SEEK_SET);
}

mi::Sint64 File_reader_impl::tell_absolute() const
{
    return m_reader->tell();
}

bool File_reader_impl::seek_absolute( mi::Sint64 pos)
{
    return m_reader->seek( pos, mi::mdl::IMDL_resource_reader::MDL_SEEK_SET);
}

mi::Sint64 File_reader_impl::get_file_size() const
{
    mi::Uint64 pos = m_reader->tell();
    m_reader->seek( 0, mi::mdl::IMDL_resource_reader::MDL_SEEK_END);
    mi::Uint64 size = m_reader->tell();
    m_reader->seek( pos, mi::mdl::IMDL_resource_reader::MDL_SEEK_SET);
    return size;
}

bool File_reader_impl::seek_end()
{
    return m_reader->seek( 0, mi::mdl::IMDL_resource_reader::MDL_SEEK_END);
}

mi::Sint64 File_reader_impl::read( char* buffer, mi::Sint64 size)
{
    return m_reader->read( buffer, size);
}

bool File_reader_impl::readline( char* buffer, mi::Sint32 size)
{
    if( size == 0)
        return false;

    mi::Size data_size    = get_file_size();
    mi::Size current      = tell_absolute();
    mi::Size offset       = 0;
    bool more_input       = current < data_size;
    bool more_output      = offset < static_cast<mi::Size>( size-1);
    bool newline_not_seen = true;
    while( more_input && more_output && newline_not_seen) {
        m_reader->read( buffer + offset, 1);
        newline_not_seen  = buffer[offset] != '\n';
        ++current;
        ++offset;
        more_input        = current < data_size;
        more_output       = offset < static_cast<mi::Size>( size-1);
    }
    buffer[offset] = '\0';
    return true;
}

mi::neuraylib::IReader* Mdl_container_callback::get_reader(
    const char* container_filename, const char* member_filename)
{
    ASSERT( M_SCENE, container_filename && member_filename);

    SYSTEM::Access_module<MDLC::Mdlc_module> mdlc_module( false);
    mi::base::Handle<mi::mdl::IMDL> mdl( mdlc_module->get_mdl());
    mi::base::Handle<mi::mdl::IInput_stream> input_stream(NULL);

    std::string container_filename_str = container_filename;
    if (container_filename_str.rfind(".mdr") != std::string::npos)
    {
        mi::base::Handle<mi::mdl::IArchive_tool> archive_tool(mdl->create_archive_tool());
        input_stream = archive_tool->get_file_content(container_filename, member_filename);
    }
    else if (container_filename_str.rfind(".mdle") != std::string::npos)
    {
        mi::base::Handle<mi::mdl::IEncapsulate_tool> mdle_tool(mdl->create_encapsulate_tool());
        input_stream = mdle_tool->get_file_content(container_filename, member_filename);
    }

    if (!input_stream)
        return 0;

    mi::base::Handle<mi::mdl::IMDL_resource_reader> file_random_access(
        input_stream->get_interface<mi::mdl::IMDL_resource_reader>());

    ASSERT( M_SCENE, file_random_access.get());
    return new File_reader_impl( file_random_access.get());
}

Mdl_image_set::Mdl_image_set(
    mi::mdl::IMDL_resource_set* set, const std::string& file_name, const std::string& container_name)
    : m_resource_set( set, mi::base::DUP_INTERFACE)
    , m_container_name(container_name)
    , m_is_container( !container_name.empty())
{
    ASSERT( M_SCENE, set->get_mdl_url(0));

    if ( set->get_count() > 1)  // uvtile/udim sequence
    {
        // construct absolute mdl file path which still contains the uvtile/udim marker
        // from original filename and the first absolute mdl url in the set
        const std::string mdl_url = set->get_mdl_url( 0);
       
        std::string marker_string;
        std::size_t p = file_name.find_last_of("/");
        if (p == std::string::npos)
            marker_string = "/" + file_name;
        else
            marker_string = file_name.substr(p);

        p = mdl_url.find_last_of("/"); 
        ASSERT(M_SCENE, p != std::string::npos); // there needs to be at least one slash
        m_mdl_file_path = mdl_url.substr(0, p) + marker_string;

    }
    else
        m_mdl_file_path = set->get_mdl_url(0);
}

mi::Size Mdl_image_set::get_length() const
{
    return m_resource_set->get_count();
}

char const* Mdl_image_set::get_mdl_file_path() const
{
    return m_mdl_file_path.empty() ? NULL : m_mdl_file_path.c_str();
}

char const * Mdl_image_set::get_container_filename() const
{
    return m_container_name.empty() ? NULL : m_container_name.c_str();
}

char const * Mdl_image_set::get_mdl_url(mi::Size i) const
{
    return m_resource_set->get_mdl_url( i);
}

char const * Mdl_image_set::get_resolved_filename(mi::Size i) const
{
    return m_is_container ? NULL : m_resource_set->get_filename( i);
}

char const * Mdl_image_set::get_container_membername(mi::Size i) const
{
    if(m_is_container)
    {
        char const* p = strstr( m_resource_set->get_filename( i), ".mdr:");
        size_t offset = 5;

        if( p == NULL) {
            p = strstr( m_resource_set->get_filename(i), ".mdle:");
            offset = 6;
        }

        if( p)
            return p + offset;
    }
    return NULL;
}

bool Mdl_image_set::get_uv_mapping(mi::Size i, mi::Sint32 &u, mi::Sint32 &v) const
{
    return m_resource_set->get_udim_mapping( i, u, v);
}

mi::neuraylib::IReader* Mdl_image_set::open_reader( mi::Size i) const
{
    mi::base::Handle<mi::mdl::IMDL_resource_reader> reader(
        m_resource_set->open_reader( i));
    if( reader)
        return new File_reader_impl(reader.get());
    return NULL;
}

bool Mdl_image_set::is_uvtile() const
{
    int u, v;
    return m_resource_set->get_udim_mapping( 0, u, v);
}

bool Mdl_image_set::is_mdl_container() const
{
    return m_is_container;
}

std::string lookup_thumbnail(
    const std::string& module_filename,
    const std::string& mdl_name,
    const IAnnotation_block* annotations,
    mi::mdl::IArchive_tool* archive_tool)
{
    std::string stripped_mdl_name, def_name, module_name;
    mi::Size p = mdl_name.find( "(");
    if( p == std::string::npos) 
        stripped_mdl_name = mdl_name;
    else // strip function signature
        stripped_mdl_name = mdl_name.substr( 0, p);

    p = stripped_mdl_name.rfind( "::");
    if( p == std::string::npos)
         // invalid mdl name
        return "";

    def_name = stripped_mdl_name.substr(p + 2);
    module_name = stripped_mdl_name.substr(0, p);

    if(annotations) {
        for( mi::Size i=0; i<annotations->get_size(); ++i) {

            mi::base::Handle<const MI::MDL::IAnnotation> anno( annotations->get_annotation(i));
            if( !anno)
                continue;
            if( strcmp( anno->get_name(), "::anno::thumbnail(string)") == 0) {

                mi::base::Handle<const MI::MDL::IExpression_list> expressions(
                    anno->get_arguments());
                ASSERT( M_SCENE, expressions->get_size() == 1);
                mi::base::Handle<const MI::MDL::IExpression_constant> expr(
                    mi::base::make_handle(expressions->get_expression( mi::Size(0)))
                    ->get_interface<const MI::MDL::IExpression_constant>());
                if( !expr)
                    break;
                mi::base::Handle<const MI::MDL::IValue_string> vstr(
                    expr->get_value<IValue_string>());
                ASSERT( M_SCENE, vstr.is_valid_interface());
                // lookup file
                std::string file = resolve_resource_filename(
                    vstr->get_value(), module_filename.c_str(),
                    module_name.c_str(), /*log_messages*/ true);
                if( !file.empty())
                    return file;
                break;
            }
        }
    }

    // construct thumbnail filename according to the "old" convention module_name.mdl_name.ext 
    const char* ext[] = {"png", "jpg", "jpeg", "PNG", "JPG", "JPEG", NULL};

    std::string file_base;
    std::string container_path;

    mi::Size p_mdr = module_filename.find(".mdr:");
    mi::Size p_mdl = module_filename.find(".mdl");

    // located in container?
    if( p_mdr != std::string::npos)
    {
        container_path = module_filename.substr( 0, p_mdr + 4);
        file_base = module_filename.substr( p_mdr + 5, p_mdl - p_mdr - 5) + "." + def_name + ".";

        // check for supported file types
        for( int i = 0; ext[i] != NULL; ++i) {
            std::string file_name = file_base + ext[i];
            mi::base::Handle<mi::mdl::IInput_stream> file(
                archive_tool->get_file_content(container_path.c_str(), file_name.c_str()));
            if( file)
                return container_path + ":" + file_name;
        }
    }
    else
    {
        file_base = module_filename.substr( 0, p_mdl) + "." + def_name + ".";

        // check for supported file types
        for( int i = 0; ext[i] != NULL; ++i) {
            std::string file_name = file_base + ext[i];
            if( DISK::is_file( file_name.c_str()))
                return file_name;
        }
    }
    return "";
}

} // namespace DETAIL

} // namespace MDL

} // namespace MI
