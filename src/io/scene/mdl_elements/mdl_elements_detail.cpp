/***************************************************************************************************
 * Copyright (c) 2012-2023, NVIDIA CORPORATION. All rights reserved.
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

#include <atomic>

#include <boost/algorithm/string/replace.hpp>
#include <boost/core/ignore_unused.hpp>

#include <mi/base/config.h>
#include <mi/base/atom.h>
#include <mi/mdl/mdl_archiver.h>
#include <mi/mdl/mdl_encapsulator.h>
#include <mi/mdl/mdl_entity_resolver.h>
#include <mi/mdl/mdl_mdl.h>
#include <mi/mdl/mdl_streams.h>
#include <mi/neuraylib/imdl_entity_resolver.h>

#include <base/system/main/access_module.h>
#include <base/hal/disk/disk.h>
#include <base/hal/hal/i_hal_ospath.h>
#include <base/lib/log/i_log_logger.h>
#include <base/lib/path/i_path.h>
#include <base/util/string_utils/i_string_utils.h>
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

std::mutex g_transaction_mutex;

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
/// logged. This method supports all three kinds of resources, including animated and/or uvtile
/// textures.
mi::mdl::IMDL_resource_set* get_resource_set(
    const char* file_path,
    const char* module_file_system_path,
    const char* module_name,
    bool log_messages,
    Execution_context* context)
{
    ASSERT( M_SCENE, file_path);

    SYSTEM::Access_module<MDLC::Mdlc_module> mdlc_module( false);
    mi::base::Handle<mi::mdl::IMDL> mdl( mdlc_module->get_mdl());
    mi::base::Handle<mi::mdl::IEntity_resolver> resolver(
        mdl->get_entity_resolver( /*module_cache*/ nullptr));
    mi::base::Handle<mi::mdl::IThread_context> ctx( create_thread_context( mdl.get(), context));

    mi::mdl::IMDL_resource_set* res_set = resolver->resolve_resource_file_name(
        file_path, module_file_system_path, module_name, /*pos*/ nullptr, ctx.get());

    if( log_messages)
        convert_and_log_messages( resolver->access_messages(), /*out_messages*/ nullptr);
    return res_set;
}

/// Calls the MDL entity resolver with the given arguments and returns the resulting reader, or \c
/// NULL in case of failure. The flag \c log_messages indicates whether error messages should be
/// logged. This method does not support animated nor uvtile textures. It should only be used for
/// light profiles and BSDF measurements.
mi::mdl::IMDL_resource_reader* get_reader(
    const char* file_path,
    const char* module_file_system_path,
    const char* module_name,
    bool log_messages,
    Execution_context* context)
{
    ASSERT( M_SCENE, file_path);

    SYSTEM::Access_module<MDLC::Mdlc_module> mdlc_module( false);
    mi::base::Handle<mi::mdl::IMDL> mdl( mdlc_module->get_mdl());
    mi::base::Handle<mi::mdl::IEntity_resolver> resolver(
        mdl->get_entity_resolver( /*module_cache*/ nullptr));
    mi::base::Handle<mi::mdl::IThread_context> ctx( create_thread_context( mdl.get(), context));

    mi::base::Handle<mi::mdl::IMDL_resource_set> res_set( resolver->resolve_resource_file_name(
        file_path, module_file_system_path, module_name, /*pos*/ nullptr, ctx.get()));

    if( log_messages)
        convert_and_log_messages( resolver->access_messages(), /*out_messages*/ nullptr);
    if( !res_set)
        return nullptr;
    if( res_set->get_count() != 1 || res_set->get_udim_mode() != mi::mdl::NO_UDIM)
        return nullptr;
    mi::base::Handle<mi::mdl::IMDL_resource_element const> elem( res_set->get_element( 0));
    return elem->open_reader( 0);
}

/// Calls the MDL entity resolver with the given arguments and returns the result of
/// mi::mdl::IMDL_resource_reader::get_filename(), or the empty string in case of failure.
std::string resolve_resource_filename(
    const char* file_path,
    const char* module_file_system_path,
    const char* module_name,
    bool log_messages,
    Execution_context* context)
{
    mi::base::Handle<mi::mdl::IMDL_resource_set> name_set( get_resource_set(
        file_path, module_file_system_path, module_name, log_messages, context));
    if( !name_set)
        return "";

    mi::base::Handle<mi::mdl::IMDL_resource_element const> elem(name_set->get_element(0));
    if (!elem)
        return "";
    const char* resolved_filename = elem->get_filename( 0);
    return std::string( resolved_filename ? resolved_filename : "");
}

/// Decomposes a resolved filename of a resource into the various pieces:
///
/// - nullptr: memory-based resources, all strings are empty
/// - not part of a container: file-based resource, returns input in \p filename
/// - part of a container: container-based resource, splits input into \p container_filename
///   and \p container_membername
void decompose_resolved_filename(
    const char* resolved_filename,
    std::string& filename,
    std::string& container_filename,
    std::string& container_membername)
{
    filename.clear();
    container_filename.clear();
    container_membername.clear();

    // memory-based resources
    if( !resolved_filename)
        return;

    // file-based resources
    if( !is_container_member( resolved_filename)) {
        filename = resolved_filename;
        return;
    }

    // container-based resources
    container_filename   = get_container_filename( resolved_filename);
    container_membername = get_container_membername( resolved_filename);
}

/// Returns a decoded module name suitable for error messages. Support \c NULL arguments.
std::string get_module_name_for_error_msg( const char* module_name)
{
    if( !module_name)
        return "(no module name available)";
    return decode_for_error_msg( module_name);
}

} // namespace

std::string unresolve_resource_filename(
    const char* filename,
    const char* module_filename,
    const char* module_name,
    Execution_context* context)
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
        file_path.c_str(), module_filename, module_name, /*log_messages*/ false, context);
    resolved_filename = HAL::Ospath::normpath_v2( resolved_filename);
    if( is_container_member( resolved_filename.c_str()))
        return "";
    if( resolved_filename != norm_filename)
        return "";

    // return absolute MDL file path in case of success
    ASSERT( M_SCENE, is_absolute_mdl_file_path( file_path));
    return file_path;
}

std::string unresolve_resource_filename(
    const char* container_filename,
    const char* container_membername,
    const char* module_filename,
    const char* module_name,
    Execution_context* context)
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
        file_path.c_str(), module_filename, module_name, /*log_messages*/ false, context);
    resolved_filename = HAL::Ospath::normpath_v2( resolved_filename);
    if( !is_container_member( resolved_filename.c_str()))
        return "";
    resolved_filename = get_container_filename( resolved_filename.c_str());
    if( resolved_filename != norm_filename)
        return "";

    // return absolute MDL file path in case of success
    ASSERT( M_SCENE, is_absolute_mdl_file_path( file_path));
    return file_path;
}

DB::Tag mdl_resource_to_tag(
    DB::Transaction* transaction,
    const mi::mdl::IValue_resource* value,
    const char* module_filename,
    const char* module_name,
    bool errors_are_warnings,
    Execution_context* context)
{
    switch( value->get_kind()) {

        case mi::mdl::IValue::VK_TEXTURE: {
            const mi::mdl::IValue_texture* texture = cast<mi::mdl::IValue_texture>( value);
            return mdl_texture_to_tag(
                transaction,
                texture,
                module_filename,
                module_name,
                errors_are_warnings,
                context);
        }

        case mi::mdl::IValue::VK_LIGHT_PROFILE: {
            const mi::mdl::IValue_light_profile* lp = cast<mi::mdl::IValue_light_profile>( value);
            return mdl_light_profile_to_tag(
                transaction,
                lp,
                module_filename,
                module_name,
                errors_are_warnings,
                context);
        }

        case mi::mdl::IValue::VK_BSDF_MEASUREMENT: {
            const mi::mdl::IValue_bsdf_measurement* bsdfm
                = cast<mi::mdl::IValue_bsdf_measurement>( value);
            return mdl_bsdf_measurement_to_tag(
                transaction,
                bsdfm,
                module_filename,
                module_name,
                errors_are_warnings,
                context);
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
    const char* module_name,
    bool errors_are_warnings,
    Execution_context* context)
{
    mi::Uint32 tag_uint32 = value->get_tag_value();

    // Check whether the tag value is set before checking whether the string value is not set.
    // Resources in compiled materials typically do not have the string value set.
    if( tag_uint32)
        return DB::Tag( tag_uint32);

    const char* file_path = value->get_string_value();

    // Fail if neither tag nor string value is set.
    if( !file_path || !file_path[0])
        return DB::Tag();

    mi::mdl::IType_texture::Shape shape
        = as<mi::mdl::IType_texture>( value->get_type())->get_shape();
    mi::Float32 gamma = convert_gamma_enum_to_float( value->get_gamma_mode());
    const char* selector = value->get_selector();
    if( selector && !selector[0])
        selector = nullptr;

    // Convert string value into tag value.
    return mdl_texture_to_tag(
        transaction,
        file_path,
        module_filename,
        module_name,
        errors_are_warnings,
        shape,
        gamma,
        selector,
        /*shared*/ true,
        context);
}

DB::Tag mdl_texture_to_tag(
    DB::Transaction* transaction,
    const char* file_path,
    const char* module_filename,
    const char* module_name,
    bool errors_are_warnings,
    mi::mdl::IType_texture::Shape shape,
    mi::Float32 gamma,
    const char* selector,
    bool shared,
    Execution_context* context)
{
    // Check for volume textures.
    std::string extension = STRING::to_lower( HAL::Ospath::get_ext( file_path));
    if( extension == ".vdb" && shape == mi::mdl::IType_texture::TS_3D) {
        return mdl_volume_texture_to_tag(
            transaction,
            file_path,
            module_filename,
            module_name,
            errors_are_warnings,
            selector,
            /*shared*/ true,
            context);
    }

    mi::base::Handle<mi::mdl::IMDL_resource_set> resource_set( get_resource_set(
        file_path, module_filename, module_name, /*log_messages*/ true, context));
    mi::base::Handle<const mi::mdl::IMDL_resource_element> elem(
        resource_set ? resource_set->get_element( 0) : nullptr);
    if( !elem) {
        std::string decoded_mdl_module_name = get_module_name_for_error_msg( module_name);
        std::string msg = std::string( "Failed to resolve \"") + file_path
            + "\" in \"" + decoded_mdl_module_name + "\".";
        if( errors_are_warnings) {
            LOG::mod_log->warning( M_SCENE, LOG::Mod_log::C_IO, "%s", msg.c_str());
            add_warning_message( context, msg.c_str());
        } else {
            add_error_message( context, msg.c_str(), -4);
        }
        return DB::Tag();
    }

    const char* first_filename = elem->get_filename( 0);
    if( !first_filename)
        first_filename = "(no filename)";
    LOG::mod_log->debug( M_SCENE, LOG::Mod_log::C_IO,
        "Resolved \"%s\" in \"%s\" to \"%s\"%s.",
        file_path,
        module_name,
        first_filename,
        resource_set->get_count() > 1 ? " ... " : "");

    // Consider the presence of a custom loading wait handle factory as a sign that the
    // application does not support callbacks under a mutex.
    bool no_callbacks_under_mutex = context && make_handle(
        context->get_interface_option<const mi::base::IInterface>(
            MDL_CTX_OPTION_LOADING_WAIT_HANDLE_FACTORY));
    if( no_callbacks_under_mutex)
        resource_set = new Local_mdl_resource_set( resource_set.get());

    Mdl_image_set image_set( resource_set.get(), file_path, selector);
    mi::base::Uuid hash = get_hash( resource_set.get());

    std::unique_lock<std::mutex> lock( g_transaction_mutex);

    mi::Sint32 result = 0;
    DB::Tag tag = TEXTURE::load_mdl_texture( transaction, &image_set, hash, shared, gamma, result);
    ASSERT( M_SCENE, result == 0 || result == -3 || result == -7 || result == -10);
    // Note: tag.is_valid() && (result != 0) is feasible.
    // TODO MDL-1136 harmonize valid tag XOR non-zero result
    ASSERT( M_SCENE, tag.is_valid() | (result != 0));

    if( result != 0) {
        std::string decoded_mdl_module_name = get_module_name_for_error_msg( module_name);
        std::string msg;
        if( result == -3) {
            msg = std::string( "No image plugin found to handle \"") + file_path
                + "\" in \"" + decoded_mdl_module_name + "\".";
        } else if( result == -7) {
            msg = std::string( "File format error in \"") + file_path
                + "\" in \"" + decoded_mdl_module_name + "\".";
        } else if( result == -10) {
            msg = std::string( "Failed to apply selector \"") + (selector ? selector : "")
                + "\" to \"" + file_path
                + "\" in \"" + decoded_mdl_module_name + "\".";
        }

        if( errors_are_warnings) {
            LOG::mod_log->warning( M_SCENE, LOG::Mod_log::C_IO, "%s", msg.c_str());
            add_warning_message( context, msg.c_str());
        } else {
            add_error_message( context, msg.c_str(), result);
        }
        return tag;
    }

    LOG::mod_log->debug( M_SCENE, LOG::Mod_log::C_IO,
        "... and mapped to texture \"%s\" (tag %u).",
        transaction->tag_to_name( tag), tag.get_uint());

    return tag;
}

DB::Tag mdl_volume_texture_to_tag(
    DB::Transaction* transaction,
    const char* file_path,
    const char* module_filename,
    const char* module_name,
    bool errors_are_warnings,
    const char* selector,
    bool shared,
    Execution_context* context)
{
    return DB::Tag();
}

DB::Tag mdl_light_profile_to_tag(
    DB::Transaction* transaction,
    const mi::mdl::IValue_light_profile* value,
    const char* module_filename,
    const char* module_name,
    bool errors_are_warnings,
    Execution_context* context)
{
    mi::Uint32 tag_uint32 = value->get_tag_value();

    // Check whether the tag value is set before checking whether the string value is not set.
    // Resources in compiled materials typically do not have the string value set.
    if( tag_uint32)
        return DB::Tag( tag_uint32);

    const char* file_path = value->get_string_value();

    // Fail if neither tag nor string value is set.
    if( !file_path || !file_path[0])
        return DB::Tag();

    // Convert string value into tag value.
    return mdl_light_profile_to_tag(
        transaction,
        file_path,
        module_filename,
        module_name,
        /*shared*/ true,
        errors_are_warnings,
        context);
}

DB::Tag mdl_light_profile_to_tag(
    DB::Transaction* transaction,
    const char* file_path,
    const char* module_filename,
    const char* module_name,
    bool shared,
    bool errors_are_warnings,
    Execution_context* context)
{
    std::string extension = STRING::to_lower( HAL::Ospath::get_ext( file_path));
    if( extension != ".ies") {
        std::string decoded_mdl_module_name = get_module_name_for_error_msg( module_name);
        std::string msg = std::string( "Invalid extension for light profile \"") + file_path
            + "\" in \"" + decoded_mdl_module_name + "\".";
        if( errors_are_warnings) {
            LOG::mod_log->warning( M_SCENE, LOG::Mod_log::C_IO, "%s", msg.c_str());
            add_warning_message( context, msg.c_str());
        } else {
            add_error_message( context, msg.c_str(), -3);
        }
        return DB::Tag();
    }

    mi::base::Handle<mi::mdl::IMDL_resource_reader> reader(
        get_reader( file_path, module_filename, module_name, /*log_messages*/ true, context));
    if( !reader) {
        std::string decoded_mdl_module_name = get_module_name_for_error_msg( module_name);
        std::string msg = std::string( "Failed to resolve \"") + file_path
            + "\" in \"" + decoded_mdl_module_name + "\".";
        if( errors_are_warnings) {
            LOG::mod_log->warning( M_SCENE, LOG::Mod_log::C_IO, "%s", msg.c_str());
            add_warning_message( context, msg.c_str());
        } else {
            add_error_message( context, msg.c_str(), -4);
        }
        return DB::Tag();
    }

    const char* resolved_filename = reader->get_filename();
    LOG::mod_log->debug( M_SCENE, LOG::Mod_log::C_IO,
        "Resolved \"%s\" in \"%s\" to \"%s\".",
        file_path, module_name, resolved_filename ? resolved_filename : "(no filename)");

    const std::string& absolute_mdl_file_path = reader->get_mdl_url();
    mi::base::Uuid hash = get_hash( reader.get());

    std::string filename;
    std::string container_filename;
    std::string container_membername;
    decompose_resolved_filename(
        resolved_filename, filename, container_filename, container_membername);

    std::unique_lock<std::mutex> lock( g_transaction_mutex);

    Resource_reader_impl wrapped_reader( reader.get());
    mi::Sint32 result = 0;
    DB::Tag tag = LIGHTPROFILE::load_mdl_lightprofile(
        transaction,
        &wrapped_reader,
        filename,
        container_filename,
        container_membername,
        absolute_mdl_file_path,
        hash,
        shared,
        result);
    ASSERT( M_SCENE, result == 0 || result == -7);
    // Note: tag.is_valid() && (result != 0) is *not* feasible.
    ASSERT( M_SCENE, tag.is_valid() ^ (result != 0));

    if( result == -7) {
        std::string decoded_mdl_module_name = get_module_name_for_error_msg( module_name);
        std::string msg = std::string( "File format error in \"") + file_path
            + "\" in \"" + decoded_mdl_module_name + "\".";
        if( errors_are_warnings) {
            LOG::mod_log->warning( M_SCENE, LOG::Mod_log::C_IO, "%s", msg.c_str());
            add_warning_message( context, msg.c_str());
        } else {
            add_error_message( context, msg.c_str(), result);
        }
        return tag;
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
    const char* module_name,
    bool errors_are_warnings,
    Execution_context* context)
{
    mi::Uint32 tag_uint32 = value->get_tag_value();

    // Check whether the tag value is set before checking whether the string value is not set.
    // Resources in compiled materials typically do not have the string value set.
    if( tag_uint32)
        return DB::Tag( tag_uint32);

    const char* file_path = value->get_string_value();

    // Fail if neither tag nor string value is set.
    if( !file_path || !file_path[0])
        return DB::Tag();

    // Convert string value into tag value.
    return mdl_bsdf_measurement_to_tag(
        transaction,
        file_path,
        module_filename,
        module_name,
        /*shared*/ true,
        errors_are_warnings,
        context);
}

DB::Tag mdl_bsdf_measurement_to_tag(
    DB::Transaction* transaction,
    const char* file_path,
    const char* module_filename,
    const char* module_name,
    bool shared,
    bool errors_are_warnings,
    Execution_context* context)
{
    std::string extension = STRING::to_lower( HAL::Ospath::get_ext( file_path));
    if( extension != ".mbsdf") {
        std::string decoded_mdl_module_name = get_module_name_for_error_msg( module_name);
        std::string msg = std::string( "Invalid extension for BSDF measurement \"") + file_path
            + "\" in \"" + decoded_mdl_module_name + "\".";
        if( errors_are_warnings) {
            LOG::mod_log->warning( M_SCENE, LOG::Mod_log::C_IO, "%s", msg.c_str());
            add_warning_message( context, msg.c_str());
        } else {
            add_error_message( context, msg.c_str(), -3);
        }
        return DB::Tag();
    }

    mi::base::Handle<mi::mdl::IMDL_resource_reader> reader(
        get_reader( file_path, module_filename, module_name, /*log_messages*/ true, context));
    if( !reader) {
        std::string decoded_mdl_module_name = get_module_name_for_error_msg( module_name);
        std::string msg = std::string( "Failed to resolve \"") + file_path
            + "\" in \"" + decoded_mdl_module_name + "\".";
        if( errors_are_warnings) {
            LOG::mod_log->warning( M_SCENE, LOG::Mod_log::C_IO, "%s", msg.c_str());
            add_warning_message( context, msg.c_str());
        } else {
            add_error_message( context, msg.c_str(), -4);
        }
        return DB::Tag();
    }

    const char* resolved_filename = reader->get_filename();
    LOG::mod_log->debug( M_SCENE, LOG::Mod_log::C_IO,
        "Resolved \"%s\" in \"%s\" to \"%s\".",
        file_path, module_name, resolved_filename ? resolved_filename : "(no filename)");

    const std::string& absolute_mdl_file_path = reader->get_mdl_url();
    mi::base::Uuid hash = get_hash( reader.get());

    std::string filename;
    std::string container_filename;
    std::string container_membername;
    decompose_resolved_filename(
        resolved_filename, filename, container_filename, container_membername);

    std::unique_lock<std::mutex> lock( g_transaction_mutex);

    Resource_reader_impl wrapped_reader( reader.get());
    mi::Sint32 result = 0;
    DB::Tag tag = BSDFM::load_mdl_bsdf_measurement(
        transaction,
        &wrapped_reader,
        filename,
        container_filename,
        container_membername,
        absolute_mdl_file_path,
        hash,
        shared,
        result);
    ASSERT( M_SCENE, result == 0 || result == -7);
    // Note: tag.is_valid() && (result != 0) is *not* feasible.
    ASSERT( M_SCENE, tag.is_valid() ^ (result != 0));

    if( result == -7) {
        std::string decoded_mdl_module_name = get_module_name_for_error_msg( module_name);
        std::string msg = std::string( "File format error in \"") + file_path
            + "\" in \"" + decoded_mdl_module_name + "\".";
        if( errors_are_warnings) {
            LOG::mod_log->warning( M_SCENE, LOG::Mod_log::C_IO, "%s", msg.c_str());
            add_warning_message( context, msg.c_str());
        } else {
            add_error_message( context, msg.c_str(), result);
        }
        return tag;
    }

    LOG::mod_log->debug( M_SCENE, LOG::Mod_log::C_IO,
        "... and mapped to scene element \"%s\" (tag %u).",
        transaction->tag_to_name( tag), tag.get_uint());
    return tag;
}

static std::atomic_uint32_t uniq_id_for_generate_suffix = 0;

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
        return nullptr;

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

Input_stream_reader_impl::Input_stream_reader_impl( mi::mdl::IInput_stream* stream)
  : m_stream( stream, mi::base::DUP_INTERFACE),
    m_eof( false)
{
}

mi::Sint64 Input_stream_reader_impl::read( char* buffer, mi::Sint64 size)
{
    mi::Sint64 read_size = 0;

    while( read_size < size) {
        int c = m_stream->read_char();
        if( c == -1) {
            m_eof = true;
            break;
        }
        buffer[read_size++] = c;
    }

    return read_size;
}

bool Input_stream_reader_impl::readline( char* buffer, mi::Sint32 size)
{
    if( size == 0)
        return false;

    mi::Sint64 read_size = 0;

    while( read_size < size-1) {
        int c = m_stream->read_char();
        if( c == -1) {
            m_eof = true;
            break;
        }
        buffer[read_size++] = c;
        if( c == '\n')
            break;
    }

    buffer[read_size++] = '\0';

    return true;
}

Output_stream_impl::Output_stream_impl( mi::neuraylib::IWriter* writer)
  : m_writer( writer, mi::base::DUP_INTERFACE),
    m_error( false)
{
}

void Output_stream_impl::write_char( char c)
{
    mi::Sint64 out_len = m_writer->write( &c, 1);
    if( out_len != 1)
        m_error = true;
}

void Output_stream_impl::write( const char* s)
{
    size_t in_len = strlen( s);
    mi::Sint64 out_len = m_writer->write( s, in_len);
    if( static_cast<size_t>(out_len) != in_len)
        m_error = true;
}

void Output_stream_impl::flush()
{
    if( !m_writer->flush())
        m_error = true;
}

bool Output_stream_impl::unput( char c)
{
    // unsupported
    return false;
}

Resource_reader_impl::Resource_reader_impl( mi::mdl::IMDL_resource_reader* reader)
  : m_reader( reader, mi::base::DUP_INTERFACE)
{
}

bool Resource_reader_impl::eof() const
{
    return tell_absolute() == get_file_size();
}

bool Resource_reader_impl::rewind()
{
    return m_reader->seek( 0, mi::mdl::IMDL_resource_reader::MDL_SEEK_SET);
}

mi::Sint64 Resource_reader_impl::tell_absolute() const
{
    return m_reader->tell();
}

bool Resource_reader_impl::seek_absolute( mi::Sint64 pos)
{
    return m_reader->seek( pos, mi::mdl::IMDL_resource_reader::MDL_SEEK_SET);
}

mi::Sint64 Resource_reader_impl::get_file_size() const
{
    mi::Uint64 pos = m_reader->tell();
    m_reader->seek( 0, mi::mdl::IMDL_resource_reader::MDL_SEEK_END);
    mi::Uint64 size = m_reader->tell();
    m_reader->seek( pos, mi::mdl::IMDL_resource_reader::MDL_SEEK_SET);
    return size;
}

bool Resource_reader_impl::seek_end()
{
    return m_reader->seek( 0, mi::mdl::IMDL_resource_reader::MDL_SEEK_END);
}

mi::Sint64 Resource_reader_impl::read( char* buffer, mi::Sint64 size)
{
    return m_reader->read( buffer, size);
}

bool Resource_reader_impl::readline( char* buffer, mi::Sint32 size)
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

Input_stream_impl::Input_stream_impl( mi::neuraylib::IReader* reader, const std::string& filename)
  : m_reader( reader, mi::base::DUP_INTERFACE),
    m_filename( filename)
{
}

int Input_stream_impl::read_char()
{
    char c;
    mi::Sint64 result = m_reader->read( &c, 1);
    return result <= 0 ? -1 : c;
}

const char* Input_stream_impl::get_filename()
{
    return !m_filename.empty() ? m_filename.c_str() : nullptr;
}

Mdle_input_stream_impl::Mdle_input_stream_impl(
    mi::neuraylib::IReader* reader, const std::string& filename)
  : Input_stream_impl( reader, filename)
{
}

int Mdle_input_stream_impl::read_char()
{
    return Input_stream_impl::read_char();
}

const char* Mdle_input_stream_impl::get_filename()
{
    return Input_stream_impl::get_filename();
}

Mdl_resource_reader_impl::Mdl_resource_reader_impl(
    mi::neuraylib::IReader* reader,
    const std::string& file_path,
    const std::string& filename,
    const mi::base::Uuid& hash)
  : m_reader( reader, mi::base::DUP_INTERFACE),
    m_file_path( file_path),
    m_filename( filename),
    m_hash( hash)
{
    ASSERT( M_SCENE, reader->supports_absolute_access());
}

Uint64 Mdl_resource_reader_impl::read( void* ptr, Uint64 size)
{
    return m_reader->read( static_cast<char*>( ptr), size);
}

Uint64 Mdl_resource_reader_impl::tell()
{
    return m_reader->tell_absolute();
}

bool Mdl_resource_reader_impl::seek( Sint64 offset, Position origin)
{
    mi::Sint64 position = 0;
    switch( origin) {
        case MDL_SEEK_SET: position = 0; break;
        case MDL_SEEK_CUR: position = m_reader->tell_absolute(); break;
        case MDL_SEEK_END: position = m_reader->get_file_size(); break;
    }

    return m_reader->seek_absolute( position + offset);
}

const char* Mdl_resource_reader_impl::get_filename() const
{
    return !m_filename.empty() ? m_filename.c_str() : nullptr;
}

const char* Mdl_resource_reader_impl::get_mdl_url() const
{
    return !m_file_path.empty() ? m_file_path.c_str() : nullptr;
}

bool Mdl_resource_reader_impl::get_resource_hash( unsigned char hash[16])
{
    return convert_hash( m_hash, hash);
}

mi::neuraylib::IReader* Mdl_container_callback::get_reader(
    const char* container_filename, const char* member_filename)
{
    ASSERT( M_SCENE, container_filename && member_filename);

    SYSTEM::Access_module<MDLC::Mdlc_module> mdlc_module( false);
    mi::base::Handle<mi::mdl::IMDL> mdl( mdlc_module->get_mdl());
    mi::base::Handle<mi::mdl::IInput_stream> input_stream;

    if (is_archive_filename(container_filename))
    {
        mi::base::Handle<mi::mdl::IArchive_tool> archive_tool(mdl->create_archive_tool());
        input_stream = archive_tool->get_file_content(container_filename, member_filename);
    }
    else if (is_mdle_filename(container_filename))
    {
        mi::base::Handle<mi::mdl::IEncapsulate_tool> mdle_tool(mdl->create_encapsulate_tool());
        input_stream = mdle_tool->get_file_content(container_filename, member_filename);
    }

    if (!input_stream)
        return nullptr;

    mi::base::Handle<mi::mdl::IMDL_resource_reader> file_random_access(
        input_stream->get_interface<mi::mdl::IMDL_resource_reader>());

    ASSERT( M_SCENE, file_random_access.get());
    return new Resource_reader_impl( file_random_access.get());
}

Mdl_image_set::Mdl_image_set(
    mi::mdl::IMDL_resource_set* set, const std::string& file_path, const char* selector)
  : m_resource_set( set, mi::base::DUP_INTERFACE),
    m_selector( selector ? selector : ""),
    m_is_container( false)
{
    mi::base::Handle<mi::mdl::IMDL_resource_element const> elem(set->get_element(0));
    const char* first_filename = elem ? elem->get_filename(0) : nullptr;
    if( first_filename) {
        m_container_filename = MDL::get_container_filename( first_filename);
        m_is_container = !m_container_filename.empty();
    }

    m_mdl_file_path = set->get_mdl_url_mask();

    m_file_format = STRING::to_lower( HAL::Ospath::get_ext( m_mdl_file_path));
    if( !m_file_format.empty() && m_file_format[0] == '.' )
        m_file_format = m_file_format.substr( 1);
}

bool Mdl_image_set::is_mdl_container() const
{
    return m_is_container;
}

const char* Mdl_image_set::get_original_filename() const
{
    return "";
}

const char* Mdl_image_set::get_container_filename() const
{
    return m_container_filename.c_str();
}

const char* Mdl_image_set::get_mdl_file_path() const
{
    return m_mdl_file_path.c_str();
}

const char* Mdl_image_set::get_selector() const
{
    return !m_selector.empty() ? m_selector.c_str() : nullptr;
}

const char* Mdl_image_set::get_image_format() const
{
    return m_file_format.c_str();
}

bool Mdl_image_set::is_uvtile() const
{
    return m_resource_set->get_udim_mode() != mi::mdl::NO_UDIM;
}

bool Mdl_image_set::is_animated() const
{
    return m_resource_set->has_sequence_marker();
}

mi::Size Mdl_image_set::get_length() const
{
    return m_resource_set->get_count();
}

mi::Size Mdl_image_set::get_frame_number( mi::Size f) const
{
    ASSERT( M_SCENE, f < get_length());

    mi::base::Handle<mi::mdl::IMDL_resource_element const> elem( m_resource_set->get_element( f));
    return elem->get_frame_number();
}

mi::Size Mdl_image_set::get_frame_length( mi::Size f) const
{
    ASSERT( M_SCENE, f < get_length());

    mi::base::Handle<mi::mdl::IMDL_resource_element const> elem( m_resource_set->get_element( f));
    return elem->get_count();
}

void Mdl_image_set::get_uvtile_uv( mi::Size f, mi::Size i, mi::Sint32 &u, mi::Sint32 &v) const
{
    ASSERT( M_SCENE, f < get_length());
    ASSERT( M_SCENE, i < get_frame_length( f));

    mi::base::Handle<mi::mdl::IMDL_resource_element const> elem( m_resource_set->get_element( f));
    if( m_resource_set->get_udim_mode() == mi::mdl::NO_UDIM) {
        u = 0;
        v = 0;
    } else {
        elem->get_udim_mapping( i, u, v);
    }
}

const char* Mdl_image_set::get_resolved_filename( mi::Size f, mi::Size i) const
{
    ASSERT( M_SCENE, f < get_length());
    ASSERT( M_SCENE, i < get_frame_length( f));

    if( m_is_container)
        return "";

    mi::base::Handle<mi::mdl::IMDL_resource_element const> elem( m_resource_set->get_element( f));
    const char* s = elem->get_filename( i);
    return s ? s : "";
}

const char* Mdl_image_set::get_container_membername( mi::Size f, mi::Size i) const
{
    ASSERT( M_SCENE, f < get_length());
    ASSERT( M_SCENE, i < get_frame_length( f));

    if( !m_is_container)
        return "";

    mi::base::Handle<mi::mdl::IMDL_resource_element const> elem( m_resource_set->get_element( f));
    const char* filename = elem->get_filename( i);
    const char* membername = MDL::get_container_membername( filename);
    ASSERT( M_SCENE, membername[0] != '\0');
    return membername;
}

mi::neuraylib::IReader* Mdl_image_set::open_reader( mi::Size f, mi::Size i) const
{
    ASSERT( M_SCENE, f < get_length());
    ASSERT( M_SCENE, i < get_frame_length( f));

    mi::base::Handle<mi::mdl::IMDL_resource_element const> elem( m_resource_set->get_element( f));
    mi::base::Handle<mi::mdl::IMDL_resource_reader> reader( elem->open_reader( i));
    return reader ? new Resource_reader_impl( reader.get()) : nullptr;
}

mi::neuraylib::ICanvas* Mdl_image_set::get_canvas( mi::Size f, mi::Size i) const
{
    return nullptr;
}

std::string lookup_thumbnail(
    const std::string& module_filename,
    const std::string& module_name,
    const std::string& def_simple_name,
    const IAnnotation_block* annotations,
    mi::mdl::IArchive_tool* archive_tool)
{
    for( mi::Size i = 0; annotations && i < annotations->get_size(); ++i) {

        mi::base::Handle<const IAnnotation> anno( annotations->get_annotation( i));
        if( !anno)
            continue;

        if( strcmp( anno->get_name(), "::anno::thumbnail(string)") != 0)
            continue;

         mi::base::Handle<const IExpression_list> expressions(
             anno->get_arguments());
         ASSERT( M_SCENE, expressions->get_size() == 1);

         mi::base::Handle<const IExpression_constant> expr(
             expressions->get_expression<IExpression_constant>( mi::Size( 0)));
         if( !expr)
             break;

         mi::base::Handle<const IValue_string> vstr( expr->get_value<IValue_string>());
         ASSERT( M_SCENE, vstr.is_valid_interface());

         // lookup file
         Execution_context context;
         std::string file = resolve_resource_filename(
             vstr->get_value(),
             module_filename.c_str(),
             module_name.c_str(),
             /*log_messages*/ true,
             &context);
         if( !file.empty())
             return file;
         break;
    }

    // construct thumbnail filename according to the "old" convention
    // module_name.def_simple_name.ext
    const char* ext[] = {"png", "jpg", "jpeg", "PNG", "JPG", "JPEG", nullptr};

    // located in container?
    if( is_archive_filename( module_filename) || is_mdle_filename( module_filename)) {

        std::string module_container_filename
            = get_container_filename( module_filename.c_str());
        std::string module_container_membername
            = get_container_membername( module_filename.c_str());

        if( !has_mdl_suffix( module_container_membername)) {
            assert( false);
            return "";
        }

        // remove ".mdl" to obtain module base name
        std::string module_container_member_basename
            = strip_dot_mdl_suffix( module_container_membername);

        // construct thumbnail base name
        std::string thumbnail_container_member_basename
            = module_container_member_basename + '.' + def_simple_name + '.';

        // check for supported file types
        for( int i = 0; ext[i] != nullptr; ++i) {
            std::string thumbnail_container_membername
                = thumbnail_container_member_basename + ext[i];
            mi::base::Handle<mi::mdl::IInput_stream> file( archive_tool->get_file_content(
                module_container_filename.c_str(), thumbnail_container_membername.c_str()));
            if( file)
                return module_container_filename + ':' + thumbnail_container_membername;
        }

        return "";
    }

    // string-based module?
    if( !has_mdl_suffix( module_filename))
        return "";

    // remove ".mdl" to obtain module base name
    std::string module_basename = strip_dot_mdl_suffix( module_filename);
    assert( !module_basename.empty());

    // construct thumbnail base name
    std::string thumbnail_basename = module_basename + '.' + def_simple_name + '.';

    // check for supported file types
    for( int i = 0; ext[i] != nullptr; ++i) {
        std::string thumbnail_filename = thumbnail_basename + ext[i];
        if( DISK::is_file( thumbnail_filename.c_str()))
            return thumbnail_filename;
    }

    return "";
}

Local_mdl_resource_element::Local_mdl_resource_element( const mi::mdl::IMDL_resource_element* other)
  : m_frame( other->get_frame_number())
{
    size_t count = other->get_count();
    m_entries.reserve(count);

    for( size_t i = 0; i < count; ++i) {
        const char *mdl_url = other->get_mdl_url(i);
        const char *filename = other->get_filename(i);

        Resource_element_entry e;
        if (mdl_url)
            e.m_mdl_url = mdl_url;
        if (filename)
            e.m_filename = filename;
        other->get_udim_mapping(i, e.m_u, e.m_v);
        mi::base::Handle<mi::mdl::IMDL_resource_reader> reader(other->open_reader(i));
        e.m_reader = new Local_mdl_resource_reader(reader.get());
        e.m_hash_valid = other->get_resource_hash(i, e.m_hash);

        m_entries.push_back(e);
    }
}

size_t Local_mdl_resource_element::get_frame_number() const
{
    return m_frame;
}

size_t Local_mdl_resource_element::get_count() const
{
    return m_entries.size();
}

const char* Local_mdl_resource_element::get_mdl_url( size_t i) const
{
    if( i >= m_entries.size())
        return nullptr;

    const Resource_element_entry &elem = m_entries[i];
    return !elem.m_mdl_url.empty() ? elem.m_mdl_url.c_str() : nullptr;
}

const char* Local_mdl_resource_element::get_filename( size_t i) const
{
    if( i >= m_entries.size())
        return nullptr;

    const Resource_element_entry &elem = m_entries[i];
    return !elem.m_filename.empty() ? elem.m_filename.c_str() : nullptr;
}

bool Local_mdl_resource_element::get_udim_mapping( size_t i, int &u, int &v) const
{
    if( i >= m_entries.size())
        return false;

    const Resource_element_entry &elem = m_entries[i];
    u = elem.m_u;
    v = elem.m_v;
    return true;
}

mi::mdl::IMDL_resource_reader* Local_mdl_resource_element::open_reader( size_t i) const
{
    if( i >= m_entries.size())
        return nullptr;

    const Resource_element_entry &elem = m_entries[i];
    if( !elem.m_reader)
        return nullptr;

    elem.m_reader->retain();
    return elem.m_reader.get();
}

bool Local_mdl_resource_element::get_resource_hash( size_t i, unsigned char hash[16]) const
{
    if( i >= m_entries.size())
        return false;

    const Resource_element_entry &elem = m_entries[i];
    if( !elem.m_hash_valid)
        return false;

    const auto &h = elem.m_hash;
    memcpy( hash, h, 16);
    return true;
}

Local_mdl_resource_set::Local_mdl_resource_set( mi::mdl::IMDL_resource_set* other)
  : m_first_frame( other->get_first_frame()),
    m_last_frame( other->get_last_frame()),
    m_has_sequence_marker( other->has_sequence_marker()),
    m_udim_mode( other->get_udim_mode())
{
    const char* mdl_url_mask  = other->get_mdl_url_mask();
    const char* filename_mask = other->get_filename_mask();

    if( mdl_url_mask)
        m_mdl_url_mask  = mdl_url_mask;
    if( filename_mask)
        m_filename_mask = filename_mask;

    size_t count = other->get_count();
    m_elements.reserve( count);

    m_element_idx.resize( m_last_frame - m_first_frame + (count > 0 ? 1 : 0), ~size_t(0));

    for( size_t i = 0; i < count; ++i) {
         mi::base::Handle<mi::mdl::IMDL_resource_element const> elem( other->get_element( i));
         size_t idx = m_elements.size();
         size_t frame = elem->get_frame_number();
         m_elements.push_back( mi::base::make_handle( new Local_mdl_resource_element( elem.get())));
         m_element_idx[frame - m_first_frame] = idx;
    }
}

const char* Local_mdl_resource_set::get_mdl_url_mask() const
{
    return !m_mdl_url_mask.empty() ? m_mdl_url_mask.c_str() : nullptr;
}

const char* Local_mdl_resource_set::get_filename_mask() const
{
    return !m_filename_mask.empty() ? m_filename_mask.c_str() : nullptr;
}

size_t Local_mdl_resource_set::get_count() const
{
    return m_elements.size();
}

bool Local_mdl_resource_set::has_sequence_marker() const
{
    return m_has_sequence_marker;
};

mi::mdl::UDIM_mode Local_mdl_resource_set::get_udim_mode() const
{
    return m_udim_mode;
};

const mi::mdl::IMDL_resource_element* Local_mdl_resource_set::get_element( size_t i) const
{
    if( i >= m_elements.size())
        return nullptr;
    const mi::mdl::IMDL_resource_element* elem = m_elements[i].get();
    elem->retain();
    return elem;
}

size_t Local_mdl_resource_set::get_first_frame() const
{
    return m_first_frame;
}

size_t Local_mdl_resource_set::get_last_frame() const
{
    return m_last_frame;
}

const mi::mdl::IMDL_resource_element* Local_mdl_resource_set::get_frame( size_t frame) const
{
    if( m_first_frame <= frame && frame <= m_last_frame) {
        size_t idx = m_element_idx[frame - m_first_frame];
        if( idx != ~size_t(0)) {
            mi::mdl::IMDL_resource_element const* elem = m_elements[idx].get();
            elem->retain();
            return elem;
        }
    }
    return nullptr;
}

Local_mdl_resource_reader::Local_mdl_resource_reader( mi::mdl::IMDL_resource_reader* other)
{
    const char* filename = other->get_filename();
    const char* mdl_url  = other->get_mdl_url();

    m_size         = 0;
    m_position     = 0;
    if( filename)
        m_filename = filename;
    if( mdl_url)
        m_mdl_url  = mdl_url;
    m_hash_valid   = other->get_resource_hash( m_hash);

    bool success = other->seek( 0, mi::mdl::IMDL_resource_reader::MDL_SEEK_END);
    if( !success)
       return;

    m_size = other->tell();

    success = other->seek( 0, mi::mdl::IMDL_resource_reader::MDL_SEEK_SET);
    if( !success)
       return;

    m_data.resize( m_size);
    mi::Uint64 count = other->read( m_data.data(), m_size);
    if( count < m_size) {
        m_size = count;
        m_data.resize( m_size);
    }
}

mi::Uint64 Local_mdl_resource_reader::read( void* ptr, mi::Uint64 size)
{
   mi::Uint64 count = std::min( size, m_size-m_position);
   memcpy( ptr, &m_data[m_position], count);
   m_position += count;
   return count;
}

bool Local_mdl_resource_reader::seek(
    mi::Sint64 offset, mi::mdl::IMDL_resource_reader::Position origin)
{
    switch( origin) {

        case mi::mdl::IMDL_resource_reader::MDL_SEEK_SET:
            if( offset > static_cast<mi::Sint64>( m_size))
                return false;
            m_position = offset;
            return true;

        case mi::mdl::IMDL_resource_reader::MDL_SEEK_CUR:
            if(    offset > static_cast<mi::Sint64>( m_size-m_position)
                || offset < - static_cast<mi::Sint64>( m_position))
                return false;
            m_position += offset;
            return true;

        case mi::mdl::IMDL_resource_reader::MDL_SEEK_END:
            if( offset > static_cast<mi::Sint64>( m_size))
                return false;
            m_position = m_size - offset;
            return true;

    }

    ASSERT( M_SCENE, false);
    return false;
}

const char* Local_mdl_resource_reader::get_filename() const
{
    return !m_filename.empty() ? m_filename.c_str() : nullptr;
}

const char* Local_mdl_resource_reader::get_mdl_url() const
{
    return !m_mdl_url.empty() ? m_mdl_url.c_str() : nullptr;
}

bool Local_mdl_resource_reader::get_resource_hash( unsigned char hash[16])
{
    if( !m_hash_valid)
        return false;

    memcpy( hash, m_hash, 16);
    return true;
}

} // namespace DETAIL

} // namespace MDL

} // namespace MI

