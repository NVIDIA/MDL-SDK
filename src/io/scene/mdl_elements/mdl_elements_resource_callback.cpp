/***************************************************************************************************
 * Copyright (c) 2012-2025, NVIDIA CORPORATION. All rights reserved.
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

/** \file
 ** \brief Source for the Resource_callback implementation.
 **/

#include "pch.h"

#include "i_mdl_elements_resource_callback.h"

#include <cassert>
#include <sstream>

#include <mi/mdl/mdl_modules.h>
#include <mi/neuraylib/ibsdf_isotropic_data.h>
#include <mi/neuraylib/ibuffer.h>
#include <mi/neuraylib/icanvas.h>
#include <mi/neuraylib/iexport_result.h>
#include <mi/neuraylib/istring.h>

#include <base/data/db/i_db_access.h>
#include <base/data/db/i_db_transaction.h>
#include <base/hal/hal/i_hal_ospath.h>
#include <io/image/image/i_image.h>
#include <io/image/image/i_image_mipmap.h>
#include <io/scene/bsdf_measurement/i_bsdf_measurement.h>
#include <io/scene/dbimage/i_dbimage.h>
#include <io/scene/lightprofile/i_lightprofile.h>
#include <io/scene/mdl_elements/i_mdl_elements_utilities.h>
#include <io/scene/mdl_elements/i_mdl_elements_module.h>
#include <io/scene/texture/i_texture.h>

namespace fs = std::filesystem;

namespace MI {

namespace MDL {

Resource_callback::Resource_callback(
    DB::Transaction* transaction,
    const mi::mdl::IModule* module,
    const char* module_name,
    const char* module_filename,
    Execution_context* context,
    mi::neuraylib::IExport_result_ext* result)
  : m_transaction( transaction),
    m_module( module, mi::base::DUP_INTERFACE),
    m_module_name( module_name),
    m_context( context),
    m_bundle_resources( context->get_option<bool>( MDL_CTX_OPTION_BUNDLE_RESOURCES)),
    m_add_module_prefix(
        context->get_option<bool>( MDL_CTX_OPTION_EXPORT_RESOURCES_WITH_MODULE_PREFIX)),
    m_keep_original_file_paths(
        context->get_option<bool>( MDL_CTX_OPTION_KEEP_ORIGINAL_RESOURCE_FILE_PATHS)),
    m_result( result, mi::base::DUP_INTERFACE),
    m_image_module( false)
{
    m_transaction->pin();

    // Map context option "handle_filename_conflicts" to m_handle_filename_conflicts enum.
    std::string handle_filename_conflicts
        = context->get_option<std::string>( MDL_CTX_OPTION_HANDLE_FILENAME_CONFLICTS);
    if( handle_filename_conflicts == "generate_unique")
        m_handle_filename_conflicts = GENERATE_UNIQUE;
    else if( handle_filename_conflicts == "overwrite_existing")
        m_handle_filename_conflicts = OVERWRITE_EXISTING;
    else if( handle_filename_conflicts == "fail_if_existing")
        m_handle_filename_conflicts = FAIL_IF_EXISTING;
    else {
        ASSERT( M_SCENE, false);
    }

    // And set m_copy_options based on it.
    m_copy_options = (m_handle_filename_conflicts == OVERWRITE_EXISTING)
        ? fs::copy_options::overwrite_existing : fs::copy_options::none;

    // Compute m_module_filename, m_module_filename_c_str, and m_path_prefix.
    if( module_filename) {
        std::error_code ec;
        fs::path path( fs::u8path( module_filename));
        path = fs::absolute( path, ec);
        m_module_filename = path.u8string();
        m_module_filename_c_str = m_module_filename.c_str();
        mi::Size length = m_module_filename.length();
        ASSERT( M_SCENE, length >= 4 && m_module_filename.substr( length-4) == ".mdl");
        m_path_prefix = m_add_module_prefix
            ? m_module_filename.substr( 0, length-4) + '_'
            : get_directory( m_module_filename) + HAL::Ospath::sep();
    }
}

Resource_callback::~Resource_callback()
{
    m_transaction->unpin();
}

namespace {

mi::Float32 convert_rtt_kind_to_gamma( mi::mdl::Resource_tag_tuple::Kind kind)
{
    switch( kind) {
        case mi::mdl::Resource_tag_tuple::RK_TEXTURE_GAMMA_DEFAULT: return 0.0f;
        case mi::mdl::Resource_tag_tuple::RK_TEXTURE_GAMMA_LINEAR:  return 1.0f;
        case mi::mdl::Resource_tag_tuple::RK_TEXTURE_GAMMA_SRGB:    return 2.2f;
        case mi::mdl::Resource_tag_tuple::RK_LIGHT_PROFILE:         return 0.0f;
        case mi::mdl::Resource_tag_tuple::RK_BSDF_MEASUREMENT:      return 0.0f;

        case mi::mdl::Resource_tag_tuple::RK_BAD:
        case mi::mdl::Resource_tag_tuple::RK_INVALID_REF:
        case mi::mdl::Resource_tag_tuple::RK_STRING:
        case mi::mdl::Resource_tag_tuple::RK_SIMPLE_GLOSSY_MULTISCATTER:
        case mi::mdl::Resource_tag_tuple::RK_BACKSCATTERING_GLOSSY_MULTISCATTER:
        case mi::mdl::Resource_tag_tuple::RK_BECKMANN_SMITH_MULTISCATTER:
        case mi::mdl::Resource_tag_tuple::RK_GGX_SMITH_MULTISCATTER:
        case mi::mdl::Resource_tag_tuple::RK_BECKMANN_VC_MULTISCATTER:
        case mi::mdl::Resource_tag_tuple::RK_GGX_VC_MULTISCATTER:
        case mi::mdl::Resource_tag_tuple::RK_WARD_GEISLER_MORODER_MULTISCATTER:
        case mi::mdl::Resource_tag_tuple::RK_SHEEN_MULTISCATTER:
        case mi::mdl::Resource_tag_tuple::RK_MICROFLAKE_SHEEN_GENERAL:
        case mi::mdl::Resource_tag_tuple::RK_MICROFLAKE_SHEEN_MULTISCATTER:
            ASSERT( M_SCENE, false);
            return 0.0f;
    }

    ASSERT( M_SCENE, false);
    return 0.0f;
}

} // namespace

const char* Resource_callback::get_resource_name(
    const mi::mdl::IValue_resource* resource,
    bool supports_strict_relative_path)
{
    return get_resource_name( resource, supports_strict_relative_path, nullptr);
}

const char* Resource_callback::get_resource_name(
    const mi::mdl::IValue_resource* resource,
    bool supports_strict_relative_path,
    Buffer_callback* buffer_callback)
{
    const char* string_value = resource->get_string_value();
    if( m_keep_original_file_paths && !m_bundle_resources)
        return string_value;

    // Resources in modules loaded from disk or string most likely do not have a valid tag in AST
    // representation (only in the DAG representation). There is not much we can do in such a case:
    // If resolving succeeds, so does unresolving, and we return the same (or an equivalent) string.
    // If resolving fails, we return \c nullptr, which triggers the default action of printing the
    // string value (and the useless tag version). So returning the string value seems to be the
    // best solution.
    DB::Tag tag( resource->get_tag_value());
    if( !tag && m_module) {

        // However, if the resources are referenced in the current module, we can get the tag from
        // this module
        DB::Tag module_tag = m_transaction->name_to_tag( m_module_name.c_str());
        if( module_tag) {

            SERIAL::Class_id class_id = m_transaction->get_class_id( module_tag);
            if( class_id != ID_MDL_MODULE) {
                add_error_resource_type( 6010, "module", module_tag);
                return nullptr;
            }

            mi::Float32 res_gamma = 0.0f;
            std::string res_selector;
            const auto* texture = mi::mdl::as<mi::mdl::IValue_texture>( resource);
            if( texture) {
                res_gamma    = convert_gamma_enum_to_float( texture->get_gamma_mode());
                res_selector = texture->get_selector();
            }

            DB::Access<Mdl_module> db_module( module_tag, m_transaction);
            for( mi::Size i = 0, n = db_module->get_resources_count(); i < n; ++i) {

                const Resource_tag_tuple* element = db_module->get_resource_tag_tuple( i);
                mi::Float32 element_gamma = convert_rtt_kind_to_gamma( element->m_kind);

                if(    element->m_mdl_file_path == string_value
                    && element_gamma            == res_gamma
                    && element->m_selector      == res_selector) {

                    tag = element->m_tag;
                    break;
                }
            }
        }
    }

    if( !tag)
        return string_value;

    // Return result for already translated tags.
    if( !m_file_paths[tag].empty())
        return m_file_paths[tag].c_str();

    std::string result;

    switch( resource->get_kind()) {
        case mi::mdl::IValue::VK_TEXTURE:
            result = handle_texture( tag, supports_strict_relative_path, buffer_callback);
            break;
        case mi::mdl::IValue::VK_LIGHT_PROFILE:
            result = handle_light_profile( tag, supports_strict_relative_path, buffer_callback);
            break;
        case mi::mdl::IValue::VK_BSDF_MEASUREMENT:
            result = handle_bsdf_measurement( tag, supports_strict_relative_path, buffer_callback);
            break;
        default:
            ASSERT( M_SCENE, false);
    }

    if( result.empty())
        return nullptr;

    m_file_paths[tag] = result;
    return m_file_paths[tag].c_str();
}

std::string Resource_callback::handle_texture(
    DB::Tag texture_tag,
    bool supports_strict_relative_path,
    Buffer_callback* buffer_callback)
{
    SERIAL::Class_id class_id = m_transaction->get_class_id( texture_tag);
    if( class_id != TEXTURE::Texture::id) {
        add_error_resource_type( 6010, "texture", texture_tag);
        return {};
    }

    DB::Access<TEXTURE::Texture> texture( texture_tag, m_transaction);
    DB::Tag image_tag( texture->get_image());
    DB::Tag volume_tag( texture->get_volume_data());
    ASSERT( M_SCENE, !image_tag || !volume_tag);

    if( image_tag) {

        // Repeat caching on the image level in addition to the texture level. Otherwise, different
        // textures pointing to the same image might result in duplicated re-exportes files.
        if( !m_file_paths[image_tag].empty())
            return m_file_paths[image_tag].c_str();

        std::string result = handle_texture_image(
            image_tag, supports_strict_relative_path, buffer_callback);
        if( result.empty())
            return {};

        m_file_paths[image_tag] = result;
        return m_file_paths[image_tag].c_str();

    } else if( volume_tag) {

        // Repeat caching on the volume level in addition to the texture level. Otherwise, different
        // textures pointing to the same volume might result in duplicated re-exportes files.
        if( !m_file_paths[volume_tag].empty())
            return m_file_paths[volume_tag].c_str();

        std::string result = handle_texture_volume(
            volume_tag, supports_strict_relative_path, buffer_callback);
        if( result.empty())
            return {};

        m_file_paths[volume_tag] = result;
        return m_file_paths[volume_tag].c_str();

    } else
        return {};
}

std::string Resource_callback::handle_texture_image(
    DB::Tag image_tag,
    bool supports_strict_relative_path,
    Buffer_callback* buffer_callback)
{
    SERIAL::Class_id class_id = m_transaction->get_class_id( image_tag);
    if( class_id != DBIMAGE::Image::id) {
        add_error_resource_type( 6010, "image", image_tag);
        return {};
    }

    DB::Access<DBIMAGE::Image> image( image_tag, m_transaction);

    // File-based images
    if( image->is_file_based()) {

        // Use original file if bundling is not requested and the file can be found via the
        // search paths. We check only the first frame/tile here.
        if( !m_bundle_resources) {

            const std::string& filename_0_0 = image->get_filename( /*frame_id*/ 0, /*uvtile_id*/ 0);
            ASSERT( M_SCENE, m_module_name.substr( 0, 5) == "mdl::");
            const std::string& file_path = DETAIL::unresolve_resource_filename(
                filename_0_0.c_str(), m_module_filename_c_str, &m_module_name[3], m_context);

            if( !file_path.empty()) {

                if( !image->is_animated() && !image->is_uvtile())
                    return file_path;

                // Reconstructing the required markers from file_path or the individual filenames
                // is quite difficult. Instead, combine the directory part of file_path with the
                // filename part of the original MDL file path or filename.
                const std::string& mdl_file_path = image->get_mdl_file_path();
                const std::string& original_filename = image->get_original_filename();
                const std::string suffix
                    = !mdl_file_path.empty() ? mdl_file_path : original_filename;
                ASSERT( M_SCENE, !suffix.empty());
                return construct_mdl_file_path( file_path, suffix);
            }
        }

        // Fail if we need to export the resource, but no module filename is given (string-based
        // export).
        if( m_module_filename.empty()) {
            add_error_string_based( 6011, "file-based", "image", image_tag);
            return {};
        }

        // Otherwise export the image.
        std::string new_filename  = export_texture_image( image.get_ptr(), buffer_callback);
        if( new_filename.empty()) {
            add_error_export_failed( 6013, "file-based", "image", image_tag);
            return {};
        }

        return make_relative( new_filename, supports_strict_relative_path);

    // Container-based images
    } else if( image->is_container_based()) {


        // Use original file if bundling is not requested and the file can be found via the search
        // paths.
        if( !m_bundle_resources) {

            const std::string& container_filename = image->get_container_filename();
            const std::string& container_membername_0_0
                = image->get_container_membername( /*frame_id*/ 0, /*uvtile_id*/ 0);
            ASSERT( M_SCENE, m_module_name.substr( 0, 5) == "mdl::");
            const std::string& file_path = DETAIL::unresolve_resource_filename(
                container_filename.c_str(),
                container_membername_0_0.c_str(),
                m_module_filename_c_str,
                &m_module_name[3],
                m_context);

            if( !file_path.empty()) {

                if( !image->is_animated() && !image->is_uvtile())
                    return file_path;

                // Reconstructing the required markers from file_path or the individual filenames
                // is quite difficult. Instead, combine the directory part of file_path with the
                // filename part of the original MDL file path or filename.
                const std::string& mdl_file_path = image->get_mdl_file_path();
                const std::string& original_filename = image->get_original_filename();
                const std::string suffix
                    = !mdl_file_path.empty() ? mdl_file_path : original_filename;
                ASSERT( M_SCENE, !suffix.empty());
                return construct_mdl_file_path( file_path, suffix);
            }
        }

        // Fail if we need to export the resource, but no module filename is given (string-based
        // export).
        if( m_module_filename.empty()) {
            add_error_string_based( 6015, "container-based", "image", image_tag);
            return {};
        }

        // Export canvas(es) with a generated filename and return that filename.
        std::string new_filename = export_texture_image( image.get_ptr(), buffer_callback);
        if( new_filename.empty()) {
            add_error_string_based( 6016, "container-based", "image", image_tag);
            return {};
        }

        return make_relative( new_filename, supports_strict_relative_path);

    // Memory-based images
    } else {

        // Fail if we need to export the resource, but no module filename is given (string-based
        // export).
        if( m_module_filename.empty()) {
            add_error_string_based( 6012, "memory-based", "image", image_tag);
            return {};
        }

        // Export canvas(es) with a generated filename and return that filename.
        std::string new_filename = export_texture_image( image.get_ptr(), buffer_callback);
        if( new_filename.empty()) {
            add_error_string_based( 6014, "memory-based", "image", image_tag);
            return {};
        }

        return make_relative( new_filename, supports_strict_relative_path);
    }
}

std::string Resource_callback::handle_texture_volume(
    DB::Tag volume_tag,
    bool supports_strict_relative_path,
    Buffer_callback* buffer_callback)
{
    ASSERT( M_SCENE, false);
    return {};
}

std::string Resource_callback::handle_light_profile(
    DB::Tag tag,
    bool supports_strict_relative_path,
    Buffer_callback* buffer_callback)
{
    SERIAL::Class_id class_id = m_transaction->get_class_id( tag);
    if( class_id != LIGHTPROFILE::Lightprofile::id) {
        add_error_resource_type( 6010, "light profile", tag);
        return {};
    }

    DB::Access<LIGHTPROFILE::Lightprofile> lp( tag, m_transaction);

    // File-based light profiles
    if( lp->is_file_based()) {

        // Use original file if bundling is not requested and the file can be found via the search
        // paths.
        if( !m_bundle_resources) {
            const std::string& filename = lp->get_filename();
            ASSERT( M_SCENE, m_module_name.substr( 0, 5) == "mdl::");
            const std::string& file_path = DETAIL::unresolve_resource_filename(
                filename.c_str(), m_module_filename_c_str, &m_module_name[3], m_context);
            if( !file_path.empty())
                return file_path;
        }

        // Fail if we need to export the resource, but no module filename is given (string-based
        // export).
        if( m_module_filename.empty()) {
            add_error_string_based( 6011, "file-based", "light profile", tag);
            return {};
        }

        // Otherwise export the light profile.
        std::string new_filename  = export_light_profile( lp.get_ptr(), buffer_callback);
        if( new_filename.empty()) {
            add_error_export_failed( 6013, "file-based", "light profile", tag);
            return {};
        }

        return make_relative( new_filename, supports_strict_relative_path);

    // Container-based light profiles
    } else if( lp->is_container_based()) {

        // Use original file if bundling is not requested and the file can be found via the search
        // paths.
        if( !m_bundle_resources) {
            const std::string& container_filename = lp->get_container_filename();
            const std::string& container_membername = lp->get_container_membername();
            ASSERT( M_SCENE, m_module_name.substr( 0, 5) == "mdl::");
            const std::string& file_path = DETAIL::unresolve_resource_filename(
                container_filename.c_str(),
                container_membername.c_str(),
                m_module_filename_c_str,
                &m_module_name[3],
                m_context);
            if( !file_path.empty())
                return file_path;
        }

        // Fail if we need to export the resource, but no module filename is given (string-based
        // export).
        if( m_module_filename.empty()) {
            add_error_string_based( 6015, "container-based", "light profile", tag);
            return {};
        }

        // Otherwise export the light profile.
        const std::string new_filename = export_light_profile( lp.get_ptr(), buffer_callback);
        if( new_filename.empty()) {
            add_error_export_failed( 6016, "container-based", "light profile", tag);
            return {};
        }

        return make_relative( new_filename, supports_strict_relative_path);

    // Memory-based light profiles
    } else {

        // Fail if we need to export the resource, but no module filename is given (string-based
        // export).
        if( m_module_filename.empty()) {
            add_error_string_based( 6012, "memory-based", "light profile", tag);
            return {};
        }

        // Otherwise export the light profile.
        const std::string new_filename = export_light_profile( lp.get_ptr(), buffer_callback);
        if( new_filename.empty()) {
            add_error_export_failed( 6014, "memory-based", "light profile", tag);
            return {};
        }

        return make_relative( new_filename, supports_strict_relative_path);
    }
}

std::string Resource_callback::handle_bsdf_measurement(
    DB::Tag tag,
    bool supports_strict_relative_path,
    Buffer_callback* buffer_callback)
{
    SERIAL::Class_id class_id = m_transaction->get_class_id( tag);
    if( class_id != BSDFM::Bsdf_measurement::id) {
        add_error_resource_type( 6010, "BSDF measurement", tag);
        return {};
    }

    DB::Access<BSDFM::Bsdf_measurement> bsdfm( tag, m_transaction);

    // File-based BSDF measurements
    if( bsdfm->is_file_based()) {

        // Use original file if bundling is not requested and the file can be found via the search
        // paths.
        if( !m_bundle_resources) {
            const std::string& filename = bsdfm->get_filename();
            ASSERT( M_SCENE, m_module_name.substr( 0, 5) == "mdl::");
            const std::string& file_path = DETAIL::unresolve_resource_filename(
                filename.c_str(), m_module_filename_c_str, &m_module_name[3], m_context);
            if( !file_path.empty())
                return file_path;
        }

        // Fail if we need to export the resource, but no module filename is given (string-based
        // export).
        if( m_module_filename.empty()) {
            add_error_string_based( 6011, "file-based", "BSDF measurement", tag);
            return {};
        }

        // Otherwise export the BSDF measurement.
        const std::string new_filename = export_bsdf_measurement( bsdfm.get_ptr(), buffer_callback);
        if( new_filename.empty()) {
            add_error_export_failed( 6013, "file-based", "BSDF measurement", tag);
            return {};
        }

        return make_relative( new_filename, supports_strict_relative_path);

    // Container-based BSDF measurements
    } else if( bsdfm->is_container_based()) {

        // Use original file if bundling is not requested and the file can be found via the search
        // paths.
        if( !m_bundle_resources) {
            const std::string& container_filename = bsdfm->get_container_filename();
            const std::string& container_membername = bsdfm->get_container_membername();
            ASSERT( M_SCENE, m_module_name.substr( 0, 5) == "mdl::");
            const std::string& file_path = DETAIL::unresolve_resource_filename(
                container_filename.c_str(),
                container_membername.c_str(),
                m_module_filename_c_str,
                &m_module_name[3],
                m_context);
            if( !file_path.empty())
                return file_path;
        }

        // Fail if we need to export the resource, but no module filename is given (string-based
        // export).
        if( m_module_filename.empty()) {
            add_error_string_based( 6015, "container-based", "BSDF measurement", tag);
            return {};
        }

        // Otherwise export the BSDF measurement.
        const std::string new_filename = export_bsdf_measurement( bsdfm.get_ptr(), buffer_callback);
        if( new_filename.empty()) {
            add_error_export_failed( 6016, "container-based", "BSDF measurement", tag);
            return {};
        }

        return make_relative( new_filename, supports_strict_relative_path);

    // Memory-based BSDF measurements
    } else {

        // Fail if we need to export the resource, but no module filename is given (string-based
        // export).
        if( m_module_filename.empty()) {
            add_error_string_based( 6012, "memory-based", "BSDF measurement", tag);
            return {};
        }

        // Otherwise export the BSDF measurement.
        const std::string new_filename = export_bsdf_measurement( bsdfm.get_ptr(), buffer_callback);
        if( new_filename.empty()) {
            add_error_export_failed( 6014, "memory-based", "BSDF measurement", tag);
            return {};
        }

        return make_relative( new_filename, supports_strict_relative_path);
    }
}

std::string Resource_callback::export_texture_image(
    const DBIMAGE::Image* image,
    Buffer_callback* buffer_callback)
{
    // Use pixel type of first frame/uvtile to derive the extension.
    const char* new_extension = nullptr;
    {
        mi::base::Handle<const IMAGE::IMipmap> mipmap(
            image->get_mipmap( m_transaction, /*frame_id*/ 0, /*uvtile_id*/ 0));
        mi::base::Handle<const mi::neuraylib::ICanvas> canvas( mipmap->get_level( 0));
        new_extension = get_extension( canvas->get_type());
    }

    bool add_sequence_marker = image->is_animated();
    bool add_uvtile_marker   = image->is_uvtile();

    // Compute number of digits required for frame numbers.
    mi::Size frame_digits = 1;
    mi::Size n = image->get_length();
    mi::Size f = n > 0 ? image->get_frame_number( n-1) : 0;
    while( f > 9) { ++frame_digits; f /= 10; }

    // Compute all tuples of frame numbers and u/v coordinates.
   Frame_uvs frame_uvs;
    for( mi::Size i = 0; i < n; ++i) {
        mi::Size m = image->get_frame_length( i);
        mi::Size frame_number = image->get_frame_number( i);
        for( mi::Size j = 0; j < m; ++j) {
            mi::Sint32 u, v;
            image->get_uvtile_uv( i, j, u, v);
            frame_uvs.push_back( {frame_number, u, v});
        }
    }

    // Figure out whether we can copy all frames/uvtiles. If yes, then we can keep the extension.
    // Otherwise we need to re-export all frames/uvtiles to match the chosen extension.
    bool copy_all = !buffer_callback;
    for( mi::Size i = 0; copy_all && (i < n); ++i) {
        mi::Size m = image->get_frame_length( i);
        for( mi::Size j = 0; copy_all && (j < m); ++j) {
            const std::string& old_filename_fuv = image->get_filename( i, j);
            if( !fs::is_regular_file( fs::u8path( old_filename_fuv)))
                copy_all = false;
        }
    }

    // Both filenames might include frame/uvtile markers.
    std::string old_filename = image->get_original_filename();
    std::string old_basename
        = !old_filename.empty() ? old_filename : image->get_mdl_file_path();
    old_basename = strip_directories( old_basename);

    std::string new_filename = get_new_resource_filename_marker(
        new_extension,
        old_basename.empty() ? nullptr : old_basename.c_str(),
        !copy_all,
        add_sequence_marker,
        add_uvtile_marker,
        frame_digits,
        frame_uvs);
    if( new_filename.empty()) {
        ASSERT( M_SCENE, m_handle_filename_conflicts == FAIL_IF_EXISTING);
        return {};
    }

    bool success = true;

    for( mi::Size i = 0; i < n; ++i) {

        mi::Size frame_number = image->get_frame_number( i);
        mi::Size m = image->get_frame_length( i);

        for( mi::Size j = 0; j < m; ++j) {

            // Actual filenames without frame/uvtile markers.
            const std::string& old_filename_fuv = image->get_filename( i, j);
            mi::Sint32 u, v;
            image->get_uvtile_uv( i, j, u, v);
            std::string new_filename_fuv
                = (add_sequence_marker || add_uvtile_marker)
                ? frame_uvtile_marker_to_string( new_filename, frame_number, u, v)
                : new_filename;
            ASSERT( M_SCENE, !new_filename_fuv.empty());

            if( buffer_callback) {
                // export via buffer callback
                assert( !copy_all);
                mi::base::Handle<const IMAGE::IMipmap> mipmap(
                    image->get_mipmap( m_transaction, i, j));
                mi::base::Handle<const mi::neuraylib::ICanvas> canvas( mipmap->get_level( 0));
                mi::base::Handle<mi::neuraylib::IBuffer> buffer(
                    m_image_module->create_buffer_from_canvas(
                        canvas.get(), new_extension+1, canvas->get_type()));
                if( buffer) {
                    std::string tmp
                        = (*buffer_callback)( buffer.get(), new_filename_fuv.c_str());
                    success &= !tmp.empty();
                } else {
                    success = false;
                }
            } else if( copy_all) {
                // copy the file
                std::error_code ec;
                success &= fs::copy_file(
                    fs::u8path( old_filename_fuv),
                    fs::u8path( new_filename_fuv),
                    m_copy_options,
                    ec);
            } else {
                // export to file
                assert( !copy_all);
                mi::base::Handle<const IMAGE::IMipmap> mipmap(
                    image->get_mipmap( m_transaction, i, j));
                mi::base::Handle<const mi::neuraylib::ICanvas> canvas( mipmap->get_level( 0));
                success &= m_image_module->export_canvas( canvas.get(), new_filename_fuv.c_str());
            }
        }
    }

    return success ? new_filename : std::string();
}


std::string Resource_callback::export_light_profile(
    const LIGHTPROFILE::Lightprofile* profile,
    Buffer_callback* buffer_callback)
{
    std::string old_filename = profile->get_filename();
    std::string old_basename
        = !old_filename.empty() ? old_filename : profile->get_mdl_file_path();
    old_basename = strip_directories( old_basename);

    // Value of use_new_extension does not matter since there is only one valid extension.
    std::string new_filename = get_new_resource_filename(
        ".ies", old_basename.empty() ? nullptr : old_basename.c_str(), /*use_new_extension*/ true);
    if( new_filename.empty()) {
        ASSERT( M_SCENE, m_handle_filename_conflicts == FAIL_IF_EXISTING);
        return {};
    }

    bool success = false;

    if( buffer_callback) {
        // export via buffer callback
        mi::base::Handle<mi::neuraylib::IBuffer> buffer(
            LIGHTPROFILE::create_buffer_from_lightprofile( m_transaction, profile));
        if( buffer) {
            new_filename = (*buffer_callback)( buffer.get(), new_filename.c_str());
            success = !new_filename.empty();
        }
    } else {
        fs::path old_path( fs::u8path( old_filename));
        std::error_code ec;
        if( fs::is_regular_file( old_path, ec)) {
            // copy the file
            success = fs::copy_file( old_path, fs::u8path( new_filename), m_copy_options, ec);
        } else {
            // export to file
            success = LIGHTPROFILE::export_to_file( m_transaction, profile, new_filename);
        }
    }

    return success ? new_filename : std::string();
}

std::string Resource_callback::export_bsdf_measurement(
    const BSDFM::Bsdf_measurement* measurement,
    Buffer_callback* buffer_callback)
{
    std::string old_filename = measurement->get_filename();
    std::string old_basename
        = !old_filename.empty() ? old_filename : measurement->get_mdl_file_path();
    old_basename = strip_directories( old_basename);

    // Value of use_new_extension does not matter since there is only one valid extension.
    std::string new_filename = get_new_resource_filename(
        ".mbsdf",
        old_basename.empty() ? nullptr : old_basename.c_str(),
        /*use_new_extension*/ true);
    if( new_filename.empty()) {
        ASSERT( M_SCENE, m_handle_filename_conflicts == FAIL_IF_EXISTING);
        return {};
    }

    mi::base::Handle<const mi::neuraylib::IBsdf_isotropic_data> refl(
        measurement->get_reflection<mi::neuraylib::IBsdf_isotropic_data>( m_transaction));
    mi::base::Handle<const mi::neuraylib::IBsdf_isotropic_data> trans(
        measurement->get_transmission<mi::neuraylib::IBsdf_isotropic_data>( m_transaction));

    bool success = false;

    if( buffer_callback) {
        // export via buffer callback
        mi::base::Handle<mi::neuraylib::IBuffer> buffer(
            BSDFM::create_buffer_from_bsdf_measurement( refl.get(), trans.get()));
        if( buffer) {
            new_filename = (*buffer_callback)( buffer.get(), new_filename.c_str());
            success = !new_filename.empty();
        }
    } else {
        fs::path old_path( fs::u8path( old_filename));
        std::error_code ec;
        if( fs::is_regular_file( old_path, ec)) {
            // copy the file
            success = fs::copy_file( old_path, fs::u8path( new_filename), m_copy_options, ec);
        } else {
            // export to file
            success = BSDFM::export_to_file( refl.get(), trans.get(), new_filename);
        }
    }

    return success ? new_filename : std::string();
}

bool Resource_callback::collision( const std::string& s, const Frame_uvs& frame_uvs)
{
    std::error_code ec;
    for( const auto& fuv: frame_uvs) {
        std::string t = frame_uvtile_marker_to_string( s, fuv.m_frame_number, fuv.m_u, fuv.m_v);
        if( fs::exists( fs::u8path( t), ec))
            return true;
    }

    return false;
}

std::string Resource_callback::get_new_resource_filename(
    const char* new_extension, const char* old_filename, bool use_new_extension)
{
    ASSERT( M_SCENE, !m_path_prefix.empty());
    ASSERT( M_SCENE, !new_extension || new_extension[0] == '.');
    ASSERT( M_SCENE, old_filename || use_new_extension);
    ASSERT( M_SCENE, !use_new_extension || new_extension);
    ASSERT( M_SCENE, !old_filename
        || (strip_directories( old_filename) == std::string( old_filename)));

    std::error_code ec;

    // Consider a filename derived from old_filename, no counter.
    std::string old_root, old_extension;
    if( old_filename) {

        HAL::Ospath::splitext( old_filename, old_root, old_extension);
        std::string s = m_path_prefix;
        s += old_root;
        s += use_new_extension ? std::string( new_extension) : old_extension;

        bool exists = fs::exists( fs::u8path( s), ec);
        if( exists && (m_handle_filename_conflicts == FAIL_IF_EXISTING))
            return {};
        if( exists && (m_handle_filename_conflicts == OVERWRITE_EXISTING))
            return s;
        if( !exists)
            return s;

        ASSERT( M_SCENE, exists && (m_handle_filename_conflicts == GENERATE_UNIQUE));
    }

    // Consider a filename with counter, either derived from old_filename or the generic "resource"
    // filename.
    mi::Uint32 local_counter = 0;
    while( true) {

        std::string s = m_path_prefix;
        s += old_filename ? old_root.c_str() : "resource";
        s += "_";
        if( old_filename)
            s += std::to_string( local_counter++);
        else
            s += std::to_string( m_counter++);
        s += use_new_extension ? std::string( new_extension) : old_extension;

        bool exists = fs::exists( fs::u8path( s), ec);
        if( exists && (m_handle_filename_conflicts == FAIL_IF_EXISTING))
            return {};
        if( exists && (m_handle_filename_conflicts == OVERWRITE_EXISTING))
            return s;
        if( !exists)
            return s;

        ASSERT( M_SCENE, exists && (m_handle_filename_conflicts == GENERATE_UNIQUE));
    }

    ASSERT( M_SCENE, false);
    return {};
}

std::string Resource_callback::get_new_resource_filename_marker(
    const char* new_extension,
    const char* old_filename,
    bool use_new_extension,
    bool add_sequence_marker,
    bool add_uvtile_marker,
    mi::Size frame_digits,
    const Frame_uvs& frame_uvs)
{
    if( !add_sequence_marker && !add_uvtile_marker)
        return get_new_resource_filename( new_extension, old_filename, use_new_extension);

    ASSERT( M_SCENE, !m_path_prefix.empty());
    ASSERT( M_SCENE, !new_extension || new_extension[0] == '.');
    ASSERT( M_SCENE, old_filename || use_new_extension);
    ASSERT( M_SCENE, !use_new_extension || new_extension);
    ASSERT( M_SCENE, !old_filename
        || (strip_directories( old_filename) == std::string( old_filename)));

    // Consider a filename (with markers) derived from old_filename, no counter.
    std::string old_root, old_extension;

    if( old_filename) {

        HAL::Ospath::splitext( old_filename, old_root, old_extension);
        std::string s = m_path_prefix;
        s += old_root;
        s += use_new_extension ? std::string( new_extension) : old_extension;

        bool exists = collision( s, frame_uvs);
        if( exists && (m_handle_filename_conflicts == FAIL_IF_EXISTING))
            return {};
        if( exists && (m_handle_filename_conflicts == OVERWRITE_EXISTING))
            return s;
        if( !exists)
            return s;

        ASSERT( M_SCENE, exists && (m_handle_filename_conflicts == GENERATE_UNIQUE));
    }

    // Compute sequence marker string (if required).
    std::string sequence_marker;
    if( !old_filename && add_sequence_marker) {
        ASSERT( M_SCENE, frame_digits > 0);
        sequence_marker = "_frame_<";
        for( mi::Size i = 0; i < frame_digits; ++i)
            sequence_marker += '#';
        sequence_marker += '>';
    }

    // Compute uvtile marker string (if required).
    std::string uvtile_marker;
    if( !old_filename && add_uvtile_marker) {
        uvtile_marker = "_uvtile_<UVTILE0>";
    }

    // Consider a filename (with markers) with counter, either derived from old_filename or the
    // generic "resource"filename.
    mi::Uint32 local_counter = 0;
    while( true) {

        std::string s = m_path_prefix;
        s += old_filename ? old_root.c_str() : "resource";
        s += "_";
        if( old_filename) {
            // Sequence and/or uvtile markers are already included in old_filename/old_root,
            // local counter is added at the end as true suffix.
            s += std::to_string( local_counter++);
        } else {
            // Counter is added as a suffix to "resource", before the sequence and/or uvtile
            // markers, which have to be added explicitly.
            s += std::to_string( m_counter++);
            s += sequence_marker;
            s += uvtile_marker;
        }
        s += use_new_extension ? std::string( new_extension) : old_extension;

        bool exists = collision( s, frame_uvs);
        if( exists && (m_handle_filename_conflicts == FAIL_IF_EXISTING))
            return {};
        if( exists && (m_handle_filename_conflicts == OVERWRITE_EXISTING))
            return s;
        if( !exists)
            return s;

        ASSERT( M_SCENE, exists && (m_handle_filename_conflicts == GENERATE_UNIQUE));
    }

    ASSERT( M_SCENE, false);
    return {};
}

std::string Resource_callback::make_relative(
    const std::string& filename, bool supports_strict_relative_path)
{
    // filename is already strict relative
    if( supports_strict_relative_path && (filename[0] == '.') && (filename[1] == '/'))
        return filename;

    ASSERT( M_SCENE, filename.substr( 0, m_path_prefix.size()) == m_path_prefix);

    std::string tmp = strip_directories( filename);
    return supports_strict_relative_path ? std::string( "./") + tmp : tmp;
}

std::string Resource_callback::strip_directories( const std::string& filename)
{
    mi::Size separator = filename.find_last_of( "/\\:");
    return separator != std::string::npos ? filename.substr( separator+1) : filename;
}

std::string Resource_callback::get_directory( const std::string& filename)
{
    mi::Size separator = filename.find_last_of( "/\\:");
    return separator != std::string::npos ? filename.substr( 0, separator) : std::string();
}

std::string Resource_callback::construct_mdl_file_path(
    const std::string& prefix, const std::string& suffix)
{
    return HAL::Ospath::dirname( prefix) + '/' + HAL::Ospath::basename( suffix);
}

const char* Resource_callback::get_extension( const char* pixel_type)
{
    std::string s = pixel_type;

    // HDR
    if( s == "Float32" || s == "Rgb_fp" || s == "Float32<3>")
        return ".exr";
    if( s == "Color" || s == "Float32<4>") // avoid EXR with alpha
        return ".tif";
    if( s == "Float32<2>" || s == "Rgbe")  // requires conversion
        return ".exr";
    if( s == "Rgbea")                      // requires conversion, avoid EXR with alpha
        return ".tif";

    // LDR
    if( s == "Rgb" || s == "Rgba" || s == "Rgb_16" || s == "Rgba_16")
        return ".png";
    if( s == "Sint8" || s == "Sint32")
        return ".tif";

    ASSERT( M_SCENE, false);
    return ".tif";
}

void Resource_callback::add_error_export_failed(
    mi::Uint32 error_number,
    const char* file_container_or_memory_based,
    const char* resource_type,
    DB::Tag resource)
{
    if( !m_result)
        return;

    std::stringstream s;
    const char* name = m_transaction->tag_to_name( resource);
    s << "Export of " << file_container_or_memory_based << ' ' << resource_type << " \"" << name
      << "\" failed.";
    m_result->message_push_back( error_number, mi::base::MESSAGE_SEVERITY_ERROR, s.str().c_str());
}

void Resource_callback::add_error_string_based(
    mi::Uint32 error_number,
    const char* file_container_or_memory_based,
    const char* resource_type,
    DB::Tag resource)
{
    if( !m_result)
        return;

    std::stringstream s;
    const char* name = m_transaction->tag_to_name( resource);
    s << "Export of " << file_container_or_memory_based << ' ' << resource_type << " \"" << name
      << "\" is not supported in string-based exports of MDL modules.";
    m_result->message_push_back( error_number, mi::base::MESSAGE_SEVERITY_ERROR, s.str().c_str());
}

void Resource_callback::add_error_resource_type(
    mi::Uint32 error_number,
    const char* resource_type,
    DB::Tag resource)
{
    if( !m_result)
        return;

    std::stringstream s;
    const char* name = m_transaction->tag_to_name( resource);
    s << "Incorrect type for " << resource_type << " resource \"" << name << "\".";
    m_result->message_push_back( error_number, mi::base::MESSAGE_SEVERITY_ERROR, s.str().c_str());
}

} // namespace MDL

} // namespace MI

