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

/** \file
 ** \brief Source for the Resource_callback implementation.
 **/

#include "pch.h"

#include "neuray_mdl_resource_callback.h"

#include <sstream>
#include <mi/mdl/mdl_modules.h>
#include <mi/neuraylib/ibsdf_isotropic_data.h>
#include <mi/neuraylib/ibuffer.h>
#include <mi/neuraylib/icanvas.h>
#include <mi/neuraylib/iexport_result.h>
#include <mi/neuraylib/istring.h>
#include <base/data/db/i_db_access.h>
#include <base/data/db/i_db_transaction.h>
#include <base/hal/disk/disk.h>
#include <base/hal/hal/i_hal_ospath.h>
#include <io/image/image/i_image.h>
#include <io/image/image/i_image_mipmap.h>
#include <io/scene/bsdf_measurement/i_bsdf_measurement.h>
#include <io/scene/dbimage/i_dbimage.h>
#include <io/scene/lightprofile/i_lightprofile.h>
#include <io/scene/mdl_elements/i_mdl_elements_utilities.h>
#include <io/scene/mdl_elements/i_mdl_elements_module.h>
#include <io/scene/texture/i_texture.h>

namespace MI {

namespace NEURAY {

Resource_callback::Resource_callback(
    DB::Transaction* transaction,
    const mi::mdl::IModule* module,
    const char* module_name,
    const char* module_filename,
    MDL::Execution_context* context,
    mi::neuraylib::IExport_result_ext* result)
  : m_transaction( transaction),
    m_module( module, mi::base::DUP_INTERFACE),
    m_module_name( module_name),
    m_bundle_resources( context->get_option<bool>( MDL_CTX_OPTION_BUNDLE_RESOURCES)),
    m_keep_original_file_paths( context->get_option<bool>( MDL_CTX_OPTION_KEEP_ORIGINAL_RESOURCE_FILE_PATHS)),
    m_result( result, mi::base::DUP_INTERFACE),
    m_module_filename_c_str( nullptr),
    m_counter( 0),
    m_image_module(false)
{
    m_transaction->pin();

    if( module_filename) {
        m_module_filename = module_filename;
        if( !DISK::is_path_absolute( m_module_filename))
            m_module_filename = HAL::Ospath::join_v2( DISK::get_cwd(), m_module_filename);
        m_module_filename_c_str = m_module_filename.c_str();
        mi::Size length = m_module_filename.length();
        ASSERT( M_NEURAY_API, length >= 4 && m_module_filename.substr( length-4) == ".mdl");
        m_path_prefix = m_module_filename.substr( 0, length-4);
    }
}

Resource_callback::~Resource_callback()
{
    m_transaction->unpin();
}

const char* Resource_callback::get_resource_name(
    const mi::mdl::IValue_resource* resource,
    bool supports_strict_relative_path,
    BufferCallback* buffer_callback)
{
    if( m_keep_original_file_paths && !m_bundle_resources)
        return resource->get_string_value();

    // Resources in modules loaded from disk or string most likely do not have a valid tag in AST
    // representation (only in the DAG representation). There is not much we can do in such a case:
    // If resolving succeeds, so does unresolving, and we return the same (or an equivalent) string.
    // If resolving fails, we return NULL, which triggers the default action of printing the string
    // value (and the useless tag version). So returning the string value seems to be the best
    // solution.
    DB::Tag tag( resource->get_tag_value());
    if( !tag && m_module) {
        // However, if the resources are references in the current module,
        // we can get the tag from this module
        DB::Tag module_tag = m_transaction->name_to_tag( m_module_name.c_str());
        if( module_tag) {
            SERIAL::Class_id class_id = m_transaction->get_class_id( module_tag);
            if( class_id != MDL::Mdl_module::id) {
                add_error_resource_type( 6010, "module", module_tag);
                return nullptr;
            }

            // Find index of the referenced resource. Use resource table in the corresponding DB
            // element to retrieve the tag. This assumes that the tags are in sync between
            // mi::mdl::IModule and the DB element.
            for( mi::Size i = 0, n = m_module->get_referenced_resources_count(); i < n; ++i) {
                const char* mdl_path = m_module->get_referenced_resource_url( i);
                if( strcmp( mdl_path, resource->get_string_value()) == 0) {
                    DB::Access<MDL::Mdl_module> db_module( module_tag, m_transaction);
                    tag = db_module->get_resource_tag( i);
                    break;
                }
            }
        }
    }

    if( !tag)
        return resource->get_string_value();

    // Return result for already translated tags.
    if( !m_file_paths[tag].empty())
        return m_file_paths[tag].c_str();

    switch (resource->get_kind()) {
    case mi::mdl::IValue::VK_TEXTURE:
        return handle_texture( tag, supports_strict_relative_path, buffer_callback);
    case mi::mdl::IValue::VK_LIGHT_PROFILE:
        return handle_light_profile( tag, supports_strict_relative_path, buffer_callback);
    case mi::mdl::IValue::VK_BSDF_MEASUREMENT:
        return handle_bsdf_measurement( tag, supports_strict_relative_path, buffer_callback);
    default:
        ASSERT( M_NEURAY_API, false);
        return nullptr;
    }
}

const char* Resource_callback::get_resource_name(
    const mi::mdl::IValue_resource* resource,
    bool supports_strict_relative_path)
{
    return get_resource_name(resource, supports_strict_relative_path, nullptr);
}

namespace
{
const char* get_uvtile_marker(const std::string& str)
{
    if (str.find("<UDIM>") != std::string::npos)
        return "<UDIM>";
    if (str.find("<UVTILE0>") != std::string::npos)
        return "<UVTILE0>";
    if (str.find("<UVTILE1>") != std::string::npos)
        return "<UVTILE1>";

    return "";
}

} // anonymous

const char* Resource_callback::handle_texture(
    DB::Tag texture_tag, 
    bool supports_strict_relative_path,
    BufferCallback* buffer_callback)
{
    SERIAL::Class_id class_id = m_transaction->get_class_id( texture_tag);
    if( class_id != TEXTURE::Texture::id) {
        add_error_resource_type( 6010, "texture", texture_tag);
        return nullptr;
    }

    DB::Access<TEXTURE::Texture> texture( texture_tag, m_transaction);
    DB::Tag image_tag( texture->get_image());
    if( !image_tag)
        return nullptr;

    class_id = m_transaction->get_class_id( image_tag);
    if( class_id != DBIMAGE::Image::id) {
        add_error_resource_type( 6010, "image", image_tag);
        return nullptr;
    }

    DB::Access<DBIMAGE::Image> image( image_tag, m_transaction);
   
    bool is_uvtile = image->is_uvtile();
    // File-based images
    if( image->is_file_based())
    {

        const std::string& filename = image->get_filename();

        std::string uvtile_marker;
        if (is_uvtile)
        {
            uvtile_marker = get_uvtile_marker(
                image->get_original_filename().empty()
                ? image->get_mdl_file_path()
                : image->get_original_filename());
            ASSERT(M_NEURAY_API, !uvtile_marker.empty());
        }

        // Use original file if bundling is not requested and the file can be found via the search
        // paths.
        if( !m_bundle_resources)
        {
            const std::string& file_path = MDL::DETAIL::unresolve_resource_filename(
                filename.c_str(), m_module_filename_c_str, &m_module_name[3]);

            if( !file_path.empty()) {
                m_file_paths[texture_tag] = is_uvtile
                    ? MDL::uvtile_string_to_marker( file_path, uvtile_marker) : file_path;
                return m_file_paths[texture_tag].c_str();
            }
        }

        // Fail if we need to export the resource, but no module filename is given (string-based export).
        if( m_module_filename.empty())
        {
            add_error_string_based(6011, "file-based", "image", image_tag);
            return nullptr;
        }

        // Otherwise copy the file(s)
        std::string new_filename;

        // to a buffer
        if( buffer_callback) {
            new_filename = export_canvases(image.get_ptr(), uvtile_marker.c_str(), buffer_callback);
            if (new_filename.empty()) {
                add_error_export_failed(6013, "file-based-to-buffer", "image", image_tag);
                return nullptr;
            }
        }
        else
        {
            // next to the exported module
            if( is_uvtile) {
                std::string basename, extension;
                HAL::Ospath::splitext(filename, basename, extension);

                std::vector<std::string> uvtile_filenames;
                generate_uvtile_filenames(
                    image.get_ptr(), uvtile_marker.c_str(), extension.c_str(), uvtile_filenames);

                for (mi::Uint32 i = 0; i < image->get_uvtile_length(); ++i) {
                    bool result = DISK::file_copy(
                        image->get_filename(i).c_str(), uvtile_filenames[i].c_str());
                    if (!result) {
                        add_error_export_failed(6013, "file-based", "image", image_tag);
                        return nullptr;
                    }
                }

                new_filename = MDL::uvtile_string_to_marker( uvtile_filenames[0], uvtile_marker);
            }
            else
            {
                std::string basename, extension;
                HAL::Ospath::splitext(filename, basename, extension);
                new_filename = get_new_resource_filename(extension.c_str(), filename.c_str());
                bool result = DISK::file_copy(filename.c_str(), new_filename.c_str());
                if (!result) {
                    add_error_export_failed(6013, "file-based", "image", image_tag);
                    return nullptr;
                }
            }
        }

        m_file_paths[texture_tag] = make_relative( new_filename, supports_strict_relative_path);
        return m_file_paths[texture_tag].c_str();

    // Archive-based images
    } else if( image->is_container_based()) {

        const std::string& container_filename = image->get_container_filename();
        const std::string& container_membername = image->get_container_membername();

        std::string uvtile_marker;
        if (is_uvtile) {
            uvtile_marker = get_uvtile_marker(image->get_mdl_file_path());
            ASSERT(M_NEURAY_API, !uvtile_marker.empty());
        }

        // Use original file if bundling is not requested and the file can be found via the search
        // paths.
        if( !m_bundle_resources) {
            const std::string& file_path = MDL::DETAIL::unresolve_resource_filename(
                container_filename.c_str(), container_membername.c_str(), m_module_filename_c_str,
                &m_module_name[3]);
            if( !file_path.empty()) {
                m_file_paths[texture_tag] = is_uvtile
                    ? MDL::uvtile_string_to_marker( file_path, uvtile_marker) : file_path;
                return m_file_paths[texture_tag].c_str();
            }
        }

        // Fail if we need to export the resource, but no module filename is given (string-based export).
        if( m_module_filename.empty()) {
            add_error_string_based( 6015, "container-based", "image", image_tag);
            return nullptr;
        }

        // Export canvas(es) with a generated filename and return that filename.
        std::string new_filename = export_canvases(
            image.get_ptr(), uvtile_marker.c_str(), buffer_callback);
        if (new_filename.empty()) {
            add_error_string_based(6016, "container-based", "image", image_tag);
            return nullptr;
        }


        m_file_paths[texture_tag] = make_relative( new_filename, supports_strict_relative_path);
        return m_file_paths[texture_tag].c_str();

    // Memory-based images
    } else {

        // Fail if we need to export the resource, but no module filename is given (string-based export).
        if( m_module_filename.empty()) {
            add_error_string_based( 6012, "memory-based", "image", image_tag);
            return nullptr;
        }

        // Export canvas(es) with a generated filename and return that filename.
        std::string new_filename = export_canvases(image.get_ptr(), "<UVTILE0>", buffer_callback);
        if (new_filename.empty()) {
            add_error_string_based(6014, "memory-based", "image", image_tag);
            return nullptr;
        }

        m_file_paths[texture_tag] = make_relative( new_filename, supports_strict_relative_path);
        return m_file_paths[texture_tag].c_str();
    }
}

const char* Resource_callback::handle_light_profile(
    DB::Tag tag, 
    bool supports_strict_relative_path,
    BufferCallback* buffer_callback)
{
    SERIAL::Class_id class_id = m_transaction->get_class_id( tag);
    if( class_id != LIGHTPROFILE::Lightprofile::id) {
        add_error_resource_type( 6010, "light profile", tag);
        return nullptr;
    }

    DB::Access<LIGHTPROFILE::Lightprofile> lp( tag, m_transaction);

    // File-based light profiles
    if( lp->is_file_based()) {

        const std::string& filename = lp->get_filename();

        // Use original file if bundling is not requested and the file can be found via the search
        // paths.
        if( !m_bundle_resources) {
            const std::string& file_path = MDL::DETAIL::unresolve_resource_filename(
                filename.c_str(), m_module_filename_c_str, &m_module_name[3]);
            if( !file_path.empty()) {
                m_file_paths[tag] = file_path;
                return m_file_paths[tag].c_str();
            }
        }

        // Fail if we need to export the resource, but no module filename is given (string-based export).
        if( m_module_filename.empty()) {
            add_error_string_based( 6011, "file-based", "light profile", tag);
            return nullptr;
        }

        // Otherwise copy the file.
        std::string new_filename = "";
        // to a buffer
        if (buffer_callback) {
            new_filename = export_light_profile(lp.get_ptr(), buffer_callback);
            if (new_filename.empty()) {
                add_error_export_failed(6013, "file-based-to-buffer", "light profile", tag);
                return nullptr;
            }
        }
        else 
        {
            // Otherwise copy the file.
            std::string basename, extension;
            HAL::Ospath::splitext(filename, basename, extension);
            new_filename = get_new_resource_filename(extension.c_str(), filename.c_str());
            bool result = DISK::file_copy(filename.c_str(), new_filename.c_str());

            if (!result) {
                add_error_export_failed(6013, "file-based", "light profile", tag);
                return nullptr;
            }
        }

        m_file_paths[tag] = make_relative( new_filename, supports_strict_relative_path);
        return m_file_paths[tag].c_str();

    // Archive-based light profiles
    } else if( lp->is_container_based()) {

        const std::string& container_filename = lp->get_container_filename();
        const std::string& container_membername = lp->get_container_membername();

        // Use original file if bundling is not requested and the file can be found via the search
        // paths.
        if( !m_bundle_resources) {
            const std::string& file_path = MDL::DETAIL::unresolve_resource_filename(
                container_filename.c_str(), container_membername.c_str(), m_module_filename_c_str,
                &m_module_name[3]);
            if( !file_path.empty()) {
                m_file_paths[tag] = file_path;
                return m_file_paths[tag].c_str();
            }
        }

        // Fail if we need to export the resource, but no module filename is given (string-based export).
        if( m_module_filename.empty()) {
            add_error_string_based( 6015, "container-based", "light profile", tag);
            return nullptr;
        }

        // Export light profile with a generated filename and return that filename.
        const std::string new_filename = export_light_profile(lp.get_ptr(), buffer_callback);
        if (new_filename.empty()) {
            add_error_export_failed( 6016, "container-based", "light profile", tag);
            return nullptr;
        }

        m_file_paths[tag] = make_relative( new_filename, supports_strict_relative_path);
        return m_file_paths[tag].c_str();

    // Memory-based light profiles
    } else {

        // Fail if we need to export the resource, but no module filename is given (string-based export).
        if( m_module_filename.empty()) {
            add_error_string_based( 6012, "memory-based", "light profile", tag);
            return nullptr;
        }

        // Export light profile with a generated filename and return that filename.
        const std::string new_filename = export_light_profile(lp.get_ptr(), buffer_callback);
        if (new_filename.empty()) {
            add_error_export_failed(6014, "memory-based", "light profile", tag);
            return nullptr;
        }

        m_file_paths[tag] = make_relative( new_filename, supports_strict_relative_path);
        return m_file_paths[tag].c_str();
    }
}

const char* Resource_callback::handle_bsdf_measurement(
    DB::Tag tag, 
    bool supports_strict_relative_path,
    BufferCallback* buffer_callback)
{
    SERIAL::Class_id class_id = m_transaction->get_class_id( tag);
    if( class_id != BSDFM::Bsdf_measurement::id) {
        add_error_resource_type( 6010, "BSDF measurement", tag);
        return nullptr;
    }

    DB::Access<BSDFM::Bsdf_measurement> bsdfm( tag, m_transaction);

    // File-based BSDF measurements
    if( bsdfm->is_file_based()) {

        const std::string& filename = bsdfm->get_filename();

        // Use original file if bundling is not requested and the file can be found via the search
        // paths.
        if( !m_bundle_resources) {
            const std::string& file_path = MDL::DETAIL::unresolve_resource_filename(
                filename.c_str(), m_module_filename_c_str, &m_module_name[3]);
            if( !file_path.empty()) {
                m_file_paths[tag] = file_path;
                return m_file_paths[tag].c_str();
            }
        }

        // Fail if we need to export the resource, but no module filename is given (string-based export).
        if( m_module_filename.empty()) {
            add_error_string_based( 6011, "file-based", "BSDF measurement", tag);
            return nullptr;
        }

        // Otherwise copy the file.
        std::string new_filename = "";
        // to a buffer
        if (buffer_callback) {
            new_filename = export_bsdf_measurement(bsdfm.get_ptr(), buffer_callback);
            if (new_filename.empty())
            {
                add_error_export_failed(6013, "file-based-to-buffer", "BSDF measurement", tag);
                return nullptr;
            }
        }
        else
        {
            // Otherwise copy the file.
            std::string basename, extension;
            HAL::Ospath::splitext( filename, basename, extension);
            new_filename
                = get_new_resource_filename( extension.c_str(), filename.c_str());
            bool result = DISK::file_copy( filename.c_str(), new_filename.c_str());

            if( !result) {
                add_error_export_failed( 6013, "file-based", "BSDF measurement", tag);
                return nullptr;
            }
        }

        m_file_paths[tag] = make_relative( new_filename, supports_strict_relative_path);
        return m_file_paths[tag].c_str();

    // Archive-based BSDF measurements
    } else if( bsdfm->is_container_based()) {

        const std::string& container_filename = bsdfm->get_container_filename();
        const std::string& container_membername = bsdfm->get_container_membername();

        // Use original file if bundling is not requested and the file can be found via the search
        // paths.
        if( !m_bundle_resources) {
            const std::string& file_path = MDL::DETAIL::unresolve_resource_filename(
                container_filename.c_str(), container_membername.c_str(), m_module_filename_c_str,
                &m_module_name[3]);
            if( !file_path.empty()) {
                m_file_paths[tag] = file_path;
                return m_file_paths[tag].c_str();
            }
        }

        // Fail if we need to export the resource, but no module filename is given (string-based export).
        if( m_module_filename.empty()) {
            add_error_string_based( 6011, "container-based", "BSDF measurement", tag);
            return nullptr;
        }

        // Export BSDF measurement with a generated filename and return that filename.
        const std::string new_filename = export_bsdf_measurement(bsdfm.get_ptr(), buffer_callback);
        if (new_filename.empty())
        {
            add_error_export_failed(6013, "memory-based", "BSDF measurement", tag);
            return nullptr;
        }

        m_file_paths[tag] = make_relative( new_filename, supports_strict_relative_path);
        return m_file_paths[tag].c_str();

    // Memory-based BSDF measurements
    } else {

        // Fail if we need to export the resource, but no module filename is given (string-based export).
        if( m_module_filename.empty()) {
            add_error_string_based( 6012, "memory-based", "BSDF measurement", tag);
            return nullptr;
        }

        // Export BSDF measurement with a generated filename and return that filename.
        DB::Access<BSDFM::Bsdf_measurement_impl> bsdfm_measurement_impl(
            bsdfm->get_impl_tag(), m_transaction);
        mi::base::Handle<const mi::neuraylib::IBsdf_isotropic_data> reflection(
            bsdfm_measurement_impl->get_reflection<mi::neuraylib::IBsdf_isotropic_data>());
        mi::base::Handle<const mi::neuraylib::IBsdf_isotropic_data> transmission(
            bsdfm_measurement_impl->get_transmission<mi::neuraylib::IBsdf_isotropic_data>());
        const std::string new_filename = get_new_resource_filename( ".mbsdf", /*old_filename*/ nullptr);
        bool result = BSDFM::export_to_file(
            reflection.get(), transmission.get(), new_filename.c_str());

        if( !result) {
            add_error_export_failed( 6014, "memory-based", "BSDF measurement", tag);
            return nullptr;
        }

        m_file_paths[tag] = make_relative( new_filename, supports_strict_relative_path);
        return m_file_paths[tag].c_str();
    }
}

std::string Resource_callback::export_canvases(
    const DBIMAGE::Image* image, 
    const char* uvtile_marker,
    BufferCallback* buffer_callback)
{
    std::string new_filename;
    std::string old_filename = image->get_filename();
    if (old_filename.empty()) old_filename = image->get_original_filename();

    if (image->is_uvtile())
    {
        mi::base::Handle<const IMAGE::IMipmap> mipmap(image->get_mipmap(m_transaction, 0));
        mi::base::Handle<const mi::neuraylib::ICanvas> canvas(mipmap->get_level(0));
        const char* extension = get_extension(canvas->get_type());

        std::vector<std::string> uvtile_filenames;
        generate_uvtile_filenames(image, uvtile_marker, extension, uvtile_filenames);
        ASSERT(M_NEURAY_API, uvtile_filenames.size() == image->get_uvtile_length());
        
        for (mi::Uint32 i = 0; i < image->get_uvtile_length(); ++i)
        {
            mipmap = image->get_mipmap(m_transaction, i);
            canvas = mipmap->get_level(0);

            bool result = false;
            if (buffer_callback)
            {
                // allow custom output streams
                mi::base::Handle<mi::neuraylib::IBuffer> buffer(
                    m_image_module->create_buffer_from_canvas(
                    canvas.get(), extension + 1, canvas->get_type()));

                if (buffer)
                {
                    uvtile_filenames[i] = (*buffer_callback)(buffer.get(), uvtile_filenames[i].c_str());
                    result = !uvtile_filenames[i].empty();
                }
            } else {
                // write to file by default
                result = m_image_module->export_canvas(canvas.get(), uvtile_filenames[i].c_str());
            }
                
            if (!result) {
                return "";
            }
        }

        new_filename = MDL::uvtile_string_to_marker( uvtile_filenames[0], uvtile_marker);
    }
    else
    {
        mi::base::Handle<const IMAGE::IMipmap> mipmap(image->get_mipmap( m_transaction));
        mi::base::Handle<const mi::neuraylib::ICanvas> canvas(mipmap->get_level( 0));
        const char* extension = get_extension( canvas->get_type());
        new_filename = get_new_resource_filename(
            extension, old_filename.empty() ? nullptr : old_filename.c_str());

        bool result = false;
        if (buffer_callback)
        {
            // allow custom output streams
            mi::base::Handle<mi::neuraylib::IBuffer> buffer(
                m_image_module->create_buffer_from_canvas(
                canvas.get(), extension + 1, canvas->get_type()));

            if (buffer)
            {
                new_filename = (*buffer_callback)(buffer.get(), new_filename.c_str());
                result = !new_filename.empty();
            }

        } else {
            // write to file by default
            result = m_image_module->export_canvas(canvas.get(), new_filename.c_str());
        }

        if (!result) {
            return "";
        }
    }
    return new_filename;
}

// Exports the profile th disk using a generic filename.
std::string Resource_callback::export_light_profile(
    const LIGHTPROFILE::Lightprofile* profile,
    BufferCallback* buffer_callback)
{
    std::string old_filename = profile->get_filename();
    if (old_filename.empty()) old_filename = profile->get_original_filename();
    std::string new_filename = get_new_resource_filename(
        ".ies", old_filename.empty() ? nullptr : old_filename.c_str());

    bool result = false;
    if (buffer_callback) {
        // allow custom output streams
        mi::base::Handle<mi::neuraylib::IBuffer> buffer(
            LIGHTPROFILE::create_buffer_from_lightprofile(m_transaction, profile));

        if (buffer) {
            new_filename = (*buffer_callback)(buffer.get(), new_filename.c_str());
            result = !new_filename.empty();
        }

    } else {
        // write to file by default
        result = LIGHTPROFILE::export_to_file(m_transaction, profile, new_filename.c_str());
    }

    return result ? new_filename : "";
}

// Exports the measured BSDF th disk using a generic filename or to a buffer.
std::string Resource_callback::export_bsdf_measurement(
    const BSDFM::Bsdf_measurement* measurement,
    BufferCallback* buffer_callback)
{
    std::string old_filename = measurement->get_filename();
    if (old_filename.empty()) old_filename = measurement->get_original_filename();
    std::string new_filename = get_new_resource_filename(
        ".mbsdf", old_filename.empty() ? nullptr : old_filename.c_str());

    DB::Access<BSDFM::Bsdf_measurement_impl> measurement_impl(
        measurement->get_impl_tag(), m_transaction);
    mi::base::Handle<const mi::neuraylib::IBsdf_isotropic_data> refl(
        measurement_impl->get_reflection<mi::neuraylib::IBsdf_isotropic_data>());
    mi::base::Handle<const mi::neuraylib::IBsdf_isotropic_data> trans(
        measurement_impl->get_transmission<mi::neuraylib::IBsdf_isotropic_data>());

    bool result = false;
    if (buffer_callback) {
        // allow custom output streams
        mi::base::Handle<mi::neuraylib::IBuffer> buffer(
            BSDFM::create_buffer_from_bsdf_measurement(refl.get(), trans.get()));

        if (buffer) {
            new_filename = (*buffer_callback)(buffer.get(), new_filename.c_str());
            result = !new_filename.empty();
        }

    } else {
        // write to file by default
        result = BSDFM::export_to_file(refl.get(), trans.get(), new_filename.c_str());
    }

    return result ? new_filename : "";
}

void Resource_callback::generate_uvtile_filenames(
    const DBIMAGE::Image* image,
    const char* uvtile_marker, 
    const char* extension, 
    std::vector<std::string>& filenames)
{
    std::string fn_base = m_path_prefix + "_resource";

    std::vector<std::string> uvs;
    mi::Size l = image->get_uvtile_length();
    filenames.resize(l);

    for (mi::Uint32 i = 0; i < l; ++i)
    {
        mi::Sint32 u, v;
        image->get_uvtile_uv(i, u, v);
        uvs.push_back(MDL::uvtile_marker_to_string(uvtile_marker, u, v));
    }
    bool fn_base_ok;
    filenames.resize(l);
    do
    {
        fn_base_ok = true;
       
        for (mi::Uint32 i = 0; i < l; ++i)
        {
            std::string test = fn_base + uvs[i] + extension;
            if (DISK::is_file(test.c_str()))
            {
                fn_base_ok = false;
                std::ostringstream ss;
                ss << m_path_prefix << "_resource_" << m_counter++;
                fn_base = ss.str();
                break;
            }
            filenames[i] = test;
        }
    } while (!fn_base_ok);
}

std::string Resource_callback::get_new_resource_filename(
    const char* extension, const char* old_filename)
{
    ASSERT( M_NEURAY_API, !m_path_prefix.empty());

    std::string s;

    if( old_filename) {
        std::string old_root, old_ext;
        HAL::Ospath::splitext(strip_directories(old_filename), old_root, old_ext);
        s = m_path_prefix + "_" + old_root + extension;
        if( !DISK::is_file( s.c_str()))
            return s;
    }
    do {
        std::ostringstream ss;
        ss << m_path_prefix << "_resource_" << m_counter++ << extension;
        s = ss.str();
    }  while( DISK::is_file( s.c_str()));

    return s;
}

std::string Resource_callback::make_relative(
    const std::string& filename, bool supports_strict_relative_path)
{
    // filename is already strict relative
    if (supports_strict_relative_path && filename[0] == '.' && filename[1] == '/')
        return filename;

    ASSERT( M_NEURAY_API, filename.substr( 0, m_path_prefix.size()) == m_path_prefix);

    std::string tmp = strip_directories( filename);
    return supports_strict_relative_path ? std::string( "./") + tmp : tmp;
}

std::string Resource_callback::strip_directories( const std::string& filename)
{
    mi::Size separator = filename.find_last_of( "/\\:");
    return separator != std::string::npos ? filename.substr( separator+1) : filename;
 }

const char* Resource_callback::get_extension( const char* pixel_type)
{
    std::string s = pixel_type;
    if( s == "Float32" || s == "Rgb_fp" || s == "Color" || s == "Float32<3>" || s == "Float32<4>")
        return ".exr";
    if( s == "Float32<2>" || s == "Rgbe" || s == "Rgbea") // HDR, requires conversion
        return ".exr";
    if( s == "Rgb" || s == "Rgba" || s == "Rgb_16" || s == "Rgba_16") // LDR
        return ".png";
    if( s == "Sint8" || s == "Sint32") // Sint8 requires conversion
        return ".tif";
    ASSERT( M_NEURAY_API, false);
    return ".exr";
}

void Resource_callback::add_error_export_failed(
    mi::Uint32 error_number,
    const char* file_container_or_memory_based,
    const char* resource_type,
    DB::Tag resource)
{
    if (!m_result) return;

    std::stringstream s;
    const char* name = m_transaction->tag_to_name( resource);
    s << "Export of " << file_container_or_memory_based << " " << resource_type << "\"" << name
      << "\" failed.";
    m_result->message_push_back( error_number, mi::base::MESSAGE_SEVERITY_ERROR, s.str().c_str());
}

void Resource_callback::add_error_string_based(
    mi::Uint32 error_number,
    const char* file_container_or_memory_based,
    const char* resource_type,
    DB::Tag resource)
{
    if (!m_result) return;

    std::stringstream s;
    const char* name = m_transaction->tag_to_name( resource);
    s << "Export of " << file_container_or_memory_based << " " << resource_type << "\"" << name
      << "\" is not supported in string-based exports of MDL modules.";
    m_result->message_push_back( error_number, mi::base::MESSAGE_SEVERITY_ERROR, s.str().c_str());
}

void Resource_callback::add_error_resource_type(
    mi::Uint32 error_number,
    const char* resource_type,
    DB::Tag resource)
{
    if (!m_result) return;

    std::stringstream s;
    const char* name = m_transaction->tag_to_name( resource);
    s << "Incorrect type for " << resource_type << " resource \"" << name << "\".";
    m_result->message_push_back( error_number, mi::base::MESSAGE_SEVERITY_ERROR, s.str().c_str());
}

} // namespace NEURAY

} // namespace MI
