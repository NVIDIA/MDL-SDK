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
 ** \brief Header for the Resource_callback implementation.
 **/

#ifndef API_API_NEURAY_NEURAY_MDL_RESOURCE_CALLBACK_H
#define API_API_NEURAY_NEURAY_MDL_RESOURCE_CALLBACK_H

#include <string>
#include <map>

#include <mi/base/handle.h>
#include <mi/mdl/mdl_printers.h>
#include <mi/mdl/mdl_values.h>
#include <base/system/main/access_module.h>
#include <boost/core/noncopyable.hpp>
#include <base/data/db/i_db_tag.h>

namespace mi { 
    namespace neuraylib { 
        class IExport_result_ext;
        class IBuffer;
    } 
}

namespace MI {

namespace BSDFM { class Bsdf_measurement; }
namespace DB { class Transaction; }
namespace DBIMAGE { class Image; }
namespace IMAGE { class Image_module; }
namespace MDL { class Execution_context; }
namespace LIGHTPROFILE { class Lightprofile; }

namespace NEURAY {

class Resource_callback
  : public mi::mdl::IMDL_exporter_resource_callback, public boost::noncopyable
{
public:

    /// Allows to provide a custom output stream
    /// \param  buffer                  resource content, ready to be written to a file
    /// \param  suggested_file_name     the filename that would be used internally
    ///
    /// \return the new file name or an empty string the indicate an error
    typedef std::function<
        std::string(mi::neuraylib::IBuffer* buffer, const char* suggested_file_name)> BufferCallback;

    Resource_callback(
        DB::Transaction* transaction,
        const mi::mdl::IModule* module,
        const char* module_name,
        const char* module_filename,
        MDL::Execution_context* context,
        mi::neuraylib::IExport_result_ext* result);

    ~Resource_callback();

    const char* get_resource_name(
        const mi::mdl::IValue_resource* resource,
        bool supports_strict_relative_path) override;

    /// if a callback is provided, exports are written to the provided streams instead of
    /// to file next to the module
    const char* get_resource_name(
        const mi::mdl::IValue_resource* resource,
        bool supports_strict_relative_path,
        BufferCallback* buffer_callback);

private:
    /// Returns the file path for a texture.
    const char* handle_texture(
        DB::Tag tag, 
        bool supports_strict_relative_path,
        BufferCallback* buffer_callback);

    /// Returns the file path for a light profile.
    const char* handle_light_profile(
        DB::Tag tag, 
        bool supports_strict_relative_path,
        BufferCallback* buffer_callback);

    /// Returns the file path for a BSDF measurement.
    const char* handle_bsdf_measurement(
        DB::Tag tag, 
        bool supports_strict_relative_path,
        BufferCallback* buffer_callback);

    /// Generates unique uv-tile filenames for the given sequence
    void generate_uvtile_filenames(
        const DBIMAGE::Image* image,
        const char* uvtile_marker,
        const char* extension,
        std::vector<std::string>& filenames);

    /// Exports all canvases of the given image to disk using a generic filename or to a buffer.
    /// In case of an uvtile/udim-sequence, the filenames follow the convention
    /// given in \p uvtile_marker.
    std::string export_canvases(
        const DBIMAGE::Image* image, 
        const char* uvtile_marker,
        BufferCallback* buffer_callback);

    /// Exports the profile th disk using a generic filename or to a buffer.
    std::string export_light_profile(
        const LIGHTPROFILE::Lightprofile* profile,
        BufferCallback* buffer_callback);

    /// Exports the measured BSDF th disk using a generic filename or to a buffer.
    std::string export_bsdf_measurement(
        const BSDFM::Bsdf_measurement* measurement,
        BufferCallback* buffer_callback);

    /// Generates a filename for resources with the given extension or based on other filename.
    ///
    /// If \p old_filename is not \p NULL and the filename derived from it does not already exist,
    /// return it. Otherwise, generate a generic filename with the given extension that does not
    /// already exist.
    ///
    /// Returns m_path_prefix + "_" + strip_directories(old_filename) or
    /// m_path_prefix + _resource_" + counter + extension .
    std::string get_new_resource_filename( const char* extension, const char* old_filename);

    /// Generates a relative file path from a filename.
    ///
    /// Assumes that the filename points to the same directory as the module being exported.
    ///
    /// Strips directories (and drive letters) from the filename. If \p
    /// supports_strict_relative_path is \c true (MDL 1.3 and up) a "./" prefix is added.
    std::string make_relative(
        const std::string& filename, bool supports_strict_relative_path);

    /// Strips directories (and drive letters) from the filename.
    static std::string strip_directories( const std::string& filename);

    /// Returns ".exr" for HDR pixel types, ".png" for LDR pixel types, and ".tif" for "Sint8" and
    /// "Sint32".
    static const char* get_extension( const char* pixel_type);

    /// Adds an error message for failed export operations of resources.
    void add_error_export_failed(
        mi::Uint32 error_number,
        const char* file_container_or_memory_based,
        const char* resource_type,
        DB::Tag resource);

    /// Adds an error message for unfulfillable export operations in string-based exports.
    void add_error_string_based(
        mi::Uint32 error_number,
        const char* file_container_or_memory_based,
        const char* resource_type,
        DB::Tag resource);

    /// Adds an error message for resources of incorrect type (DB element has wrong class ID).
    void add_error_resource_type(
        mi::Uint32 error_number,
        const char* resource_type,
        DB::Tag resource);

    /// The DB transaction to be used.
    DB::Transaction* m_transaction;

    /// The MDL module to be exported.
    mi::base::Handle<const mi::mdl::IModule> m_module;

    /// DB name of the MDL module to be exported.
    std::string m_module_name;

    /// New URI of the MDL module to be exported (or empty for string-based exports).
    std::string m_module_uri;

    /// Flag that indicates whether resources are bundled with the exported MDL module.
    bool m_bundle_resources;
    
    /// Flag that indicates whether IValue_resource data should be returned as is (unless
    /// m_bundle_resources is set).
    bool m_keep_original_file_paths;

    /// Error messages are added to this export result.
    mi::base::Handle<mi::neuraylib::IExport_result_ext> m_result;

    /// New filename of the MDL module to be exported (or empty for string-based exports).
    std::string m_module_filename;

    /// New filename of the MDL module to be exported (or NULL for string-based exports).
    const char* m_module_filename_c_str;

    /// Path prefix for resource names.
    std::string m_path_prefix;

    /// Counter for resource names.
    mi::Uint32 m_counter;

    /// Caches all translations to avoid multiple exports of the same resource.
    std::map<DB::Tag, std::string> m_file_paths;

    /// Access to the IMAGE module.
    SYSTEM::Access_module<IMAGE::Image_module> m_image_module;

};

} // namespace NEURAY

} // namespace MI

#endif // API_API_NEURAY_NEURAY_MDL_RESOURCE_CALLBACK_H
