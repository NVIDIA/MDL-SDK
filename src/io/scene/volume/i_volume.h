/***************************************************************************************************
 * Copyright (c) 2007-2022, NVIDIA CORPORATION. All rights reserved.
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
/// \brief Handles the DB element Volume_data.

#ifndef IO_SCENE_VOLUME_I_VOLUME_H
#define IO_SCENE_VOLUME_I_VOLUME_H

#include <io/scene/scene/i_scene_scene_element.h>
#include <base/data/db/i_db_tag.h>

#include <mi/neuraylib/ivolume.h>
#include <mi/neuraylib/iserializer.h>


namespace nv {
namespace index {
class IVolume_direct_host;
class IVolume_direct_cuda;
}}

namespace MI {
namespace SERIAL { class Serializer; class Deserializer; }
namespace CUDA { class Device_id; }

namespace VOLUME {


/// The class ID for the #Volume_data class.
static constexpr SERIAL::Class_id ID_VOLUME_DATA = 0x5f566474;// '_Vdt'


/// The volume data class.
///
/// Volume data references a volume data file.
class Volume_data : public SCENE::Scene_element<Volume_data, ID_VOLUME_DATA>
{
public:

    Volume_data() = default;

    void get_scene_element_references(DB::Tag_set* result) const;

    mi::Sint32 reset_file(
            DB::Transaction* transaction,
            const char* filename,
            const char* selector);

    mi::Sint32 reset_reader(
        DB::Transaction* transaction,
        mi::neuraylib::IReader* reader,
        const char* format,
        const char* selector);

    mi::Sint32 reset_mdl_file_path(
            DB::Transaction* transaction,
            const std::string& filename,
            const std::string& container_filename,
            const std::string& container_membername,
            const std::string& mdl_file_path,
            const std::string& selector);

    const std::string& get_filename() const { return m_resolved_filename; }

    const std::string& get_original_filename() const { return m_original_filename; }

    const std::string& get_mdl_file_path() const { return m_mdl_file_path; }

    const std::string& get_selector() const { return m_selector; }

    const SERIAL::Serializable* serialize(SERIAL::Serializer* serializer) const;

    SERIAL::Serializable* deserialize(SERIAL::Deserializer* deserializer);

    // TODO dump(), get_size(), get_journal_flags() missing

    const mi::Float32_4_4& get_transform() const;

    const mi::Voxel_block& get_data_bounds() const;

    mi::base::Handle<const nv::index::IVolume_direct_host> get_data(DB::Transaction*) const;

    mi::base::Handle<nv::index::IVolume_direct_cuda> upload(
            const CUDA::Device_id&,
            DB::Transaction*) const;

    static mi::base::Handle<nv::index::IVolume_direct_cuda> upload(
            const nv::index::IVolume_direct_host& volume,
            const CUDA::Device_id& device);

private:

    std::string m_original_filename;
    std::string m_resolved_filename;

    /// The absolute MDL file path of this volume.
    std::string m_mdl_file_path;

    /// The MDL container that contains the data of this DB element
    ///
    /// Non-empty exactly for container-based volumes.
    std::string m_resolved_container_filename;

    /// The MDL container member that contains the data of this DB element.
    ///
    /// Non-empty exactly for container-based volumes.
    std::string m_resolved_container_membername;

    std::string m_selector;
    mi::Voxel_block m_aabb;
    mi::Float32_4_4 m_trafo{1.f};
    mi::neuraylib::Tag m_data_t;


    mi::Sint32 load_data(DB::Transaction*, mi::neuraylib::IReader* reader = nullptr, const char* format = nullptr);
};

/// Loads a default volume texture and stores it in the DB.
///
/// Used by the MDL integration to process volume textures that appear in default arguments
/// (similar to the lightprofile and BSDF measurement loaders). A fixed mapping from the resolved
/// filename to DB element name is used to detect already loaded textures/volumes. In such a case,
/// the tag of the existing DB element is returned.
///
/// \param transaction           The DB transaction to be used.
/// \param filename              The resolved filename (for file-based volume textures).
/// \param container_filename    The resolved filename of the container itself (for container-based
///                              volume textures).
/// \param container_membername  The relative filename of the BSDF measuement in the container (for
///                              container-based volume textures).
/// \param mdl_file_path         The MDL file path.
/// \param selector              The name of the selector inside the volume texture file.
/// \param shared_proxy          Indicates whether a possibly already existing DB element for
///                              that resource should simply be reused (the decision is based on
///                              the DB element name derived from \p filename.
///                              Otherwise, an independent DB element is
///                              created, even if the resource has already been loaded.
/// \return                      The tag of that texture (invalid in case of failures).
DB::Tag load_mdl_volume_texture(
    DB::Transaction* transaction,
    const std::string& filename,
    const std::string& container_filename,
    const std::string& container_membername,
    const std::string& mdl_file_path,
    const std::string& selector,
    bool shared_proxy);

} // namespace TEXTURE

} // namespace MI

#endif // IO_SCENE_VOLUME_I_VOLUME_H
