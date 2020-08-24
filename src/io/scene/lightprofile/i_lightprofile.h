/***************************************************************************************************
 * Copyright (c) 2007-2020, NVIDIA CORPORATION. All rights reserved.
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

#ifndef IO_SCENE_LIGHTPROFILE_I_LIGHTPROFILE_H
#define IO_SCENE_LIGHTPROFILE_I_LIGHTPROFILE_H

#include <mi/neuraylib/ilightprofile.h>
#include <vector>
#include <base/data/db/i_db_journal_type.h>
#include <io/scene/scene/i_scene_scene_element.h>

namespace mi { namespace neuraylib { class IBuffer; class IReader; class IWriter; } }

namespace MI {

namespace SERIAL { class Serializer; class Deserializer; }

namespace LIGHTPROFILE {

class Lightprofile_impl;

/// The class ID for the #Lightprofile class.
static const SERIAL::Class_id ID_LIGHTPROFILE = 0x5f4c7066; // '_Lpf'

// The DB proxy class for the scene element light profile.
//
// There are two DB classes for the scene element BSDF measurement: a proxy class (Lightprofile)
// and an implementation class (Lightprofile_impl). The implementation class holds the bulk data
// and related properties, but no filename-related information or attributes. The proxy class holds
// the filename-related information and attribues, and caches trivial properties from the
// implementation class for efficiency reasons. Several instances of the proxy class might reference
// the same instance of the implementation class. The split between proxy and implementation class
// is \em not visible to API users.
class Lightprofile : public SCENE::Scene_element<Lightprofile, ID_LIGHTPROFILE>
{
public:

    /// Default constructor.
    Lightprofile();

    Lightprofile& operator=( const Lightprofile&) = delete;

    // methods of mi::neuraylib::ILightprofile

    /// Imports a light profile from a file.
    ///
    /// \param transaction           The DB transaction to be used (to create the implementation
    ///                              class in the DB).
    /// \param original_filename     The filename of the light profile. The resource search paths
    ///                              are used to locate the file.
    /// \param resolution_phi        The desired resolution in phi-direction. The special value 0
    ///                              represents the resolution from the file itself.
    /// \param resolution_theta      The desired resolution in theta-direction. The special value 0
    ///                              represents the resolution from the file itself.
    /// \param degree                The interpolation method to use.
    /// \param flags                 Flags to be used when interpreting the file data,
    ///                              see #mi::neuraylib::Lightprofile_flags for details.
    /// \return
    ///                              -  0: Success.
    ///                              - -2: Failure to resolve the given filename, e.g., the file
    ///                                    does not exist.
    ///                              - -3: \p degree or \p flags is invalid (exactly one of
    ///                                    #mi::neuraylib::LIGHTPROFILE_CLOCKWISE or
    ///                                    #mi::neuraylib::LIGHTPROFILE_COUNTER_CLOCKWISE must be
    ///                                    #set).
    ///                              - -4: File format error.
    ///                              - -5: \p resolution_phi or \p resolution_theta is invalid
    ///                                    (must not be 1).
    mi::Sint32 reset_file(
        DB::Transaction* transaction,
        const std::string& original_filename,
        mi::Uint32 resolution_phi = 0,
        mi::Uint32 resolution_theta = 0,
        mi::neuraylib::Lightprofile_degree degree = mi::neuraylib::LIGHTPROFILE_HERMITE_BASE_1,
        mi::Uint32 flags = mi::neuraylib::LIGHTPROFILE_COUNTER_CLOCKWISE);

    /// Imports a light profile from a reader.
    ///
    /// \param transaction           The DB transaction to be used (to create the implementation
    ///                              class in the DB).
    /// \param reader                The reader for the light profile.
    /// \param resolution_phi        See #reset_file().
    /// \param resolution_theta      See #reset_file().
    /// \param degree                See #reset_file().
    /// \param flags                 See #reset_file().
    /// \return                      See #reset_file() (-2 not possible here).
    mi::Sint32 reset_reader(
        DB::Transaction* transaction,
        mi::neuraylib::IReader* reader,
        mi::Uint32 resolution_phi = 0,
        mi::Uint32 resolution_theta = 0,
        mi::neuraylib::Lightprofile_degree degree = mi::neuraylib::LIGHTPROFILE_HERMITE_BASE_1,
        mi::Uint32 flags = mi::neuraylib::LIGHTPROFILE_COUNTER_CLOCKWISE);

    /// Imports a light profile from an container (used by MDL integration).
    ///
    /// \param transaction             The DB transaction to be used (to create the implementation
    ///                                class in the DB).
    /// \param reader                  The reader for the light profile.
    /// \param filename                The resolved filename (for file-based light profiles).
    /// \param container_filename      The resolved filename of the container itself (for container-
    ///                                based light profiles).
    /// \param container_membername    The relative filename of the light profile in the container
    ///                                (for container-based light profiles).
    /// \param impl_hash               Hash of the data in the implementation class. Use {0,0,0,0} if
    ///                                hash is not known.
    /// \param mdl_file_path           The MDL file path.
    /// \param resolution_phi          See #reset_file().
    /// \param resolution_theta        See #reset_file().
    /// \param degree                  See #reset_file().
    /// \param flags                   See #reset_file().
    /// \return                        See #reset_file() (-2 not possible here).
    mi::Sint32 reset_mdl(
        DB::Transaction* transaction,
        mi::neuraylib::IReader* reader,
        const std::string& filename,
        const std::string& container_filename,
        const std::string& container_membername,
        const std::string& mdl_file_path,
        const mi::base::Uuid& impl_hash,
        mi::Uint32 resolution_phi = 0,
        mi::Uint32 resolution_theta = 0,
        mi::neuraylib::Lightprofile_degree degree = mi::neuraylib::LIGHTPROFILE_HERMITE_BASE_1,
        mi::Uint32 flags = mi::neuraylib::LIGHTPROFILE_COUNTER_CLOCKWISE);

    const std::string& get_filename() const { return m_resolved_filename; }

    const std::string& get_original_filename() const { return m_original_filename; }

    const std::string& get_mdl_file_path() const { return m_mdl_file_path; }

    mi::Uint32 get_resolution_phi() const { return m_cached_resolution_phi; }

    mi::Uint32 get_resolution_theta() const { return m_cached_resolution_theta; }

    mi::neuraylib::Lightprofile_degree get_degree() const { return m_cached_degree; }

    mi::Uint32 get_flags() const { return m_cached_flags; }

    mi::Float32 get_phi( mi::Uint32 index) const;

    mi::Float32 get_theta( mi::Uint32 index) const;

    mi::Float32 get_data(
        DB::Transaction* transaction, mi::Uint32 index_phi, mi::Uint32 index_theta) const;

    const mi::Float32* get_data( DB::Transaction* transaction ) const;

    mi::Float32 get_candela_multiplier() const { return m_cached_candela_multiplier; }

    mi::Float32 sample(
        DB::Transaction* transaction, mi::Float32 phi, mi::Float32 theta, bool candela) const;

    // methods of SERIAL::Serializable

    const SERIAL::Serializable* serialize( SERIAL::Serializer* serializer) const;

    SERIAL::Serializable* deserialize( SERIAL::Deserializer* deserializer);

    void dump() const;

    // methods of DB::Element_base

    size_t get_size() const;

    DB::Journal_type get_journal_flags() const;

    Uint bundle( DB::Tag* /*results*/, Uint /*size*/) const { return 0; }

    // methods of SCENE::Scene_element_base

    void get_scene_element_references( DB::Tag_set* result) const;

    // internal methods

    /// Returns the power of the light profile.
    ///
    /// Used to fold the MDL function df::lightprofile_power().
    mi::Float32 get_power() const { return m_cached_power; }

    /// Returns the maximum value of the light profile.
    ///
    /// Used to fold the MDL function df::lightprofile_maximum().
    mi::Float32 get_maximum() const { return get_candela_multiplier(); }

    /// Indicates whether this light profile contains valid light profile data.
    bool is_valid() const { return m_cached_is_valid; }

    /// Indicates whether this light profile is file-based.
    bool is_file_based() const { return !m_resolved_filename.empty(); }

    /// Indicates whether this light profile is container-based.
    bool is_container_based() const { return !m_resolved_container_filename.empty(); }

    /// Indicates whether this light profile is memory-based.
    bool is_memory_based() const { return !is_file_based() && !is_container_based(); }

    /// Returns the container file name for container-based light profiles, and \c NULL otherwise.
    const std::string& get_container_filename() const { return m_resolved_container_filename; }

    /// Returns the container member name for container-based light profiles, and \c NULL otherwise.
    const std::string& get_container_membername() const { return m_resolved_container_membername; }

    /// Retuns the tag of the implementation class.
    ///
    /// Might be an invalid tag after default construction.
    DB::Tag get_impl_tag() const { return m_impl_tag; }

    /// Indicates whether a hash for the implementation class is available.
    bool is_impl_hash_valid() const { return m_impl_hash != mi::base::Uuid{0,0,0,0}; }

    /// Returns the hash of the implementation class (or default-constructed hash if invalid).
    const mi::base::Uuid& get_impl_hash() const { return m_impl_hash; }

private:
    /// Sets a light profile from a reader.
    ///
    /// Implements the common functionality for all \c reset_*() methods above. The parameter
    /// \c log_identifier is only to be used for log messages.
    mi::Sint32 reset_shared(
        DB::Transaction* transaction,
        mi::neuraylib::IReader* reader,
        const std::string& log_identifier,
        const mi::base::Uuid& impl_hash,
        mi::Uint32 resolution_phi = 0,
        mi::Uint32 resolution_theta = 0,
        mi::neuraylib::Lightprofile_degree degree = mi::neuraylib::LIGHTPROFILE_HERMITE_BASE_1,
        mi::Uint32 flags = mi::neuraylib::LIGHTPROFILE_COUNTER_CLOCKWISE);

    /// Set up all cached values based on the values in \p impl.
    void setup_cached_values( const Lightprofile_impl* impl);

    /// The file (or MDL file path) that contains the data of this DB element.
    ///
    /// Non-empty for file-based light profiles.
    ///
    /// This is the filename as it has been passed into reset_file().
    std::string m_original_filename;

    /// The file that contains the data of this DB element.
    ///
    /// Non-empty for file-based light profiles.
    ///
    /// This is the filename as it has been resolved in reset_file() or deserialize().
    std::string m_resolved_filename;

    /// The container that contains the data of this DB element
    ///
    /// Non-empty exactly for container-based light profiles.
    std::string m_resolved_container_filename;

    /// The container member that contains the data of this DB element.
    ///
    /// Non-empty exactly for container-based light profiles.
    std::string m_resolved_container_membername;

    /// The MDL file path.
    std::string m_mdl_file_path;

    /// The implementation class that holds the bulk data and non-filename related properties.
    DB::Tag m_impl_tag;

    /// Hash of the data in the implementation class.
    mi::base::Uuid m_impl_hash;

    // Non-filename related properties from the implementation class (cached here for efficiency).
    //@{

    mi::Uint32  m_cached_resolution_phi;
    mi::Uint32  m_cached_resolution_theta;
    mi::neuraylib::Lightprofile_degree m_cached_degree;
    mi::Uint32  m_cached_flags;
    mi::Float32 m_cached_start_phi;
    mi::Float32 m_cached_start_theta;
    mi::Float32 m_cached_delta_phi;
    mi::Float32 m_cached_delta_theta;
    mi::Float32 m_cached_candela_multiplier;
    mi::Float32 m_cached_power;
    bool        m_cached_is_valid;

    //@}
};

/// The class ID for the #Lightprofile_impl class.
static const SERIAL::Class_id ID_LIGHTPROFILE_IMPL = 0x5f4c7069; // '_Lpi'

// The DB implementation class for the scene element light profile.
//
// There are two DB classes for the scene element BSDF measurement: a proxy class (Lightprofile)
// and an implementation class (Lightprofile_impl). The implementation class holds the bulk data
// and related properties, but no filename-related information or attributes. The proxy class holds
// the filename-related information and attribues, and caches trivial properties from the
// implementation class for efficiency reasons. Several instances of the proxy class might reference
// the same instance of the implementation class. The split between proxy and implementation class
// is \em not visible to API users.
class Lightprofile_impl : public SCENE::Scene_element<Lightprofile_impl, ID_LIGHTPROFILE_IMPL>
{
public:

    /// Default constructor.
    ///
    /// Should only be used for derserialization.
    Lightprofile_impl();

    /// Constructor.
    Lightprofile_impl(
        mi::Uint32 resolution_phi,
        mi::Uint32 resolution_theta,
        mi::neuraylib::Lightprofile_degree degree,
        mi::Uint32 flags,
        mi::Float32 start_phi,
        mi::Float32 start_theta,
        mi::Float32 delta_phi,
        mi::Float32 delta_theta,
        const std::vector<mi::Float32>& data);

    Lightprofile_impl& operator=( const Lightprofile_impl&) = delete;

    // methods of mi::neuraylib::ILightprofile

    mi::Uint32 get_resolution_phi() const { return m_resolution_phi; }

    mi::Uint32 get_resolution_theta() const { return m_resolution_theta; }

    mi::neuraylib::Lightprofile_degree get_degree() const { return m_degree; }

    mi::Uint32 get_flags() const { return m_flags; }

    mi::Float32 get_phi( mi::Uint32 index) const;

    mi::Float32 get_theta( mi::Uint32 index) const;

    mi::Float32 get_data( mi::Uint32 index_phi, mi::Uint32 index_theta) const;

    const mi::Float32* get_data() const;

    mi::Float32 get_candela_multiplier() const { return m_candela_multiplier; }

    mi::Float32 sample( mi::Float32 phi, mi::Float32 theta, bool candela) const;

    // methods of SERIAL::Serializable

    const SERIAL::Serializable* serialize( SERIAL::Serializer* serializer) const;

    SERIAL::Serializable* deserialize( SERIAL::Deserializer* deserializer);

    void dump() const;

    // methods of DB::Element_base

    size_t get_size() const;

    DB::Journal_type get_journal_flags() const;

    Uint bundle( DB::Tag* /*results*/, Uint /*size*/) const { return 0; }

    // methods of SCENE::Scene_element_base

    void get_scene_element_references( DB::Tag_set* result) const { }

    // internal methods

    /// Returns the power of the light profile.
    ///
    /// Used to fold the MDL function df::lightprofile_power().
    mi::Float32 get_power() const { return m_power; }

    /// Returns the maximum value of the light profile.
    ///
    /// Used to fold the MDL function df::lightprofile_maximum().
    mi::Float32 get_maximum() const { return get_candela_multiplier(); }

    /// Indicates whether this light profile contains valid light profile data.
    bool is_valid() const { return !m_data.empty(); }

    mi::Float32 get_start_phi() const { return m_start_phi; }

    mi::Float32 get_start_theta() const { return m_start_theta; }

    mi::Float32 get_delta_phi() const { return m_delta_phi; }

    mi::Float32 get_delta_theta() const { return m_delta_theta; }

private:

    // All members below are essentially const, but cannot be declared as such due to deserialize().

    // Constructor arguments.
    mi::Uint32 m_resolution_phi;
    mi::Uint32 m_resolution_theta;
    mi::neuraylib::Lightprofile_degree m_degree;
    mi::Uint32 m_flags;
    mi::Float32 m_start_phi;
    mi::Float32 m_start_theta;
    mi::Float32 m_delta_phi;
    mi::Float32 m_delta_theta;
    std::vector<mi::Float32> m_data;

    // Computed data
    mi::Float32 m_candela_multiplier;
    mi::Float32 m_power;
};

/// Exports the light profile to a file.
///
/// \param transaction    The DB transaction to be used.
/// \param lightprofile   The light profile to export.
/// \param filename       The filename to use.
/// \return               \c true in case of success, \c false otherwise.
bool export_to_file(
    DB::Transaction* transaction, const Lightprofile* lightprofile, const std::string& filename);

/// Exports the light profile to a buffer.
///
/// \param transaction    The DB transaction to be used.
/// \param lightprofile   The light profile to export.
/// \return               The buffer in case of success, NULL otherwise.
mi::neuraylib::IBuffer* create_buffer_from_lightprofile(
    DB::Transaction* transaction, const Lightprofile* lightprofile);

/// Loads a default light profile and stores it in the DB.
///
/// Used by the MDL integration to process light profiles that appear in default arguments (similar
/// to the texture and BSDF measurement loaders). A fixed mapping from the filenames to DB element
/// name is used to detect already loaded light profiles. In such a case, the tag of the existing
/// DB element is returned (but never for memory-based light profiles).
///
/// \param transaction           The DB transaction to be used.
/// \param reader                The reader to be used to obtain the light profile. Needs to
///                              support absolute access.
/// \param filename              The resolved filename (for file-based light profiles).
/// \param container_filename    The resolved filename of the container itself (for container-based
///                              light profiles).
/// \param container_membername  The relative filename of the light profile in the container (for
///                              container-based light profiles).
/// \param mdl_file_path         The MDL file path.
/// \param impl_hash             Hash of the data in the implementation DB element. Use {0,0,0,0} if
///                              hash is not known.
/// \param shared_proxy          Indicates whether a possibly already existing proxy DB element for
///                              that resource should simply be reused (the decision is based on
///                              \c container_filename and \c container_membername, not on
///                              \c impl_hash). Otherwise, an independent proxy DB element is
///                              created, even if the resource has already been loaded.
/// \return                      The tag of that light profile (invalid in case of failures).
DB::Tag load_mdl_lightprofile(
    DB::Transaction* transaction,
    mi::neuraylib::IReader* reader,
    const std::string& filename,
    const std::string& container_filename,
    const std::string& container_membername,
    const std::string& mdl_file_path,
    const mi::base::Uuid& impl_hash,
    bool shared_proxy);

} // namespace LIGHTPROFILE

} // namespace MI

#endif // IO_SCENE_LIGHTPROFILE_I_LIGHTPROFILE_H
