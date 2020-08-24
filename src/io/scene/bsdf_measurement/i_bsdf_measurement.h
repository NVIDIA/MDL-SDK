/***************************************************************************************************
 * Copyright (c) 2013-2020, NVIDIA CORPORATION. All rights reserved.
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

#ifndef IO_SCENE_BSDF_MEASUREMENT_I_BSDF_MEASUREMENT_H
#define IO_SCENE_BSDF_MEASUREMENT_I_BSDF_MEASUREMENT_H

#include <mi/base/handle.h>
#include <base/data/db/i_db_journal_type.h>
#include <io/scene/scene/i_scene_scene_element.h>

namespace mi { namespace neuraylib { class IBuffer; class IBsdf_isotropic_data; class IReader; } }

namespace MI {

namespace DB { class Transaction; }
namespace SERIAL { class Serializer; class Deserializer; }

namespace BSDFM {

class Bsdf_measurement_impl;

/// The class ID for the #Bsdf_measurement class.
static const SERIAL::Class_id ID_BSDF_MEASUREMENT = 0x5f427364; // '_Bsd'

// The DB proxy class for the scene element light profile.
//
// There are two DB classes for the scene element BSDF measurement: a proxy class (Bsdf_measurement)
// and an implementation class (Bsdf_measurement_impl). The implementation class holds the bulk data
// and related properties, but no filename-related information or attributes. The proxy class holds
// the filename-related information and attribues, and caches trivial properties from the
// implementation class for efficiency reasons. Several instances of the proxy class might reference
// the same instance of the implementation class. The split between proxy and implementation class
// is \em not visible to API users.
class Bsdf_measurement : public SCENE::Scene_element<Bsdf_measurement, ID_BSDF_MEASUREMENT>
{
public:

    /// Default constructor.
    ///
    /// No dummy BSDF data is set for reflection and transmission.
    Bsdf_measurement();

    Bsdf_measurement& operator=( const Bsdf_measurement&) = delete;

    // methods of mi::neuraylib::IBsdf_measurement

    /// Imports a BSDF measurement from a file.
    ///
    /// \param transaction           The DB transaction to be used (to create the implementation
    ///                              class in the DB).
    /// \param original_filename     The filename of the BSDF measurement. The resource search paths
    ///                              are used to locate the file.
    /// \return
    ///                              -  0: Success.
    ///                              - -2: Failure to resolve the given filename, e.g., the file
    ///                                    does not exist.
    ///                              - -3: Invalid file format or invalid filename extension (only
    ///                                    \c .mbsdf is supported).
    Sint32 reset_file( DB::Transaction* transaction, const std::string& original_filename);

    /// Imports a BSDF measurement from a reader.
    ///
    /// \param transaction           The DB transaction to be used (to create the implementation
    ///                              class in the DB).
    /// \param reader                The reader for the BSDF measurement.
    /// \return
    ///                              -  0: Success.
    ///                              - -3: Invalid file format.
    Sint32 reset_reader( DB::Transaction* transaction, mi::neuraylib::IReader* reader);

    /// Imports a BSDF measurement from a reader (used by MDL integration).
    ///
    /// \param transaction           The DB transaction to be used (to create the implementation
    ///                              class in the DB).
    /// \param reader                The reader for the BSDF measurement.
    /// \param filename              The resolved filename (for file-based BSDF measurements).
    /// \param container_filename    The resolved filename of the container itself (for container-based
    ///                              BSDF measurements).
    /// \param container_membername  The relative filename of the BSDF measuement in the container (for
    ///                              container-based BSDF measurements).
    /// \param mdl_file_path         The MDL file path.
    /// \param impl_hash             Hash of the data in the implementation class. Use {0,0,0,0} if
    ///                              hash is not known.
    /// \return
    ///                              -  0: Success.
    ///                              - -3: Invalid file format or invalid filename extension (only
    ///                                    \c .mbsdf is supported).
    Sint32 reset_mdl(
        DB::Transaction* transaction,
        mi::neuraylib::IReader* reader,
        const std::string& filename,
        const std::string& container_filename,
        const std::string& container_membername,
        const std::string& mdl_file_path,
        const mi::base::Uuid& impl_hash);

    const std::string& get_filename() const { return m_resolved_filename; }

    const std::string& get_original_filename() const { return m_original_filename; }

    const std::string& get_mdl_file_path() const { return m_mdl_file_path; }

    void set_reflection(
        DB::Transaction* transaction, const mi::neuraylib::IBsdf_isotropic_data* reflection);

    const mi::base::IInterface* get_reflection( DB::Transaction* transaction) const;

    template <class T>
    const T* get_reflection( DB::Transaction* transaction) const
    {
        const mi::base::IInterface* ptr_iinterface = get_reflection( transaction);
        if( !ptr_iinterface)
            return nullptr;
        const T* ptr_T = static_cast<const T*>( ptr_iinterface->get_interface( typename T::IID()));
        ptr_iinterface->release();
        return ptr_T;
    }

    void set_transmission(
        DB::Transaction* transaction, const mi::neuraylib::IBsdf_isotropic_data* bsdf_data);

    const mi::base::IInterface* get_transmission( DB::Transaction* transmission) const;

    template <class T>
    const T* get_transmission( DB::Transaction* transaction) const
    {
        const mi::base::IInterface* ptr_iinterface = get_transmission( transaction);
        if( !ptr_iinterface)
            return nullptr;
        const T* ptr_T = static_cast<const T*>( ptr_iinterface->get_interface( typename T::IID()));
        ptr_iinterface->release();
        return ptr_T;
    }

    // methods of SERIAL::Serializable

    const SERIAL::Serializable* serialize( SERIAL::Serializer* serializer) const;

    SERIAL::Serializable* deserialize( SERIAL::Deserializer* deserializer);

    void dump() const;

    // methods of DB::Element_base

    size_t get_size() const;

    DB::Journal_type get_journal_flags() const;

    Uint bundle( DB::Tag* results, Uint size) const { return 0; }

    // methods of SCENE::Scene_element_base

    void get_scene_element_references( DB::Tag_set* result) const;

    // internal methods

    /// Indicates whether this BSDF measurement contains valid reflection or transmission data.
    bool is_valid() const { return m_cached_is_valid; }

    /// Indicates whether this BSDF measurement is file-based.
    bool is_file_based() const { return !m_resolved_filename.empty(); }

    /// Indicates whether this BSDF measurement is container-based.
    bool is_container_based() const { return !m_resolved_container_filename.empty(); }

    /// Indicates whether this BSDF measurement is memory-based.
    bool is_memory_based() const { return !is_file_based() && !is_container_based(); }

    /// Returns the container file name for container-based BSDF measurements,
    /// and \c NULL otherwise.
    const std::string& get_container_filename() const { return m_resolved_container_filename; }

    /// Returns the container member name for container-based BSDF measurements,
    /// and \c NULL otherwise.
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
    /// Set a BSDF measurement from two isotropic data sets.
    ///
    /// Implements the common functionality for all \c reset_*() and \c set_*() methods above.
    void reset_shared(
        DB::Transaction* transaction,
        const mi::neuraylib::IBsdf_isotropic_data* reflection,
        const mi::neuraylib::IBsdf_isotropic_data* transmission,
        const mi::base::Uuid& impl_hash);

    /// Set up all cached values based on the values in \p impl.
    void setup_cached_values( const Bsdf_measurement_impl* impl);

    /// The file (or MDL file path) that contains the data of this DB element.
    ///
    /// Non-empty for file-based BSDF measurements.
    ///
    /// This is the filename as it has been passed into reset_file().
    std::string m_original_filename;

    /// The file that contains the data of this DB element.
    ///
    /// Non-empty exactly for file-based BSDF measurements.
    ///
    /// This is the filename as it has been resolved in reset_file() or deserialize().
    std::string m_resolved_filename;

    /// The container that contains the data of this DB element
    ///
    /// Non-empty exactly for container-based BSDF measurements.
    std::string m_resolved_container_filename;

    /// The container member that contains the data of this DB element.
    ///
    /// Non-empty exactly for container-based BSDF measurements.
    std::string m_resolved_container_membername;

    /// The MDL file path.
    std::string m_mdl_file_path;

    /// The implementation class that holds the bulk data and non-filename related properties.
    DB::Tag m_impl_tag;

    /// Hash of the data in the implementation class.
    mi::base::Uuid m_impl_hash;

    // Non-filename related properties from the implementation class (cached here for efficiency).
    //@{

    bool m_cached_is_valid;

    //@}
};

/// The class ID for the #Bsdf_measurement class.
static const SERIAL::Class_id ID_BSDF_MEASUREMENT_IMPL = 0x5f427369; // '_Bsi'

// The DB implementation class for the scene element light profile.
//
// There are two DB classes for the scene element BSDF measurement: a proxy class (Bsdf_measurement)
// and an implementation class (Bsdf_measurement_impl). The implementation class holds the bulk data
// and related properties, but no filename-related information or attributes. The proxy class holds
// the filename-related information and attribues, and caches trivial properties from the
// implementation class for efficiency reasons. Several instances of the proxy class might reference
// the same instance of the implementation class. The split between proxy and implementation class
// is \em not visible to API users.
class Bsdf_measurement_impl
  : public SCENE::Scene_element<Bsdf_measurement_impl, ID_BSDF_MEASUREMENT_IMPL>
{
public:

    /// Default constructor.
    ///
    /// Should only be used for derserialization.
    Bsdf_measurement_impl();

    /// Constructor.
    Bsdf_measurement_impl(
        const mi::neuraylib::IBsdf_isotropic_data* reflection,
        const mi::neuraylib::IBsdf_isotropic_data* transmission);

    /// Copy constructor.
    ///
    /// Explicit trivial copy constructor because the implicitly generated one requires the full
    /// definition of mi::neuraylib::IBsdf_isotropic_data (at least Visual Studio implicitly
    /// generates the default copy constructor as soon as it is needed *inline*).
    Bsdf_measurement_impl( const Bsdf_measurement_impl& other);

    Bsdf_measurement_impl& operator=( const Bsdf_measurement_impl&) = delete;

    /// Destructor.
    ///
    /// Explicit trivial destructor because the implicitly generated one requires the full
    /// definition of mi::neuraylib::IBsdf_isotropic_data.
    ~Bsdf_measurement_impl();

    // methods of mi::neuraylib::IBsdf_measurement

    const mi::base::IInterface* get_reflection() const;

    template <class T>
    const T* get_reflection() const
    {
        const mi::base::IInterface* ptr_iinterface = get_reflection();
        if( !ptr_iinterface)
            return nullptr;
        const T* ptr_T = static_cast<const T*>( ptr_iinterface->get_interface( typename T::IID()));
        ptr_iinterface->release();
        return ptr_T;
    }

    const mi::base::IInterface* get_transmission() const;

    template <class T>
    const T* get_transmission() const
    {
        const mi::base::IInterface* ptr_iinterface = get_transmission();
        if( !ptr_iinterface)
            return nullptr;
        const T* ptr_T = static_cast<const T*>( ptr_iinterface->get_interface( typename T::IID()));
        ptr_iinterface->release();
        return ptr_T;
    }

    // methods of SERIAL::Serializable

    const SERIAL::Serializable* serialize( SERIAL::Serializer* serializer) const;

    SERIAL::Serializable* deserialize( SERIAL::Deserializer* deserializer);

    void dump() const;

    // methods of DB::Element_base

    size_t get_size() const;

    DB::Journal_type get_journal_flags() const;

    Uint bundle( DB::Tag* results, Uint size) const { return 0; }

    // methods of SCENE::Scene_element_base

    void get_scene_element_references( DB::Tag_set* result) const { }

    // internal methods

    /// Indicates whether this BSDF measurement contains valid reflection or transmission data.
    bool is_valid() const
    {
        return m_reflection.is_valid_interface() || m_transmission.is_valid_interface();
    }


private:
    /// Serializes an instance of #mi::neuraylib::IBsdf_isotropic_data.
    static void serialize_bsdf_data(
        SERIAL::Serializer* serializer, const mi::neuraylib::IBsdf_isotropic_data* bsdf_data);

    /// Deserializes an instance of #mi::neuraylib::IBsdf_isotropic_data.
    static mi::neuraylib::IBsdf_isotropic_data* deserialize_bsdf_data(
        SERIAL::Deserializer* deserializer);

    // All members below are essentially const, but cannot be declared as such due to deserialize().

    /// The BSDF data for the reflection.
    mi::base::Handle<const mi::neuraylib::IBsdf_isotropic_data> m_reflection;

    /// The BSDF data for the transmission.
    mi::base::Handle<const mi::neuraylib::IBsdf_isotropic_data> m_transmission;
};

/// Imports BSDF data from a file.
///
/// \param filename             The file to import (already resolved against search paths).
/// \param[out] reflection      The imported BSDF data for the reflection (or \c NULL if there is no
///                             BSDF data for the reflection.
/// \param[out] transmission    The imported BSDF data for the reflection (or \c NULL if there is no
///                             BSDF data for the transmission.
/// \return                     \c true in case of success, \c false otherwise.
bool import_from_file(
    const std::string& filename,
    mi::base::Handle<mi::neuraylib::IBsdf_isotropic_data>& reflection,
    mi::base::Handle<mi::neuraylib::IBsdf_isotropic_data>& transmission);

/// Imports BSDF data from a reader.
///
/// \param reader               The reader to import from.
/// \param[out] reflection      The imported BSDF data for the reflection (or \c NULL if there is no
///                             BSDF data for the reflection.
/// \param[out] transmission    The imported BSDF data for the reflection (or \c NULL if there is no
///                             BSDF data for the transmission.
/// \return                     \c true in case of success, \c false otherwise.
bool import_from_reader(
    mi::neuraylib::IReader* reader,
    mi::base::Handle<mi::neuraylib::IBsdf_isotropic_data>& reflection,
    mi::base::Handle<mi::neuraylib::IBsdf_isotropic_data>& transmission);

/// Exports the BSDF data to a file.
///
/// \param reflection     The BSDF data to export for the reflection. Can be \p NULL.
/// \param transmission   The BSDF data to export for the transmission. Can be \p NULL.
/// \param filename       The filename to use.
/// \return               \c true in case of success, \c false otherwise.
bool export_to_file(
    const mi::neuraylib::IBsdf_isotropic_data* reflection,
    const mi::neuraylib::IBsdf_isotropic_data* transmission,
    const std::string& filename);

/// Exports the BSDF data to a buffer.
///
/// \param reflection     The BSDF data to export for the reflection. Can be \p NULL.
/// \param transmission   The BSDF data to export for the transmission. Can be \p NULL.
/// \return               The buffer in case of success, NULL otherwise.
mi::neuraylib::IBuffer* create_buffer_from_bsdf_measurement(
    const mi::neuraylib::IBsdf_isotropic_data* reflection,
    const mi::neuraylib::IBsdf_isotropic_data* transmission);

/// Loads a default BSDF measurement and stores it in the DB.
///
/// Used by the MDL integration to process BSDF measurements that appear in default arguments
/// (similar to the texture and light profile loaders). A fixed mapping from the filenames to DB
/// element name is used to detect already loaded BSDF measurements. In such a case, the tag of the
/// existing DB element is returned (but never for memory-based BSDF measurements).
///
/// \param transaction           The DB transaction to be used.
/// \param reader                The reader to be used to obtain the BSDF measurement. Needs to
///                              support absolute access.
/// \param filename              The resolved filename (for file-based BSDF measurements).
/// \param container_filename    The resolved filename of the container itself (for container-based
///                              BSDF measurements).
/// \param container_membername  The relative filename of the BSDF measuement in the container (for
///                              container-based BSDF measurements).
/// \param mdl_file_path         The MDL file path.
/// \param impl_hash             Hash of the data in the implementation DB element. Use {0,0,0,0} if
///                              hash is not known.
/// \param shared_proxy          Indicates whether a possibly already existing proxy DB element for
///                              that resource should simply be reused (the decision is based on
///                              \c container_filename and \c container_membername, not on
///                              \c impl_hash). Otherwise, an independent proxy DB element is
///                              created, even if the resource has already been loaded.
/// \return                      The tag of that BSDF measurement (invalid in case of failures).
DB::Tag load_mdl_bsdf_measurement(
    DB::Transaction* transaction,
    mi::neuraylib::IReader* reader,
    const std::string& filename,
    const std::string& container_filename,
    const std::string& container_membername,
    const std::string& mdl_file_path,
    const mi::base::Uuid& impl_hash,
    bool shared_proxy);

} // namespace BSDFM

} // namespace MI

#endif // IO_SCENE_BSDF_MEASUREMENT_I_BSDF_MEASUREMENT_H
