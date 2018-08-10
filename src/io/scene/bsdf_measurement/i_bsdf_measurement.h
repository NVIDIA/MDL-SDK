/***************************************************************************************************
 * Copyright (c) 2013-2018, NVIDIA CORPORATION. All rights reserved.
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

namespace mi { namespace neuraylib { class IBsdf_isotropic_data; class IReader; } }

namespace MI {

namespace SERIAL { class Serializer; class Deserializer; }

namespace BSDFM {

/// The class ID for the #Bsdf_measurement class.
static const SERIAL::Class_id ID_BSDF_MEASUREMENT = 0x5f427364; // '_Bsd'

class Bsdf_measurement : public SCENE::Scene_element<Bsdf_measurement, ID_BSDF_MEASUREMENT>
{
public:

    /// Default constructor.
    ///
    /// No dummy BSDF data is set for reflection and transmission.
    Bsdf_measurement();

    /// Copy constructor.
    ///
    /// Explicit trivial copy constructor because the implicitly generated one requires the full
    /// definition of mi::neuraylib::IBsdf_isotropic_data (at least Visual Studio implicitly
    /// generates the default copy constructor as soon as it is needed *inline*).
    Bsdf_measurement( const Bsdf_measurement& other);

    /// Destructor.
    ///
    /// Explicit trivial destructor because the implicitly generated one requires the full
    /// definition of mi::neuraylib::IBsdf_isotropic_data.
    ~Bsdf_measurement();

    // methods of mi::neuraylib::IBsdf_measurement

    /// Imports a BSDF measurement from a file.
    ///
    /// \param original_filename     The filename of the BSDF measurement. The resource search paths
    ///                              are used to locate the file.
    /// \return
    ///                              -  0: Success.
    ///                              - -2: Failure to resolve the given filename, e.g., the file
    ///                                    does not exist.
    ///                              - -3: Invalid file format or invalid filename extension (only
    ///                                    \c .mbsdf is supported).
    Sint32 reset_file( const std::string& original_filename);

    /// Imports a BSDF measurement from a reader.
    ///
    /// \param reader                The reader for the BSDF measurement.
    /// \return
    ///                              -  0: Success.
    ///                              - -3: Invalid file format.
    Sint32 reset_reader( mi::neuraylib::IReader* reader);

    /// Imports a BSDF measurement from a file (used by MDL integration).
    ///
    /// \param resolved_filename     The resolved filename of the BSDF measurement. The MDL
    ///                              integration passes an already resolved filename here since it
    ///                              uses its own filename resolution rules.
    /// \param mdl_file_path         The MDL file path.
    /// \return
    ///                              -  0: Success.
    ///                              - -2: Failure to resolve the given filename, e.g., the file
    ///                                    does not exist.
    ///                              - -3: Invalid file format or invalid filename extension (only
    ///                                    \c .mbsdf is supported).
    Sint32 reset_file_mdl(
        const std::string& resolved_filename, const std::string& mdl_file_path);

    /// Imports a BSDF measurement from an archive (used by MDL integration).
    ///
    /// \param reader                The reader for the BSDF measurement.
    /// \param archive_filename      The resolved archive filename.
    /// \param archive_membername    The resolved archive member name.
    /// \param mdl_file_path         The MDL file path.
    /// \return
    ///                              -  0: Success.
    ///                              - -3: Invalid file format or invalid filename extension (only
    ///                                    \c .mbsdf is supported).
    Sint32 reset_archive_mdl(
        mi::neuraylib::IReader* reader,
        const std::string& archive_filename,
        const std::string& archive_membername,
        const std::string& mdl_file_path);

    const std::string& get_filename() const;

    const std::string& get_original_filename() const;

    const std::string& get_mdl_file_path() const;

    void set_reflection( const mi::neuraylib::IBsdf_isotropic_data* bsdf_data);

    const mi::base::IInterface* get_reflection() const;

    template <class T>
    const T* get_reflection() const
    {
        const mi::base::IInterface* ptr_iinterface = get_reflection();
        if( !ptr_iinterface)
            return 0;
        const T* ptr_T = static_cast<const T*>( ptr_iinterface->get_interface( typename T::IID()));
        ptr_iinterface->release();
        return ptr_T;
    }

    void set_transmission( const mi::neuraylib::IBsdf_isotropic_data* bsdf_data);

    const mi::base::IInterface* get_transmission() const;

    template <class T>
    const T* get_transmission() const
    {
        const mi::base::IInterface* ptr_iinterface = get_transmission();
        if( !ptr_iinterface)
            return 0;
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

    Uint bundle( DB::Tag* results, Uint size) const;

    // methods of SCENE::Scene_element_base

    void get_scene_element_references( DB::Tag_set* result) const;

    // internal methods

    /// Indicates whether this BSDF measurement contains valid reflection or transmission data.
    bool is_valid() const
    {
        return m_reflection.is_valid_interface() || m_transmission.is_valid_interface();
    }

    /// Indicates whether this BSDF measurement is file-based.
    bool is_file_based() const { return !m_resolved_filename.empty(); }

    /// Indicates whether this BSDF measurement is archive-based.
    bool is_archive_based() const { return !m_resolved_archive_filename.empty(); }

    /// Indicates whether this BSDF measurement is memory-based.
    bool is_memory_based() const { return !is_file_based() && !is_archive_based(); }

    /// Returns the archive file name for archive-based BSDF measurements, and \c NULL otherwise.
    const std::string& get_archive_filename() const { return m_resolved_archive_filename;; }

    /// Returns the archive member name for archive-based BSDF measurements, and \c NULL otherwise.
    const std::string& get_archive_membername() const { return m_resolved_archive_membername; }

private:
    /// Comments on DB::Element_base and DB::Element say that the copy constructor is needed.
    /// But the assignment operator is not implemented, although usually, they are implemented both
    /// or none. Let's make the assignment operator private for now.
    Bsdf_measurement& operator=( const Bsdf_measurement&);

    /// Serializes an instance of #mi::neuraylib::IBsdf_isotropic_data.
    static void serialize_bsdf_data(
        SERIAL::Serializer* serializer, const mi::neuraylib::IBsdf_isotropic_data* bsdf_data);

    /// Deserializes an instance of #mi::neuraylib::IBsdf_isotropic_data.
    static mi::neuraylib::IBsdf_isotropic_data* deserialize_bsdf_data(
        SERIAL::Deserializer* deserializer);

    /// Dumps some data about an instance of #mi::neuraylib::IBsdf_isotropic_data.
    ///
    /// Used by the various reset_*() methods and dump().
    static std::string dump_data( const mi::neuraylib::IBsdf_isotropic_data* data);

    /// The BSDF data for the reflection.
    mi::base::Handle<const mi::neuraylib::IBsdf_isotropic_data> m_reflection;

    /// The BSDF data for the transmission.
    mi::base::Handle<const mi::neuraylib::IBsdf_isotropic_data> m_transmission;

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

    /// The archive that contains the data of this DB element
    ///
    /// Non-empty exactly for archive-based BSDF measurements.
    std::string m_resolved_archive_filename;

    /// The archive member that contains the data of this DB element.
    ///
    /// Non-empty exactly for archive-based BSDF measurements.
    std::string m_resolved_archive_membername;

    /// The MDL file path.
    std::string m_mdl_file_path;
};

/// Imports BSDF data from a file.
///
/// \param filename            The file to import (already resolved against search paths).
/// \param[out] reflection     The imported BSDF data for the reflection. The incoming value must be
///                            \p NULL. The reference count of the outgoing value has already been
///                            increased for the caller (similar as for return values). Note that
///                            the outgoing value is \p NULL if there is no BSDF data for the
///                            reflection.
/// \param[out] transmission   The imported BSDF data for the reflection. The incoming value must be
///                            \p NULL. The reference count of the outgoing value has already been
///                            increased for the caller (similar as for return values). Note that
///                            the outgoing value is \p NULL if there is no BSDF data for the
///                            transmission.
/// \return                    \c true in case of success, \c false otherwise.
bool import_from_file(
    const std::string& filename,
    mi::neuraylib::IBsdf_isotropic_data*& reflection,
    mi::neuraylib::IBsdf_isotropic_data*& transmission);

/// Imports BSDF data from a reader.
///
/// \param reader              The reader to import from.
/// \param[out] reflection     The imported BSDF data for the reflection. The incoming value must be
///                            \p NULL. The reference count of the outgoing value has already been
///                            increased for the caller (similar as for return values). Note that
///                            the outgoing value is \p NULL if there is no BSDF data for the
///                            reflection.
/// \param[out] transmission   The imported BSDF data for the reflection. The incoming value must be
///                            \p NULL. The reference count of the outgoing value has already been
///                            increased for the caller (similar as for return values). Note that
///                            the outgoing value is \p NULL if there is no BSDF data for the
///                            transmission.
/// \return                    \c true in case of success, \c false otherwise.
bool import_from_reader(
    mi::neuraylib::IReader* reader,
    mi::neuraylib::IBsdf_isotropic_data*& reflection,
    mi::neuraylib::IBsdf_isotropic_data*& transmission);

/// Imports BSDF data from a reader.
///
/// \param reader              The reader to import from.
/// \param archive_filename    The resolved filename of the archive itself.
/// \param archive_membername  The relative filename of the BSDF measurement in the archive.
/// \param[out] reflection     The imported BSDF data for the reflection. The incoming value must be
///                            \p NULL. The reference count of the outgoing value has already been
///                            increased for the caller (similar as for return values). Note that
///                            the outgoing value is \p NULL if there is no BSDF data for the
///                            reflection.
/// \param[out] transmission   The imported BSDF data for the reflection. The incoming value must be
///                            \p NULL. The reference count of the outgoing value has already been
///                            increased for the caller (similar as for return values). Note that
///                            the outgoing value is \p NULL if there is no BSDF data for the
///                            transmission.
/// \return                    \c true in case of success, \c false otherwise.
bool import_from_reader(
    mi::neuraylib::IReader* reader,
    const std::string& archive_filename,
    const std::string& archive_membername,
    mi::neuraylib::IBsdf_isotropic_data*& reflection,
    mi::neuraylib::IBsdf_isotropic_data*& transmission);

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

/// Loads a default BSDF measurement and stores it in the DB.
///
/// Used by the MDL integration to process BSDF measurements that appear in default arguments
/// (similar to the texture and light profile loaders). A fixed mapping from the resolved filename
/// to DB element name is used to detect already loaded BSDF measurements. In such a case, the tag
/// of the existing DB element is returned.
///
/// \param transaction         The DB transaction to be used.
/// \param resolved_filename   The resolved filename of the BSDF measurement.
/// \param mdl_file_path       The MDL file path.
/// \param shared              Indicates whether a possibly already existing DB element for that
///                            resource should simply be reused. Otherwise, an independent DB
///                            element is created, even if the resource has already been loaded.
/// \return                    The tag of that BSDF measurement (invalid in case of failures).
DB::Tag load_mdl_bsdf_measurement(
    DB::Transaction* transaction,
    const std::string& resolved_filename,
    const std::string& mdl_file_path,
    bool shared);

/// Loads a default BSDF measurement and stores it in the DB.
///
/// Used by the MDL integration to process BSDF measurements that appear in default arguments
/// (similar to the texture and light profile loaders). A fixed mapping from the archive and member
/// filenames to DB element name is used to detect already loaded BSDF measurements. In such a case,
/// the tag of the existing DB element is returned.
///
/// \param transaction         The DB transaction to be used.
/// \param reader              The reader to be used to obtain the BSDF measurement. Needs to
///                            support absolute access.
/// \param archive_filename    The resolved filename of the archive itself.
/// \param archive_membername  The relative filename of the BSDF measurement in the archive.
/// \param mdl_file_path       The MDL file path.
/// \param shared              Indicates whether a possibly already existing DB element for that
///                            resource should simply be reused. Otherwise, an independent DB
///                            element is created, even if the resource has already been loaded.
/// \return                    The tag of that BSDF measurement (invalid in case of failures).
DB::Tag load_mdl_bsdf_measurement(
    DB::Transaction* transaction,
    mi::neuraylib::IReader* reader,
    const std::string& archive_filename,
    const std::string& archive_membername,
    const std::string& mdl_file_path,
    bool shared);

} // namespace BSDFM

} // namespace MI

#endif // IO_SCENE_BSDF_MEASUREMENT_I_BSDF_MEASUREMENT_H
