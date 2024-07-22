/***************************************************************************************************
 * Copyright (c) 2013-2024, NVIDIA CORPORATION. All rights reserved.
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

#include "pch.h"

#include "i_bsdf_measurement.h"

#include <filesystem>
#include <sstream>

#include <mi/neuraylib/bsdf_isotropic_data.h>
#include <mi/neuraylib/ibsdf_isotropic_data.h>
#include <mi/neuraylib/ireader.h>

#include <base/system/main/access_module.h>
#include <base/hal/disk/disk_file_reader_writer_impl.h>
#include <base/hal/disk/disk_memory_reader_writer_impl.h>
#include <base/hal/hal/i_hal_ospath.h>
#include <base/lib/config/config.h>
#include <base/lib/log/i_log_logger.h>
#include <base/lib/log/i_log_assert.h>
#include <base/lib/path/i_path.h>
#include <base/data/serial/i_serializer.h>
#include <base/data/db/i_db_access.h>
#include <base/data/db/i_db_transaction.h>
#include <base/util/registry/i_config_registry.h>
#include <base/util/string_utils/i_string_utils.h>
#include <io/scene/scene/i_scene_journal_types.h>
#include <io/scene/mdl_elements/mdl_elements_detail.h>

namespace fs = std::filesystem;

namespace MI {

namespace BSDFM {

namespace {

const char* magic_header = "NVIDIA ARC MBSDF V1\n";
const char* magic_data = "MBSDF_DATA=\n";
const char* magic_reflection = "MBSDF_DATA_REFLECTION=\n";
const char* magic_transmission = "MBSDF_DATA_TRANSMISSION=\n";

// Returns a string representation of mi::base::Uuid
std::string hash_to_string( const mi::base::Uuid& hash)
{
    char buffer[35];
    snprintf( buffer, sizeof( buffer), "0x%08x%08x%08x%08x",
              hash.m_id1, hash.m_id2, hash.m_id3, hash.m_id4);
    return buffer;
}

// Dumps some data about an instance of #mi::neuraylib::IBsdf_isotropic_data.
std::string dump_data( const mi::neuraylib::IBsdf_isotropic_data* data)
{
    if( !data)
        return "none";

    std::ostringstream s;
    s << "type \"" << (data->get_type() == mi::neuraylib::BSDF_SCALAR ? "Scalar" : "Rgb");
    s << "\", res. theta " << data->get_resolution_theta();
    s << ", res. phi " << data->get_resolution_phi();
    return s.str();
}

} // namespace

Bsdf_measurement::Bsdf_measurement()
  : m_impl_tag( DB::Tag()),
    m_impl_hash{0,0,0,0},
    m_cached_is_valid( false)
{
}

mi::Sint32 Bsdf_measurement::reset_file(
    DB::Transaction* transaction, const std::string& original_filename)
{
    SYSTEM::Access_module<PATH::Path_module> m_path_module( false);
    const std::string resolved_filename
        = m_path_module->search( PATH::RESOURCE, original_filename);
#if 0
    LOG::mod_log->info( M_SCENE, LOG::Mod_log::C_IO,
        "Configured resource search path:");
    const std::vector<std::string>& search_path
        = m_path_module->get_search_path( PATH::RESOURCE);
    for( mi::Size i = 0; i < search_path.size(); ++i)
         LOG::mod_log->info( M_SCENE, LOG::Mod_log::C_IO,
             "  [%" FMT_MI_SIZE "]: %s", i, search_path[i].c_str());
    LOG::mod_log->info( M_SCENE, LOG::Mod_log::C_IO,
        "Request: %s", original_filename.c_str());
    LOG::mod_log->info( M_SCENE, LOG::Mod_log::C_IO,
        "Result: %s", resolved_filename.empty() ? "(failed)" : resolved_filename.c_str());
#endif

    if( resolved_filename.empty())
        return -4;

    mi::base::Handle<mi::neuraylib::IBsdf_isotropic_data> reflection;
    mi::base::Handle<mi::neuraylib::IBsdf_isotropic_data> transmission;
    mi::Sint32 result = import_from_file( resolved_filename, reflection, transmission);
    if( result != 0)
        return result;

    const mi::base::Uuid impl_hash{0,0,0,0};
    reset_shared( transaction, reflection.get(), transmission.get(), impl_hash);

    m_original_filename = original_filename;
    m_resolved_filename = resolved_filename;
    m_resolved_container_filename.clear();
    m_resolved_container_membername.clear();
    m_mdl_file_path.clear();

    std::ostringstream s;
    s << "Loading BSDF measurement \"" << m_resolved_filename.c_str()
      << "\", reflection: " << dump_data( reflection.get())
      << ", transmission: " << dump_data( transmission.get());
    LOG::mod_log->info( M_SCENE, LOG::Mod_log::C_IO, "%s", s.str().c_str());

    return 0;
}

mi::Sint32 Bsdf_measurement::reset_reader(
    DB::Transaction* transaction, mi::neuraylib::IReader* reader)
{
    mi::base::Handle<mi::neuraylib::IBsdf_isotropic_data> reflection;
    mi::base::Handle<mi::neuraylib::IBsdf_isotropic_data> transmission;
    mi::Sint32 result = import_from_reader( reader, reflection, transmission);
    if( result != 0)
        return result;

    const mi::base::Uuid impl_hash{0,0,0,0};
    reset_shared( transaction, reflection.get(), transmission.get(), impl_hash);

    m_original_filename.clear();
    m_resolved_filename.clear();
    m_resolved_container_filename.clear();
    m_resolved_container_membername.clear();
    m_mdl_file_path.clear();

    std::ostringstream s;
    s << "Loading memory-based BSDF measurement"
      << ", reflection: " << dump_data( reflection.get())
      << ", transmission: " << dump_data( transmission.get());
    LOG::mod_log->info( M_SCENE, LOG::Mod_log::C_IO, "%s", s.str().c_str());

    return 0;
}

mi::Sint32 Bsdf_measurement::reset_mdl(
    DB::Transaction* transaction,
    mi::neuraylib::IReader* reader,
    const std::string& filename,
    const std::string& container_filename,
    const std::string& container_membername,
    const std::string& mdl_file_path,
    const mi::base::Uuid& impl_hash)
{
    mi::base::Handle<mi::neuraylib::IBsdf_isotropic_data> reflection;
    mi::base::Handle<mi::neuraylib::IBsdf_isotropic_data> transmission;
    mi::Sint32 result = import_from_reader( reader, reflection, transmission);
    if( result != 0)
        return result;

    reset_shared( transaction, reflection.get(), transmission.get(), impl_hash);

    m_original_filename.clear();
    m_resolved_filename = filename;
    m_resolved_container_filename = container_filename;
    m_resolved_container_membername = container_membername;
    m_mdl_file_path = mdl_file_path;

    std::ostringstream s;
    s << "Loading BSDF measurement ";
    if( !m_resolved_filename.empty())
        s << '\"' << m_resolved_filename << '\"';
    else if( !m_resolved_container_filename.empty())
        s << '\"' << m_resolved_container_membername << "\" in \""
          << m_resolved_container_filename << '\"';
    else
        s << "from MDL file path \"" << m_mdl_file_path << '\"';
    s << ", reflection: " << dump_data( reflection.get());
    s << ", transmission: " << dump_data( transmission.get());
    LOG::mod_log->info( M_SCENE, LOG::Mod_log::C_IO, "%s", s.str().c_str());

    return 0;
}

void Bsdf_measurement::set_reflection(
    DB::Transaction* transaction, const mi::neuraylib::IBsdf_isotropic_data* reflection)
{
    m_original_filename.clear();
    m_resolved_filename.clear();
    m_resolved_container_filename.clear();
    m_resolved_container_membername.clear();
    m_mdl_file_path.clear();

    mi::base::Handle<const mi::neuraylib::IBsdf_isotropic_data> transmission(
        get_transmission<mi::neuraylib::IBsdf_isotropic_data>( transaction));
    const mi::base::Uuid impl_hash{0,0,0,0};
    reset_shared( transaction, reflection, transmission.get(), impl_hash);
}

const mi::base::IInterface* Bsdf_measurement::get_reflection( DB::Transaction* transaction) const
{
   if( !m_impl_tag)
        return nullptr;

    DB::Access<Bsdf_measurement_impl> impl( m_impl_tag, transaction);
    return impl->get_reflection();
}

void Bsdf_measurement::set_transmission(
    DB::Transaction* transaction, const mi::neuraylib::IBsdf_isotropic_data* transmission)
{
    m_original_filename.clear();
    m_resolved_filename.clear();
    m_resolved_container_filename.clear();
    m_resolved_container_membername.clear();
    m_mdl_file_path.clear();


    mi::base::Handle<const mi::neuraylib::IBsdf_isotropic_data> reflection(
        get_reflection<mi::neuraylib::IBsdf_isotropic_data>( transaction));
    const mi::base::Uuid impl_hash{0,0,0,0};
    reset_shared( transaction, reflection.get(), transmission, impl_hash);
}

const mi::base::IInterface* Bsdf_measurement::get_transmission( DB::Transaction* transaction) const
{
   if( !m_impl_tag)
        return nullptr;

    DB::Access<Bsdf_measurement_impl> impl( m_impl_tag, transaction);
    return impl->get_transmission();
}

const SERIAL::Serializable* Bsdf_measurement::serialize( SERIAL::Serializer* serializer) const
{
    Scene_element_base::serialize( serializer);

    SERIAL::write( serializer, serializer->is_remote() ? "" : m_original_filename);
    SERIAL::write( serializer, serializer->is_remote() ? "" : m_resolved_filename);
    SERIAL::write( serializer, serializer->is_remote() ? "" : m_resolved_container_filename);
    SERIAL::write( serializer, serializer->is_remote() ? "" : m_resolved_container_membername);
    SERIAL::write( serializer, serializer->is_remote() ? "" : m_mdl_file_path);
    SERIAL::write( serializer, HAL::Ospath::sep());

    SERIAL::write( serializer, m_impl_tag);
    SERIAL::write( serializer, m_impl_hash);

    SERIAL::write( serializer, m_cached_is_valid);

    return this + 1;
}

SERIAL::Serializable* Bsdf_measurement::deserialize( SERIAL::Deserializer* deserializer)
{
    Scene_element_base::deserialize( deserializer);

    SERIAL::read( deserializer, &m_original_filename);
    SERIAL::read( deserializer, &m_resolved_filename);
    SERIAL::read( deserializer, &m_resolved_container_filename);
    SERIAL::read( deserializer, &m_resolved_container_membername);
    SERIAL::read( deserializer, &m_mdl_file_path);
    std::string serializer_sep;
    SERIAL::read( deserializer, &serializer_sep);

    SERIAL::read( deserializer, &m_impl_tag);
    SERIAL::read( deserializer, &m_impl_hash);

    SERIAL::read( deserializer, &m_cached_is_valid);

    // Adjust m_original_filename and m_resolved_filename for this host.
    std::error_code ec;
    if( !m_original_filename.empty()) {

        if( serializer_sep != HAL::Ospath::sep()) {
            m_original_filename
                = HAL::Ospath::convert_to_platform_specific_path( m_original_filename);
            m_resolved_filename
                = HAL::Ospath::convert_to_platform_specific_path( m_resolved_filename);
        }

        // Re-resolve filename if it is not meaningful for this host. If unsuccessful, clear value
        // (no error since we no longer require all resources to be present on all nodes).
        if( !fs::is_regular_file( fs::u8path( m_resolved_filename), ec)) {
            SYSTEM::Access_module<PATH::Path_module> m_path_module( false);
            m_resolved_filename = m_path_module->search( PATH::MDL, m_original_filename);
            if( m_resolved_filename.empty())
                m_resolved_filename = m_path_module->search(
                    PATH::RESOURCE, m_original_filename);
        }

    }

    // Adjust m_resolved_container_filename and m_resolved_container_membername for this host.
    if( !m_resolved_container_filename.empty()) {

        m_resolved_container_filename
            = HAL::Ospath::convert_to_platform_specific_path( m_resolved_container_filename);
        m_resolved_container_membername
            = HAL::Ospath::convert_to_platform_specific_path( m_resolved_container_membername);
        if( !fs::is_regular_file( fs::u8path( m_resolved_container_filename), ec)) {
            m_resolved_container_filename.clear();
            m_resolved_container_membername.clear();
        }

    } else
        ASSERT( M_SCENE, m_resolved_container_membername.empty());

    return this + 1;
}

void Bsdf_measurement::dump() const
{
    std::ostringstream s;

    s << "Original filename: " << m_original_filename << std::endl;
    s << "Resolved filename: " << m_resolved_filename << std::endl;
    s << "Resolved container filename: " << m_resolved_container_filename << std::endl;
    s << "Resolved container membername: " << m_resolved_container_membername << std::endl;
    s << "MDL file path: " << m_mdl_file_path << std::endl;

    s << "Implementation tag: " << m_impl_tag.get_uint() << std::endl;
    s << "Implementation hash: " << hash_to_string( m_impl_hash) << std::endl;

    s << "Is valid (cached): " << (m_cached_is_valid  ? "true" : "false") << std::endl;

    LOG::mod_log->info( M_SCENE, LOG::Mod_log::C_DATABASE, "%s", s.str().c_str());
}

size_t Bsdf_measurement::get_size() const
{
    return sizeof( *this)
        + dynamic_memory_consumption( m_original_filename)
        + dynamic_memory_consumption( m_resolved_filename)
        + dynamic_memory_consumption( m_resolved_container_filename)
        + dynamic_memory_consumption( m_resolved_container_membername)
        + dynamic_memory_consumption( m_mdl_file_path);
}

DB::Journal_type Bsdf_measurement::get_journal_flags() const
{
    return DB::Journal_type(
        SCENE::JOURNAL_CHANGE_FIELD.get_type() |
        SCENE::JOURNAL_CHANGE_SHADER_ATTRIBUTE.get_type());
}

void Bsdf_measurement::get_scene_element_references( DB::Tag_set* result) const
{
    if( m_impl_tag)
        result->insert( m_impl_tag);
}

void Bsdf_measurement::reset_shared(
    DB::Transaction* transaction,
    const mi::neuraylib::IBsdf_isotropic_data* reflection,
    const mi::neuraylib::IBsdf_isotropic_data* transmission,
    const mi::base::Uuid& impl_hash)
{
    // if impl_hash is valid, check whether implementation class exists already
    std::string impl_name;
    if( impl_hash != mi::base::Uuid{0,0,0,0}) {
        impl_name = "MI_default_bsdf_measurement_impl_" + hash_to_string( impl_hash);
        m_impl_tag = transaction->name_to_tag( impl_name.c_str());
        if( m_impl_tag) {
            m_impl_hash = impl_hash;
            DB::Access<Bsdf_measurement_impl> impl( m_impl_tag, transaction);
            setup_cached_values( impl.get_ptr());
            return;
        }
    }

    auto* impl = new Bsdf_measurement_impl( reflection, transmission);

    setup_cached_values( impl);

    // We do not know the scope in which the instance of the proxy class ends up. Therefore, we have
    // to pick the global scope for the instance of the implementation class. Make sure to use
    // a DB name for the implementation class exactly for valid hashes.
    ASSERT( M_SCENE, impl_name.empty() ^ (impl_hash != mi::base::Uuid{0,0,0,0}));
    m_impl_tag = transaction->store_for_reference_counting(
        impl, !impl_name.empty() ? impl_name.c_str() : nullptr, /*privacy_level*/ 0);
    m_impl_hash = impl_hash;
}

void Bsdf_measurement::setup_cached_values( const Bsdf_measurement_impl* impl)
{
    m_cached_is_valid = impl->is_valid();
}

Bsdf_measurement_impl::Bsdf_measurement_impl() = default;

Bsdf_measurement_impl::Bsdf_measurement_impl(
    const mi::neuraylib::IBsdf_isotropic_data* reflection,
    const mi::neuraylib::IBsdf_isotropic_data* transmission)
{
    m_reflection = make_handle_dup( reflection);
    m_transmission = make_handle_dup( transmission);
}

Bsdf_measurement_impl::Bsdf_measurement_impl( const Bsdf_measurement_impl& other)
  : SCENE::Scene_element<Bsdf_measurement_impl, ID_BSDF_MEASUREMENT_IMPL>( other)
{
    m_reflection = other.m_reflection;
    m_transmission = other.m_transmission;
}

Bsdf_measurement_impl::~Bsdf_measurement_impl() = default;

const mi::base::IInterface* Bsdf_measurement_impl::get_reflection() const
{
    if( !m_reflection)
        return nullptr;

    m_reflection->retain();
    return m_reflection.get();
}

const mi::base::IInterface* Bsdf_measurement_impl::get_transmission() const
{
    if( !m_transmission)
        return nullptr;

    m_transmission->retain();
    return m_transmission.get();
}


const SERIAL::Serializable* Bsdf_measurement_impl::serialize( SERIAL::Serializer* serializer) const
{
    Scene_element_base::serialize( serializer);

    serialize_bsdf_data( serializer, m_reflection.get());
    serialize_bsdf_data( serializer, m_transmission.get());

    return this + 1;
}

SERIAL::Serializable* Bsdf_measurement_impl::deserialize( SERIAL::Deserializer* deserializer)
{
    Scene_element_base::deserialize( deserializer);

    m_reflection = deserialize_bsdf_data( deserializer);
    m_transmission = deserialize_bsdf_data( deserializer); //-V656 PVS

    return this + 1;
}

void Bsdf_measurement_impl::dump() const
{
    std::ostringstream s;

    s << "Reflection: " << dump_data( m_reflection.get()) << std::endl;
    s << "Transmission: " << dump_data( m_transmission.get()) << std::endl;

    LOG::mod_log->info( M_SCENE, LOG::Mod_log::C_DATABASE, "%s", s.str().c_str());
}

size_t Bsdf_measurement_impl::get_size() const
{
    size_t result = sizeof( *this);

    // For memory-based BSDF measurements it is unclear whether the actual data should be counted
    // here or not (data exclusively owned by us or not).

    result += 2 * sizeof( mi::neuraylib::Bsdf_isotropic_data);
    result += 2 * sizeof( mi::neuraylib::Bsdf_buffer);

    mi::Uint32 resolution_theta;
    mi::Uint32 resolution_phi;
    mi::neuraylib::Bsdf_type type;
    mi::Size size;

    resolution_theta = m_reflection ? m_reflection->get_resolution_theta() : 0;
    resolution_phi   = m_reflection ? m_reflection->get_resolution_phi()   : 0;
    type             = m_reflection ? m_reflection->get_type() : mi::neuraylib::BSDF_SCALAR;
    size             = static_cast<mi::Size>( resolution_theta) * resolution_theta * resolution_phi;
    if( type == mi::neuraylib::BSDF_RGB)
        size *= 3;
    result += size * sizeof( mi::Float32);

    resolution_theta = m_transmission ? m_transmission->get_resolution_theta() : 0;
    resolution_phi   = m_transmission ? m_transmission->get_resolution_phi()   : 0;
    type             = m_transmission ? m_transmission->get_type() : mi::neuraylib::BSDF_SCALAR;
    size             = static_cast<mi::Size>( resolution_theta) * resolution_theta * resolution_phi;
    if( type == mi::neuraylib::BSDF_RGB)
        size *= 3;
    result += size * sizeof( mi::Float32);

    return result;
}

DB::Journal_type Bsdf_measurement_impl::get_journal_flags() const
{
    return DB::Journal_type(
        SCENE::JOURNAL_CHANGE_FIELD.get_type() |
        SCENE::JOURNAL_CHANGE_SHADER_ATTRIBUTE.get_type());
}

void Bsdf_measurement_impl::serialize_bsdf_data(
    SERIAL::Serializer* serializer, const mi::neuraylib::IBsdf_isotropic_data* bsdf_data)
{
    bool exists = bsdf_data != nullptr;
    SERIAL::write( serializer, exists);
    if( !exists)
        return;

    const mi::Uint32 resolution_theta   = bsdf_data->get_resolution_theta();
    const mi::Uint32 resolution_phi     = bsdf_data->get_resolution_phi();
    const mi::neuraylib::Bsdf_type type = bsdf_data->get_type();

    SERIAL::write( serializer, resolution_theta);
    SERIAL::write( serializer, resolution_phi);
    SERIAL::write( serializer, static_cast<mi::Uint32>( type));

    mi::Size size = static_cast<mi::Size>( resolution_theta) * resolution_theta * resolution_phi;
    if( type == mi::neuraylib::BSDF_RGB)
        size *= 3;
    mi::base::Handle<const mi::neuraylib::IBsdf_buffer> buffer( bsdf_data->get_bsdf_buffer());
    const mi::Float32* data = buffer->get_data();
    serializer->write( reinterpret_cast<const char*>( data), size * sizeof( mi::Float32));
}

mi::neuraylib::IBsdf_isotropic_data* Bsdf_measurement_impl::deserialize_bsdf_data(
    SERIAL::Deserializer* deserializer)
{
    bool exists;
    SERIAL::read( deserializer, &exists);
    if( !exists)
        return nullptr;

    mi::Uint32 resolution_theta;
    mi::Uint32 resolution_phi;
    mi::Uint32 type;

    SERIAL::read( deserializer, &resolution_theta);
    SERIAL::read( deserializer, &resolution_phi);
    SERIAL::read( deserializer, &type);

    auto* bsdf_data = new mi::neuraylib::Bsdf_isotropic_data(
        resolution_theta, resolution_phi, static_cast<mi::neuraylib::Bsdf_type>( type));
    mi::base::Handle<mi::neuraylib::Bsdf_buffer> bsdf_buffer( bsdf_data->get_bsdf_buffer());
    mi::Float32* data = bsdf_buffer->get_data();

    mi::Size size = static_cast<mi::Size>( resolution_theta) * resolution_theta * resolution_phi;
    if( type == mi::neuraylib::BSDF_RGB)
        size *= 3;
    deserializer->read( reinterpret_cast<char*>( data), size * sizeof( mi::Float32));

    return bsdf_data;
}
namespace {

bool read( mi::neuraylib::IReader* reader, char* buffer, Sint64 size)
{
    return reader->read( buffer, size) == size;
}

mi::neuraylib::IBsdf_isotropic_data* import_data_from_reader( mi::neuraylib::IReader* reader)
{
    // type
    mi::Uint32 type_uint32;
    if( !read( reader, reinterpret_cast<char*>( &type_uint32), sizeof( mi::Uint32)))
        return nullptr;
    if( type_uint32 > 1)
        return nullptr;
    const mi::neuraylib::Bsdf_type type
        = type_uint32 == 0 ? mi::neuraylib::BSDF_SCALAR : mi::neuraylib::BSDF_RGB;

    // resolution_theta
    mi::Uint32 resolution_theta;
    if( !read( reader, reinterpret_cast<char*>( &resolution_theta), sizeof( mi::Uint32)))
        return nullptr;
    if( resolution_theta == 0)
        return nullptr;

    // resolution_phi
    mi::Uint32 resolution_phi;
    if( !read( reader, reinterpret_cast<char*>( &resolution_phi), sizeof( mi::Uint32)))
        return nullptr;
    if( resolution_phi == 0)
        return nullptr;

    // data
    mi::base::Handle<mi::neuraylib::Bsdf_isotropic_data> bsdf_data(
        new mi::neuraylib::Bsdf_isotropic_data( resolution_theta, resolution_phi, type));
    mi::base::Handle<mi::neuraylib::Bsdf_buffer> bsdf_buffer( bsdf_data->get_bsdf_buffer());
    mi::Float32* data = bsdf_buffer->get_data();
    mi::Size size = static_cast<mi::Size>( resolution_theta) * resolution_theta * resolution_phi;
    if( type == mi::neuraylib::BSDF_RGB)
        size *= 3;
    if( !read( reader, reinterpret_cast<char*>( data), size * sizeof( mi::Float32)))
        return nullptr;

    return bsdf_data.extract();
}

bool import_measurement_from_reader(
    mi::neuraylib::IReader* reader,
    mi::base::Handle<mi::neuraylib::IBsdf_isotropic_data>& reflection,
    mi::base::Handle<mi::neuraylib::IBsdf_isotropic_data>& transmission)
{
    char buffer[1024];
    std::string buffer_str;

    reflection.reset();
    transmission.reset();

    // header
    if( !reader->readline( buffer, sizeof( buffer)))
        return false;
    buffer_str = buffer;
    if( buffer_str != magic_header)
        return false;

    // skip metadata
    do {
        if( !reader->readline( buffer, sizeof( buffer)))
            return false;
        if( reader->eof())
            return true;
        buffer_str = buffer;
    } while(    buffer_str != magic_data
             && buffer_str != magic_reflection
             && buffer_str != magic_transmission);

    // reflection data
    if( buffer_str == magic_data || buffer_str == magic_reflection) {
        reflection = import_data_from_reader( reader);
        if( !reflection)
            return false;
        if( !reader->readline( buffer, sizeof( buffer)))
            return false;
        if( reader->eof())
            return true;
        buffer_str = buffer;
    }

    // transmission data
    if( buffer_str == magic_transmission) {
        transmission = import_data_from_reader( reader);
        if( !transmission)
            return false;
        if( !reader->readline( buffer, sizeof( buffer)))
            return false;
        if( reader->eof())
            return true;
    }

    return false;
}

} // namespace

mi::Sint32 import_from_file(
    const std::string& filename,
    mi::base::Handle<mi::neuraylib::IBsdf_isotropic_data>& reflection,
    mi::base::Handle<mi::neuraylib::IBsdf_isotropic_data>& transmission)
{
    std::string extension = STRING::to_lower( HAL::Ospath::get_ext( filename));
    if( extension != ".mbsdf")
        return -3;

    DISK::File_reader_impl reader;
    if( !reader.open( filename.c_str()))
        return -5;

    return import_measurement_from_reader( &reader, reflection, transmission) ? 0 : -7;
}

mi::Sint32 import_from_reader(
    mi::neuraylib::IReader* reader,
    mi::base::Handle<mi::neuraylib::IBsdf_isotropic_data>& reflection,
    mi::base::Handle<mi::neuraylib::IBsdf_isotropic_data>& transmission)
{
    return import_measurement_from_reader( reader, reflection, transmission) ? 0 : -7;
}

namespace {

bool export_to_file(
    mi::neuraylib::IWriter* writer, const mi::neuraylib::IBsdf_isotropic_data* bsdf_data)
{
    const mi::neuraylib::Bsdf_type type = bsdf_data->get_type();
    const mi::Uint32 resolution_theta   = bsdf_data->get_resolution_theta();
    const mi::Uint32 resolution_phi     = bsdf_data->get_resolution_phi();
    const mi::Uint32 type_uint32        = type == mi::neuraylib::BSDF_SCALAR ? 0 : 1;

    if( !writer->write( reinterpret_cast<const char*>( &type_uint32), 4))
        return false;
    if( !writer->write( reinterpret_cast<const char*>( &resolution_theta), 4))
        return false;
    if( !writer->write( reinterpret_cast<const char*>( &resolution_phi), 4))
        return false;

    mi::Size size = static_cast<mi::Size>( resolution_theta) * resolution_theta * resolution_phi;
    if( type == mi::neuraylib::BSDF_RGB)
        size *= 3;
    mi::base::Handle<const mi::neuraylib::IBsdf_buffer> buffer( bsdf_data->get_bsdf_buffer());
    const mi::Float32* data = buffer->get_data();
    if( !writer->write( reinterpret_cast<const char*>( data), size * sizeof( mi::Float32)))
        return false;

    return true;
}

} // anonymous

mi::neuraylib::IBuffer* create_buffer_from_bsdf_measurement(
    const mi::neuraylib::IBsdf_isotropic_data* reflection,
    const mi::neuraylib::IBsdf_isotropic_data* transmission)
{
    DISK::Memory_writer_impl writer;

    bool success = true;
    success &= writer.writeline( magic_header);

    if( reflection) {
        success &= writer.writeline( magic_reflection);
        success &= export_to_file( &writer, reflection);
    }

    if( transmission) {
        success &= writer.writeline( magic_transmission);
        success &= export_to_file( &writer, transmission);
    }

    if( !success)
        return nullptr;

    mi::neuraylib::IBuffer* buffer = writer.get_buffer();
    return buffer;
}

bool export_to_file(
    const mi::neuraylib::IBsdf_isotropic_data* reflection,
    const mi::neuraylib::IBsdf_isotropic_data* transmission,
    const std::string& filename)
{
    DISK::File_writer_impl writer;
    if( !writer.open(filename.c_str()))
        return false;

    bool success = true;
    success &= writer.writeline( magic_header);

    if( reflection) {
        success &= writer.writeline( magic_reflection);
        success &= export_to_file( &writer, reflection);
    }

    if( transmission) {
        success &= writer.writeline( magic_transmission);
        success &= export_to_file( &writer, transmission);
    }

    writer.close();
    return success;
}

DB::Tag load_mdl_bsdf_measurement(
    DB::Transaction* transaction,
    mi::neuraylib::IReader* reader,
    const std::string& filename,
    const std::string& container_filename,
    const std::string& container_membername,
    const std::string& mdl_file_path,
    const mi::base::Uuid& impl_hash,
    bool shared_proxy,
    mi::Sint32& result)
{
    if( !reader) {
        result = -1;
        return {};
    }

    std::string identifier;
    if( !filename.empty()) {
        identifier = filename;
    } else if( !container_filename.empty()) {
        identifier = container_filename + "_" + container_membername;
    } else if( !mdl_file_path.empty()) {
        identifier = "mfp_" + mdl_file_path;
    } else {
        identifier = "without_name";
        // Never share the proxy for memory-based resources.
        shared_proxy = false;
    }

    std::string db_name = shared_proxy ? "MI_default_" : "";
    db_name += "bsdf_measurement_" + identifier;
    if( !shared_proxy)
        db_name = MDL::DETAIL::generate_unique_db_name( transaction, db_name.c_str());

    DB::Tag tag = transaction->name_to_tag( db_name.c_str());
    if( tag) {
        result = 0;
        return tag;
    }

    auto bsdfm = std::make_unique<Bsdf_measurement>();
    result = bsdfm->reset_mdl( transaction,
        reader, filename, container_filename, container_membername, mdl_file_path, impl_hash);
    ASSERT( M_BSDF_MEASUREMENT, result == 0 || result == -7);
    if( result != 0)
        return {};

    tag = transaction->store_for_reference_counting(
        bsdfm.release(), db_name.c_str(), transaction->get_scope()->get_level());
    result = 0;
    return tag;
}

} // namespace BSDFM

} // namespace MI
