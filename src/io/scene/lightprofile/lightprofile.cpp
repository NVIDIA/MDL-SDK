/***************************************************************************************************
 * Copyright (c) 2007-2024, NVIDIA CORPORATION. All rights reserved.
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

#include "i_lightprofile.h"

#include <mi/math/function.h>
#include <base/system/main/access_module.h>
#include <base/hal/disk/disk_file_reader_writer_impl.h>
#include <base/hal/disk/disk_memory_reader_writer_impl.h>
#include <base/hal/hal/i_hal_ospath.h>
#include <base/lib/log/i_log_assert.h>
#include <base/lib/log/i_log_logger.h>
#include <base/lib/mem/i_mem_consumption.h>
#include <base/lib/path/i_path.h>
#include <base/data/db/i_db_access.h>
#include <base/data/db/i_db_transaction.h>
#include <base/data/serial/i_serializer.h>
#include <base/util/string_utils/i_string_utils.h>
#include <io/scene/scene/i_scene_journal_types.h>
#include <io/scene/mdl_elements/mdl_elements_detail.h>

#include <sstream>


namespace MI {

namespace LIGHTPROFILE {

namespace {

// Returns a string representation of mi::base::Uuid
std::string hash_to_string( const mi::base::Uuid& hash)
{
    char buffer[35];
    snprintf( buffer, sizeof( buffer), "0x%08x%08x%08x%08x",
              hash.m_id1, hash.m_id2, hash.m_id3, hash.m_id4);
    return buffer;
}

std::string get_log_identifier(
    const std::string& filename,
    const std::string& container_filename,
    const std::string& container_membername,
    const std::string& mdl_file_path)
{
    std::string result;
    if( !filename.empty())
        return std::string( "light profile \"") + filename + '\"';
    if( !container_filename.empty())
        return std::string( "light profile \"") + container_membername + "\" in \""
            + container_filename + '\"';
    if( !mdl_file_path.empty())
        return std::string( "light profile from MDL file path \"") + mdl_file_path + '\"';
    return "memory- or reader-based light profile (no name available)";
}

} // namespace

// implemented in lightprofile_ies_parser.cpp
bool setup_lightprofile(
    mi::neuraylib::IReader* reader,
    const std::string& log_identifier,
    mi::neuraylib::Lightprofile_degree degree,
    mi::Uint32 flags,
    mi::Uint32& resolution_phi,
    mi::Uint32& resolution_theta,
    mi::Float32& start_phi,
    mi::Float32& start_theta,
    mi::Float32& delta_phi,
    mi::Float32& delta_theta,
    std::vector<mi::Float32>& data);

Lightprofile::Lightprofile()
  : m_impl_tag( DB::Tag()),
    m_impl_hash{0,0,0,0},
    m_cached_resolution_phi( 0),
    m_cached_resolution_theta( 0),
    m_cached_degree( mi::neuraylib::LIGHTPROFILE_HERMITE_BASE_1),
    m_cached_flags( mi::neuraylib::LIGHTPROFILE_COUNTER_CLOCKWISE),
    m_cached_start_phi( 0.0f),
    m_cached_start_theta( 0.0f),
    m_cached_delta_phi( 0.0f),
    m_cached_delta_theta( 0.0f),
    m_cached_candela_multiplier( 0.0f),
    m_cached_power( 0.0f),
    m_cached_is_valid( false)
{
}

mi::Sint32 Lightprofile::reset_file(
    DB::Transaction* transaction,
    const std::string& original_filename,
    mi::Uint32 resolution_phi,
    mi::Uint32 resolution_theta,
    mi::neuraylib::Lightprofile_degree degree,
    mi::Uint32 flags)
{
    std::string extension = STRING::to_lower( HAL::Ospath::get_ext( original_filename));
    if( extension != ".ies")
        return -3;

    SYSTEM::Access_module<PATH::Path_module> m_path_module( false);
    const std::string resolved_filename
        = m_path_module->search( PATH::RESOURCE, original_filename);
#if 0
    LOG::mod_log->info( M_SCENE, LOG::Mod_log::C_IO,
        "Configured resource search paths:");
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

    // create reader for resolved_filename
    DISK::File_reader_impl reader;
    if( !reader.open( resolved_filename.c_str()))
        return -5;

    const std::string log_identifier = get_log_identifier( resolved_filename, {}, {}, {});
    const mi::base::Uuid impl_hash{0,0,0,0};
    const mi::Sint32 result = reset_shared( transaction,
        &reader, log_identifier, impl_hash, resolution_phi, resolution_theta, degree, flags);
    if( result != 0)
        return result;

    m_original_filename = original_filename;
    m_resolved_filename = resolved_filename;
    m_resolved_container_filename.clear();
    m_resolved_container_membername.clear();
    m_mdl_file_path.clear();

    return 0;
}

mi::Sint32 Lightprofile::reset_reader(
    DB::Transaction* transaction,
    mi::neuraylib::IReader* reader,
    mi::Uint32 resolution_phi,
    mi::Uint32 resolution_theta,
    mi::neuraylib::Lightprofile_degree degree,
    mi::Uint32 flags)
{
    const std::string log_identifier = get_log_identifier( {}, {}, {}, {});
    const mi::base::Uuid impl_hash{0,0,0,0};
    const mi::Sint32 result = reset_shared( transaction,
        reader, log_identifier, impl_hash, resolution_phi, resolution_theta, degree, flags);
    if( result != 0)
        return result;

    m_original_filename.clear();
    m_resolved_filename.clear();
    m_resolved_container_filename.clear();
    m_resolved_container_membername.clear();
    m_mdl_file_path.clear();

    return 0;
}

mi::Sint32 Lightprofile::reset_mdl(
    DB::Transaction* transaction,
    mi::neuraylib::IReader* reader,
    const std::string& filename,
    const std::string& container_filename,
    const std::string& container_membername,
    const std::string& mdl_file_path,
    const mi::base::Uuid& impl_hash,
    mi::Uint32 resolution_phi,
    mi::Uint32 resolution_theta,
    mi::neuraylib::Lightprofile_degree degree,
    mi::Uint32 flags)
{
    std::string log_identifier = get_log_identifier(
        filename, container_filename, container_membername, mdl_file_path);
    const mi::Sint32 result = reset_shared( transaction,
        reader, log_identifier, impl_hash, resolution_phi, resolution_theta, degree, flags);
    if( result != 0)
        return result;

    m_original_filename.clear();
    m_resolved_filename = filename;
    m_resolved_container_filename = container_filename;
    m_resolved_container_membername = container_membername;
    m_mdl_file_path = mdl_file_path;

    return 0;
}

mi::Sint32 Lightprofile::reset_shared(
    DB::Transaction* transaction,
    mi::neuraylib::IReader* reader,
    const std::string& log_identifier,
    const mi::base::Uuid& impl_hash,
    mi::Uint32 resolution_phi,
    mi::Uint32 resolution_theta,
    mi::neuraylib::Lightprofile_degree degree,
    mi::Uint32 flags)
{
    // reject invalid flags
    const bool cw_set  = (flags & mi::neuraylib::LIGHTPROFILE_CLOCKWISE        ) != 0;
    const bool ccw_set = (flags & mi::neuraylib::LIGHTPROFILE_COUNTER_CLOCKWISE) != 0;
    if( flags >= 16 || (cw_set && ccw_set) || (!cw_set && !ccw_set))
        return -13;

    // reject invalid degree
    if(    degree != mi::neuraylib::LIGHTPROFILE_HERMITE_BASE_1
        && degree != mi::neuraylib::LIGHTPROFILE_HERMITE_BASE_3)
        return -14;

    // reject invalid resolutions
    if( resolution_phi == 1 || resolution_theta == 1)
        return -15;

    // if impl_hash is valid, check whether implementation class exists already
    std::string impl_name;
    if( impl_hash != mi::base::Uuid{0,0,0,0}) {
        impl_name = "MI_default_lightprofile_impl_" + hash_to_string( impl_hash);
        m_impl_tag = transaction->name_to_tag( impl_name.c_str());
        if( m_impl_tag) {
            m_impl_hash = impl_hash;
            DB::Access<Lightprofile_impl> impl( m_impl_tag, transaction);
            setup_cached_values( impl.get_ptr());
            return 0;
        }
    }

    mi::Float32 start_phi;
    mi::Float32 start_theta;
    mi::Float32 delta_phi;
    mi::Float32 delta_theta;
    std::vector<mi::Float32> data;

    // parse and interpolate data
    const bool success = setup_lightprofile(
        reader, log_identifier, degree, flags, resolution_phi, resolution_theta,
        start_phi, start_theta, delta_phi, delta_theta, data);

    // handle file format errors
    if( !success)
        return -7;

    Lightprofile_impl* impl = new Lightprofile_impl(
        resolution_phi, resolution_theta, degree, flags,
        start_phi, start_theta, delta_phi, delta_theta, data);

    setup_cached_values( impl);

    // We do not know the scope in which the instance of the proxy class ends up. Therefore, we have
    // to pick the global scope for the instance of the implementation class. Make sure to use
    // a DB name for the implementation class exactly for valid hashes.
    ASSERT( M_SCENE, impl_name.empty() ^ (impl_hash != mi::base::Uuid{0,0,0,0}));
    m_impl_tag = transaction->store_for_reference_counting(
        impl, !impl_name.empty() ? impl_name.c_str() : nullptr, /*privacy_level*/ 0);
    m_impl_hash = impl_hash;

    return 0;
}

mi::Float32 Lightprofile::get_phi( mi::Uint32 index) const
{
    if( index >= m_cached_resolution_phi)
        return 0.0;

    return m_cached_start_phi + index * m_cached_delta_phi;
}

mi::Float32 Lightprofile::get_theta( mi::Uint32 index) const
{
    if( index >= m_cached_resolution_theta)
        return 0.0f;

    return m_cached_start_theta + index * m_cached_delta_theta;
}

mi::Float32 Lightprofile::get_data(
    DB::Transaction* transaction, mi::Uint32 index_phi, mi::Uint32 index_theta) const
{
    if( index_phi >= m_cached_resolution_phi || index_theta >= m_cached_resolution_theta)
        return 0.0f;

    if( !m_impl_tag)
        return 0.0f;

    DB::Access<Lightprofile_impl> impl( m_impl_tag, transaction);
    return impl->get_data( index_phi, index_theta);
}

const mi::Float32* Lightprofile::get_data( DB::Transaction* transaction ) const
{
    if( !m_impl_tag)
        return nullptr;

    DB::Access<Lightprofile_impl> impl( m_impl_tag, transaction);
    return impl->get_data();
}

mi::Float32 Lightprofile::sample(
    DB::Transaction* transaction, mi::Float32 phi, mi::Float32 theta, bool candela) const
{
    if( !m_impl_tag)
        return 0.0f;

    DB::Access<Lightprofile_impl> impl( m_impl_tag, transaction);
    return impl->sample( phi, theta, candela);
}

const SERIAL::Serializable* Lightprofile::serialize( SERIAL::Serializer* serializer) const
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

    SERIAL::write( serializer, m_cached_resolution_phi);
    SERIAL::write( serializer, m_cached_resolution_theta);
    SERIAL::write( serializer, static_cast<mi::Uint32>( m_cached_degree));
    SERIAL::write( serializer, m_cached_flags);
    SERIAL::write( serializer, m_cached_start_phi);
    SERIAL::write( serializer, m_cached_start_theta);
    SERIAL::write( serializer, m_cached_delta_phi);
    SERIAL::write( serializer, m_cached_delta_theta);
    SERIAL::write( serializer, m_cached_candela_multiplier);
    SERIAL::write( serializer, m_cached_power);
    SERIAL::write( serializer, m_cached_is_valid);

    return this + 1;
}

SERIAL::Serializable* Lightprofile::deserialize( SERIAL::Deserializer* deserializer)
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

    SERIAL::read( deserializer, &m_cached_resolution_phi);
    SERIAL::read( deserializer, &m_cached_resolution_theta);
    mi::Uint32 degree;
    SERIAL::read( deserializer, &degree);
    m_cached_degree = static_cast<mi::neuraylib::Lightprofile_degree>( degree);
    SERIAL::read( deserializer, &m_cached_flags);
    SERIAL::read( deserializer, &m_cached_start_phi);
    SERIAL::read( deserializer, &m_cached_start_theta);
    SERIAL::read( deserializer, &m_cached_delta_phi);
    SERIAL::read( deserializer, &m_cached_delta_theta);
    SERIAL::read( deserializer, &m_cached_candela_multiplier);
    SERIAL::read( deserializer, &m_cached_power);
    SERIAL::read( deserializer, &m_cached_is_valid);

    // Adjust m_original_filename and m_resolved_filename for this host.
    if( !m_original_filename.empty()) {

       if( serializer_sep != HAL::Ospath::sep()) {
            m_original_filename
                = HAL::Ospath::convert_to_platform_specific_path( m_original_filename);
            m_resolved_filename
                = HAL::Ospath::convert_to_platform_specific_path( m_resolved_filename);
        }

        // Re-resolve filename if it is not meaningful for this host. If unsuccessful, clear value
        // (no error since we no longer require all resources to be present on all nodes).
        if( !DISK::is_file( m_resolved_filename.c_str())) {
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
        if( !DISK::is_file( m_resolved_container_filename.c_str())) {
            m_resolved_container_filename.clear();
            m_resolved_container_membername.clear();
        }

    } else
        ASSERT( M_SCENE, m_resolved_container_membername.empty());

    return this + 1;
}

void Lightprofile::dump() const
{
    std::ostringstream s;

    s << "Original filename: " << m_original_filename << std::endl;
    s << "Resolved filename: " << m_resolved_filename << std::endl;
    s << "Resolved container filename: " << m_resolved_container_filename << std::endl;
    s << "Resolved container membername: " << m_resolved_container_membername << std::endl;
    s << "MDL file path: " << m_mdl_file_path << std::endl;

    s << "Implementation tag: " << m_impl_tag.get_uint() << std::endl;
    s << "Implementation hash: " << hash_to_string( m_impl_hash) << std::endl;

    s << "Phi resolution (cached): " << m_cached_resolution_phi
      << ", start (cached): " << m_cached_start_phi
      << ", delta (cached): " << m_cached_delta_phi << std::endl;
    s << "Theta resolution (cached): " << m_cached_resolution_theta
      << ", start (cached): " << m_cached_start_theta
      << ", delta (cached): " << m_cached_delta_theta << std::endl;

    s << "Degree (cached): " << m_cached_degree << std::endl;
    s << "Flags (cached): " << m_cached_flags << std::endl;
    s << "Candela multiplier (cached): " << m_cached_candela_multiplier << std::endl;
    s << "Power (cached): " << m_cached_power << std::endl;
    s << "Is valid (cached): " << (m_cached_is_valid  ? "true" : "false") << std::endl;

    LOG::mod_log->info( M_SCENE, LOG::Mod_log::C_DATABASE, "%s", s.str().c_str());
}

size_t Lightprofile::get_size() const
{
    return sizeof( *this)
        + dynamic_memory_consumption( m_original_filename)
        + dynamic_memory_consumption( m_resolved_filename)
        + dynamic_memory_consumption( m_resolved_container_filename)
        + dynamic_memory_consumption( m_resolved_container_membername)
        + dynamic_memory_consumption( m_mdl_file_path);
}

DB::Journal_type Lightprofile::get_journal_flags() const
{
    return DB::Journal_type(
        SCENE::JOURNAL_CHANGE_FIELD.get_type() |
        SCENE::JOURNAL_CHANGE_SHADER_ATTRIBUTE.get_type());
}

void Lightprofile::get_scene_element_references( DB::Tag_set* result) const
{
    if( m_impl_tag)
        result->insert( m_impl_tag);
}

void Lightprofile::setup_cached_values( const Lightprofile_impl* impl)
{
    m_cached_degree             = impl->get_degree();
    m_cached_flags              = impl->get_flags();
    m_cached_resolution_phi     = impl->get_resolution_phi();
    m_cached_resolution_theta   = impl->get_resolution_theta();
    m_cached_start_phi          = impl->get_start_phi();
    m_cached_start_theta        = impl->get_start_theta();
    m_cached_delta_phi          = impl->get_delta_phi();
    m_cached_delta_theta        = impl->get_delta_theta();
    m_cached_candela_multiplier = impl->get_candela_multiplier();
    m_cached_power              = impl->get_power();
    m_cached_is_valid           = impl->is_valid();
}

Lightprofile_impl::Lightprofile_impl()
  : m_resolution_phi( 0),
    m_resolution_theta( 0),
    m_degree( mi::neuraylib::LIGHTPROFILE_HERMITE_BASE_1),
    m_flags( mi::neuraylib::LIGHTPROFILE_COUNTER_CLOCKWISE),
    m_start_phi( 0.0f),
    m_start_theta( 0.0f),
    m_delta_phi( 0.0f),
    m_delta_theta( 0.0f),
    m_candela_multiplier( 0.0f),
    m_power( 0.0f)
{
}

Lightprofile_impl::Lightprofile_impl(
    mi::Uint32 resolution_phi,
    mi::Uint32 resolution_theta,
    mi::neuraylib::Lightprofile_degree degree,
    mi::Uint32 flags,
    mi::Float32 start_phi,
    mi::Float32 start_theta,
    mi::Float32 delta_phi,
    mi::Float32 delta_theta,
    const std::vector<mi::Float32>& data)
  : m_resolution_phi( resolution_phi),
    m_resolution_theta( resolution_theta),
    m_degree( degree),
    m_flags( flags),
    m_start_phi( start_phi),
    m_start_theta( start_theta),
    m_delta_phi( delta_phi),
    m_delta_theta( delta_theta),
    m_data( data)
{
    ASSERT( M_LIGHTPROFILE, m_data.size() == m_resolution_phi * m_resolution_theta);

    // normalize m_data, save multiplier in m_candela_multiplier
    m_candela_multiplier = 0.0f;
    for( mi::Size i = 0; i < m_data.size(); ++i)
        if( m_data[i] > m_candela_multiplier)
            m_candela_multiplier = m_data[i];
    if( m_candela_multiplier > 0.0f) {
        const mi::Float32 recipr = 1.0f / m_candela_multiplier;
        for( mi::Size i = 0; i < m_data.size(); ++i)
            m_data[i] *= recipr;
    }

    // compute power (integrate using one average value per grid cell)
    m_power = 0.0f;
    for( mi::Size p = 0; p < m_resolution_phi-1; ++p) {
        mi::Float32 cos_theta0 = cosf( m_start_theta);
        mi::Size offs0 = p * m_resolution_theta;
        mi::Size offs1 = (p + 1) * m_resolution_theta;
        for( mi::Uint32 t = 0; t < m_resolution_theta-1; ++t,++offs0,++offs1) {
            const mi::Float32 cos_theta1 = cosf( m_start_theta + (float)(t+1) * m_delta_theta);
            const mi::Float32 mu_theta = cos_theta0 - cos_theta1;
            const mi::Float32 value = m_data[offs0]
                                    + m_data[offs0 + 1]
                                    + m_data[offs1]
                                    + m_data[offs1 + 1];
            m_power += value * mu_theta;
            cos_theta0 = cos_theta1;
        }
    }
    // take into account the maximum value, phi range, and that we have summed up four values per
    // grid cell
    m_power *= m_candela_multiplier * m_delta_phi * 0.25f;
}

mi::Float32 Lightprofile_impl::get_phi( mi::Uint32 index) const
{
    return index >= m_resolution_phi ? 0.0f : m_start_phi + index * m_delta_phi;
}

mi::Float32 Lightprofile_impl::get_theta( mi::Uint32 index) const
{
    return index >= m_resolution_theta ? 0.0f : m_start_theta + index * m_delta_theta;
}

mi::Float32 Lightprofile_impl::get_data( mi::Uint32 index_phi, mi::Uint32 index_theta) const
{
    if( index_phi >= m_resolution_phi || index_theta >= m_resolution_theta)
        return 0.0f;

    return m_data[index_phi * m_resolution_theta + index_theta];
}

const mi::Float32* Lightprofile_impl::get_data() const
{
    return !m_data.empty() ? m_data.data() : nullptr;
}

mi::Float32 Lightprofile_impl::sample( mi::Float32 phi, mi::Float32 theta, bool candela) const
{
    if( m_data.empty())
        return 0.0f;

          mi::Float32 phi_offset   = phi   - m_start_phi;
    const mi::Float32 theta_offset = theta - m_start_theta;

    // reduce phi_offset mod 2*pi
    phi_offset -= floor( phi_offset / (float)(2*M_PI)) * (float)(2*M_PI);
    if( phi_offset < 0.0f || phi_offset > (float)(2*M_PI-0.0001))
        phi_offset = 0.0f;

    mi::Difference phi_index   = static_cast<mi::Difference>( floor( phi_offset   / m_delta_phi));
    mi::Difference theta_index = static_cast<mi::Difference>( floor( theta_offset / m_delta_theta));
    if( phi_index < 0 || theta_index < 0)
        return 0.0f;
    if( phi_index   == m_resolution_phi-1   && phi_offset   <= phi_index  *m_delta_phi   + 0.0001f)
        phi_index   -= 1;
    if( theta_index == m_resolution_theta-1 && theta_offset <= theta_index*m_delta_theta + 0.0001f)
        theta_index -= 1;
    if( phi_index >= m_resolution_phi-1 || theta_index >= m_resolution_theta-1)
        return 0.0f;

    mi::Float32 u = (phi_offset   - phi_index  *m_delta_phi  ) / m_delta_phi;
    mi::Float32 v = (theta_offset - theta_index*m_delta_theta) / m_delta_theta;
    u = mi::math::clamp( u, 0.0f, 1.0f);
    v = mi::math::clamp( v, 0.0f, 1.0f);

    // bilinear interpolation
    const mi::Size index0 =  phi_index    * m_resolution_theta + theta_index;
    const mi::Size index1 = (phi_index+1) * m_resolution_theta + theta_index;
    ASSERT( M_LIGHTPROFILE, index0+1 < m_data.size() && index1+1 < m_data.size());
    const mi::Float32 value = (1.0f-u) * ((1.0f-v) * m_data[index0] + v * m_data[index0 + 1])
                            +       u  * ((1.0f-v) * m_data[index1] + v * m_data[index1 + 1]);

    return candela ? value * m_candela_multiplier : value;
}

const SERIAL::Serializable* Lightprofile_impl::serialize( SERIAL::Serializer* serializer) const
{
    Scene_element_base::serialize( serializer);

    SERIAL::write( serializer, m_resolution_phi);
    SERIAL::write( serializer, m_resolution_theta);
    SERIAL::write_enum( serializer, m_degree);
    SERIAL::write( serializer, m_flags);
    SERIAL::write( serializer, m_start_phi);
    SERIAL::write( serializer, m_start_theta);
    SERIAL::write( serializer, m_delta_phi);
    SERIAL::write( serializer, m_delta_theta);

    SERIAL::write( serializer, m_data);
    SERIAL::write( serializer, m_candela_multiplier);
    SERIAL::write( serializer, m_power);

    return this + 1;
}

SERIAL::Serializable* Lightprofile_impl::deserialize( SERIAL::Deserializer* deserializer)
{
    Scene_element_base::deserialize( deserializer);

    SERIAL::read( deserializer, &m_resolution_phi);
    SERIAL::read( deserializer, &m_resolution_theta);
    SERIAL::read_enum( deserializer, &m_degree);
    SERIAL::read( deserializer, &m_flags);
    SERIAL::read( deserializer, &m_start_phi);
    SERIAL::read( deserializer, &m_start_theta);
    SERIAL::read( deserializer, &m_delta_phi);
    SERIAL::read( deserializer, &m_delta_theta);

    SERIAL::read( deserializer, &m_data);
    SERIAL::read( deserializer, &m_candela_multiplier);
    SERIAL::read( deserializer, &m_power);

    return this + 1;
}

void Lightprofile_impl::dump() const
{
    std::ostringstream s;

    s << "Phi resolution: " << m_resolution_phi
      << ", start: " << m_start_phi
      << ", delta: " << m_delta_phi << std::endl;
    s << "Theta resolution: " << m_resolution_theta
      << ", start: " << m_start_theta
      << ", delta: " << m_delta_theta << std::endl;

    s << "Degree: " << m_degree << std::endl;
    s << "Flags: " << m_flags << std::endl;
    s << "Candela multiplier: " << m_candela_multiplier << std::endl;
    s << "Power: " << m_power << std::endl;

    LOG::mod_log->info( M_SCENE, LOG::Mod_log::C_DATABASE, "%s", s.str().c_str());
}

size_t Lightprofile_impl::get_size() const
{
    return sizeof( *this)
        + dynamic_memory_consumption( m_data);
}

DB::Journal_type Lightprofile_impl::get_journal_flags() const
{
    return DB::Journal_type(
        SCENE::JOURNAL_CHANGE_FIELD.get_type() |
        SCENE::JOURNAL_CHANGE_SHADER_ATTRIBUTE.get_type());
}

namespace {

float round_3_digits( float x) { return mi::math::round( x*1000.0f)/1000.0f; }

bool export_to_writer(
    DB::Transaction* transaction, const Lightprofile* lightprofile, mi::neuraylib::IWriter* writer)
{
    const DB::Tag impl_tag = lightprofile->get_impl_tag();

    // reject default-constructed instances
    if( !impl_tag)
        return false;

    DB::Access<Lightprofile_impl> impl( impl_tag, transaction);

    const mi::Uint32 resolution_phi   = impl->get_resolution_phi();
    const mi::Uint32 resolution_theta = impl->get_resolution_theta();

    const mi::Float32 first_phi
        = round_3_digits( mi::math::degrees( impl->get_phi( 0)));
    const mi::Float32 last_phi
        = round_3_digits( mi::math::degrees( impl->get_phi( resolution_phi-1)));

    bool type_c;
    bool sampling;

    if( first_phi == 0.f && last_phi == 360.f) {
        type_c     = true;
        sampling   = false;
    } else if( first_phi == 270.f && last_phi == 450.f) {
        type_c     = false;
        sampling   = false;
    } else if( first_phi == 270.f && last_phi == 630.f) {
        // The range from -90 to 270 is not valid for IES files. Under certain conditions we still
        // have the original angles as a subset, but in general we need to re-sample the data to
        // obtain values corresponding to angles in a valid range.
        type_c     = true;
        sampling   = true;
    } else
        return false;

    // version
    if( !writer->writeline( "IESNA91\r\n"))
        return false;

    if( !writer->writeline( "TILT=NONE\r\n"))
        return false;

    // number of lamps, lumens, candela multiplier, resolution theta, resolution phi,
    // photometric type, units type, width, length, height
    {
    const std::string line = STRING::formatted_string( "1 0 %g %u %u %u 1 0 0 0\r\n",
        impl->get_candela_multiplier(), resolution_theta, resolution_phi, type_c ? 1 : 2);
    if( line.empty() || !writer->writeline( line.c_str()))
        return false;
    }

    // ballast factor, ballast lamp factor, input watts
    if( !writer->writeline( "1 1 0\r\n"))
        return false;

    // theta values
    for( mi::Uint32 j = 0; j < resolution_theta; ++j) {
        const mi::Float32 theta = static_cast<mi::Float32>( type_c
            ? mi::math::degrees( (float)M_PI - impl->get_theta( resolution_theta-1-j))
            : mi::math::degrees( impl->get_theta( j) - (float)(M_PI/2.0)));

        const std::string line = STRING::formatted_string( "%g ", round_3_digits( theta));
        if( line.empty() || !writer->writeline( line.c_str()))
            return false;

        if( (j + 1) % 14 == 0 && j + 1 < resolution_theta)
            if( !writer->writeline( "\r\n"))
                return false;
    }
    if( !writer->writeline("\r\n"))
        return false;

    // phi values
    const float scale = (float)(360.0 / (resolution_phi - 1));
    for( mi::Uint32 i = 0; i < resolution_phi; ++i) {
        const mi::Float32 phi = sampling
            ? (float)i * scale
            : 360.0f - mi::math::degrees( impl->get_phi( resolution_phi-1-i));

        const std::string line = STRING::formatted_string( "%g ", round_3_digits( phi));
        if( line.empty() || !writer->writeline(line.c_str()))
            return false;

        if( (i + 1) % 14 == 0 && i + 1 < resolution_phi)
            if( !writer->writeline( "\r\n"))
                return false;
    }
    if( !writer->writeline( "\r\n"))
        return false;

    // candela values
    const float scalec = (float)((2.0 * M_PI) / (resolution_phi - 1));
    for( mi::Uint32 i = 0; i < resolution_phi; ++i) {
        const mi::Float32 phi = (float)(resolution_phi-1 - i) * scalec;
        for( mi::Uint32 j = 0; j < resolution_theta; ++j) {
             mi::Float32 value;
             if( sampling) {
                 const mi::Float32 theta = impl->get_theta( resolution_theta-1-j);
                 value = impl->sample( phi, theta, false);
             } else if( type_c)
                 value = impl->get_data( resolution_phi-1-i, resolution_theta-1-j);
             else
                 value = impl->get_data( resolution_phi-1-i, j);

            const std::string line = STRING::formatted_string( "%f ", value);
            if( line.empty() || !writer->writeline(line.c_str()))
                return false;

             if( (j + 1) % 14 == 0 && j + 1 < resolution_theta)
                if( !writer->writeline( "\r\n"))
                    return false;
        }
        if( !writer->writeline( "\r\n"))
            return false;
    }

    return true;
}

} // anonymous


bool export_to_file(
    DB::Transaction* transaction, const Lightprofile* lightprofile, const std::string& filename)
{
    DISK::File_writer_impl writer;
    if( !writer.open(filename.c_str()))
        return false;

    const bool success = export_to_writer( transaction, lightprofile, &writer);
    writer.close();
    return success;
}

mi::neuraylib::IBuffer* create_buffer_from_lightprofile(
    DB::Transaction* transaction, const Lightprofile* lightprofile)
{
    DISK::Memory_writer_impl writer;

    if( !export_to_writer( transaction, lightprofile, &writer))
        return nullptr;

    mi::neuraylib::IBuffer* buffer = writer.get_buffer();
    return buffer;
}

DB::Tag load_mdl_lightprofile(
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
        return DB::Tag();
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
    db_name += "lightprofile_" + identifier;
    if( !shared_proxy)
        db_name = MDL::DETAIL::generate_unique_db_name( transaction, db_name.c_str());

    DB::Tag tag = transaction->name_to_tag( db_name.c_str());
    if( tag) {
        result = 0;
        return tag;
    }

    auto lp = std::make_unique<Lightprofile>();
    result = lp->reset_mdl( transaction,
        reader, filename, container_filename, container_membername, mdl_file_path, impl_hash);
    ASSERT( M_LIGHTPROFILE, result == 0 || result == -7);
    if( result != 0)
        return DB::Tag();

    tag = transaction->store_for_reference_counting(
        lp.release(), db_name.c_str(), transaction->get_scope()->get_level());
    result = 0;
    return tag;
}

} // namespace LIGHTPROFILE

} // namespace MI
