/***************************************************************************************************
 * Copyright (c) 2010-2020, NVIDIA CORPORATION. All rights reserved.
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
 ** \brief Source for the ILightprofile implementation.
 **/

#include "pch.h"

#include "neuray_lightprofile_impl.h"

#include <io/scene/lightprofile/i_lightprofile.h>
#include <io/scene/scene/i_scene_journal_types.h>

namespace MI {

namespace NEURAY {

DB::Element_base* Lightprofile_impl::create_db_element(
    mi::neuraylib::ITransaction* transaction,
    mi::Uint32 argc,
    const mi::base::IInterface* argv[])
{
    if( !transaction)
        return nullptr;
    if( argc != 0)
        return nullptr;
    return new LIGHTPROFILE::Lightprofile;
}

mi::base::IInterface* Lightprofile_impl::create_api_class(
    mi::neuraylib::ITransaction* transaction,
    mi::Uint32 argc,
    const mi::base::IInterface* argv[])
{
    if( !transaction)
        return nullptr;
    if( argc != 0)
        return nullptr;
    return (new Lightprofile_impl())->cast_to_major();
}

mi::neuraylib::Element_type Lightprofile_impl::get_element_type() const
{
    return mi::neuraylib::ELEMENT_TYPE_LIGHTPROFILE;
}

mi::Sint32 Lightprofile_impl::reset_file(
    const char* filename,
    mi::Uint32 resolution_phi,
    mi::Uint32 resolution_theta,
    mi::neuraylib::Lightprofile_degree degree,
    mi::Uint32 flags)
{
    if( !filename)
        return -1;

    mi::Sint32 result = get_db_element()->reset_file(
        get_db_transaction(), filename, resolution_phi, resolution_theta, degree, flags);
    if( result == 0) {
        add_journal_flag( SCENE::JOURNAL_CHANGE_FIELD);
        add_journal_flag( SCENE::JOURNAL_CHANGE_SHADER_ATTRIBUTE);
    }
    return result;
}

mi::Sint32 Lightprofile_impl::reset_reader(
    mi::neuraylib::IReader* reader,
    mi::Uint32 resolution_phi,
    mi::Uint32 resolution_theta,
    mi::neuraylib::Lightprofile_degree degree,
    mi::Uint32 flags)
{
    if( !reader)
        return -1;

    mi::Sint32 result = get_db_element()->reset_reader(
        get_db_transaction(), reader, resolution_phi, resolution_theta, degree, flags);
    if( result == 0) {
        add_journal_flag( SCENE::JOURNAL_CHANGE_FIELD);
        add_journal_flag( SCENE::JOURNAL_CHANGE_SHADER_ATTRIBUTE);
    }
    return result;
}

const char* Lightprofile_impl::get_filename() const
{
    const std::string& filename = get_db_element()->get_filename();
    return filename.empty() ? nullptr : filename.c_str();
}

const char* Lightprofile_impl::get_original_filename() const
{
    const std::string& filename = get_db_element()->get_original_filename();
    return filename.empty() ? nullptr : filename.c_str();
}

mi::Uint32 Lightprofile_impl::get_resolution_phi() const
{
    return get_db_element()->get_resolution_phi();
}

mi::Uint32 Lightprofile_impl::get_resolution_theta() const
{
    return get_db_element()->get_resolution_theta();
}

mi::neuraylib::Lightprofile_degree Lightprofile_impl::get_degree() const
{
    return get_db_element()->get_degree();
}

mi::Uint32 Lightprofile_impl::get_flags() const
{
    return get_db_element()->get_flags();
}

mi::Float32 Lightprofile_impl::get_phi( mi::Uint32 index) const
{
    return get_db_element()->get_phi( index);
}

mi::Float32 Lightprofile_impl::get_theta( mi::Uint32 index) const
{
    return get_db_element()->get_theta( index);
}

mi::Float32 Lightprofile_impl::get_data( mi::Uint32 index_phi, mi::Uint32 index_theta) const
{
    return get_db_element()->get_data( get_db_transaction(), index_phi, index_theta);
}

const mi::Float32* Lightprofile_impl::get_data() const
{
    return get_db_element()->get_data( get_db_transaction());
}

mi::Float64 Lightprofile_impl::get_candela_multiplier() const
{
    return get_db_element()->get_candela_multiplier();
}

mi::Float32 Lightprofile_impl::sample( mi::Float32 phi, mi::Float32 theta, bool candela) const
{
    return get_db_element()->sample( get_db_transaction(), phi, theta, candela);
}

} // namespace NEURAY

} // namespace MI
