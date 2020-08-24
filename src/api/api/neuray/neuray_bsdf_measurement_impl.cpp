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

/** \file
 ** \brief Source for the IBsdf_measurement implementation.
 **/

#include "pch.h"

#include "neuray_bsdf_measurement_impl.h"

#include <mi/neuraylib/ibsdf_isotropic_data.h>

#include <io/scene/scene/i_scene_journal_types.h>
#include <io/scene/bsdf_measurement/i_bsdf_measurement.h>

namespace MI {

namespace NEURAY {

DB::Element_base* Bsdf_measurement_impl::create_db_element(
    mi::neuraylib::ITransaction* transaction,
    mi::Uint32 argc,
    const mi::base::IInterface* argv[])
{
    if( !transaction)
        return nullptr;
    if( argc != 0)
        return nullptr;
    return new BSDFM::Bsdf_measurement;
}

mi::base::IInterface* Bsdf_measurement_impl::create_api_class(
    mi::neuraylib::ITransaction* transaction,
    mi::Uint32 argc,
    const mi::base::IInterface* argv[])
{
    if( !transaction)
        return nullptr;
    if( argc != 0)
        return nullptr;
    return (new Bsdf_measurement_impl())->cast_to_major();
}

mi::neuraylib::Element_type Bsdf_measurement_impl::get_element_type() const
{
    return mi::neuraylib::ELEMENT_TYPE_BSDF_MEASUREMENT;
}

mi::Sint32 Bsdf_measurement_impl::reset_file( const char* filename)
{
    if( !filename)
        return -1;

    mi::Sint32 result = get_db_element()->reset_file( get_db_transaction(), filename);
    if( result == 0)
        add_journal_flag( SCENE::JOURNAL_CHANGE_SHADER_ATTRIBUTE);
    return result;
}

mi::Sint32 Bsdf_measurement_impl::reset_reader( mi::neuraylib::IReader* reader)
{
    if( !reader)
        return -1;

    mi::Sint32 result = get_db_element()->reset_reader( get_db_transaction(), reader);
    if( result == 0)
        add_journal_flag( SCENE::JOURNAL_CHANGE_SHADER_ATTRIBUTE);
    return result;
}

const char* Bsdf_measurement_impl::get_filename() const
{
    const std::string& filename = get_db_element()->get_filename();
    return filename.empty() ? nullptr : filename.c_str();
}

const char* Bsdf_measurement_impl::get_original_filename() const
{
    const std::string& filename = get_db_element()->get_original_filename();
    return filename.empty() ? nullptr : filename.c_str();
}

mi::Sint32 Bsdf_measurement_impl::set_reflection(
    const mi::neuraylib::IBsdf_isotropic_data* bsdf_data)
{
    if( bsdf_data) {
        if( bsdf_data->get_resolution_theta() <= 0 || bsdf_data->get_resolution_phi() <= 0)
            return -2;
        mi::neuraylib::Bsdf_type type = bsdf_data->get_type();
        if( type != mi::neuraylib::BSDF_RGB && type != mi::neuraylib::BSDF_SCALAR)
            return -2;
    }

    get_db_element()->set_reflection( get_db_transaction(), bsdf_data);
    add_journal_flag( SCENE::JOURNAL_CHANGE_SHADER_ATTRIBUTE);
    return 0;
}

const mi::base::IInterface* Bsdf_measurement_impl::get_reflection() const
{
    return get_db_element()->get_reflection( get_db_transaction());
}

mi::Sint32 Bsdf_measurement_impl::set_transmission(
    const mi::neuraylib::IBsdf_isotropic_data* bsdf_data)
{
    if( bsdf_data) {
        if( bsdf_data->get_resolution_theta() <= 0 || bsdf_data->get_resolution_phi() <= 0)
            return -2;
        mi::neuraylib::Bsdf_type type = bsdf_data->get_type();
        if( type != mi::neuraylib::BSDF_RGB && type != mi::neuraylib::BSDF_SCALAR)
            return -2;
    }

    get_db_element()->set_transmission( get_db_transaction(), bsdf_data);
    add_journal_flag( SCENE::JOURNAL_CHANGE_SHADER_ATTRIBUTE);
    return 0;
}

const mi::base::IInterface* Bsdf_measurement_impl::get_transmission() const
{
    return get_db_element()->get_transmission( get_db_transaction());
}

} // namespace NEURAY

} // namespace MI
