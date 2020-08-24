/***************************************************************************************************
 * Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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

#include "mdl_elements_annotation_definition_proxy.h"
#include "i_mdl_elements_utilities.h"


#include <base/lib/log/i_log_logger.h>
#include <base/data/serial/i_serializer.h>
#include <io/scene/scene/i_scene_journal_types.h>

namespace MI {

namespace MDL {

Mdl_annotation_definition_proxy::Mdl_annotation_definition_proxy()
  : m_module_db_name()
{
}

Mdl_annotation_definition_proxy::Mdl_annotation_definition_proxy(
    const char* module_name)
  : m_module_db_name( get_db_name( module_name))
{
}

const SERIAL::Serializable* Mdl_annotation_definition_proxy::serialize( SERIAL::Serializer* serializer) const
{
    Scene_element_base::serialize( serializer);

    serializer->write( m_module_db_name);

    return this + 1;
}

SERIAL::Serializable* Mdl_annotation_definition_proxy::deserialize( SERIAL::Deserializer* deserializer)
{
    Scene_element_base::deserialize( deserializer);

    deserializer->read( &m_module_db_name);

    return this + 1;
}

void Mdl_annotation_definition_proxy::dump() const
{
    std::ostringstream s;
    s << "Module DB name: \"" << m_module_db_name << "\"" << std::endl;
    LOG::mod_log->info( M_SCENE, LOG::Mod_log::C_DATABASE, "%s", s.str().c_str());
}

size_t Mdl_annotation_definition_proxy::get_size() const
{
    return sizeof( *this)
        + SCENE::Scene_element<Mdl_annotation_definition_proxy, 
                               Mdl_annotation_definition_proxy::id>::get_size()
            - sizeof( SCENE::Scene_element<Mdl_annotation_definition_proxy,
                                           Mdl_annotation_definition_proxy::id>)
        + dynamic_memory_consumption( m_module_db_name);
}

DB::Journal_type Mdl_annotation_definition_proxy::get_journal_flags() const
{
    return SCENE::JOURNAL_CHANGE_NOTHING;
}

} // namespace MDL

} // namespace MI
