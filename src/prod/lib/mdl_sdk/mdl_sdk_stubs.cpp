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

/// \file
/// \brief Stubs for the MDL SDK library.

#include "pch.h"

#include <io/scene/scene/i_scene_scene_element_base.h>

namespace MI {
    
// needed because MDL and other scene elements are used without linking the SCENE module
namespace SCENE {

std::string get_class_name( SERIAL::Class_id id)
{
    // TODO implement this (used by the API page for the admin HTTP server only)
    return "";
}

std::string Scene_element_base::get_class_name() const
{
    return SCENE::get_class_name( this->get_class_id());
}

void Scene_element_base::swap( Scene_element_base& other)
{
    std::swap( m_attributes, other.m_attributes);
}

size_t Scene_element_base::get_size() const { return 0; }

const SERIAL::Serializable* Scene_element_base::serialize( SERIAL::Serializer* serializer) const
{
    return this + 1;
}

SERIAL::Serializable* Scene_element_base::deserialize( SERIAL::Deserializer* deserializer)
{
    return this + 1;
}

void Scene_element_base::get_references( DB::Tag_set* result) const
{
    get_attributes()->get_references( result);
    get_scene_element_references( result);
}

} // namespace SCENE

} // namespace MI
