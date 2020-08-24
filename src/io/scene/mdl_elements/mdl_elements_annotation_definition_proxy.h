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

#ifndef IO_SCENE_MDL_ELEMENTS_ANNOTATION_DEFINITION_PROXY_H
#define IO_SCENE_MDL_ELEMENTS_ANNOTATION_DEFINITION_PROXY_H

#include <string>

#include <io/scene/scene/i_scene_scene_element.h>

namespace MI {

namespace MDL {

/// The class ID for the #Mdl_annotation_definition_proxy class.
static const SERIAL::Class_id ID_MDL_ANNOTATION_DEFINITION_PROXY = 0x5f4d6164; // '_Mad'

class Mdl_annotation_definition_proxy
  : public SCENE::Scene_element<Mdl_annotation_definition_proxy,
                                ID_MDL_ANNOTATION_DEFINITION_PROXY>
{
public:

    /// Default constructor.
    ///
    /// Does not create a valid instance, to be used by the deserializer only.
    Mdl_annotation_definition_proxy();

    /// Constructor.
    Mdl_annotation_definition_proxy( const char* module_name);

    /// Copy constructor.
    Mdl_annotation_definition_proxy( const Mdl_annotation_definition_proxy& other) = default;

    Mdl_annotation_definition_proxy& operator=( const Mdl_annotation_definition_proxy&) = delete;

    // methods corresponding to mi::neuraylib::IMaterial_instance

    const char* get_db_module_name() const { return m_module_db_name.c_str(); }

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

private:

    std::string m_module_db_name;                ///< The DB name of the module.
};

} // namespace MDL

} // namespace MI

#endif // IO_SCENE_MDL_ELEMENTS_ANNOTATION_DEFINITION_PROXY_H
