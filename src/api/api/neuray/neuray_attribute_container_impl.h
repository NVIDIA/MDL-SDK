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
 ** \brief Header for the IAttribute_container implementation.
 **/

#ifndef API_API_NEURAY_NEURAY_ATTRIBUTE_CONTAINER_IMPL_H
#define API_API_NEURAY_NEURAY_ATTRIBUTE_CONTAINER_IMPL_H

#include <mi/neuraylib/iattribute_container.h>

#include "neuray_db_element_impl.h"
#include "neuray_attribute_set_impl.h"

#include <io/scene/scene/i_scene_scene_element.h>

namespace MI {

namespace NEURAY {

class Attribute_container;

class Attribute_container_impl
  : public Attribute_set_impl<Db_element_impl<mi::neuraylib::IAttribute_container,
                                              Attribute_container> >
{
public:

    typedef Attribute_set_impl<Db_element_impl<mi::neuraylib::IAttribute_container,
                                               Attribute_container> > Parent_type;

    static DB::Element_base* create_db_element(
        mi::neuraylib::ITransaction* transaction,
        mi::Uint32 argc,
        const mi::base::IInterface* argv[]);

    static mi::base::IInterface* create_api_class(
        mi::neuraylib::ITransaction* transaction,
        mi::Uint32 argc,
        const mi::base::IInterface* argv[]);

    // public API methods

    // IAttribute_set methods

    /// Override implementation in Attribute_set_impl to skip the type check.
    mi::IData* create_attribute( const char* name, const char* type_name);

    // IScene_element methods

    mi::neuraylib::Element_type get_element_type() const;

    // internal methods

    /// See Attribute_set_impl<T>::set_attribute_set().
    void set_attribute_set(
        ATTR::Attribute_set* attribute_set, const mi::base::IInterface* owner);

    /// See Attribute_set_impl<T>::set_attribute_set().
    void set_attribute_set(
        const ATTR::Attribute_set* attribute_set, const mi::base::IInterface* owner) const;
};

const SERIAL::Class_id ID_ATTRIBUTE_CONTAINER = 0x5F415443; // '_ATC'

class Attribute_container : public SCENE::Scene_element<Attribute_container, ID_ATTRIBUTE_CONTAINER>
{
public:
    size_t get_size() const;

    /// Calls SCENE::Scene_element::serialize().
    const SERIAL::Serializable* serialize( SERIAL::Serializer *serializer) const;

    /// Calls SCENE::Scene_element::deserialize().
    SERIAL::Serializable* deserialize( SERIAL::Deserializer* deserializer);

    /// Leaves \p result unchanged.
    void get_scene_element_references( DB::Tag_set* result) const;

    /// Returns "Attribute_container".
    std::string get_class_name() const;
};

} // namespace NEURAY

} // namespace MI

#endif // API_API_NEURAY_NEURAY_ATTRIBUTE_CONTAINER_IMPL_H
