/***************************************************************************************************
 * Copyright (c) 2015-2024, NVIDIA CORPORATION. All rights reserved.
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
 ** \brief Header for the IMaterial_instance implementation.
 **/

#ifndef API_API_NEURAY_NEURAY_MATERIAL_INSTANCE_IMPL_H
#define API_API_NEURAY_NEURAY_MATERIAL_INSTANCE_IMPL_H

#include <mi/neuraylib/imaterial_instance.h>

#include <mi/base/handle.h>
#include <mi/base/interface_implement.h>
#include <mi/neuraylib/ifunction_call.h>

namespace MI {

namespace MDL { class Mdl_function_call; }

namespace NEURAY {

/// This class implements MDL material instances.
///
/// Note that this class does not use the usual mixins, but implements all methods by forwarding to
/// the wrapped function call (with the exception of get_element_type() and get_interface()).
class Material_instance_impl final
  : public mi::base::Interface_implement<mi::neuraylib::IMaterial_instance>
{
public:

    static mi::neuraylib::IMaterial_instance* create_api_class(
        mi::neuraylib::IFunction_call* impl);

    static const mi::neuraylib::IMaterial_instance* create_api_class(
        const mi::neuraylib::IFunction_call* impl);

    // public API methods (IInterface)

    const mi::base::IInterface* get_interface( const mi::base::Uuid& interface_id) const;

    mi::base::IInterface* get_interface( const mi::base::Uuid& interface_id);

    // public API methods (IAttribute_set)

    mi::IData* create_attribute( const char* name, const char* type) final;

    bool destroy_attribute( const char* name) final;

    const mi::IData* access_attribute( const char* name) const final;

    mi::IData* edit_attribute( const char* name) final;

    bool is_attribute( const char* name) const final;

    const char* get_attribute_type_name( const char* name) const final;

    mi::Sint32 set_attribute_propagation(
        const char* name, mi::neuraylib::Propagation_type value) final;

    mi::neuraylib::Propagation_type get_attribute_propagation( const char* name) const final;

    const char* enumerate_attributes( mi::Sint32 index) const final;

    // public API methods (IScene_element)

    mi::neuraylib::Element_type get_element_type() const final;

    // public API methods (IMaterial_instance)

    mi::neuraylib::ICompiled_material* create_compiled_material(
        mi::Uint32 flags,
        mi::neuraylib::IMdl_execution_context* context) const final;

    // internal (part of IDb_element, although not derived from it)

    const MDL::Mdl_function_call* get_db_element() const;

    MDL::Mdl_function_call* get_db_element();

private:
    Material_instance_impl( mi::neuraylib::IFunction_call* impl);

    mi::base::Handle<mi::neuraylib::IFunction_call> m_impl;
};

} // namespace NEURAY

} // namespace MI

#endif // API_API_NEURAY_NEURAY_MATERIAL_INSTANCE_IMPL_H
