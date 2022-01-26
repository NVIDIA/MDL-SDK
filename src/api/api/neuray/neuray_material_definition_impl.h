/***************************************************************************************************
 * Copyright (c) 2015-2022, NVIDIA CORPORATION. All rights reserved.
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
 ** \brief Header for the IMaterial_definition implementation.
 **/

#ifndef API_API_NEURAY_MATERIAL_DEFINITION_IMPL_H
#define API_API_NEURAY_MATERIAL_DEFINITION_IMPL_H

#include <mi/neuraylib/imaterial_definition.h>

#include <mi/base/handle.h>
#include <mi/base/interface_implement.h>
#include <mi/neuraylib/ifunction_definition.h>

namespace MI {

namespace MDL { class Mdl_function_definition; }

namespace NEURAY {

/// This class implements an MDL material definition.
///
/// Note that this class does not use the usual mixins, but implements all methods by forwarding to
/// the wrapped function definition (with the exception of get_element_type() and get_interface()).
class Material_definition_impl final
    : public mi::base::Interface_implement<mi::neuraylib::IMaterial_definition>
{
public:

    static mi::neuraylib::IMaterial_definition* create_api_class(
        mi::neuraylib::IFunction_definition* impl);

    static const mi::neuraylib::IMaterial_definition* create_api_class(
        const mi::neuraylib::IFunction_definition* impl);

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

    // public API methods (IMaterial_definition)

    const char* get_module() const final;

    const char* get_mdl_name() const final;

    const char* get_mdl_module_name() const final;

    const char* get_mdl_simple_name() const final;

    const char* get_mdl_parameter_type_name( mi::Size index) const final;

    const char* get_prototype() const final;

    void get_mdl_version(
        mi::neuraylib::Mdl_version& since, mi::neuraylib::Mdl_version& removed) const final;

    mi::neuraylib::IFunction_definition::Semantics get_semantic() const final;

    bool is_exported() const final;

    const mi::neuraylib::IType* get_return_type() const final;

    mi::Size get_parameter_count() const final;

    const char* get_parameter_name( mi::Size index) const final;

    mi::Size get_parameter_index( const char* name) const final;

    const mi::neuraylib::IType_list* get_parameter_types() const final;

    const mi::neuraylib::IExpression_list* get_defaults() const final;

    const mi::neuraylib::IExpression_list* get_enable_if_conditions() const final;

    mi::Size get_enable_if_users( mi::Size index) const final;

    mi::Size get_enable_if_user( mi::Size index, mi::Size u_index) const final;

    const mi::neuraylib::IAnnotation_block* get_annotations() const final;

    const mi::neuraylib::IAnnotation_block* get_return_annotations() const final;

    const mi::neuraylib::IAnnotation_list* get_parameter_annotations() const final;

    const char* get_thumbnail() const final;

    bool is_valid( mi::neuraylib::IMdl_execution_context* context) const final;

    const mi::neuraylib::IExpression_direct_call* get_body() const final;

    mi::Size get_temporary_count() const final;

    const mi::neuraylib::IExpression* get_temporary( mi::Size index) const final;

    const char* get_temporary_name( mi::Size index) const final;

    mi::neuraylib::IMaterial_instance* create_material_instance(
        const mi::neuraylib::IExpression_list* arguments,
        mi::Sint32* errors) const final;

    // internal (part of IDb_element, although not derived from it)

    const MDL::Mdl_function_definition* get_db_element() const;

    MDL::Mdl_function_definition* get_db_element();

private:
    Material_definition_impl( mi::neuraylib::IFunction_definition* impl);

    mi::base::Handle<mi::neuraylib::IFunction_definition> m_impl;
};

} // namespace NEURAY

} // namespace MI

#endif // API_API_NEURAY_MATERIAL_DEFINITION_IMPL_H
