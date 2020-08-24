/***************************************************************************************************
 * Copyright (c) 2015-2020, NVIDIA CORPORATION. All rights reserved.
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
 ** \brief Source for the ICompiled_material implementation.
 **/

#include "pch.h"

#include <mi/base/handle.h>

#include "neuray_compiled_material_impl.h"
#include "neuray_expression_impl.h"
#include "neuray_mdl_execution_context_impl.h"
#include "neuray_transaction_impl.h"
#include "neuray_value_impl.h"
#include "neuray_string_impl.h"

#include <io/scene/mdl_elements/i_mdl_elements_compiled_material.h>
#include <io/scene/scene/i_scene_journal_types.h>

namespace MI {

namespace NEURAY {

DB::Element_base* Compiled_material_impl::create_db_element(
    mi::neuraylib::ITransaction* transaction,
    mi::Uint32 argc,
    const mi::base::IInterface* argv[])
{
    if( argc != 0)
        return nullptr;
    return new MDL::Mdl_compiled_material;
}

mi::base::IInterface* Compiled_material_impl::create_api_class(
    mi::neuraylib::ITransaction* transaction,
    mi::Uint32 argc,
    const mi::base::IInterface* argv[])
{
    if( argc != 0)
        return nullptr;
    return (new Compiled_material_impl())->cast_to_major();
}

mi::neuraylib::Element_type Compiled_material_impl::get_element_type() const
{
    return mi::neuraylib::ELEMENT_TYPE_COMPILED_MATERIAL;
}

const mi::neuraylib::IExpression_direct_call* Compiled_material_impl::get_body() const
{
   mi::base::Handle<Expression_factory> ef( get_transaction()->get_expression_factory());
   mi::base::Handle<const MDL::IExpression_direct_call> result_int( get_db_element()->get_body());
   return ef->create<mi::neuraylib::IExpression_direct_call>(
       result_int.get(), this->cast_to_major());
}

mi::Size Compiled_material_impl::get_temporary_count() const
{
    return get_db_element()->get_temporary_count();
}

const mi::neuraylib::IExpression* Compiled_material_impl::get_temporary(
    mi::Size index) const
{
   mi::base::Handle<Expression_factory> ef( get_transaction()->get_expression_factory());
   mi::base::Handle<const MDL::IExpression> result_int( get_db_element()->get_temporary( index));
   return ef->create( result_int.get(), this->cast_to_major());
}

mi::Float32 Compiled_material_impl::get_mdl_meters_per_scene_unit() const
{
    return get_db_element()->get_mdl_meters_per_scene_unit();
}

mi::Float32 Compiled_material_impl::get_mdl_wavelength_min() const
{
    return get_db_element()->get_mdl_wavelength_min();
}

mi::Float32 Compiled_material_impl::get_mdl_wavelength_max() const
{
    return get_db_element()->get_mdl_wavelength_max();
}

bool Compiled_material_impl::depends_on_state_transform() const
{
    return get_db_element()->depends_on_state_transform();
}

bool Compiled_material_impl::depends_on_state_object_id() const
{
    return get_db_element()->depends_on_state_object_id();
}

bool Compiled_material_impl::depends_on_global_distribution() const
{
    return get_db_element()->depends_on_global_distribution();
}

bool Compiled_material_impl::depends_on_uniform_scene_data() const
{
    return get_db_element()->depends_on_uniform_scene_data();
}

mi::Size Compiled_material_impl::get_referenced_scene_data_count() const
{
    return get_db_element()->get_referenced_scene_data_count();
}

char const *Compiled_material_impl::get_referenced_scene_data_name( mi::Size index) const
{
    return get_db_element()->get_referenced_scene_data_name( index);
}

mi::Size Compiled_material_impl::get_parameter_count() const
{
    return get_db_element()->get_parameter_count();
}

const char* Compiled_material_impl::get_parameter_name( mi::Size index) const
{
    return get_db_element()->get_parameter_name( index);
}

const mi::neuraylib::IValue* Compiled_material_impl::get_argument( mi::Size index) const
{
   mi::base::Handle<Value_factory> vf( get_transaction()->get_value_factory());
   mi::base::Handle<const MDL::IValue> result_int( get_db_element()->get_argument( index));
   return vf->create( result_int.get(), this->cast_to_major());
}

mi::base::Uuid Compiled_material_impl::get_hash() const
{
    return get_db_element()->get_hash();
}

mi::base::Uuid Compiled_material_impl::get_slot_hash( mi::neuraylib::Material_slot slot) const
{
    return get_db_element()->get_slot_hash( slot);
}

const mi::neuraylib::IExpression* Compiled_material_impl::lookup_sub_expression(
    const char* path) const
{
    mi::base::Handle<Expression_factory> ef(
        get_transaction()->get_expression_factory());
    mi::base::Handle<const MDL::IExpression> result_int(
        get_db_element()->lookup_sub_expression( path));
    return ef->create( result_int.get(), this->cast_to_major());
}

const mi::IString* Compiled_material_impl::get_connected_function_db_name(
    const char* material_instance_name,
    mi::Size parameter_index,
    mi::Sint32* errors) const
{
    mi::Sint32 dummy_error = 0;
    if (!errors) errors = &dummy_error;

    if (!material_instance_name) {
        *errors = -1;
        return nullptr;
    }
    if (parameter_index >= get_parameter_count()) {
        *errors = -2;
        return nullptr;
    }
    DB::Transaction* transaction = get_db_transaction();
    DB::Tag material_instance_tag = transaction->name_to_tag(material_instance_name);
    if (material_instance_tag.is_invalid()) {
        *errors = -1;
        return nullptr;
    }
    DB::Tag call_tag = get_db_element()->get_connected_function_db_name(
        transaction, material_instance_tag, get_parameter_name(parameter_index));
    if (call_tag.is_invalid()) {
        *errors = -3;
        return nullptr;
    }
    mi::IString* result = new String_impl();
    result->set_c_str(transaction->tag_to_name(call_tag));

    *errors = 0;
    return result;
}

static mi::neuraylib::Material_opacity int_opacity_to_opacity(
    mi::mdl::IGenerated_code_dag::IMaterial_instance::Opacity opacity)
{
    switch (opacity) {
    case mi::mdl::IGenerated_code_dag::IMaterial_instance::OPACITY_OPAQUE:
        return mi::neuraylib::OPACITY_OPAQUE;
    case mi::mdl::IGenerated_code_dag::IMaterial_instance::OPACITY_TRANSPARENT:
        return mi::neuraylib::OPACITY_TRANSPARENT;
    case mi::mdl::IGenerated_code_dag::IMaterial_instance::OPACITY_UNKNOWN:
        return mi::neuraylib::OPACITY_UNKNOWN;
    default:
        break;
    }
    return mi::neuraylib::OPACITY_UNKNOWN;
}

mi::neuraylib::Material_opacity Compiled_material_impl::get_opacity() const
{
    return int_opacity_to_opacity(get_db_element()->get_opacity());
}

mi::neuraylib::Material_opacity Compiled_material_impl::get_surface_opacity() const
{
    return int_opacity_to_opacity(get_db_element()->get_surface_opacity());
}

bool Compiled_material_impl::get_cutout_opacity(mi::Float32 *cutout_opacity) const
{
    return get_db_element()->get_cutout_opacity(cutout_opacity);
}

bool Compiled_material_impl::is_valid(mi::neuraylib::IMdl_execution_context* context) const
{
    MDL::Execution_context default_context;
    return get_db_element()->is_valid(
        get_db_transaction(),
        unwrap_and_clear_context(context, default_context));
}

void Compiled_material_impl::swap( MDL::Mdl_compiled_material& rhs)
{
    get_db_element()->swap( rhs);
}

} // namespace NEURAY

} // namespace MI
