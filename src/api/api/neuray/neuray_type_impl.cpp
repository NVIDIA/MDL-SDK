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
 ** \brief Source for the IType implementation.
 **/

#include "pch.h"

#include "neuray_type_impl.h"

#include "neuray_class_factory.h"
#include "neuray_expression_impl.h"

namespace MI {

namespace NEURAY {

mi::Uint32 int_modifiers_to_ext_modifiers( mi::Uint32 modifiers)
{
    ASSERT( M_SCENE, (modifiers
        & ~MDL::IType_alias::MK_UNIFORM
        & ~MDL::IType_alias::MK_VARYING) == 0);

    mi::Uint32 result = 0;
    if( modifiers & MDL::IType_alias::MK_UNIFORM) result |= mi::neuraylib::IType_alias::MK_UNIFORM;
    if( modifiers & MDL::IType_alias::MK_VARYING) result |= mi::neuraylib::IType_alias::MK_VARYING;
    return result;
}

mi::Uint32 ext_modifiers_to_int_modifiers( mi::Uint32 modifiers)
{
    ASSERT( M_SCENE, (modifiers
        & ~mi::neuraylib::IType_alias::MK_UNIFORM
        & ~mi::neuraylib::IType_alias::MK_VARYING) == 0);

    mi::Uint32 result = 0;
    if( modifiers & mi::neuraylib::IType_alias::MK_UNIFORM) result |= MDL::IType_alias::MK_UNIFORM;
    if( modifiers & mi::neuraylib::IType_alias::MK_VARYING) result |= MDL::IType_alias::MK_VARYING;
    return result;
}

mi::neuraylib::IType_enum::Predefined_id int_enum_id_to_ext_enum_id(
    MDL::IType_enum::Predefined_id enum_id)
{
    switch( enum_id) {
        case MDL::IType_enum::EID_USER:
            return mi::neuraylib::IType_enum::EID_USER;
        case MDL::IType_enum::EID_TEX_GAMMA_MODE:
            return mi::neuraylib::IType_enum::EID_TEX_GAMMA_MODE;
        case MDL::IType_enum::EID_INTENSITY_MODE:
            return mi::neuraylib::IType_enum::EID_INTENSITY_MODE;
        case MDL::IType_enum::EID_FORCE_32_BIT:
            return mi::neuraylib::IType_enum::EID_FORCE_32_BIT;
    }

    ASSERT( M_SCENE, false);
    return mi::neuraylib::IType_enum::EID_USER;
}

MDL::IType_enum::Predefined_id ext_enum_id_to_int_enum_id(
    mi::neuraylib::IType_enum::Predefined_id enum_id)
{
    switch( enum_id) {
        case mi::neuraylib::IType_enum::EID_USER:
            return MDL::IType_enum::EID_USER;
        case mi::neuraylib::IType_enum::EID_TEX_GAMMA_MODE:
            return MDL::IType_enum::EID_TEX_GAMMA_MODE;
        case mi::neuraylib::IType_enum::EID_INTENSITY_MODE:
            return MDL::IType_enum::EID_INTENSITY_MODE;
        case mi::neuraylib::IType_enum::EID_FORCE_32_BIT:
            return MDL::IType_enum::EID_FORCE_32_BIT;
    }

    ASSERT( M_SCENE, false);
    return MDL::IType_enum::EID_USER;
}

mi::neuraylib::IType_struct::Predefined_id int_struct_id_to_ext_struct_id(
    MDL::IType_struct::Predefined_id struct_id)
{
    switch( struct_id) {
        case MDL::IType_struct::SID_USER:
            return mi::neuraylib::IType_struct::SID_USER;
        case MDL::IType_struct::SID_MATERIAL_EMISSION:
            return mi::neuraylib::IType_struct::SID_MATERIAL_EMISSION;
        case MDL::IType_struct::SID_MATERIAL_SURFACE:
            return mi::neuraylib::IType_struct::SID_MATERIAL_SURFACE;
        case MDL::IType_struct::SID_MATERIAL_VOLUME:
            return mi::neuraylib::IType_struct::SID_MATERIAL_VOLUME;
        case MDL::IType_struct::SID_MATERIAL_GEOMETRY:
            return mi::neuraylib::IType_struct::SID_MATERIAL_GEOMETRY;
        case MDL::IType_struct::SID_MATERIAL:
            return mi::neuraylib::IType_struct::SID_MATERIAL;
        case MDL::IType_struct::SID_FORCE_32_BIT:
            return mi::neuraylib::IType_struct::SID_FORCE_32_BIT;
    }

    ASSERT( M_SCENE, false);
    return mi::neuraylib::IType_struct::SID_USER;
}

MDL::IType_struct::Predefined_id ext_struct_id_to_int_struct_id(
    mi::neuraylib::IType_struct::Predefined_id struct_id)
{
    switch( struct_id) {
    case mi::neuraylib::IType_struct::SID_USER:
            return MDL::IType_struct::SID_USER;
        case mi::neuraylib::IType_struct::SID_MATERIAL_EMISSION:
            return MDL::IType_struct::SID_MATERIAL_EMISSION;
        case mi::neuraylib::IType_struct::SID_MATERIAL_SURFACE:
            return MDL::IType_struct::SID_MATERIAL_SURFACE;
        case mi::neuraylib::IType_struct::SID_MATERIAL_VOLUME:
            return MDL::IType_struct::SID_MATERIAL_VOLUME;
        case mi::neuraylib::IType_struct::SID_MATERIAL_GEOMETRY:
            return MDL::IType_struct::SID_MATERIAL_GEOMETRY;
        case mi::neuraylib::IType_struct::SID_MATERIAL:
            return MDL::IType_struct::SID_MATERIAL;
        case mi::neuraylib::IType_struct::SID_FORCE_32_BIT:
            return MDL::IType_struct::SID_FORCE_32_BIT;
    }

    ASSERT( M_SCENE, false);
    return MDL::IType_struct::SID_USER;
}

mi::neuraylib::IType_texture::Shape int_shape_to_ext_shape( MDL::IType_texture::Shape shape)
{
    switch( shape) {
        case MDL::IType_texture::TS_2D:
            return mi::neuraylib::IType_texture::TS_2D;
        case MDL::IType_texture::TS_3D:
            return mi::neuraylib::IType_texture::TS_3D;
        case MDL::IType_texture::TS_CUBE:
            return mi::neuraylib::IType_texture::TS_CUBE;
        case MDL::IType_texture::TS_PTEX:
            return mi::neuraylib::IType_texture::TS_PTEX;
        case MDL::IType_texture::TS_BSDF_DATA:
            return mi::neuraylib::IType_texture::TS_BSDF_DATA;
        case MDL::IType_texture::TS_FORCE_32_BIT:
            return mi::neuraylib::IType_texture::TS_FORCE_32_BIT;
    }

    ASSERT( M_SCENE, false);
    return mi::neuraylib::IType_texture::TS_2D;
}

MDL::IType_texture::Shape ext_shape_to_int_shape( mi::neuraylib::IType_texture::Shape shape)
{
    switch( shape) {
        case mi::neuraylib::IType_texture::TS_2D:
            return MDL::IType_texture::TS_2D;
        case mi::neuraylib::IType_texture::TS_3D:
            return MDL::IType_texture::TS_3D;
        case mi::neuraylib::IType_texture::TS_CUBE:
            return MDL::IType_texture::TS_CUBE;
        case mi::neuraylib::IType_texture::TS_PTEX:
            return MDL::IType_texture::TS_PTEX;
        case mi::neuraylib::IType_texture::TS_BSDF_DATA:
            return MDL::IType_texture::TS_BSDF_DATA;
        case mi::neuraylib::IType_texture::TS_FORCE_32_BIT:
            return MDL::IType_texture::TS_FORCE_32_BIT;
    }

    ASSERT( M_SCENE, false);
    return MDL::IType_texture::TS_2D;
}

const MDL::IType* get_internal_type( const mi::neuraylib::IType* type)
{
    if( !type)
        return nullptr;
    mi::base::Handle<const IType_wrapper> type_wrapper( type->get_interface<IType_wrapper>());
    if( !type_wrapper)
        return nullptr;
    return type_wrapper->get_internal_type();
}

const MDL::IType_list* get_internal_type_list( const mi::neuraylib::IType_list* type_list)
{
    if( !type_list)
        return nullptr;
    mi::base::Handle<const IType_list_wrapper> type_list_wrapper(
        type_list->get_interface<IType_list_wrapper>());
    if( !type_list_wrapper)
        return nullptr;
    return type_list_wrapper->get_internal_type_list();
}

template <class T, class I>
Type_base<T, I>::~Type_base() { }

template <class E, class I>
mi::Uint32 Type_base<E, I>::get_all_type_modifiers() const
{
    return int_modifiers_to_ext_modifiers( m_type->get_all_type_modifiers());
}

template <class E, class I>
const mi::neuraylib::IType* Type_base<E, I>::skip_all_type_aliases() const
{
    mi::base::Handle<const MDL::IType> result( m_type->skip_all_type_aliases());
    return m_tf->create( result.get(), m_owner.get());
}

const mi::neuraylib::IType* Type_alias::get_aliased_type() const
{
    mi::base::Handle<const MDL::IType> result_int( m_type->get_aliased_type());
    return m_tf->create( result_int.get(), m_owner.get());
}

mi::Uint32 Type_alias::get_type_modifiers() const
{
    return int_modifiers_to_ext_modifiers( m_type->get_type_modifiers());
}

mi::Sint32 Type_enum::get_value_code( mi::Size index, mi::Sint32* errors) const
{
    return m_type->get_value_code( index, errors);
}

mi::neuraylib::IType_enum::Predefined_id Type_enum::get_predefined_id() const
{
    return int_enum_id_to_ext_enum_id( m_type->get_predefined_id());
}

const mi::neuraylib::IAnnotation_block* Type_enum::get_annotations() const
{
    mi::base::Handle<const MDL::IAnnotation_block> result_int( m_type->get_annotations());
    mi::base::Handle<Expression_factory> ef(
        s_class_factory->create_expression_factory( m_transaction.get()));
    return ef->create_annotation_block( result_int.get(), m_owner.get());
}

const mi::neuraylib::IAnnotation_block* Type_enum::get_value_annotations( mi::Size index) const
{
    mi::base::Handle<const MDL::IAnnotation_block> result_int(
        m_type->get_value_annotations( index));
    mi::base::Handle<Expression_factory> ef(
        s_class_factory->create_expression_factory( m_transaction.get()));
    return ef->create_annotation_block( result_int.get(), m_owner.get());
}

const mi::neuraylib::IType* Type_vector::get_component_type( mi::Size index) const
{
    mi::base::Handle<const MDL::IType> result_int( m_type->get_component_type( index));
    return m_tf->create( result_int.get(), m_owner.get());
}

const mi::neuraylib::IType_atomic* Type_vector::get_element_type() const
{
    mi::base::Handle<const MDL::IType_atomic> result_int( m_type->get_element_type());
    return m_tf->create<mi::neuraylib::IType_atomic>( result_int.get(), m_owner.get());
}

const mi::neuraylib::IType* Type_matrix::get_component_type( mi::Size index) const
{
    mi::base::Handle<const MDL::IType> result_int( m_type->get_component_type( index));
    return m_tf->create( result_int.get(), m_owner.get());
}

const mi::neuraylib::IType_vector* Type_matrix::get_element_type() const
{
    mi::base::Handle<const MDL::IType_vector> result_int( m_type->get_element_type());
    return m_tf->create<mi::neuraylib::IType_vector>( result_int.get(), m_owner.get());
}

const mi::neuraylib::IType* Type_color::get_component_type( mi::Size index) const
{
    mi::base::Handle<const MDL::IType> result_int( m_type->get_component_type( index));
    return m_tf->create( result_int.get(), m_owner.get());
}

const mi::neuraylib::IType* Type_array::get_component_type( mi::Size index) const
{
    mi::base::Handle<const MDL::IType> result_int( m_type->get_component_type( index));
    return m_tf->create( result_int.get(), m_owner.get());
}

const mi::neuraylib::IType* Type_array::get_element_type() const
{
    mi::base::Handle<const MDL::IType> result_int( m_type->get_element_type());
    return m_tf->create( result_int.get(), m_owner.get());
}

const mi::neuraylib::IType* Type_struct::get_component_type( mi::Size index) const
{
    mi::base::Handle<const MDL::IType> result_int( m_type->get_component_type( index));
    return m_tf->create( result_int.get(), m_owner.get());
}

const mi::neuraylib::IType* Type_struct::get_field_type( mi::Size index) const
{
    mi::base::Handle<const MDL::IType> result_int( m_type->get_field_type( index));
    return m_tf->create( result_int.get(), m_owner.get());
}

mi::neuraylib::IType_struct::Predefined_id Type_struct::get_predefined_id() const
{
    return int_struct_id_to_ext_struct_id( m_type->get_predefined_id());
}

const mi::neuraylib::IAnnotation_block* Type_struct::get_annotations() const
{
    mi::base::Handle<const MDL::IAnnotation_block> result_int( m_type->get_annotations());
    mi::base::Handle<Expression_factory> ef(
        s_class_factory->create_expression_factory( m_transaction.get()));
    return ef->create_annotation_block( result_int.get(), m_owner.get());
}

const mi::neuraylib::IAnnotation_block* Type_struct::get_field_annotations( mi::Size index) const
{
    mi::base::Handle<const MDL::IAnnotation_block> result_int(
        m_type->get_field_annotations( index));
    mi::base::Handle<Expression_factory> ef(
        s_class_factory->create_expression_factory( m_transaction.get()));
    return ef->create_annotation_block( result_int.get(), m_owner.get());
}

mi::neuraylib::IType_texture::Shape Type_texture::get_shape() const
{
    return int_shape_to_ext_shape( m_type->get_shape());
}

const mi::neuraylib::IType* Type_list::get_type( mi::Size index) const
{
    mi::base::Handle<const MDL::IType> result_int( m_type_list->get_type( index));
    return m_tf->create( result_int.get(), m_owner.get());
}

const mi::neuraylib::IType* Type_list::get_type( const char* name) const
{
    mi::base::Handle<const MDL::IType> result_int( m_type_list->get_type( name));
    return m_tf->create( result_int.get(), m_owner.get());
}

mi::Sint32 Type_list::set_type( mi::Size index, const mi::neuraylib::IType* type)
{
    mi::base::Handle<const MDL::IType> type_int( get_internal_type( type));
    return m_type_list->set_type( index, type_int.get());
}

mi::Sint32 Type_list::set_type( const char* name, const mi::neuraylib::IType* type)
{
    mi::base::Handle<const MDL::IType> type_int( get_internal_type( type));
    return m_type_list->set_type( name, type_int.get());
}

mi::Sint32 Type_list::add_type( const char* name, const mi::neuraylib::IType* type)
{
    mi::base::Handle<const MDL::IType> type_int( get_internal_type( type));
    return m_type_list->add_type( name, type_int.get());
}

const MDL::IType_list* Type_list::get_internal_type_list() const
{
    m_type_list->retain();
    return m_type_list.get();
}

const mi::neuraylib::IType_alias* Type_factory::create_alias(
    const mi::neuraylib::IType* type, mi::Uint32 modifiers, const char* symbol) const
{
    mi::base::Handle<const MDL::IType> type_int( get_internal_type( type));
    mi::Uint32 modifiers_int = ext_modifiers_to_int_modifiers( modifiers);
    mi::base::Handle<const MDL::IType_alias> result_int(
        m_tf->create_alias( type_int.get(), modifiers_int, symbol));
    return create<mi::neuraylib::IType_alias>( result_int.get(), /*owner*/ nullptr);
}

const mi::neuraylib::IType_bool* Type_factory::create_bool() const
{
    mi::base::Handle<const MDL::IType_bool> result_int( m_tf->create_bool());
    return create<mi::neuraylib::IType_bool>( result_int.get(), /*owner*/ nullptr);
}

const mi::neuraylib::IType_int* Type_factory::create_int() const
{
    mi::base::Handle<const MDL::IType_int> result_int( m_tf->create_int());
    return create<mi::neuraylib::IType_int>( result_int.get(), /*owner*/ nullptr);
}

const mi::neuraylib::IType_enum* Type_factory::create_enum( const char* symbol) const
{
    mi::base::Handle<const MDL::IType_enum> result_int( m_tf->create_enum( symbol));
    return create<mi::neuraylib::IType_enum>( result_int.get(), /*owner*/ nullptr);
}

const mi::neuraylib::IType_float* Type_factory::create_float() const
{
    mi::base::Handle<const MDL::IType_float> result_int( m_tf->create_float());
    return create<mi::neuraylib::IType_float>( result_int.get(), /*owner*/ nullptr);
}

const mi::neuraylib::IType_double* Type_factory::create_double() const
{
    mi::base::Handle<const MDL::IType_double> result_int( m_tf->create_double());
    return create<mi::neuraylib::IType_double>( result_int.get(), /*owner*/ nullptr);
}

const mi::neuraylib::IType_string* Type_factory::create_string() const
{
    mi::base::Handle<const MDL::IType_string> result_int( m_tf->create_string());
    return create<mi::neuraylib::IType_string>( result_int.get(), /*owner*/ nullptr);
}

const mi::neuraylib::IType_vector* Type_factory::create_vector(
    const mi::neuraylib::IType_atomic* element_type, mi::Size size) const
{
    mi::base::Handle<const MDL::IType_atomic> element_type_int(
        get_internal_type<MDL::IType_atomic>( element_type));
    mi::base::Handle<const MDL::IType_vector> result_int(
        m_tf->create_vector( element_type_int.get(), size));
    return create<mi::neuraylib::IType_vector>( result_int.get(), /*owner*/ nullptr);
}

const mi::neuraylib::IType_matrix* Type_factory::create_matrix(
    const mi::neuraylib::IType_vector* column_type, mi::Size columns) const
{
    mi::base::Handle<const MDL::IType_vector> column_type_int(
        get_internal_type<MDL::IType_vector>( column_type));
    mi::base::Handle<const MDL::IType_matrix> result_int(
        m_tf->create_matrix( column_type_int.get(), columns));
    return create<mi::neuraylib::IType_matrix>( result_int.get(), /*owner*/ nullptr);
}

const mi::neuraylib::IType_color* Type_factory::create_color() const
{
    mi::base::Handle<const MDL::IType_color> result_int( m_tf->create_color());
    return create<mi::neuraylib::IType_color>( result_int.get(), /*owner*/ nullptr);
}

const mi::neuraylib::IType_array* Type_factory::create_immediate_sized_array(
    const mi::neuraylib::IType* element_type, mi::Size size) const
{
    mi::base::Handle<const MDL::IType> element_type_int( get_internal_type( element_type));
    mi::base::Handle<const MDL::IType_array> result_int(
        m_tf->create_immediate_sized_array( element_type_int.get(), size));
    return create<mi::neuraylib::IType_array>( result_int.get(), /*owner*/ nullptr);
}

const mi::neuraylib::IType_array* Type_factory::create_deferred_sized_array(
    const mi::neuraylib::IType* element_type, const char* size) const
{
    mi::base::Handle<const MDL::IType> element_type_int( get_internal_type( element_type));
    mi::base::Handle<const MDL::IType_array> result_int(
        m_tf->create_deferred_sized_array( element_type_int.get(), size));
    return create<mi::neuraylib::IType_array>( result_int.get(), /*owner*/ nullptr);
}

const mi::neuraylib::IType_struct* Type_factory::create_struct( const char* symbol) const
{
    mi::base::Handle<const MDL::IType_struct> result_int( m_tf->create_struct( symbol));
    return create<mi::neuraylib::IType_struct>( result_int.get(), /*owner*/ nullptr);
}

const mi::neuraylib::IType_texture* Type_factory::create_texture(
    mi::neuraylib::IType_texture::Shape shape) const
{
    MDL::IType_texture::Shape shape_int = ext_shape_to_int_shape( shape);
    mi::base::Handle<const MDL::IType_texture> result_int( m_tf->create_texture( shape_int));
    return create<mi::neuraylib::IType_texture>( result_int.get(), /*owner*/ nullptr);
}

const mi::neuraylib::IType_light_profile* Type_factory::create_light_profile() const
{
    mi::base::Handle<const MDL::IType_light_profile> result_int( m_tf->create_light_profile());
    return create<mi::neuraylib::IType_light_profile>( result_int.get(), /*owner*/ nullptr);
}

const mi::neuraylib::IType_bsdf_measurement* Type_factory::create_bsdf_measurement() const
{
    mi::base::Handle<const MDL::IType_bsdf_measurement> result_int(
        m_tf->create_bsdf_measurement());
    return create<mi::neuraylib::IType_bsdf_measurement>( result_int.get(), /*owner*/ nullptr);
}

const mi::neuraylib::IType_bsdf* Type_factory::create_bsdf() const
{
    mi::base::Handle<const MDL::IType_bsdf> result_int( m_tf->create_bsdf());
    return create<mi::neuraylib::IType_bsdf>( result_int.get(), /*owner*/ nullptr);
}

const mi::neuraylib::IType_hair_bsdf* Type_factory::create_hair_bsdf() const
{
    mi::base::Handle<const MDL::IType_hair_bsdf> result_int( m_tf->create_hair_bsdf());
    return create<mi::neuraylib::IType_hair_bsdf>( result_int.get(), /*owner*/ nullptr);
}

const mi::neuraylib::IType_edf* Type_factory::create_edf() const
{
    mi::base::Handle<const MDL::IType_edf> result_int( m_tf->create_edf());
    return create<mi::neuraylib::IType_edf>( result_int.get(), /*owner*/ nullptr);
}

const mi::neuraylib::IType_vdf* Type_factory::create_vdf() const
{
    mi::base::Handle<const MDL::IType_vdf> result_int( m_tf->create_vdf());
    return create<mi::neuraylib::IType_vdf>( result_int.get(), /*owner*/ nullptr);
}

mi::neuraylib::IType_list* Type_factory::create_type_list() const
{
    mi::base::Handle<MDL::IType_list> result_int( m_tf->create_type_list());
    return create_type_list( result_int.get(), /*owner*/ nullptr);
}

const mi::neuraylib::IType_enum* Type_factory::get_predefined_enum(
    mi::neuraylib::IType_enum::Predefined_id id) const
{
    MDL::IType_enum::Predefined_id id_int = ext_enum_id_to_int_enum_id( id);
    mi::base::Handle<const MDL::IType_enum> result_int( m_tf->get_predefined_enum( id_int));
    return create<mi::neuraylib::IType_enum>( result_int.get(), /*owner*/ nullptr);
}

const mi::neuraylib::IType_struct* Type_factory::get_predefined_struct(
    mi::neuraylib::IType_struct::Predefined_id id) const
{
    MDL::IType_struct::Predefined_id id_int = ext_struct_id_to_int_struct_id( id);
    mi::base::Handle<const MDL::IType_struct> result_int( m_tf->get_predefined_struct( id_int));
    return create<mi::neuraylib::IType_struct>( result_int.get(), /*owner*/ nullptr);
}

mi::Sint32 Type_factory::compare(
    const mi::neuraylib::IType* lhs, const mi::neuraylib::IType* rhs) const
{
    mi::base::Handle<const MDL::IType> lhs_int( get_internal_type( lhs));
    mi::base::Handle<const MDL::IType> rhs_int( get_internal_type( rhs));
    return m_tf->compare( lhs_int.get(), rhs_int.get());
}

mi::Sint32 Type_factory::compare(
    const mi::neuraylib::IType_list* lhs, const mi::neuraylib::IType_list* rhs) const
{
    mi::base::Handle<const MDL::IType_list> lhs_int( get_internal_type_list( lhs));
    mi::base::Handle<const MDL::IType_list> rhs_int( get_internal_type_list( rhs));
    return m_tf->compare( lhs_int.get(), rhs_int.get());
}

mi::Sint32 Type_factory::is_compatible(
    const mi::neuraylib::IType* src, const mi::neuraylib::IType* dst) const
{
    mi::base::Handle<const MDL::IType> src_int( get_internal_type(src));
    mi::base::Handle<const MDL::IType> dst_int( get_internal_type(dst));
    return m_tf->is_compatible( src_int.get(), dst_int.get());
}

const mi::IString* Type_factory::dump( const mi::neuraylib::IType* type, mi::Size depth) const
{
    mi::base::Handle<const MDL::IType> type_int( get_internal_type( type));
    return m_tf->dump( type_int.get(), depth);
}

const mi::IString* Type_factory::dump( const mi::neuraylib::IType_list* list, mi::Size depth) const
{
    mi::base::Handle<const MDL::IType_list> list_int( get_internal_type_list( list));
    return m_tf->dump( list_int.get(), depth);
}

const mi::neuraylib::IType* Type_factory::create(
    const MDL::IType* type, const mi::base::IInterface* owner) const
{
    if( !type)
        return nullptr;

    MDL::IType::Kind kind = type->get_kind();

    switch( kind) {
        case MDL::IType::TK_BOOL: {
            mi::base::Handle<const MDL::IType_bool> t( type->get_interface<MDL::IType_bool>());
            return new Type_bool( this, t.get(), owner);
        }
        case MDL::IType::TK_ALIAS: {
            mi::base::Handle<const MDL::IType_alias> t( type->get_interface<MDL::IType_alias>());
            return new Type_alias( this, t.get(), owner);
        }
        case MDL::IType::TK_INT: {
            mi::base::Handle<const MDL::IType_int> t( type->get_interface<MDL::IType_int>());
            return new Type_int( this, t.get(), owner);
        }
        case MDL::IType::TK_ENUM: {
            mi::base::Handle<const MDL::IType_enum> t( type->get_interface<MDL::IType_enum>());
            return new Type_enum( this, m_transaction.get(), t.get(), owner);
        }
        case MDL::IType::TK_FLOAT: {
            mi::base::Handle<const MDL::IType_float> t( type->get_interface<MDL::IType_float>());
            return new Type_float( this, t.get(), owner);
        }
        case MDL::IType::TK_DOUBLE: {
            mi::base::Handle<const MDL::IType_double> t( type->get_interface<MDL::IType_double>());
            return new Type_double( this, t.get(), owner);
        }
        case MDL::IType::TK_STRING: {
            mi::base::Handle<const MDL::IType_string> t( type->get_interface<MDL::IType_string>());
            return new Type_string( this, t.get(), owner);
        }
        case MDL::IType::TK_VECTOR: {
            mi::base::Handle<const MDL::IType_vector> t( type->get_interface<MDL::IType_vector>());
            return new Type_vector( this, t.get(), owner);
        }
        case MDL::IType::TK_MATRIX: {
            mi::base::Handle<const MDL::IType_matrix> t( type->get_interface<MDL::IType_matrix>());
            return new Type_matrix( this, t.get(), owner);
        }
        case MDL::IType::TK_COLOR: {
            mi::base::Handle<const MDL::IType_color> t( type->get_interface<MDL::IType_color>());
            return new Type_color( this, t.get(), owner);
        }
        case MDL::IType::TK_ARRAY: {
            mi::base::Handle<const MDL::IType_array> t( type->get_interface<MDL::IType_array>());
            return new Type_array( this, t.get(), owner);
        }
        case MDL::IType::TK_STRUCT: {
            mi::base::Handle<const MDL::IType_struct> t( type->get_interface<MDL::IType_struct>());
            return new Type_struct( this, m_transaction.get(), t.get(), owner);
        }
        case MDL::IType::TK_TEXTURE: {
            mi::base::Handle<const MDL::IType_texture> t(
                type->get_interface<MDL::IType_texture>());
            return new Type_texture( this, t.get(), owner);
        }
        case MDL::IType::TK_LIGHT_PROFILE: {
            mi::base::Handle<const MDL::IType_light_profile> t(
                type->get_interface<MDL::IType_light_profile>());
            return new Type_light_profile( this, t.get(), owner);
        }
        case MDL::IType::TK_BSDF_MEASUREMENT: {
            mi::base::Handle<const MDL::IType_bsdf_measurement> t(
                type->get_interface<MDL::IType_bsdf_measurement>());
            return new Type_bsdf_measurement( this, t.get(), owner);
        }
        case MDL::IType::TK_BSDF: {
            mi::base::Handle<const MDL::IType_bsdf> t( type->get_interface<MDL::IType_bsdf>());
            return new Type_bsdf( this, t.get(), owner);
        }
        case MDL::IType::TK_HAIR_BSDF: {
            mi::base::Handle<const MDL::IType_hair_bsdf> t(
                type->get_interface<MDL::IType_hair_bsdf>());
            return new Type_hair_bsdf(this, t.get(), owner);
        }
        case MDL::IType::TK_EDF: {
            mi::base::Handle<const MDL::IType_edf> t( type->get_interface<MDL::IType_edf>());
            return new Type_edf( this, t.get(), owner);
        }
        case MDL::IType::TK_VDF: {
            mi::base::Handle<const MDL::IType_vdf> t( type->get_interface<MDL::IType_vdf>());
            return new Type_vdf( this, t.get(), owner);
        }
        case MDL::IType::TK_FORCE_32_BIT:
            ASSERT( M_SCENE, false);
            return nullptr;
    };

   ASSERT( M_SCENE, false);
   return nullptr;
}

mi::neuraylib::IType_list* Type_factory::create_type_list(
    MDL::IType_list* type_list, const mi::base::IInterface* owner) const
{
    if( !type_list)
        return nullptr;
    return new Type_list( this, type_list, owner);
}

const mi::neuraylib::IType_list* Type_factory::create_type_list(
    const MDL::IType_list* type_list, const mi::base::IInterface* owner) const
{
    return create_type_list( const_cast<MDL::IType_list*>( type_list), owner);
}

} // namespace NEURAY

} // namespace MI
