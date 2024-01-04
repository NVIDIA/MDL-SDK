/***************************************************************************************************
 * Copyright (c) 2015-2023, NVIDIA CORPORATION. All rights reserved.
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
/// \brief      Source for the IType hierarchy and IType_factory implementation.

#include "pch.h"

#include "mdl_elements_type.h"

#include <mi/neuraylib/istring.h>

#include <cstring>
#include <sstream>

#include <boost/core/ignore_unused.hpp>

#include <base/system/main/access_module.h>
#include <base/system/stlext/i_stlext_likely.h>
#include <base/util/string_utils/i_string_lexicographic_cast.h>
#include <base/lib/log/i_log_logger.h>
#include <base/lib/mem/i_mem_consumption.h>
#include <base/data/serial/i_serializer.h>
#include <mdl/integration/mdlnr/i_mdlnr.h>

#include "mdl_elements_utilities.h"

namespace MI {

namespace MDL {

namespace TYPES {

template <class I>
class Interface_implement_singleton : public I
{
public:
    Interface_implement_singleton<I>& operator=( const Interface_implement_singleton<I>& other)
    {
        // Note: no call of operator= on m_refcount
        // avoid warning
        (void) other;
        return *this;
    }

    /// Always returns 1.
    virtual mi::Uint32 retain() const final { return 1; }

    /// Always returns 1.
    virtual mi::Uint32 release() const final { return 1; }

    virtual const mi::base::IInterface* get_interface( const mi::base::Uuid& interface_id) const
    {
        return I::get_interface_static( this, interface_id);
    }

    virtual mi::base::IInterface* get_interface( const mi::base::Uuid& interface_id)
    {
        return I::get_interface_static( this, interface_id);
    }

    mi::base::Uuid get_iid() const
    {
        return typename I::IID();
    }

protected:
    virtual ~Interface_implement_singleton() {}
};


template <class T>
class Type_base : public mi::base::Interface_implement<T>
{
public:
    IType::Kind get_kind() const final { return T::s_kind; }

    mi::Uint32 get_all_type_modifiers() const override { return 0; }

    const IType* skip_all_type_aliases() const override
    {
        this->retain();
        return this;
    }
};


template <class T>
class Type_base_immutable : public Interface_implement_singleton<T>
{
public:
    IType::Kind get_kind() const final { return T::s_kind; }

    mi::Uint32 get_all_type_modifiers() const final { return 0; }

    const IType* skip_all_type_aliases() const final
    {
        this->retain();
        return this;
    }

    mi::Size get_memory_consumption() const final { return 0; }
};


class Type_alias : public Type_base<IType_alias>
{
public:
    Type_alias(
        const IType* aliased_type, mi::Sint32 modifiers, const char* symbol)
      : m_aliased_type( aliased_type, mi::base::DUP_INTERFACE),
        m_modifiers( modifiers),
        m_symbol( symbol ? symbol : "")
    {
    }

    mi::Uint32 get_all_type_modifiers() const final
    {
        mi::base::Handle<const IType> aliased_type( this->get_aliased_type());
        return this->get_type_modifiers() | aliased_type->get_all_type_modifiers();
    }

    const IType* skip_all_type_aliases() const final
    {
        mi::base::Handle<const IType> aliased_type( this->get_aliased_type());
        return aliased_type->skip_all_type_aliases();
    }

    const IType* get_aliased_type() const final
    {
        if( !m_aliased_type)
            return nullptr;
        m_aliased_type->retain();
        return m_aliased_type.get();
    }

    mi::Uint32 get_type_modifiers() const final { return m_modifiers; }

    const char* get_symbol() const final
    {
        return m_symbol.empty() ? nullptr : m_symbol.c_str();
    }

    mi::Size get_memory_consumption() const final
    {
        return sizeof( *this)
            + dynamic_memory_consumption( m_aliased_type)
            + dynamic_memory_consumption( m_symbol);
    }

private:
    const mi::base::Handle<const IType> m_aliased_type;
    const mi::Sint32 m_modifiers;
    const std::string m_symbol;
};


class Type_bool final : public Type_base_immutable<IType_bool>
{
public:
    // user defined default constructor required to be const-default-constructible
    Type_bool() {}
};


class Type_int final : public Type_base_immutable<IType_int>
{
public:
    // user defined default constructor required to be const-default-constructible
    Type_int() {}
};


class Type_enum final : public Type_base<IType_enum>
{
public:
    Type_enum(
        Type_factory* owner,
        const char* symbol,
        IType_enum::Predefined_id id,
        const IType_enum::Values& values,
        const mi::base::Handle<const IAnnotation_block>& annotations,
        const IType_enum::Value_annotations& value_annotations)
      : m_owner( owner, mi::base::DUP_INTERFACE),
        m_symbol( symbol),
        m_predefined_id( id),
        m_values( values),
        m_annotations( annotations),
        m_value_annotations( value_annotations)
    {
        ASSERT( M_SCENE,
            m_value_annotations.size() == m_values.size() || m_value_annotations.empty());
    }

    const char* get_symbol() const final { return m_symbol.c_str(); }

    mi::Size get_size() const final { return m_values.size(); }

    const char* get_value_name( mi::Size index) const final
    {
        if( index >= m_values.size())
            return nullptr;
        return m_values[index].first.c_str();
    }

    mi::Sint32 get_value_code( mi::Size index, mi::Sint32* errors) const final
    {
        if( index >= m_values.size()) {
            if( errors) *errors = -1;
            return 0;
        }

        if( errors) *errors = 0;
        return m_values[index].second;
    }

    mi::Size find_value( const char* name) const final
    {
        if( !name)
            return static_cast<mi::Size>( -1);

        for( mi::Size i = 0; i < m_values.size(); ++i)
            if( m_values[i].first == name)
                return i;

        return static_cast<mi::Size>( -1);
    }

    mi::Size find_value( mi::Sint32 code) const final
    {
        for( mi::Size i = 0, n = m_values.size(); i < n; ++i)
            if( m_values[i].second == code)
                return i;

        return static_cast<mi::Size>( -1);
    }

    IType_enum::Predefined_id get_predefined_id() const final
    { return m_predefined_id; }

    const IAnnotation_block* get_annotations() const final
    {
        if( !m_annotations)
            return nullptr;

        m_annotations->retain();
        return m_annotations.get();
    }

    const IAnnotation_block* get_value_annotations( mi::Size index) const final
    {
        if( index >= m_value_annotations.size() || !m_value_annotations[index])
            return nullptr;

        m_value_annotations[index]->retain();
        return m_value_annotations[index].get();
    }

    mi::Size get_memory_consumption() const final
    {
        return sizeof( *this)
            + dynamic_memory_consumption( m_symbol)
            + dynamic_memory_consumption( m_values);
    }

    mi::Uint32 release() const final
    {
        mi::Uint32 count = --refcount();
        if( count > 0)
            return count;

        // TODO MDL-957 The remainder of this method is not thread-safe.
        std::unique_lock<Type_factory> lock( *m_owner.get());
        count = refcount();
        if( count > 0)
            return count;

        m_owner->unregister_enum_type( this);

        lock.unlock();
        delete this;
        return 0;
    }

private:
    mutable mi::base::Handle<Type_factory> m_owner;
    const std::string m_symbol;
    const IType_enum::Predefined_id m_predefined_id;
    const IType_enum::Values m_values;
    const mi::base::Handle<const IAnnotation_block> m_annotations;
    const IType_enum::Value_annotations m_value_annotations;
};


class Type_float final : public Type_base_immutable<IType_float>
{
public:
    // user defined default constructor required to be const-default-constructible
    Type_float() {}
};


class Type_double final : public Type_base_immutable<IType_double>
{
public:
    // user defined default constructor required to be const-default-constructible
    Type_double() {}
};


class Type_string final : public Type_base_immutable<IType_string>
{
public:
    // user defined default constructor required to be const-default-constructible
    Type_string() {}
};


class Type_vector final : public Type_base_immutable<IType_vector>
{
public:
    Type_vector( const IType_atomic* element_type, mi::Size size)
      : m_element_type( element_type, mi::base::DUP_INTERFACE),
        m_size( size)
    {
    }

    const IType* get_component_type( mi::Size index) const final
    {
        if( index < m_size)
            return get_element_type();
        return nullptr;
    }

    mi::Size get_size() const final { return m_size; }

    const IType_atomic* get_element_type() const final
    {
        m_element_type->retain();
        return m_element_type.get();
    }

private:
    const mi::base::Handle<const IType_atomic> m_element_type;
    const mi::Size m_size;
};


class Type_matrix final : public Type_base_immutable<IType_matrix>
{
public:
    Type_matrix( const IType_vector* element_type, mi::Size columns)
      : m_element_type( element_type, mi::base::DUP_INTERFACE),
       m_columns( columns)
    {
    }

    const IType* get_component_type( mi::Size index) const final
    {
        if( index < m_columns)
            return get_element_type();
        return nullptr;
    }

    mi::Size get_size() const final { return m_columns; }

    const IType_vector* get_element_type() const final
    {
        m_element_type->retain();
        return m_element_type.get();
    }

private:
    const mi::base::Handle<const IType_vector> m_element_type;
    const mi::Size m_columns;
};


class Type_color final : public Type_base_immutable<IType_color>
{
public:
    Type_color( const IType_float* component_type)
      : m_component_type( component_type, mi::base::DUP_INTERFACE)
    {
    }

    const IType* get_component_type( mi::Size index) const final
    {
        if( index >= s_compound_size)
            return nullptr;
        m_component_type->retain();
        return m_component_type.get();
    }

    mi::Size get_size() const final { return s_compound_size; }

private:
    const mi::base::Handle<const IType_float> m_component_type;
    static const mi::Size s_compound_size = 3;
};


class Type_array final : public Type_base<IType_array>
{
public:
    Type_array(
        const IType* element_type, mi::Size size)
      : m_element_type( element_type, mi::base::DUP_INTERFACE),
        m_immediate_sized( true),
        m_immediate_size( size),
        m_deferred_size()
    {
    }

    Type_array(
        const IType* element_type, const char* deferred_size)
      : m_element_type( element_type, mi::base::DUP_INTERFACE),
        m_immediate_sized( false),
        m_immediate_size( -1),
        m_deferred_size( deferred_size)
    {
    }

    const IType* get_component_type( mi::Size index) const final
    {
        if( m_immediate_sized && index >= m_immediate_size)
            return nullptr;
        return get_element_type();
    }

    mi::Size get_size() const final { return m_immediate_sized ? m_immediate_size : -1; }

    const IType* get_element_type() const final
    {
        m_element_type->retain();
        return m_element_type.get();
    }

    bool is_immediate_sized() const final { return m_immediate_sized; }

    const char* get_deferred_size() const final
    {
        return m_immediate_sized ? nullptr : m_deferred_size.c_str();
    }

    mi::Size get_memory_consumption() const final
    {
        return sizeof( *this)
            + dynamic_memory_consumption( m_element_type)
            + dynamic_memory_consumption( m_deferred_size);
    }

private:
    const mi::base::Handle<const IType> m_element_type;
    const bool m_immediate_sized;
    const mi::Size m_immediate_size;
    const std::string m_deferred_size;
};


class Type_struct final : public Type_base<IType_struct>
{
    using Base = Type_base<IType_struct>;

public:
    Type_struct(
        Type_factory *owner,
        const char* symbol,
        IType_struct::Predefined_id id,
        const IType_struct::Fields& fields,
        const mi::base::Handle<const IAnnotation_block>& annotations,
        const IType_struct::Field_annotations& field_annotations)
      : m_owner( owner, mi::base::DUP_INTERFACE),
        m_symbol( symbol),
        m_predefined_id( id),
        m_fields( fields),
        m_annotations( annotations),
        m_field_annotations( field_annotations)
    {
        ASSERT( M_SCENE,
            m_field_annotations.size() == m_fields.size() || m_field_annotations.empty());
    }

    const IType* get_component_type( mi::Size index) const final
    {
        return get_field_type( index);
    }

    mi::Size get_size() const final { return m_fields.size(); }

    const char* get_symbol() const final { return m_symbol.c_str(); }

    const IType* get_field_type( mi::Size index) const final
    {
        if( index >= m_fields.size())
            return nullptr;
        m_fields[index].first->retain();
        return m_fields[index].first.get();
    }

    const char* get_field_name( mi::Size index) const final
    {
        if( index >= m_fields.size())
            return nullptr;
        return m_fields[index].second.c_str();
    }

    mi::Size find_field( const char* name) const final
    {
        if( !name)
            return -1;

        for( mi::Size i = 0, n = m_fields.size(); i < n; ++i)
            if( m_fields[i].second == name)
                return i;

        return -1;
    }

    IType_struct::Predefined_id get_predefined_id() const final { return m_predefined_id; }

    const IAnnotation_block* get_annotations() const final
    {
        if( !m_annotations)
            return nullptr;

        m_annotations->retain();
        return m_annotations.get();
    }

    const IAnnotation_block* get_field_annotations( mi::Size index) const final
    {
        if( index >= m_field_annotations.size() || !m_field_annotations[index])
            return nullptr;

        m_field_annotations[index]->retain();
        return m_field_annotations[index].get();
    }

    mi::Size get_memory_consumption() const final
    {
        return sizeof( *this)
            + dynamic_memory_consumption( m_symbol)
            + dynamic_memory_consumption( m_fields);
    }

    mi::Uint32 release() const final
    {
        mi::Uint32 count = --refcount();
        if( count > 0)
            return count;

        // TODO MDL-957 The remainder of this method is not thread-safe.
        std::unique_lock<Type_factory> lock( *m_owner.get());
        count = refcount();
        if( count > 0)
            return count;

        m_owner->unregister_struct_type( this);

        lock.unlock();
        delete this;
        return 0;
    }

private:
    mutable mi::base::Handle<Type_factory> m_owner;
    const std::string m_symbol;
    const IType_struct::Predefined_id m_predefined_id;
    const IType_struct::Fields m_fields;
    const mi::base::Handle<const IAnnotation_block> m_annotations;
    const IType_struct::Field_annotations m_field_annotations;
};


class Type_texture final : public Type_base_immutable<IType_texture>
{
public:
    Type_texture( IType_texture::Shape shape)
    : m_shape( shape)
    {
    }

    IType_texture::Shape get_shape() const final { return m_shape; }

private:
    const IType_texture::Shape m_shape;
};


class Type_light_profile final : public Type_base_immutable<IType_light_profile>
{
public:
    // user defined default constructor required to be const-default-constructible
    Type_light_profile() {}
};


class Type_bsdf_measurement final : public Type_base_immutable<IType_bsdf_measurement>
{
public:
    // user defined default constructor required to be const-default-constructible
    Type_bsdf_measurement() {}
};


class Type_bsdf final : public Type_base_immutable<IType_bsdf>
{
public:
    // user defined default constructor required to be const-default-constructible
    Type_bsdf() {}
};


class Type_hair_bsdf final : public Type_base_immutable<IType_hair_bsdf>
{
public:
    // user defined default constructor required to be const-default-constructible
    Type_hair_bsdf() {}
};


class Type_edf final : public Type_base_immutable<IType_edf>
{
public:
    // user defined default constructor required to be const-default-constructible
    Type_edf() {}
};


class Type_vdf final : public Type_base_immutable<IType_vdf>
{
public:
    // user defined default constructor required to be const-default-constructible
    Type_vdf() {}
};


static const Type_bool             the_bool_type;
static const Type_int              the_int_type;
static const Type_float            the_float_type;
static const Type_double           the_double_type;
static const Type_color            the_color_type( &the_float_type);
static const Type_string           the_string_type;
static const Type_texture          the_texture_2d_type( IType_texture::TS_2D);
static const Type_texture          the_texture_3d_type( IType_texture::TS_3D);
static const Type_texture          the_texture_cube_type( IType_texture::TS_CUBE);
static const Type_texture          the_texture_ptex_type( IType_texture::TS_PTEX);
static const Type_texture          the_texture_bsdf_data_type( IType_texture::TS_BSDF_DATA);
static const Type_bsdf             the_bsdf_type;
static const Type_hair_bsdf        the_hair_bsdf_type;
static const Type_vdf              the_vdf_type;
static const Type_edf              the_edf_type;
static const Type_light_profile    the_light_profile_type;
static const Type_bsdf_measurement the_bsdf_measurement_type;

static const Type_vector           the_bool_2_type( &the_bool_type, 2);
static const Type_vector           the_bool_3_type( &the_bool_type, 3);
static const Type_vector           the_bool_4_type( &the_bool_type, 4);

static const Type_vector           the_int_2_type( &the_int_type, 2);
static const Type_vector           the_int_3_type( &the_int_type, 3);
static const Type_vector           the_int_4_type( &the_int_type, 4);

static const Type_vector           the_float_2_type( &the_float_type, 2);
static const Type_vector           the_float_3_type( &the_float_type, 3);
static const Type_vector           the_float_4_type( &the_float_type, 4);

static const Type_vector          the_double_2_type( &the_double_type, 2);
static const Type_vector          the_double_3_type( &the_double_type, 3);
static const Type_vector          the_double_4_type( &the_double_type, 4);

static const Type_matrix          the_float_2x2_type( &the_float_2_type, 2);
static const Type_matrix          the_float_2x3_type( &the_float_2_type, 3);
static const Type_matrix          the_float_2x4_type( &the_float_2_type, 4);

static const Type_matrix          the_float_3x2_type( &the_float_3_type, 2);
static const Type_matrix          the_float_3x3_type( &the_float_3_type, 3);
static const Type_matrix          the_float_3x4_type( &the_float_3_type, 4);

static const Type_matrix          the_float_4x2_type( &the_float_4_type, 2);
static const Type_matrix          the_float_4x3_type( &the_float_4_type, 3);
static const Type_matrix          the_float_4x4_type( &the_float_4_type, 4);

static const Type_matrix          the_double_2x2_type( &the_double_2_type, 2);
static const Type_matrix          the_double_2x3_type( &the_double_2_type, 3);
static const Type_matrix          the_double_2x4_type( &the_double_2_type, 4);

static const Type_matrix          the_double_3x2_type( &the_double_3_type, 2);
static const Type_matrix          the_double_3x3_type( &the_double_3_type, 3);
static const Type_matrix          the_double_3x4_type( &the_double_3_type, 4);

static const Type_matrix          the_double_4x2_type( &the_double_4_type, 2);
static const Type_matrix          the_double_4x3_type( &the_double_4_type, 3);
static const Type_matrix          the_double_4x4_type( &the_double_4_type, 4);

}  // TYPES


Type_list::Type_list( mi::Size initial_capacity)
{
    m_index_name.reserve( initial_capacity);
    m_types.reserve( initial_capacity);
}

mi::Size Type_list::get_size() const
{
    return m_types.size();
}

mi::Size Type_list::get_index( const char* name) const
{
    if( !name)
        return static_cast<mi::Size>( -1);

    // For typical list sizes a linear search is much faster than maintaining a map from names to
    // indices.
    for( mi::Size i = 0; i < m_index_name.size(); ++i)
        if( m_index_name[i] == name)
            return i;

    return static_cast<mi::Size>( -1);
}

const char* Type_list::get_name( mi::Size index) const
{
    if( index >= m_index_name.size())
        return nullptr;
    return m_index_name[index].c_str();
}

const IType* Type_list::get_type( mi::Size index) const
{
    if( index >= m_types.size())
        return nullptr;
    m_types[index]->retain();
    return m_types[index].get();
}

const IType* Type_list::get_type( const char* name) const
{
    if( !name)
        return nullptr;
    mi::Size index = get_index( name);
    if( index == static_cast<mi::Size>( -1))
        return nullptr;
    return get_type( index);
}

mi::Sint32 Type_list::set_type( mi::Size index, const IType* type)
{
    if( !type)
        return -1;
    if( index >= m_types.size())
        return -2;
    m_types[index] = make_handle_dup( type);
    return 0;
}

mi::Sint32 Type_list::set_type( const char* name, const IType* type)
{
    if( !name || !type)
        return -1;
    mi::Size index = get_index( name);
    if( index == static_cast<mi::Size>( -1))
        return -2;
    m_types[index] = make_handle_dup( type);
    return 0;
}

mi::Sint32 Type_list::add_type( const char* name, const IType* type)
{
    if( !name || !type)
        return -1;
    mi::Size index = get_index( name);
    if( index != static_cast<mi::Size>( -1))
        return -2;
    m_types.push_back( make_handle_dup( type));
    m_index_name.push_back( name);
    return 0;
}

void Type_list::add_type_unchecked( const char* name, const IType* type)
{
    m_types.push_back( make_handle_dup( type));
    m_index_name.push_back( name);
}

mi::Size Type_list::get_memory_consumption() const
{
    return sizeof( *this)
        + dynamic_memory_consumption( m_index_name)
        + dynamic_memory_consumption( m_types);
}


const IType_alias* Type_factory::create_alias(
    const IType* type, mi::Uint32 modifiers, const char* symbol) const
{
    return type ? new TYPES::Type_alias( type, modifiers, symbol) : nullptr;
}

const IType_bool* Type_factory::create_bool() const
{
    return &TYPES::the_bool_type;
}

const IType_int* Type_factory::create_int() const
{
    return &TYPES::the_int_type;
}

const IType_enum* Type_factory::create_enum( const char* symbol) const
{
    if( !symbol)
        return nullptr;

    std::shared_lock<std::shared_mutex> lock( m_mutex);

    Weak_enum_symbol_map::const_iterator it = m_enum_symbols.find( symbol);
    if( it == m_enum_symbols.end())
        return nullptr;

    it->second->retain();
    return it->second;
}

const IType_float* Type_factory::create_float() const { return &TYPES::the_float_type; }

const IType_double* Type_factory::create_double() const { return &TYPES::the_double_type; }

const IType_string* Type_factory::create_string() const { return &TYPES::the_string_type; }

const IType_vector* Type_factory::create_vector(
    const IType_atomic* element_type, mi::Size size) const
{
    if( !element_type || size < 2 || size > 4)
        return nullptr;

    IType::Kind element_kind = element_type->get_kind();
    switch( element_kind) {
        case IType::TK_BOOL:
            switch( size) {
                case 2: return &TYPES::the_bool_2_type;
                case 3: return &TYPES::the_bool_3_type;
                case 4: return &TYPES::the_bool_4_type;
            }
            break;
        case IType::TK_INT:
            switch( size) {
                case 2: return &TYPES::the_int_2_type;
                case 3: return &TYPES::the_int_3_type;
                case 4: return &TYPES::the_int_4_type;
            }
            break;
        case IType::TK_FLOAT:
            switch( size) {
                case 2: return &TYPES::the_float_2_type;
                case 3: return &TYPES::the_float_3_type;
                case 4: return &TYPES::the_float_4_type;
            }
            break;
        case IType::TK_DOUBLE:
            switch( size) {
                case 2: return &TYPES::the_double_2_type;
                case 3: return &TYPES::the_double_3_type;
                case 4: return &TYPES::the_double_4_type;
            }
            break;
        default:
            break;
    }
    return nullptr;
}

const IType_matrix* Type_factory::create_matrix(
    const IType_vector* column_type, mi::Size columns) const
{
    if( !column_type || columns < 2 || columns > 4)
        return nullptr;

    mi::base::Handle<const IType> element_type(
        column_type->get_element_type());
    IType::Kind element_kind = element_type->get_kind();
    switch( element_kind) {
        case IType::TK_FLOAT:
            switch( column_type->get_size()) {
                case 2:
                    switch( columns) {
                        case 2: return &TYPES::the_float_2x2_type;
                        case 3: return &TYPES::the_float_2x3_type;
                        case 4: return &TYPES::the_float_2x4_type;
                    }
                    break;
                case 3:
                    switch( columns) {
                        case 2: return &TYPES::the_float_3x2_type;
                        case 3: return &TYPES::the_float_3x3_type;
                        case 4: return &TYPES::the_float_3x4_type;
                    }
                    break;
                case 4:
                    switch( columns) {
                        case 2: return &TYPES::the_float_4x2_type;
                        case 3: return &TYPES::the_float_4x3_type;
                        case 4: return &TYPES::the_float_4x4_type;
                    }
                    break;
            }
            break;
        case IType::TK_DOUBLE:
            switch( column_type->get_size()) {
                case 2:
                    switch( columns) {
                        case 2: return &TYPES::the_double_2x2_type;
                        case 3: return &TYPES::the_double_2x3_type;
                        case 4: return &TYPES::the_double_2x4_type;
                    }
                    break;
                case 3:
                    switch( columns) {
                        case 2: return &TYPES::the_double_3x2_type;
                        case 3: return &TYPES::the_double_3x3_type;
                        case 4: return &TYPES::the_double_3x4_type;
                    }
                    break;
                case 4:
                    switch( columns) {
                        case 2: return &TYPES::the_double_4x2_type;
                        case 3: return &TYPES::the_double_4x3_type;
                        case 4: return &TYPES::the_double_4x4_type;
                    }
                    break;
            }
            break;
        default:
            break;
    }
    return nullptr;
}

const IType_color* Type_factory::create_color() const
{
    return &TYPES::the_color_type;
}

const IType_array* Type_factory::create_immediate_sized_array(
    const IType* element_type, mi::Size size) const
{
    return element_type ? new TYPES::Type_array( element_type, size) : nullptr;
}

const IType_array* Type_factory::create_deferred_sized_array(
    const IType* element_type, const char* size) const
{
    if( !element_type || !size)
        return nullptr;

    SYSTEM::Access_module<MDLC::Mdlc_module> mdlc_module( false);
    mi::base::Handle<mi::mdl::IMDL> mdl( mdlc_module->get_mdl());
    if( !mdl->is_valid_mdl_identifier( size))
        return nullptr;

    return new TYPES::Type_array( element_type, size);
}

const IType_struct* Type_factory::create_struct( const char* symbol) const
{
    if( !symbol)
        return nullptr;

    std::shared_lock<std::shared_mutex> lock( m_mutex);

    Weak_struct_symbol_map::const_iterator it = m_struct_symbols.find( symbol);
    if( it == m_struct_symbols.end())
        return nullptr;

    it->second->retain();
    return it->second;
}

const IType_texture* Type_factory::create_texture(
    IType_texture::Shape shape) const
{
    switch( shape) {
    case IType_texture::TS_2D:        return &TYPES::the_texture_2d_type;
    case IType_texture::TS_3D:        return &TYPES::the_texture_3d_type;
    case IType_texture::TS_CUBE:      return &TYPES::the_texture_cube_type;
    case IType_texture::TS_PTEX:      return &TYPES::the_texture_ptex_type;
    case IType_texture::TS_BSDF_DATA: return &TYPES::the_texture_bsdf_data_type;
    case IType_texture::TS_FORCE_32_BIT:
        break;
    }
    return nullptr;
}

const IType_light_profile* Type_factory::create_light_profile() const
{
    return &TYPES::the_light_profile_type;
}

const IType_bsdf_measurement* Type_factory::create_bsdf_measurement() const
{
    return &TYPES::the_bsdf_measurement_type;
}

const IType_bsdf* Type_factory::create_bsdf() const
{
    return &TYPES::the_bsdf_type;
}

const IType_hair_bsdf* Type_factory::create_hair_bsdf() const
{
    return &TYPES::the_hair_bsdf_type;
}

const IType_edf* Type_factory::create_edf() const
{
    return &TYPES::the_edf_type;
}

const IType_vdf* Type_factory::create_vdf() const
{
    return &TYPES::the_vdf_type;
}

IType_list* Type_factory::create_type_list( mi::Size initial_capacity) const
{
    return new Type_list( initial_capacity);
}

const IType_enum* Type_factory::get_predefined_enum(
    IType_enum::Predefined_id id) const
{
    std::shared_lock<std::shared_mutex> lock( m_mutex);

    Weak_enum_id_map::const_iterator it = m_enum_ids.find( id);
    if( it == m_enum_ids.end())
        return nullptr;

    it->second->retain();
    return it->second;
}

const IType_struct* Type_factory::get_predefined_struct(
    IType_struct::Predefined_id id) const
{
    std::shared_lock<std::shared_mutex> lock( m_mutex);

    Weak_struct_id_map::const_iterator it = m_struct_ids.find( id);
    if( it == m_struct_ids.end())
        return nullptr;

    it->second->retain();
    return it->second;
}

IType_list* Type_factory::clone( const IType_list* list) const
{
    if( !list)
        return nullptr;

    mi::Size n = list->get_size();
    IType_list* result = create_type_list( n);
    for( mi::Size i = 0; i < n; ++i) {
        mi::base::Handle<const IType> type( list->get_type( i));
        const char* name = list->get_name( i);
        result->add_type_unchecked( name, type.get());
    }
    return result;
}

mi::Sint32 Type_factory::compare( const IType* lhs, const IType* rhs) const
{
    return compare_static( lhs, rhs);
}

mi::Sint32 Type_factory::compare( const IType_list* lhs, const IType_list* rhs) const
{
    return compare_static( lhs, rhs);
}

mi::Sint32 Type_factory::is_compatible( const IType* src, const IType* dst) const
{
    if( !src || !dst)
        return -1;

    if( compare_static( src, dst) == 0)
        return 1;

    IType::Kind src_kind = src->get_kind();
    IType::Kind dst_kind = dst->get_kind();

    if( src_kind == IType::TK_STRUCT && dst_kind == IType::TK_STRUCT) {

        mi::base::Handle<const IType_struct> src_str( src->get_interface<IType_struct>());
        mi::base::Handle<const IType_struct> dst_str( dst->get_interface<IType_struct>());

        // For compatibility, all field types need to be pairwise compatible.

        if( src_str->get_size() != dst_str->get_size())
            return -1;

        for( mi::Size i = 0, n = src_str->get_size(); i < n; ++i) {
            mi::base::Handle<const IType> src_field_type( src_str->get_field_type( i));
            mi::base::Handle<const IType> dst_field_type( dst_str->get_field_type( i));
            if( is_compatible( src_field_type.get(), dst_field_type.get()) < 0)
                return -1;
        }
        return 0;

    } else if( src_kind == IType::TK_ENUM && dst_kind == IType::TK_ENUM) {

        mi::base::Handle<const IType_enum> src_e( src->get_interface<IType_enum>());
        mi::base::Handle<const IType_enum> dst_e( dst->get_interface<IType_enum>());

        // For compatibility, the sets of value codes need to be equal.

        std::set<mi::Uint32> src_codes;
        for( mi::Size i = 0, n = src_e->get_size(); i < n; ++i)
            src_codes.insert( src_e->get_value_code( i));
        std::set<mi::Uint32> dst_codes;
        for( mi::Size i = 0, n = dst_e->get_size(); i < n; ++i)
            dst_codes.insert( dst_e->get_value_code( i));
        return src_codes == dst_codes ? 0 : -1;

    } else if( src_kind == IType::TK_ARRAY && dst_kind == IType::TK_ARRAY) {

        mi::base::Handle<const IType_array> src_a( src->get_interface<IType_array>());
        mi::base::Handle<const IType_array> dst_a( dst->get_interface<IType_array>());

        // For compatibility, both need to be either immediate-sized or deferred-sized,
        // sizes need to match and element types need to be compatible.

        if( src_a->get_size() != dst_a->get_size())
            return -1;

        mi::base::Handle<const IType> src_a_elem_type( src_a->get_element_type());
        mi::base::Handle<const IType> dst_a_elem_type( dst_a->get_element_type());
        return is_compatible( src_a_elem_type.get(), dst_a_elem_type.get());
    }

    return -1;
}

namespace {

std::string get_prefix( mi::Size depth)
{
    std::string prefix;
    for( mi::Size i = 0; i < depth; ++i)
        prefix += "    ";
    return prefix;
}

} // namespace

const mi::IString* Type_factory::dump( const IType* type, mi::Size depth) const
{
    std::ostringstream s;
    dump( type, depth, s);
    return create_istring( s.str().c_str());
}

const mi::IString* Type_factory::dump( const IType_list* list, mi::Size depth) const
{
    std::ostringstream s;
    dump( list, depth, s);
    return create_istring( s.str().c_str());
}

std::string Type_factory::get_mdl_type_name( const IType* type) const
{
    return get_mdl_type_name_static( type);
}

namespace {

/// Converts a vector or matrix size from char to int. Returns 0 in case of errors.
mi::Size get_size( char c)
{
    switch( c) {
        case '2': return 2;
        case '3': return 3;
        case '4': return 4;
        default:  return 0;
    }
}

} // namespace

const IType* Type_factory::create_from_mdl_type_name( const char* name) const
{
    if( !name)
        return nullptr;

    std::string s = name;
    if( s.empty())
        return nullptr;

    // IType_alias
    // TODO support named aliases
    if( s.substr( 0, 8) == "uniform ") {
        mi::base::Handle<const IType> aliased_type( create_from_mdl_type_name( s.c_str() + 8));
        if( !aliased_type)
            return nullptr;
        return create_alias( aliased_type.get(), IType::MK_UNIFORM, nullptr);
    }
    if( s.substr( 0, 8) == "varying ") {
        mi::base::Handle<const IType> aliased_type( create_from_mdl_type_name( s.c_str() + 8));
        if( !aliased_type)
            return nullptr;
        return create_alias( aliased_type.get(), IType::MK_VARYING, nullptr);
    }

    // IType_array
    size_t n = s.size();
    if( s[n-1] == ']') {
        size_t left_bracket = s.rfind( '[');
        if( left_bracket == std::string::npos)
            return nullptr;
        std::string element_type_name = s.substr( 0, left_bracket);
        mi::base::Handle<const IType> element_type(
            create_from_mdl_type_name( element_type_name.c_str()));
        if( !element_type)
            return nullptr;
        std::string size_name = s.substr( left_bracket+1, n-1-left_bracket-1);
        STLEXT::Likely<mi::Size> size_likely = STRING::lexicographic_cast_s<mi::Size>( size_name);
        if( size_likely.get_status())
            return create_immediate_sized_array( element_type.get(), *size_likely.get_ptr());
        else
            return create_deferred_sized_array( element_type.get(), size_name.c_str());
    }

    // IType_atomic without IType_enum, and IType_color
    if( s == "bool")
         return create_bool();
    if( s == "int")
         return create_int();
    if( s == "float")
         return create_float();
    if( s == "double")
         return create_double();
    if( s == "color")
         return create_color();
    if( s == "string")
         return create_string();

    // IType_resource
    if( s == "texture_2d")
        return create_texture( IType_texture::TS_2D);
    if( s == "texture_3d")
        return create_texture( IType_texture::TS_3D);
    if( s == "texture_cube")
        return create_texture( IType_texture::TS_CUBE);
    if( s == "texture_ptex")
        return create_texture( IType_texture::TS_PTEX);
    if( s == "light_profile")
        return create_light_profile();
    if( s == "bsdf_measurement")
        return create_bsdf_measurement();

    // IType_df
    if( s == "bsdf")
        return create_bsdf();
    if( s == "hair_bsdf")
        return create_hair_bsdf();
    if( s == "edf")
        return create_edf();
    if( s == "vdf")
        return create_vdf();

    // IType_vector and IType_matrix
    const char* element_type_names[] =  { "bool", "int", "float", "double" };
    for( const char* element_type_name: element_type_names) {

        size_t m = strlen( element_type_name);
        if( strncmp( s.c_str(), element_type_name, m) != 0)
            continue;

        // IType_vector
        if( n == m+1) {
            mi::Size size = get_size( s[m]);
            if( size == 0)
                 return nullptr;
            mi::base::Handle<const IType> element_type(
                create_from_mdl_type_name( element_type_name));
            mi::base::Handle<const IType_atomic> element_type_atomic(
                element_type->get_interface<IType_atomic>());
            return create_vector( element_type_atomic.get(), size);
        }

        // IType_matrix
        if( n == m+3) {
            // element types "bool" and "int" are not allowed for matrices
            if( m == 4 || m == 3)
                return nullptr;
            mi::Size columns = get_size( s[m]);
            mi::Size rows    = get_size( s[m+2]);
            if( columns == 0 || rows == 0 || s[m+1] != 'x')
                 return nullptr;
            mi::base::Handle<const IType> element_type(
                create_from_mdl_type_name( element_type_name));
            mi::base::Handle<const IType_atomic> element_type_atomic(
                element_type->get_interface<IType_atomic>());
            mi::base::Handle<const IType_vector> vector_type(
                create_vector( element_type_atomic.get(), rows));
            return create_matrix( vector_type.get(), columns);
        }
    }

    // IType_enum and IType_struct
    s = prefix_builtin_type_name( s.c_str());
    const IType_enum* type_enum = create_enum( s.c_str());
    if( type_enum)
         return type_enum;
    const IType_struct* type_struct = create_struct( s.c_str());
    if( type_struct)
         return type_struct;

    return nullptr;
}

const IType_enum* Type_factory::create_enum(
    const char* symbol,
    IType_enum::Predefined_id id,
    const IType_enum::Values& values,
    mi::base::Handle<const IAnnotation_block>& annotations,
    const IType_enum::Value_annotations& value_annotations,
    mi::Sint32* errors)
{
    if( !symbol || values.empty()) {
        *errors = -1;
        return nullptr;
    }
    if( !is_absolute( symbol)) {
        *errors = -2;
        return nullptr;
    }

    {
        std::shared_lock<std::shared_mutex> lock( m_mutex);
        const IType_enum* result = lookup_enum( symbol, id, values, errors);
        if( result || ( *errors != 0))
            return result;
    }
    {
        // Acquire unique lock in order to modify the maps.
        std::unique_lock<std::shared_mutex> lock( m_mutex);
        // Repeat lookup. Another thread might have modified the maps between release of the shared
        // lock and acquisition of the unique lock.
        const IType_enum* result = lookup_enum( symbol, id, values, errors);
        if( result || ( *errors != 0))
            return result;

        // Modify maps.
        const IType_enum* type = new TYPES::Type_enum(
            this, symbol, id, values, annotations, value_annotations);
        m_enum_symbols[symbol] = type;
        if( id != IType_enum::EID_USER) {
            ASSERT( M_SCENE, !m_enum_ids[id]);
            m_enum_ids[id] = type;
        }

        *errors = 0;
        return type;
    }
}

const IType_struct* Type_factory::create_struct(
    const char* symbol,
    IType_struct::Predefined_id id,
    const IType_struct::Fields& fields,
    mi::base::Handle<const IAnnotation_block>& annotations,
    const IType_struct::Field_annotations& field_annotations,
    mi::Sint32* errors)
{
    if( !symbol || fields.empty()) {
        *errors = -1;
        return nullptr;
    }
    if( !is_absolute( symbol)) {
        *errors = -2;
        return nullptr;
    }

    {
        std::shared_lock<std::shared_mutex> lock( m_mutex);
        const IType_struct* result = lookup_struct( symbol, id, fields, errors);
        if( result || ( *errors != 0))
            return result;
    }
    {
        // Acquire unique lock in order to modify the maps.
        std::unique_lock<std::shared_mutex> lock( m_mutex);
        // Repeat lookup. Another thread might have modified the maps between release of the shared
        // lock and acquisition of the unique lock.
        const IType_struct* result = lookup_struct( symbol, id, fields, errors);
        if( result || ( *errors != 0))
            return result;

        // Modify maps.
        const IType_struct* type = new TYPES::Type_struct(
            this, symbol, id, fields, annotations, field_annotations);
        m_struct_symbols[symbol] = type;
        if( id != IType_struct::SID_USER) {
            ASSERT( M_SCENE, !m_struct_ids[id]);
            m_struct_ids[id] = type;
        }

        *errors = 0;
        return type;
    }
}

void Type_factory::serialize( SERIAL::Serializer* serializer, const IType* type) const
{
    IType::Kind kind = type->get_kind();
    mi::Uint32 kind_as_uint32 = kind;
    SERIAL::write( serializer, kind_as_uint32);

    switch( kind) {

        case IType::TK_BOOL:
        case IType::TK_INT:
        case IType::TK_FLOAT:
        case IType::TK_DOUBLE:
        case IType::TK_STRING:
        case IType::TK_COLOR:
        case IType::TK_LIGHT_PROFILE:
        case IType::TK_BSDF_MEASUREMENT:
        case IType::TK_BSDF:
        case IType::TK_HAIR_BSDF:
        case IType::TK_EDF:
        case IType::TK_VDF:
            return;

        case IType::TK_ALIAS: {
            mi::base::Handle<const IType_alias> type_alias( type->get_interface<IType_alias>());
            mi::base::Handle<const IType> aliased_type( type_alias->get_aliased_type());
            SERIAL::write( serializer, type_alias->get_type_modifiers());
            const char* symbol = type_alias->get_symbol();
            SERIAL::write( serializer, symbol ? symbol : "");
            serialize( serializer, aliased_type.get());
            return;
        }

        case IType::TK_ENUM: {
            mi::base::Handle<IExpression_factory> ef( get_expression_factory());
            mi::base::Handle<const IType_enum> type_enum( type->get_interface<IType_enum>());
            SERIAL::write( serializer, std::string( type_enum->get_symbol()));
            SERIAL::write( serializer, type_enum->get_predefined_id());
            mi::Size count = type_enum->get_size();
            SERIAL::write( serializer, count);
            for( mi::Size i = 0; i < count; ++i) {
                SERIAL::write( serializer, std::string( type_enum->get_value_name( i)));
                SERIAL::write( serializer, type_enum->get_value_code( i));
            }
            mi::base::Handle<const IAnnotation_block> annotations( type_enum->get_annotations());
            ef->serialize_annotation_block( serializer, annotations.get());
            for( mi::Size i = 0; i < count; ++i) {
                mi::base::Handle<const IAnnotation_block> value_annotations(
                    type_enum->get_value_annotations( i));
                ef->serialize_annotation_block( serializer, value_annotations.get());
            }
            return;
        }

        case IType::TK_VECTOR:
        case IType::TK_MATRIX: {
            mi::base::Handle<const IType_compound> type_compound(
                type->get_interface<IType_compound>());
            mi::base::Handle<const IType> component_type( type_compound->get_component_type( 0));
            SERIAL::write( serializer, type_compound->get_size());
            serialize( serializer, component_type.get());
            return;
        }

        case IType::TK_ARRAY: {
            mi::base::Handle<const IType_array> type_array( type->get_interface<IType_array>());
            mi::base::Handle<const IType> element_type( type_array->get_element_type());
            bool immediate_sized = type_array->is_immediate_sized();
            SERIAL::write( serializer, immediate_sized);
            if( immediate_sized)
                SERIAL::write( serializer, type_array->get_size());
            else
                SERIAL::write( serializer, std::string( type_array->get_deferred_size()));
            serialize( serializer, element_type.get());
            return;
        }

        case IType::TK_STRUCT: {
            mi::base::Handle<IExpression_factory> ef( get_expression_factory());
            mi::base::Handle<const IType_struct> type_struct( type->get_interface<IType_struct>());
            SERIAL::write( serializer, std::string( type_struct->get_symbol()));
            SERIAL::write( serializer, type_struct->get_predefined_id());
            mi::Size count = type_struct->get_size();
            SERIAL::write( serializer, count);
            for( mi::Size i = 0; i < count; ++i) {
                mi::base::Handle<const IType> field_type( type_struct->get_field_type( i));
                serialize( serializer, field_type.get());
                SERIAL::write( serializer, std::string( type_struct->get_field_name( i)));
            }
            mi::base::Handle<const IAnnotation_block> annotations( type_struct->get_annotations());
            ef->serialize_annotation_block( serializer, annotations.get());
            for( mi::Size i = 0; i < count; ++i) {
                mi::base::Handle<const IAnnotation_block> field_annotations(
                    type_struct->get_field_annotations( i));
                ef->serialize_annotation_block( serializer, field_annotations.get());
            }
            return;
        }

        case IType::TK_TEXTURE: {
            mi::base::Handle<const IType_texture> type_texture(
                type->get_interface<IType_texture>());
            mi::Uint32 shape = type_texture->get_shape();
            SERIAL::write( serializer, shape);
            return;
        }

        case IType::TK_FORCE_32_BIT:
        ASSERT( M_SCENE, false);
    }

    ASSERT( M_SCENE, false);
}

const IType* Type_factory::deserialize( SERIAL::Deserializer* deserializer)
{
    mi::Uint32 kind_as_uint32;
    SERIAL::read( deserializer, &kind_as_uint32);
    IType::Kind kind = static_cast<IType::Kind>( kind_as_uint32);

    switch( kind) {

        case IType::TK_BOOL:             return create_bool();
        case IType::TK_INT:              return create_int();
        case IType::TK_FLOAT:            return create_float();
        case IType::TK_DOUBLE:           return create_double();
        case IType::TK_STRING:           return create_string();
        case IType::TK_COLOR:            return create_color();
        case IType::TK_LIGHT_PROFILE:    return create_light_profile();
        case IType::TK_BSDF_MEASUREMENT: return create_bsdf_measurement();
        case IType::TK_BSDF:             return create_bsdf();
        case IType::TK_HAIR_BSDF:        return create_hair_bsdf();
        case IType::TK_EDF:              return create_edf();
        case IType::TK_VDF:              return create_vdf();

        case IType::TK_ALIAS: {
            mi::Uint32 modifiers;
            SERIAL::read( deserializer, &modifiers);
            std::string symbol;
            SERIAL::read( deserializer, &symbol);
            mi::base::Handle<const IType> aliased_type( deserialize( deserializer));
            ASSERT( M_SCENE, aliased_type);
            const IType* result = create_alias(
                aliased_type.get(), modifiers, symbol.empty() ? nullptr : symbol.c_str());
            return result;
        }

        case IType::TK_ENUM: {

            mi::base::Handle<IExpression_factory> ef( get_expression_factory());

            std::string symbol;
            SERIAL::read( deserializer, &symbol);
            mi::Uint32 id_uint32;
            SERIAL::read( deserializer, &id_uint32);
            IType_enum::Predefined_id id = static_cast<IType_enum::Predefined_id>( id_uint32);
            mi::Size count;
            SERIAL::read( deserializer, &count);
            IType_enum::Values values( count);
            for( mi::Size i = 0; i < count; ++i) {
                SERIAL::read( deserializer, &values[i].first);
                SERIAL::read( deserializer, &values[i].second);
            }
            mi::base::Handle<const IAnnotation_block> annotations(
                ef->deserialize_annotation_block( deserializer));
            IType_enum::Value_annotations value_annotations( count);
            for( mi::Size i = 0; i < count; ++i)
                value_annotations[i] = ef->deserialize_annotation_block( deserializer);

            mi::Sint32 errors = 0;
            const IType_enum* result = create_enum(
                symbol.c_str(), id, values, annotations, value_annotations, &errors);
            if( errors != 0 || !result) {
                ASSERT( M_SCENE, errors != 0 && !result);
                LOG::mod_log->error( M_SCENE, LOG::Mod_log::C_DATABASE,
                    "Type mismatch for enum \"%s\".", symbol.c_str());
                return nullptr;
            }
            return result;
        }

        case IType::TK_VECTOR: {
            mi::Size size;
            SERIAL::read( deserializer, &size);
            mi::base::Handle<const IType> component_type( deserialize( deserializer));
            mi::base::Handle<const IType_atomic> component_type_atomic(
                component_type->get_interface<IType_atomic>());
            ASSERT( M_SCENE, component_type_atomic);
            return create_vector( component_type_atomic.get(), size);
        }

        case IType::TK_MATRIX: {
            mi::Size size;
            SERIAL::read( deserializer, &size);
            mi::base::Handle<const IType> component_type( deserialize( deserializer));
            mi::base::Handle<const IType_vector> component_type_vector(
                component_type->get_interface<IType_vector>());
            ASSERT( M_SCENE, component_type_vector);
            return create_matrix( component_type_vector.get(), size);
        }

        case IType::TK_ARRAY: {
            bool is_immediate_sized;
            SERIAL::read( deserializer, &is_immediate_sized);
            if( is_immediate_sized) {
                mi::Size size;
                SERIAL::read( deserializer, &size);
                mi::base::Handle<const IType> element_type( deserialize( deserializer));
                ASSERT( M_SCENE, element_type);
                return create_immediate_sized_array( element_type.get(), size);
            } else {
                std::string size;
                SERIAL::read( deserializer, &size);
                mi::base::Handle<const IType> element_type( deserialize( deserializer));
                ASSERT( M_SCENE, element_type);
                return create_deferred_sized_array( element_type.get(), size.c_str());
            }
        }

        case IType::TK_STRUCT: {

            mi::base::Handle<IExpression_factory> ef( get_expression_factory());

            std::string symbol;
            SERIAL::read( deserializer, &symbol);
            mi::Uint32 id_uint32;
            SERIAL::read( deserializer, &id_uint32);
            IType_struct::Predefined_id id = static_cast<IType_struct::Predefined_id>( id_uint32);
            mi::Size count;
            SERIAL::read( deserializer, &count);
            IType_struct::Fields fields( count);
            for( mi::Size i = 0; i < count; ++i) {
                fields[i].first = deserialize( deserializer);
                SERIAL::read( deserializer, &fields[i].second);
            }
            mi::base::Handle<const IAnnotation_block> annotations(
                ef->deserialize_annotation_block( deserializer));
            IType_struct::Field_annotations field_annotations( count);
            for( mi::Size i = 0; i < count; ++i)
                field_annotations[i] = ef->deserialize_annotation_block( deserializer);

            mi::Sint32 errors = 0;
            const IType_struct* result = create_struct(
                symbol.c_str(), id, fields, annotations, field_annotations, &errors);
            if( errors != 0 || !result) {
                ASSERT( M_SCENE, errors != 0 && !result);
                LOG::mod_log->error( M_SCENE, LOG::Mod_log::C_DATABASE,
                    "Type mismatch for struct \"%s\".", symbol.c_str());
                return nullptr;
            }
            return result;
        }

        case IType::TK_TEXTURE: {
            mi::Uint32 shape_as_uint32;
            SERIAL::read( deserializer, &shape_as_uint32);
            IType_texture::Shape shape = static_cast<IType_texture::Shape>( shape_as_uint32);
            return create_texture( shape);
        }

        case IType::TK_FORCE_32_BIT:
            ASSERT( M_SCENE, false);
            return nullptr;
        }

    ASSERT( M_SCENE, false);
    return nullptr;
}

void Type_factory::serialize_list( SERIAL::Serializer* serializer, const IType_list* list) const
{
    const Type_list* list_impl = static_cast<const Type_list*>( list);

    write( serializer, list_impl->m_index_name);

    mi::Size size = list_impl->m_types.size();
    SERIAL::write( serializer, size);
    for( mi::Size i = 0; i < size; ++i)
        serialize( serializer, list_impl->m_types[i].get());
}

IType_list* Type_factory::deserialize_list( SERIAL::Deserializer* deserializer)
{
    Type_list* list_impl = new Type_list( /*initial_capacity*/ 0);

    read( deserializer, &list_impl->m_index_name);

    mi::Size size;
    SERIAL::read( deserializer, &size);
    list_impl->m_types.resize( size);
    for( mi::Size i = 0; i < size; ++i)
        list_impl->m_types[i] = deserialize( deserializer);

    return list_impl;
}

mi::Sint32 Type_factory::compare_static( const IType* lhs, const IType* rhs)
{
    if( lhs == rhs)
        return 0;

    if( !lhs && !rhs) return  0;
    if( !lhs &&  rhs) return -1;
    if(  lhs && !rhs) return +1;
    ASSERT( M_SCENE, lhs && rhs);

    IType::Kind lhs_kind = lhs->get_kind(); //-V522 PVS
    IType::Kind rhs_kind = rhs->get_kind(); //-V522 PVS

    if( lhs_kind < rhs_kind) return -1;
    if( lhs_kind > rhs_kind) return +1;
    ASSERT( M_SCENE, lhs_kind == rhs_kind);

    switch( lhs_kind) {

        case IType::TK_BOOL:
        case IType::TK_INT:
        case IType::TK_FLOAT:
        case IType::TK_DOUBLE:
        case IType::TK_STRING:
        case IType::TK_COLOR:
        case IType::TK_LIGHT_PROFILE:
        case IType::TK_BSDF_MEASUREMENT:
        case IType::TK_BSDF:
        case IType::TK_HAIR_BSDF:
        case IType::TK_EDF:
        case IType::TK_VDF:
            return 0;
        case IType::TK_ALIAS: {
            mi::base::Handle<const IType_alias> lhs_alias(
                lhs->get_interface<IType_alias>());
            mi::base::Handle<const IType_alias> rhs_alias(
                rhs->get_interface<IType_alias>());
            return compare_static( lhs_alias.get(), rhs_alias.get());
        }
        case IType::TK_ENUM: {
            mi::base::Handle<const IType_enum> lhs_enum(
                lhs->get_interface<IType_enum>());
            mi::base::Handle<const IType_enum> rhs_enum(
                rhs->get_interface<IType_enum>());
            mi::Sint32 result = strcmp( lhs_enum->get_symbol(), rhs_enum->get_symbol());
            if( result < 0) return -1;
            if( result > 0) return +1;
            return 0;
        }
        case IType::TK_VECTOR:
        case IType::TK_MATRIX: {
            mi::base::Handle<const IType_compound> lhs_compound(
                lhs->get_interface<IType_compound>());
            mi::base::Handle<const IType_compound> rhs_compound(
                rhs->get_interface<IType_compound>());
            return compare_static( lhs_compound.get(), rhs_compound.get());
        }
        case IType::TK_ARRAY: {
            mi::base::Handle<const IType_array> lhs_array(
                lhs->get_interface<IType_array>());
            mi::base::Handle<const IType_array> rhs_array(
                rhs->get_interface<IType_array>());
            return compare_static( lhs_array.get(), rhs_array.get());
        }
        case IType::TK_STRUCT: {
            mi::base::Handle<const IType_struct> lhs_struct(
                lhs->get_interface<IType_struct>());
            mi::base::Handle<const IType_struct> rhs_struct(
                rhs->get_interface<IType_struct>());
            mi::Sint32 result = strcmp( lhs_struct->get_symbol(), rhs_struct->get_symbol());
            if( result < 0) return -1;
            if( result > 0) return +1;
            return 0;
        }
        case IType::TK_TEXTURE: {
            mi::base::Handle<const IType_texture> lhs_texture(
                lhs->get_interface<IType_texture>());
            mi::base::Handle<const IType_texture> rhs_texture(
                rhs->get_interface<IType_texture>());
            return compare_static( lhs_texture.get(), rhs_texture.get());
        }
        case IType::TK_FORCE_32_BIT:
            ASSERT( M_SCENE, false);
            return 0;
    }

    ASSERT( M_SCENE, false);
    return 0;
}

mi::Sint32 Type_factory::compare_static( const IType_list* lhs, const IType_list* rhs)
{
    if( !lhs && !rhs) return  0;
    if( !lhs &&  rhs) return -1;
    if(  lhs && !rhs) return +1;
    ASSERT( M_SCENE, lhs && rhs);

    mi::Size lhs_n = lhs->get_size(); //-V522 PVS
    mi::Size rhs_n = rhs->get_size(); //-V522 PVS
    if( lhs_n < rhs_n) return -1;
    if( lhs_n > rhs_n) return +1;

    for( mi::Size i = 0; i < lhs_n; ++i) {
        const char* lhs_name = lhs->get_name( i);
        const char* rhs_name = rhs->get_name( i);
        mi::Sint32 result = strcmp( lhs_name, rhs_name);
        if( result < 0) return -1;
        if( result > 0) return +1;
        mi::base::Handle<const IType> lhs_type( lhs->get_type( i));
        mi::base::Handle<const IType> rhs_type( rhs->get_type( i));
        result = compare_static( lhs_type.get(), rhs_type.get());
        if( result != 0)
            return result;
    }

    return 0;
}

std::string Type_factory::get_dump_type_name( const IType* type, bool include_aliased_type)
{
    IType::Kind kind = type->get_kind();

    switch( kind) {

        case IType::TK_ALIAS: {
            mi::base::Handle<const IType_alias> type_alias(
                type->get_interface<IType_alias>());
            std::ostringstream result;
            result << "alias";
            const char* symbol = type_alias->get_symbol();
            if( symbol)
                result << " \"" << symbol << '\"';
            mi::Sint32 modifiers = type_alias->get_type_modifiers();
            if( modifiers & IType::MK_UNIFORM)
                result << " uniform";
            if( modifiers & IType::MK_VARYING)
                result << " varying";
            if( include_aliased_type) {
                mi::base::Handle<const IType> aliased_type(
                    type_alias->get_aliased_type());
                result << ' ' << get_dump_type_name( aliased_type.get(), include_aliased_type);
            }
            return result.str();
        }
        case IType::TK_ENUM: {
            mi::base::Handle<const IType_enum> type_enum(
                type->get_interface<IType_enum>());
            std::string result = "enum \"";
            result += type_enum->get_symbol();
            result += '\"';
            return result;
        }
        case IType::TK_STRUCT: {
            mi::base::Handle<const IType_struct> type_struct(
                type->get_interface<IType_struct>());
            std::string result = "struct \"";
            result += type_struct->get_symbol();
            result += '\"';
            return result;
        }

        case IType::TK_BOOL:
        case IType::TK_INT:
        case IType::TK_FLOAT:
        case IType::TK_DOUBLE:
        case IType::TK_STRING:
        case IType::TK_VECTOR:
        case IType::TK_MATRIX:
        case IType::TK_COLOR:
        case IType::TK_ARRAY:
        case IType::TK_TEXTURE:
        case IType::TK_LIGHT_PROFILE:
        case IType::TK_BSDF_MEASUREMENT:
        case IType::TK_BSDF:
        case IType::TK_HAIR_BSDF:
        case IType::TK_EDF:
        case IType::TK_VDF:
        case IType::TK_FORCE_32_BIT:
            return get_mdl_type_name_static( type);

    }

    ASSERT( M_SCENE, false);
    return std::string();
}

std::string Type_factory::get_mdl_type_name_static( const IType* type)
{
    IType::Kind kind = type->get_kind();

    switch( kind) {

        case IType::TK_BOOL:             return "bool";
        case IType::TK_INT:              return "int";
        case IType::TK_FLOAT:            return "float";
        case IType::TK_DOUBLE:           return "double";
        case IType::TK_STRING:           return "string";
        case IType::TK_COLOR:            return "color";
        case IType::TK_LIGHT_PROFILE:    return "light_profile";
        case IType::TK_BSDF_MEASUREMENT: return "bsdf_measurement";
        case IType::TK_BSDF:             return "bsdf";
        case IType::TK_HAIR_BSDF:        return "hair_bsdf";
        case IType::TK_EDF:              return "edf";
        case IType::TK_VDF:              return "vdf";

        case IType::TK_ALIAS: {
            mi::base::Handle<const IType_alias> type_alias(
                type->get_interface<IType_alias>());
            // TODO support named aliases
            // const char* symbol = type_alias->get_symbol();
            // if( symbol)
            //     return symbol;
            std::string result;
            bool found = false;
            switch( type_alias->get_type_modifiers()) {
                case IType::MK_NONE:         found = true; break;
                case IType::MK_UNIFORM:      found = true; result += "uniform "; break;
                case IType::MK_VARYING:      found = true; result += "varying "; break;
                case IType::MK_FORCE_32_BIT: break;
            }
            ASSERT( M_SCENE, found); // Handle combination of flags if this fails.
            boost::ignore_unused( found);
            mi::base::Handle<const IType> aliased_type(
                type_alias->get_aliased_type());
            result += get_mdl_type_name_static( aliased_type.get());
            return result;
        }
        case IType::TK_ENUM: {
            mi::base::Handle<const IType_enum> type_enum(
                type->get_interface<IType_enum>());
            const char* symbol = type_enum->get_symbol();
            switch( type_enum->get_predefined_id()) {
                case IType_enum::EID_USER:
                case IType_enum::EID_TEX_GAMMA_MODE:
                    return symbol;
                case IType_enum::EID_INTENSITY_MODE:
                    return remove_prefix_for_builtin_type_name( symbol, /*compare_string*/ false);
                case IType_enum::EID_FORCE_32_BIT:
                    break;
            }
            ASSERT( M_SCENE, false);
            return std::string();
        }
        case IType::TK_VECTOR: {
            mi::base::Handle<const IType_vector> type_vector(
                type->get_interface<IType_vector>());
            mi::base::Handle<const IType> element_type(
                type_vector->get_element_type());
            std::ostringstream result;
            result << get_mdl_type_name_static( element_type.get());
            result << type_vector->get_size();
            return result.str();
        }
        case IType::TK_MATRIX: {
            mi::base::Handle<const IType_matrix> type_matrix(
                type->get_interface<IType_matrix>());
            mi::base::Handle<const IType_vector> vector_type(
                type_matrix->get_element_type());
            mi::base::Handle<const IType> element_type(
                vector_type->get_element_type());
            std::ostringstream result;
            result << get_mdl_type_name_static( element_type.get());
            result << type_matrix->get_size() << 'x' << vector_type->get_size();
            return result.str();
        }
        case IType::TK_STRUCT: {
            mi::base::Handle<const IType_struct> type_struct(
                type->get_interface<IType_struct>());
            const char* symbol = type_struct->get_symbol();
            switch( type_struct->get_predefined_id()) {
                case IType_struct::SID_USER:
                    return symbol;
                case IType_struct::SID_MATERIAL_EMISSION:
                case IType_struct::SID_MATERIAL_SURFACE:
                case IType_struct::SID_MATERIAL_VOLUME:
                case IType_struct::SID_MATERIAL_GEOMETRY:
                case IType_struct::SID_MATERIAL:
                    return remove_prefix_for_builtin_type_name( symbol, /*compare_string*/ false);
                case IType_struct::SID_FORCE_32_BIT:
                    break;
            }
            ASSERT( M_SCENE, false);
            return std::string();
        }
        case IType::TK_ARRAY: {
            mi::base::Handle<const IType_array> type_array(
                type->get_interface<IType_array>());
            mi::base::Handle<const IType> element_type(
                type_array->get_element_type());
            std::ostringstream result;
            result << get_mdl_type_name_static( element_type.get()) << '[';
            if( type_array->is_immediate_sized())
                result << type_array->get_size();
            else
                result <<  type_array->get_deferred_size();
            result << ']';
            return result.str();
        }
        case IType::TK_TEXTURE: {
            mi::base::Handle<const IType_texture> type_texture(
                type->get_interface<IType_texture>());
            IType_texture::Shape shape = type_texture->get_shape();
            switch( shape) {
                case IType_texture::TS_2D:   return "texture_2d";
                case IType_texture::TS_3D:   return "texture_3d";
                case IType_texture::TS_CUBE: return "texture_cube";
                case IType_texture::TS_PTEX: return "texture_ptex";
                case IType_texture::TS_BSDF_DATA:
                    ASSERT( M_SCENE, false); return std::string();
                case IType_texture::TS_FORCE_32_BIT:
                    ASSERT( M_SCENE, false); return std::string();
            }
            ASSERT( M_SCENE, false);
            return std::string();
        }
        case IType::TK_FORCE_32_BIT:
            ASSERT( M_SCENE, false);
            return std::string();
    }

    ASSERT( M_SCENE, false);
    return std::string();
}

void Type_factory::unregister_enum_type( const IType_enum* type)
{
    m_enum_symbols.erase( type->get_symbol());
    m_enum_ids.erase( type->get_predefined_id());
}

void Type_factory::unregister_struct_type( const IType_struct* type)
{
    m_struct_symbols.erase( type->get_symbol());
    m_struct_ids.erase( type->get_predefined_id());
}

mi::Sint32 Type_factory::compare_static(
    const IType_alias* lhs, const IType_alias* rhs)
{
    if( lhs == rhs)
        return 0;

    mi::Sint32 lhs_modifiers = lhs->get_type_modifiers();
    mi::Sint32 rhs_modifiers = rhs->get_type_modifiers();
    if( lhs_modifiers < rhs_modifiers) return -1;
    if( lhs_modifiers > rhs_modifiers) return +1;

    const char* lhs_symbol = lhs->get_symbol();
    const char* rhs_symbol = rhs->get_symbol();

    if( !lhs_symbol &&  rhs_symbol) return -1;
    if(  lhs_symbol && !rhs_symbol) return +1;
    if(  lhs_symbol &&  rhs_symbol) {
        mi::Sint32 result = strcmp( lhs_symbol, rhs_symbol);
        if( result < 0) return -1;
        if( result > 0) return +1;
    }

    mi::base::Handle<const IType> lhs_aliased_type( lhs->get_aliased_type());
    mi::base::Handle<const IType> rhs_aliased_type( rhs->get_aliased_type());
    return compare_static( lhs_aliased_type.get(), rhs_aliased_type.get());
}

mi::Sint32 Type_factory::compare_static(
    const IType_compound* lhs, const IType_compound* rhs)
{
    if( lhs == rhs)
        return 0;

    ASSERT( M_SCENE, lhs->get_kind() != IType_array::s_kind);
    ASSERT( M_SCENE, rhs->get_kind() != IType_array::s_kind);
    ASSERT( M_SCENE, lhs->get_kind() != IType_struct::s_kind);
    ASSERT( M_SCENE, rhs->get_kind() != IType_struct::s_kind);

    // comparing component types first results in a nicer order, but sizes first is faster

    mi::Size lhs_compound_size = lhs->get_size();
    mi::Size rhs_compound_size = rhs->get_size();
    if( lhs_compound_size < rhs_compound_size) return -1;
    if( lhs_compound_size > rhs_compound_size) return +1;

    mi::base::Handle<const IType> lhs_component_type( lhs->get_component_type( 0));
    mi::base::Handle<const IType> rhs_component_type( rhs->get_component_type( 0));
    return compare_static( lhs_component_type.get(), rhs_component_type.get());
}

mi::Sint32 Type_factory::compare_static(
    const IType_array* lhs, const IType_array* rhs)
{
    if( lhs == rhs)
        return 0;

    bool lhs_immediate_sized = lhs->is_immediate_sized();
    bool rhs_immediate_sized = rhs->is_immediate_sized();
    if(  lhs_immediate_sized && !rhs_immediate_sized) return -1;
    if( !lhs_immediate_sized &&  rhs_immediate_sized) return +1;
    ASSERT( M_SCENE, lhs_immediate_sized == rhs_immediate_sized);

    // comparing component types first results in a nicer order, but sizes first is faster

    if( lhs_immediate_sized) {
        mi::Size lhs_immediate_size = lhs->get_size();
        mi::Size rhs_immediate_size = rhs->get_size();
        if( lhs_immediate_size < rhs_immediate_size) return -1;
        if( lhs_immediate_size > rhs_immediate_size) return +1;
    } else {
        const char* lhs_deferred_size = lhs->get_deferred_size();
        const char* rhs_deferred_size = rhs->get_deferred_size();
        mi::Sint32 result = strcmp( lhs_deferred_size, rhs_deferred_size);
        if( result != 0)
            return result;
    }

    mi::base::Handle<const IType> lhs_component_type( lhs->get_component_type( 0));
    mi::base::Handle<const IType> rhs_component_type( rhs->get_component_type( 0));
    return compare_static( lhs_component_type.get(), rhs_component_type.get());
}

mi::Sint32 Type_factory::compare_static(
    const IType_texture* lhs, const IType_texture* rhs)
{
    if( lhs == rhs)
        return 0;

    IType_texture::Shape lhs_shape = lhs->get_shape();
    IType_texture::Shape rhs_shape = rhs->get_shape();
    if( lhs_shape < rhs_shape) return -1;
    if( lhs_shape > rhs_shape) return +1;
    return 0;
}

void Type_factory::dump(
    const IType* type, mi::Size depth, std::ostringstream& s)
{
    if( !type)
        return;

    std::string name = get_dump_type_name( type, /*include_aliased_type*/ false);
    ASSERT( M_SCENE, !name.empty());

    IType::Kind kind = type->get_kind();

    switch( kind) {

        case IType::TK_BOOL:
        case IType::TK_INT:
        case IType::TK_FLOAT:
        case IType::TK_DOUBLE:
        case IType::TK_STRING:
        case IType::TK_VECTOR:
        case IType::TK_MATRIX:
        case IType::TK_COLOR:
        case IType::TK_ARRAY:
        case IType::TK_TEXTURE:
        case IType::TK_LIGHT_PROFILE:
        case IType::TK_BSDF_MEASUREMENT:
        case IType::TK_BSDF:
        case IType::TK_HAIR_BSDF:
        case IType::TK_EDF:
        case IType::TK_VDF:
            s << name;
            return;
        case IType::TK_ALIAS: {
            mi::base::Handle<const IType_alias> type_alias(
                type->get_interface<IType_alias>());
            s << name << '\n' << get_prefix( depth + 1);
            mi::base::Handle<const IType> aliased_type(
                type_alias->get_aliased_type());
            dump( aliased_type.get(), depth + 1, s);
            return;
        }
        case IType::TK_ENUM: {
            mi::base::Handle<const IType_enum> type_enum(
                type->get_interface<IType_enum>());
            const std::string& prefix = get_prefix( depth);
            s << name << " {\n";
            mi::Size n = type_enum->get_size();
            for( mi::Size i = 0; i < n; ++i)
                s << prefix << "    "
                << type_enum->get_value_name( i) << " = "
                << type_enum->get_value_code( i) << ",\n";
            s << prefix << '}';
            return;
        }
        case IType::TK_STRUCT: {
            mi::base::Handle<const IType_struct> type_struct(
                type->get_interface<IType_struct>());
            s << name << " {\n";
            mi::Size n = type_struct->get_size();
            const std::string& prefix = get_prefix( depth);
            for( mi::Size i = 0; i < n; ++i) {
                s << prefix << "    ";
                mi::base::Handle<const IType> field_type(
                    type_struct->get_field_type( i));
                dump( field_type.get(), depth + 1, s);
                s << ' ' << type_struct->get_field_name( i) << ";\n";
            }
            s << prefix << '}';
            return;
        }
        case IType::TK_FORCE_32_BIT:
            ASSERT( M_SCENE, false);
            return;
    }

    ASSERT( M_SCENE, false);
}

void Type_factory::dump(
    const IType_list* list, mi::Size depth, std::ostringstream& s)
{
    if( !list)
        return;

    mi::Size n = list->get_size();
    s << "type_list [";
    s << (n > 0 ? '\n' : ' ');

    const std::string& prefix = get_prefix( depth);
    for( mi::Size i = 0; i < n; ++i) {
        mi::base::Handle<const IType> type( list->get_type( i));
        s << prefix << "    "  << i << ": " << list->get_name( i) << " = ";
        dump( type.get(), depth + 1, s);
        s << '\n';
    }

    s << (n > 0 ? prefix : "") << ']';
}

const IType_enum* Type_factory::lookup_enum(
    const char* symbol,
    IType_enum::Predefined_id id,
    const IType_enum::Values& values,
    mi::Sint32* errors)
{
    if( m_struct_symbols.find( symbol) != m_struct_symbols.end()) {
        *errors = -3;
        return nullptr;
    }

    Weak_enum_symbol_map::const_iterator it = m_enum_symbols.find( symbol);
    if( m_enum_symbols.find( symbol) == m_enum_symbols.end()) {
        *errors = 0;
        return nullptr;
    }

    const IType_enum* type_enum = it->second;
    if( !equivalent_enum_types( type_enum, id, values)) {
        *errors = -4;
        return nullptr;
    }

    *errors = 0;
    type_enum->retain();
    return type_enum;
}

const IType_struct* Type_factory::lookup_struct(
    const char* symbol,
    IType_struct::Predefined_id id,
    const IType_struct::Fields& fields,
    mi::Sint32* errors)
{
    if( m_enum_symbols.find( symbol) != m_enum_symbols.end()) {
        *errors = -3;
        return nullptr;
    }

    Weak_struct_symbol_map::const_iterator it = m_struct_symbols.find( symbol);
    if( it == m_struct_symbols.end()) {
        *errors = 0;
        return nullptr;
    }

    const IType_struct* type_struct = it->second;
    if( !equivalent_struct_types( type_struct, id, fields)) {
        *errors = -4;
        return nullptr;
    }

    *errors = 0;
    type_struct->retain();
    return type_struct;
}

bool Type_factory::equivalent_enum_types(
    const IType_enum* type, IType_enum::Predefined_id id, const IType_enum::Values& values)
{
    mi::Size size = type->get_size();
    if( values.size() != size)
        return false;

    if( id != type->get_predefined_id())
        return false;

    for( mi::Size i = 0; i < size; ++i) {
        if( values[i].first != type->get_value_name( i))
            return false;
        if( values[i].second != type->get_value_code( i))
            return false;
    }

    return true;
}

bool Type_factory::equivalent_struct_types(
    const IType_struct* type, IType_struct::Predefined_id id, const IType_struct::Fields& fields)
{
    mi::Size size = type->get_size();
    if( fields.size() != size)
        return false;

    if( id != type->get_predefined_id())
        return false;

    for( mi::Size i = 0; i < size; ++i) {
        mi::base::Handle<const IType> field_type( type->get_field_type( i));
        if( compare_static( fields[i].first.get(), field_type.get()) != 0)
            return false;
        if( fields[i].second != type->get_field_name( i))
            return false;
    }

    return true;
}

} // namespace MDL

} // namespace MI
