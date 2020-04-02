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
 ** \brief Header for the IType implementation.
 **/

#ifndef API_API_NEURAY_NEURAY_TYPE_IMPL_H
#define API_API_NEURAY_NEURAY_TYPE_IMPL_H

#include <mi/neuraylib/itype.h>

#include <mi/base/interface_implement.h>
#include <mi/neuraylib/itransaction.h>

#include <base/lib/log/i_log_assert.h>
#include <io/scene/mdl_elements/i_mdl_elements_type.h>

namespace MI {

namespace NEURAY {

class Type_factory;

/// Converts MI::MDL::IType_alias modifiers into mi::neuraylib::IType_alias modifiers.
mi::Uint32 int_modifiers_to_ext_modifiers( mi::Uint32 modifiers);

/// Converts mi::neuraylb::IType_alias modifiers into MI::MDL::IType_alias modifiers.
mi::Uint32 ext_modifiers_to_int_modifiers( mi::Uint32 modifiers);

/// Converts predefined MI::MDL::IType_enum IDs into mi::neuraylib::IType_enum IDs.
mi::neuraylib::IType_enum::Predefined_id int_enum_id_to_ext_enum_id(
    MDL::IType_enum::Predefined_id enum_id);

/// Converts predefined mi::neuraylib::IType_enum IDs into MI::MDL::IType_enum IDs.
MDL::IType_enum::Predefined_id ext_enum_id_to_int_enum_id(
    mi::neuraylib::IType_enum::Predefined_id enum_id);

/// Converts predefined MI::MDL::IType_struct IDs into mi::neuraylib::IType_struct IDs.
mi::neuraylib::IType_struct::Predefined_id int_struct_id_to_ext_struct_id(
    MDL::IType_struct::Predefined_id struct_id);

/// Converts predefined mi::neuraylib::IType_struct IDs into MI::MDL::IType_struct IDs.
MDL::IType_struct::Predefined_id ext_struct_id_to_int_struct_id(
    mi::neuraylib::IType_struct::Predefined_id struct_id);

/// Converts MI::MDL::IType_texture shapes into mi::neuraylib::IType_texture shapes.
mi::neuraylib::IType_texture::Shape int_shape_to_ext_shape( MDL::IType_texture::Shape shape);

/// Converts mi::neuraylib::IType_texture shapes into MI::MDL::IType_texture shapes.
MDL::IType_texture::Shape ext_shape_to_int_shape( mi::neuraylib::IType_texture::Shape shape);

/// All implementations of mi::neuraylib::IType also implement this interface which allows access
/// to the wrapped MI::MDL::IType instance.
class IType_wrapper : public
    mi::base::Interface_declare<0x569703bd,0x0f21,0x456c,0x82,0x5c,0x26,0xd7,0x5d,0xca,0x4b,0xcf>
{
public:
    /// Returns the internal type wrapped by this wrapper.
    virtual const MDL::IType* get_internal_type() const = 0;
};

/// Returns the internal type wrapped by \p type (or \c NULL if \p type is \c NULL or not a
/// wrapper).
const MDL::IType* get_internal_type( const mi::neuraylib::IType* type);

/// Returns the internal type wrapped by \p type (or \c NULL if \p type is \c NULL or not a
/// wrapper).
template <class T>
const T* get_internal_type( const mi::neuraylib::IType* type)
{
    mi::base::Handle<const MDL::IType> ptr_type( get_internal_type( type));
    if( !ptr_type)
        return 0;
    return static_cast<const T*>( ptr_type->get_interface( typename T::IID()));
}

/// The implementation of mi::neuraylib::IType_list also implement this interface which allows
/// access to the wrapped MI::MDL::IType_list instance.
class IType_list_wrapper : public
    mi::base::Interface_declare<0xbae7dba4,0xba2f,0x44fc,0xbc,0xfd,0xef,0x1c,0xf9,0x22,0xfa,0xe2>
{
public:
    /// Returns the internal type list wrapped by this wrapper.
    virtual const MDL::IType_list* get_internal_type_list() const = 0;
};

/// Returns the internal type_list wrapped by \p type_list (or \c NULL if \p type_list is \c NULL or
/// not a wrapper).
const MDL::IType_list* get_internal_type_list( const mi::neuraylib::IType_list* type_list);

/// Wrapper that implements an external type interface by wrapping an internal one.
///
/// \tparam E   The external interface implemented by this class.
/// \tparam I   The internal interface wrapped by this class.
template <class E, class I>
class Type_base : public mi::base::Interface_implement_2<E, IType_wrapper>
{
public:
    typedef E External_type;
    typedef I Internal_type;
    typedef Type_base<E, I> Base;

    Type_base( const Type_factory* tf, const I* type, const mi::base::IInterface* owner)
      : m_tf( tf, mi::base::DUP_INTERFACE),
        m_type( type, mi::base::DUP_INTERFACE),
        m_owner( owner, mi::base::DUP_INTERFACE)
    {
        ASSERT( M_NEURAY_API, tf);
        ASSERT( M_NEURAY_API, type);
    }

    ~Type_base(); // trivial, just because Type_factory is forward declared

    // public API methods

    mi::neuraylib::IType::Kind get_kind() const { return E::s_kind; };

    mi::Uint32 get_all_type_modifiers() const;

    const mi::neuraylib::IType* skip_all_type_aliases() const;

    // internal methods (IType_wrapper)

    const I* get_internal_type() const { m_type->retain(); return m_type.get(); }

protected:
    const mi::base::Handle<const Type_factory> m_tf;
    const mi::base::Handle<const I> m_type;
    const mi::base::Handle<const mi::base::IInterface> m_owner;
};


class Type_alias : public Type_base<mi::neuraylib::IType_alias, MDL::IType_alias>
{
public:
    Type_alias(
        const Type_factory* tf, const Internal_type* type, const mi::base::IInterface* owner)
      : Base( tf, type, owner) { }

    const mi::neuraylib::IType* get_aliased_type() const;

    mi::Uint32 get_type_modifiers() const;

    const char* get_symbol() const { return m_type->get_symbol(); }
};

class Type_bool : public Type_base<mi::neuraylib::IType_bool, MDL::IType_bool>
{
public:
    Type_bool(
        const Type_factory* tf, const Internal_type* type, const mi::base::IInterface* owner)
      : Base( tf, type, owner) { }
};


class Type_int : public Type_base<mi::neuraylib::IType_int, MDL::IType_int>
{
public:
    Type_int(
        const Type_factory* tf, const Internal_type* type, const mi::base::IInterface* owner)
      : Base( tf, type, owner) { }
};


class Type_enum : public Type_base<mi::neuraylib::IType_enum, MDL::IType_enum>
{
public:
    Type_enum(
        const Type_factory* tf,
        mi::neuraylib::ITransaction* transaction,
        const Internal_type* type,
        const mi::base::IInterface* owner)
      : Base( tf, type, owner)
      , m_transaction(transaction, mi::base::DUP_INTERFACE) { }

    const char* get_symbol() const { return m_type->get_symbol(); }

    mi::Size get_size() const { return m_type->get_size(); }

    const char* get_value_name( mi::Size index) const { return m_type->get_value_name( index); }

    mi::Sint32 get_value_code( mi::Size index, mi::Sint32* errors) const;

    mi::Size find_value( char const* name) const { return m_type->find_value( name); }

    mi::Size find_value( mi::Sint32 code) const { return m_type->find_value( code); }

    mi::neuraylib::IType_enum::Predefined_id get_predefined_id() const;

    const mi::neuraylib::IAnnotation_block* get_annotations() const;

    const mi::neuraylib::IAnnotation_block* get_value_annotations( mi::Size index) const;

private:
    const mi::base::Handle<mi::neuraylib::ITransaction> m_transaction;
};


class Type_float : public Type_base<mi::neuraylib::IType_float, MDL::IType_float>
{
public:
    Type_float(
        const Type_factory* tf, const Internal_type* type, const mi::base::IInterface* owner)
      : Base( tf, type, owner) { }
};


class Type_double : public Type_base<mi::neuraylib::IType_double, MDL::IType_double>
{
public:
    Type_double(
        const Type_factory* tf, const Internal_type* type, const mi::base::IInterface* owner)
      : Base( tf, type, owner) { }
};


class Type_string : public Type_base<mi::neuraylib::IType_string, MDL::IType_string>
{
public:
    Type_string(
        const Type_factory* tf, const Internal_type* type, const mi::base::IInterface* owner)
      : Base( tf, type, owner) { }
};


class Type_vector : public Type_base<mi::neuraylib::IType_vector, MDL::IType_vector>
{
public:
    Type_vector(
        const Type_factory* tf, const Internal_type* type, const mi::base::IInterface* owner)
      : Base( tf, type, owner) { }

    const mi::neuraylib::IType* get_component_type( mi::Size index) const;

    mi::Size get_size() const { return m_type->get_size(); }

    const mi::neuraylib::IType_atomic* get_element_type() const;
};


class Type_matrix : public Type_base<mi::neuraylib::IType_matrix, MDL::IType_matrix>
{
public:
    Type_matrix(
        const Type_factory* tf, const Internal_type* type, const mi::base::IInterface* owner)
      : Base( tf, type, owner) { }

    const mi::neuraylib::IType* get_component_type( mi::Size index) const;

    mi::Size get_size() const { return m_type->get_size(); }

    const mi::neuraylib::IType_vector* get_element_type() const;
};


class Type_color : public Type_base<mi::neuraylib::IType_color, MDL::IType_color>
{
public:
    Type_color(
        const Type_factory* tf, const Internal_type* type, const mi::base::IInterface* owner)
      : Base( tf, type, owner) { }

    const mi::neuraylib::IType* get_component_type( mi::Size index) const;

    mi::Size get_size() const { return m_type->get_size(); }
};


class Type_array : public Type_base<mi::neuraylib::IType_array, MDL::IType_array>
{
public:
    Type_array(
        const Type_factory* tf, const Internal_type* type, const mi::base::IInterface* owner)
      : Base( tf, type, owner) { }

    const mi::neuraylib::IType* get_component_type( mi::Size index) const;

    mi::Size get_size() const { return m_type->get_size(); }

    const mi::neuraylib::IType* get_element_type() const;

    bool is_immediate_sized() const { return m_type->is_immediate_sized(); }

    const char* get_deferred_size() const { return m_type->get_deferred_size(); }
};


class Type_struct : public Type_base<mi::neuraylib::IType_struct, MDL::IType_struct>
{
public:
    Type_struct(
        const Type_factory* tf,
        mi::neuraylib::ITransaction* transaction,
        const Internal_type* type,
        const mi::base::IInterface* owner)
      : Base( tf, type, owner)
      , m_transaction(transaction, mi::base::DUP_INTERFACE) { }

    const mi::neuraylib::IType* get_component_type( mi::Size index) const;

    mi::Size get_size() const { return m_type->get_size(); }

    const char* get_symbol() const { return m_type->get_symbol(); }

    const mi::neuraylib::IType* get_field_type( mi::Size index) const;

    const char* get_field_name( mi::Size index) const { return m_type->get_field_name( index); }

    mi::Size find_field( const char* name) const { return m_type->find_field( name); }

    mi::neuraylib::IType_struct::Predefined_id get_predefined_id() const;

    const mi::neuraylib::IAnnotation_block* get_annotations() const;

    const mi::neuraylib::IAnnotation_block* get_field_annotations( mi::Size index) const;

private:
    const mi::base::Handle<mi::neuraylib::ITransaction> m_transaction;
};


class Type_texture : public Type_base<mi::neuraylib::IType_texture, MDL::IType_texture>
{
public:
    Type_texture(
        const Type_factory* tf, const Internal_type* type, const mi::base::IInterface* owner)
      : Base( tf, type, owner) { }

    mi::neuraylib::IType_texture::Shape get_shape() const;
};


class Type_light_profile
  : public Type_base<mi::neuraylib::IType_light_profile, MDL::IType_light_profile>
{
public:
    Type_light_profile(
        const Type_factory* tf, const Internal_type* type, const mi::base::IInterface* owner)
      : Base( tf, type, owner) { }
};


class Type_bsdf_measurement
  : public Type_base<mi::neuraylib::IType_bsdf_measurement, MDL::IType_bsdf_measurement>
{
public:
    Type_bsdf_measurement(
        const Type_factory* tf, const Internal_type* type, const mi::base::IInterface* owner)
      : Base( tf, type, owner) { }
};


class Type_bsdf : public Type_base<mi::neuraylib::IType_bsdf, MDL::IType_bsdf>
{
public:
    Type_bsdf(
        const Type_factory* tf, const Internal_type* type, const mi::base::IInterface* owner)
      : Base( tf, type, owner) { }
};


class Type_hair_bsdf : public Type_base<mi::neuraylib::IType_hair_bsdf, MDL::IType_hair_bsdf>
{
public:
    Type_hair_bsdf(
        const Type_factory* tf, const Internal_type* type, const mi::base::IInterface* owner)
      : Base(tf, type, owner) { }
};


class Type_edf : public Type_base<mi::neuraylib::IType_edf, MDL::IType_edf>
{
public:
    Type_edf(
        const Type_factory* tf, const Internal_type* type, const mi::base::IInterface* owner)
      : Base( tf, type, owner) { }
};


class Type_vdf : public Type_base<mi::neuraylib::IType_vdf, MDL::IType_vdf>
{
public:
    Type_vdf(
        const Type_factory* tf, const Internal_type* type, const mi::base::IInterface* owner)
      : Base( tf, type, owner) { }
};


class Type_list
  : public mi::base::Interface_implement_2<mi::neuraylib::IType_list, IType_list_wrapper>
{
public:
    Type_list(
        const Type_factory* tf, MDL::IType_list* type_list, const mi::base::IInterface* owner)
      : m_tf( tf, mi::base::DUP_INTERFACE),
        m_type_list( type_list, mi::base::DUP_INTERFACE),
        m_owner( owner, mi::base::DUP_INTERFACE)
    {
        ASSERT( M_NEURAY_API, m_tf);
        ASSERT( M_NEURAY_API, m_type_list);
    }

    // public API methods

    mi::Size get_size() const { return m_type_list->get_size(); }

    mi::Size get_index( const char* name) const { return m_type_list->get_index( name); }

    const char* get_name( mi::Size index) const { return m_type_list->get_name( index); }

    const mi::neuraylib::IType* get_type( mi::Size index) const;

    const mi::neuraylib::IType* get_type( const char* name) const;

    mi::Sint32 set_type( mi::Size index, const mi::neuraylib::IType* type);

    mi::Sint32 set_type( const char* name, const mi::neuraylib::IType* type);

    mi::Sint32 add_type( const char* name, const mi::neuraylib::IType* type);

    // internal methods (IType_wrapper)

    const MDL::IType_list* get_internal_type_list() const;

private:
    const mi::base::Handle<const Type_factory> m_tf;
    const mi::base::Handle<MDL::IType_list> m_type_list;
    const mi::base::Handle<const mi::base::IInterface> m_owner;
};

class Type_factory : public mi::base::Interface_implement<mi::neuraylib::IType_factory>
{
public:
    Type_factory( mi::neuraylib::ITransaction* transaction)
      : m_transaction( transaction, mi::base::DUP_INTERFACE), m_tf( MDL::get_type_factory()) { }

    // public API methods

    const mi::neuraylib::IType_alias* create_alias(
        const mi::neuraylib::IType* type,
        mi::Uint32 modifiers,
        const char* symbol) const override;

    const mi::neuraylib::IType_bool* create_bool() const override;

    const mi::neuraylib::IType_int* create_int() const override;

    const mi::neuraylib::IType_enum* create_enum( const char* symbol) const override;

    const mi::neuraylib::IType_float* create_float() const override;

    const mi::neuraylib::IType_double* create_double() const override;

    const mi::neuraylib::IType_string* create_string() const override;

    const mi::neuraylib::IType_vector* create_vector(
        const mi::neuraylib::IType_atomic* element_type, mi::Size size) const override;

    const mi::neuraylib::IType_matrix* create_matrix(
        const mi::neuraylib::IType_vector* column_type, mi::Size columns) const override;

    const mi::neuraylib::IType_color* create_color() const override;

    const mi::neuraylib::IType_array* create_immediate_sized_array(
        const mi::neuraylib::IType* element_type, mi::Size size) const override;

    const mi::neuraylib::IType_array* create_deferred_sized_array(
        const mi::neuraylib::IType* element_type, const char* size) const override;

    const mi::neuraylib::IType_struct* create_struct( const char* symbol) const override;

    const mi::neuraylib::IType_texture* create_texture(
        mi::neuraylib::IType_texture::Shape shape) const override;

    const mi::neuraylib::IType_light_profile* create_light_profile() const override;

    const mi::neuraylib::IType_bsdf_measurement* create_bsdf_measurement() const override;

    const mi::neuraylib::IType_bsdf* create_bsdf() const override;

    const mi::neuraylib::IType_hair_bsdf* create_hair_bsdf() const override;

    const mi::neuraylib::IType_edf* create_edf() const override;

    const mi::neuraylib::IType_vdf* create_vdf() const override;

    mi::neuraylib::IType_list* create_type_list() const override;

    const mi::neuraylib::IType_enum* get_predefined_enum(
        mi::neuraylib::IType_enum::Predefined_id id) const override;

    const mi::neuraylib::IType_struct* get_predefined_struct(
        mi::neuraylib::IType_struct::Predefined_id id) const override;

    mi::Sint32 compare(
        const mi::neuraylib::IType* lhs,
        const mi::neuraylib::IType* rhs) const override;

    mi::Sint32 compare(
        const mi::neuraylib::IType_list* lhs,
        const mi::neuraylib::IType_list* rhs) const override;

    mi::Sint32 is_compatible(
        const mi::neuraylib::IType* src,
        const mi::neuraylib::IType* dst) const override;

    const mi::IString* dump(
        const mi::neuraylib::IType* type,
        mi::Size depth) const override;

    const mi::IString* dump(
        const mi::neuraylib::IType_list* list,
        mi::Size depth) const override;

    // internal methods

    /// Returns the wrapped internal type factory.
    MDL::IType_factory* get_internal_type_factory() const { m_tf->retain(); return m_tf.get(); }

    /// Creates a wrapper for the internal type.
    const mi::neuraylib::IType* create(
        const MDL::IType* type, const mi::base::IInterface* owner) const;

    /// Creates a wrapper for the internal type.
    template <class T>
    const T* create( const MDL::IType* type, const mi::base::IInterface* owner) const
    {
        mi::base::Handle<const mi::neuraylib::IType> ptr_type( create( type, owner));
        if( !ptr_type)
            return 0;
        return static_cast<const T*>( ptr_type->get_interface( typename T::IID()));
    }

    /// Creates a wrapper for the internal type list.
    mi::neuraylib::IType_list* create_type_list(
        MDL::IType_list* type_list, const mi::base::IInterface* owner) const;

    /// Creates a wrapper for the internal type list.
    const mi::neuraylib::IType_list* create_type_list(
        const MDL::IType_list* type_list, const mi::base::IInterface* owner) const;

private:
    const mi::base::Handle<mi::neuraylib::ITransaction> m_transaction;
    const mi::base::Handle<MDL::IType_factory> m_tf;
};

} // namespace NEURAY

} // namespace MI

#endif // API_API_NEURAY_NEURAY_TYPE_IMPL_H
