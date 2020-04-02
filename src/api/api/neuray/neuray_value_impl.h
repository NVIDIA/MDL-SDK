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
 ** \brief Header for the IValue implementation.
 **/

#ifndef API_API_NEURAY_NEURAY_VALUE_IMPL_H
#define API_API_NEURAY_NEURAY_VALUE_IMPL_H

#include <mi/neuraylib/ivalue.h>

#include <mi/base/interface_implement.h>
#include <mi/neuraylib/itransaction.h>

#include <base/lib/log/i_log_assert.h>
#include <io/scene/mdl_elements/i_mdl_elements_value.h>

#include "neuray_type_impl.h" // for external IType_factory implementation

namespace MI {

namespace NEURAY {

class Value_factory;

/// All implementations of mi::neuraylib::IValue also implement this interface which allows access
/// to the wrapped MI::MDL::IValue instance.
class IValue_wrapper : public
    mi::base::Interface_declare<0xa3cecca4,0x7701,0x472e,0x9e,0x02,0x91,0x7e,0xd0,0xb3,0xae,0xe9>
{
public:
    /// Returns the internal value wrapped by this wrapper.
    virtual MDL::IValue* get_internal_value() = 0;

    /// Returns the internal value wrapped by this wrapper.
    virtual const MDL::IValue* get_internal_value() const = 0;
};

/// Returns the internal value wrapped by \p value (or \c NULL if \p value is \c NULL or not a
/// wrapper).
MDL::IValue* get_internal_value( mi::neuraylib::IValue* value);

/// Returns the internal value wrapped by \p value (or \c NULL if \p value is \c NULL or not a
/// wrapper).
template <class T>
T* get_internal_value( mi::neuraylib::IValue* value)
{
    mi::base::Handle<MDL::IValue> ptr_value( get_internal_value( value));
    if( !ptr_value)
        return 0;
    return static_cast<T*>( ptr_value->get_interface( typename T::IID()));
}

/// Returns the internal value wrapped by \p value (or \c NULL if \p value is \c NULL or not a
/// wrapper).
const MDL::IValue* get_internal_value( const mi::neuraylib::IValue* value);

/// Returns the internal value wrapped by \p value (or \c NULL if \p value is \c NULL or not a
/// wrapper).
template <class T>
const T* get_internal_value( const mi::neuraylib::IValue* value)
{
    mi::base::Handle<const MDL::IValue> ptr_value( get_internal_value( value));
    if( !ptr_value)
        return 0;
    return static_cast<const T*>( ptr_value->get_interface( typename T::IID()));
}

/// The implementation of mi::neuraylib::IValue_list also implement this interface which allows
/// access to the wrapped MI::MDL::IValue_list instance.
class IValue_list_wrapper : public
    mi::base::Interface_declare<0x701e2a18,0x4fd7,0x46bc,0x92,0xe5,0x5c,0x76,0x24,0xdd,0x65,0x9e>
{
public:
    /// Returns the internal value list wrapped by this wrapper.
    virtual MDL::IValue_list* get_internal_value_list() = 0;

    /// Returns the internal value list wrapped by this wrapper.
    virtual const MDL::IValue_list* get_internal_value_list() const = 0;
};

/// Returns the internal value_list wrapped by \p value_list (or \c NULL if \p value_list is \c NULL
/// or not a wrapper).
MDL::IValue_list* get_internal_value_list( mi::neuraylib::IValue_list* value_list);

/// Returns the internal value_list wrapped by \p value_list (or \c NULL if \p value_list is \c NULL
/// or not a wrapper).
const MDL::IValue_list* get_internal_value_list( const mi::neuraylib::IValue_list* value_list);

/// Wrapper that implements an external value interface by wrapping an internal one.
///
/// \tparam E   The external value interface implemented by this class.
/// \tparam I   The internal value interface wrapped by this class.
/// \tparam T   The external type interface corresponding to E.
template <class E, class I, class T>
class Value_base : public mi::base::Interface_implement_2<E, IValue_wrapper>
{
public:
    typedef E External_value;
    typedef I Internal_value;
    typedef T External_type;
    typedef Value_base<E, I, T> Base;

    Value_base( const Value_factory* vf, I* value, const mi::base::IInterface* owner)
      : m_vf( vf, mi::base::DUP_INTERFACE),
        m_value( value, mi::base::DUP_INTERFACE),
        m_owner( owner, mi::base::DUP_INTERFACE)
    {
        ASSERT( M_NEURAY_API, vf);
        ASSERT( M_NEURAY_API, value);
    }

    ~Value_base(); // trivial, just because Value_factory is forward declared

    // public API methods

    mi::neuraylib::IValue::Kind get_kind() const { return E::s_kind; };

    const T* get_type() const;

    // internal methods (IValue_wrapper)

    I* get_internal_value() { m_value->retain(); return m_value.get(); }

    const I* get_internal_value() const { m_value->retain(); return m_value.get(); }

protected:
    const mi::base::Handle<const Value_factory> m_vf;
    const mi::base::Handle<I> m_value;
    const mi::base::Handle<const mi::base::IInterface> m_owner;
};


class Value_bool
  : public Value_base<mi::neuraylib::IValue_bool, MDL::IValue_bool, mi::neuraylib::IType_bool>
{
public:
    Value_bool(
        const Value_factory* vf, Internal_value* value, const mi::base::IInterface* owner)
      : Base( vf, value, owner) { }

    bool get_value() const { return m_value->get_value(); }

    void set_value( bool value) { m_value->set_value( value); }
};

class Value_int
  : public Value_base<mi::neuraylib::IValue_int, MDL::IValue_int, mi::neuraylib::IType_int>
{
public:
    Value_int(
        const Value_factory* vf, Internal_value* value, const mi::base::IInterface* owner)
      : Base( vf, value, owner) { }

    mi::Sint32 get_value() const { return m_value->get_value(); }

    void set_value( mi::Sint32 value) { m_value->set_value( value); }
};


class Value_enum
  : public Value_base<mi::neuraylib::IValue_enum, MDL::IValue_enum, mi::neuraylib::IType_enum>
{
public:
    Value_enum(
        const Value_factory* vf, Internal_value* value, const mi::base::IInterface* owner)
      : Base( vf, value, owner) { }

    mi::Sint32 get_value() const { return m_value->get_value(); }

    const char* get_name() const { return m_value->get_name(); }

    mi::Size get_index() const { return m_value->get_index(); }

    mi::Sint32 set_value( mi::Sint32 value) { return m_value->set_value( value); }

    mi::Sint32 set_name( const char* name) { return m_value->set_name( name); }

    mi::Sint32 set_index( mi::Size index) { return m_value->set_index( index); }
};


class Value_float
  : public Value_base<mi::neuraylib::IValue_float, MDL::IValue_float, mi::neuraylib::IType_float>
{
public:
    Value_float(
        const Value_factory* vf, Internal_value* value, const mi::base::IInterface* owner)
      : Base( vf, value, owner) { }

    Float32 get_value() const { return m_value->get_value(); }

    void set_value( Float32 value) { m_value->set_value( value); }
};


class Value_double
  : public Value_base<mi::neuraylib::IValue_double, MDL::IValue_double, mi::neuraylib::IType_double>
{
public:
    Value_double(
        const Value_factory* vf, Internal_value* value, const mi::base::IInterface* owner)
      : Base( vf, value, owner) { }

    Float64 get_value() const { return m_value->get_value(); }

    void set_value( Float64 value) { m_value->set_value( value); }
};


class Value_string
  : public Value_base<mi::neuraylib::IValue_string, MDL::IValue_string, mi::neuraylib::IType_string>
{
public:
    Value_string(
        const Value_factory* vf, Internal_value* value, const mi::base::IInterface* owner)
      : Base( vf, value, owner) { }

    const char* get_value() const { return m_value->get_value(); }

    void set_value( const char* value) { m_value->set_value( value); }
};


class Value_string_localized
    : public Value_base<mi::neuraylib::IValue_string_localized, MDL::IValue_string_localized, mi::neuraylib::IType_string>
{
public:
    Value_string_localized(
        const Value_factory* vf, Internal_value* value, const mi::base::IInterface* owner)
        : Base( vf, value, owner) { }

    const char* get_value() const { return m_value->get_value(); }

    void set_value( const char* value) { m_value->set_value(value); }

    const char* get_original_value() const { return m_value->get_original_value(); }

    void set_original_value( const char* value) { m_value->set_original_value( value); }
};


class Value_vector
  : public Value_base<mi::neuraylib::IValue_vector, MDL::IValue_vector, mi::neuraylib::IType_vector>
{
public:
    Value_vector(
        const Value_factory* vf, Internal_value* value, const mi::base::IInterface* owner)
      : Base( vf, value, owner) { }

    mi::Size get_size() const { return m_value->get_size(); }

    const mi::neuraylib::IValue_atomic* get_value( mi::Size index) const;

    mi::neuraylib::IValue_atomic* get_value( mi::Size index);

    mi::Sint32 set_value( mi::Size index, mi::neuraylib::IValue* value);
};


class Value_matrix
  : public Value_base<mi::neuraylib::IValue_matrix, MDL::IValue_matrix, mi::neuraylib::IType_matrix>
{
public:
    Value_matrix(
        const Value_factory* vf, Internal_value* value, const mi::base::IInterface* owner)
      : Base( vf, value, owner) { }

    mi::Size get_size() const { return m_value->get_size(); }

    const mi::neuraylib::IValue_vector* get_value( mi::Size index) const;

    mi::neuraylib::IValue_vector* get_value( mi::Size index);

    mi::Sint32 set_value( mi::Size index, mi::neuraylib::IValue* value);
};


class Value_color
  : public Value_base<mi::neuraylib::IValue_color, MDL::IValue_color, mi::neuraylib::IType_color>
{
public:
    Value_color(
        const Value_factory* vf, Internal_value* value, const mi::base::IInterface* owner)
      : Base( vf, value, owner) { }

    mi::Size get_size() const { return m_value->get_size(); }

    const mi::neuraylib::IValue_float* get_value( mi::Size index) const;

    mi::neuraylib::IValue_float* get_value( mi::Size index);

    mi::Sint32 set_value( mi::Size index, mi::neuraylib::IValue* value);

    mi::Sint32 set_value( mi::Size index, mi::neuraylib::IValue_float* value);
};


class Value_array
  : public Value_base<mi::neuraylib::IValue_array, MDL::IValue_array, mi::neuraylib::IType_array>
{
public:
    Value_array(
        const Value_factory* vf, Internal_value* value, const mi::base::IInterface* owner)
      : Base( vf, value, owner) { }

    mi::Size get_size() const { return m_value->get_size(); }

    const mi::neuraylib::IValue* get_value( mi::Size index) const;

    mi::neuraylib::IValue* get_value( mi::Size index);

    mi::Sint32 set_value( mi::Size index, mi::neuraylib::IValue* value);

    mi::Sint32 set_size( mi::Size size) { return m_value->set_size( size); }
};


class Value_struct
  : public Value_base<mi::neuraylib::IValue_struct, MDL::IValue_struct, mi::neuraylib::IType_struct>
{
public:
    Value_struct(
        const Value_factory* vf, Internal_value* value, const mi::base::IInterface* owner)
      : Base( vf, value, owner) { }

    mi::Size get_size() const { return m_value->get_size(); }

    const mi::neuraylib::IValue* get_value( mi::Size index) const;

    mi::neuraylib::IValue* get_value( mi::Size index);

    mi::Sint32 set_value( mi::Size index, mi::neuraylib::IValue* value);

    const mi::neuraylib::IValue* get_field( const char* name) const;

    mi::neuraylib::IValue* get_field( const char* name);

    mi::Sint32 set_field( const char* name, mi::neuraylib::IValue* value);
};


class Value_texture
  : public Value_base<mi::neuraylib::IValue_texture,
        MDL::IValue_texture, mi::neuraylib::IType_texture>
{
public:
    Value_texture(
        const Value_factory* vf,
        mi::neuraylib::ITransaction* transaction,
        Internal_value* value,
        const mi::base::IInterface* owner)
      : Base( vf, value, owner), m_transaction( transaction, mi::base::DUP_INTERFACE) { }

    const char* get_value() const;

    mi::Sint32 set_value( const char* value);

    const char* get_file_path() const;

    mi::Float32 get_gamma() const;

private:
    const mi::base::Handle<mi::neuraylib::ITransaction> m_transaction;
};


class Value_light_profile
  : public Value_base<mi::neuraylib::IValue_light_profile,
        MDL::IValue_light_profile, mi::neuraylib::IType_light_profile>
{
public:
    Value_light_profile(
        const Value_factory* vf,
        mi::neuraylib::ITransaction* transaction,
        Internal_value* value,
        const mi::base::IInterface* owner)
      : Base( vf, value, owner), m_transaction( transaction, mi::base::DUP_INTERFACE) { }

    const char* get_value() const;

    mi::Sint32 set_value( const char* value);

    const char* get_file_path() const;

private:
    const mi::base::Handle<mi::neuraylib::ITransaction> m_transaction;
};


class Value_bsdf_measurement
  : public Value_base<mi::neuraylib::IValue_bsdf_measurement,
        MDL::IValue_bsdf_measurement, mi::neuraylib::IType_bsdf_measurement>
{
public:
    Value_bsdf_measurement(
        const Value_factory* vf,
        mi::neuraylib::ITransaction* transaction,
        Internal_value* value,
        const mi::base::IInterface* owner)
      : Base( vf, value, owner), m_transaction( transaction, mi::base::DUP_INTERFACE) { }

    const char* get_value() const;

    mi::Sint32 set_value( const char* value);

    const char* get_file_path() const;

private:
    const mi::base::Handle<mi::neuraylib::ITransaction> m_transaction;
};


class Value_invalid_df
  : public Value_base<mi::neuraylib::IValue_invalid_df,
        MDL::IValue_invalid_df, mi::neuraylib::IType_reference>
{
public:
    Value_invalid_df(
        const Value_factory* vf, Internal_value* value, const mi::base::IInterface* owner)
      : Base( vf, value, owner) { }
};


class Value_list
  : public mi::base::Interface_implement_2<mi::neuraylib::IValue_list, IValue_list_wrapper>
{
public:
    Value_list(
        const Value_factory* vf,
        MDL::IValue_list* value_list,
        const mi::base::IInterface* owner)
      : m_vf( vf, mi::base::DUP_INTERFACE),
        m_value_list( value_list, mi::base::DUP_INTERFACE),
        m_owner( owner, mi::base::DUP_INTERFACE)
    {
        ASSERT( M_NEURAY_API, m_vf);
        ASSERT( M_NEURAY_API, m_value_list);
    }

    // public API methods

    mi::Size get_size() const { return m_value_list->get_size(); }

    mi::Size get_index( const char* name) const { return m_value_list->get_index( name); }

    const char* get_name( mi::Size index) const { return m_value_list->get_name( index); }

    const mi::neuraylib::IValue* get_value( mi::Size index) const;

    const mi::neuraylib::IValue* get_value( const char* name) const;

    mi::Sint32 set_value( mi::Size index, const mi::neuraylib::IValue* value);

    mi::Sint32 set_value( const char* name, const mi::neuraylib::IValue* value);

    mi::Sint32 add_value( const char* name, const mi::neuraylib::IValue* value);

    // internal methods (IValue_list_wrapper)

    MDL::IValue_list* get_internal_value_list();

    const MDL::IValue_list* get_internal_value_list() const;

private:
    const mi::base::Handle<const Value_factory> m_vf;
    const mi::base::Handle<MDL::IValue_list> m_value_list;
    const mi::base::Handle<const mi::base::IInterface> m_owner;
};

class Value_factory : public mi::base::Interface_implement<mi::neuraylib::IValue_factory>
{
public:
    /// Constructor
    ///
    /// Most methods require a valid transaction, but #dump() works without transaction (but
    /// providing a transaction will produce better output).
    Value_factory( mi::neuraylib::ITransaction* transaction, mi::neuraylib::IType_factory* tf)
      : m_transaction( transaction, mi::base::DUP_INTERFACE),
        m_vf( MDL::get_value_factory()),
        m_tf( tf, mi::base::DUP_INTERFACE) { }

    // public API methods

    mi::neuraylib::IType_factory* get_type_factory() const { m_tf->retain(); return m_tf.get(); }

    mi::neuraylib::IValue_bool* create_bool( bool value) const;

    mi::neuraylib::IValue_int* create_int( mi::Sint32 value) const;

    mi::neuraylib::IValue_enum* create_enum(
        const mi::neuraylib::IType_enum* type, mi::Size index) const;

    mi::neuraylib::IValue_float* create_float( mi::Float32 value) const;

    mi::neuraylib::IValue_double* create_double( mi::Float64 value) const;

    mi::neuraylib::IValue_string* create_string( const char* value) const;

    mi::neuraylib::IValue_string_localized* create_string_localized( const char* value, const char* original) const;

    mi::neuraylib::IValue_vector* create_vector( const mi::neuraylib::IType_vector* type) const;

    mi::neuraylib::IValue_matrix* create_matrix( const mi::neuraylib::IType_matrix* type) const;

    mi::neuraylib::IValue_color* create_color(
        mi::Float32 red, mi::Float32 green, mi::Float32 blue) const;

    mi::neuraylib::IValue_array* create_array( const mi::neuraylib::IType_array* type) const;

    mi::neuraylib::IValue_struct* create_struct( const mi::neuraylib::IType_struct* type) const;

    mi::neuraylib::IValue_texture* create_texture(
        const mi::neuraylib::IType_texture* type, const char* value) const;

    mi::neuraylib::IValue_light_profile* create_light_profile( const char* value) const;

    mi::neuraylib::IValue_bsdf_measurement* create_bsdf_measurement( const char* value) const;

    mi::neuraylib::IValue_invalid_df* create_invalid_df(
        const mi::neuraylib::IType_reference* type) const;

    mi::neuraylib::IValue* create( const mi::neuraylib::IType* type) const;

    mi::neuraylib::IValue_list* create_value_list() const;

    mi::neuraylib::IValue* clone( const mi::neuraylib::IValue* value) const;

    mi::neuraylib::IValue_list* clone( const mi::neuraylib::IValue_list* value_list) const;

    mi::Sint32 compare( const mi::neuraylib::IValue* lhs, const mi::neuraylib::IValue* rhs) const;

    mi::Sint32 compare(
        const mi::neuraylib::IValue_list* lhs, const mi::neuraylib::IValue_list* rhs) const;

    const mi::IString* dump(
        const mi::neuraylib::IValue* value, const char* name, mi::Size depth) const;

    const mi::IString* dump(
        const mi::neuraylib::IValue_list* list, const char* name, mi::Size depth) const;

    // internal methods

    /// Returns the wrapped internal value factory.
    MDL::IValue_factory* get_internal_value_factory() const { m_vf->retain(); return m_vf.get(); }

    /// Creates a wrapper for the internal value.
    mi::neuraylib::IValue* create( MDL::IValue* value, const mi::base::IInterface* owner) const;

    /// Creates a wrapper for the internal value.
    template <class T>
    T* create( MDL::IValue* value, const mi::base::IInterface* owner) const
    {
        mi::base::Handle<mi::neuraylib::IValue> ptr_value( create( value, owner));
        if( !ptr_value)
            return 0;
        return static_cast<T*>( ptr_value->get_interface( typename T::IID()));
    }


    /// Creates a wrapper for the internal value.
    const mi::neuraylib::IValue* create(
        const MDL::IValue* value, const mi::base::IInterface* owner) const;

    /// Creates a wrapper for the internal value.
    template <class T>
    const T* create( const MDL::IValue* value, const mi::base::IInterface* owner) const
    {
        mi::base::Handle<const mi::neuraylib::IValue> ptr_value( create( value, owner));
        if( !ptr_value)
            return 0;
        return static_cast<const T*>( ptr_value->get_interface( typename T::IID()));
    }

    /// Creates a wrapper for the internal value list.
    mi::neuraylib::IValue_list* create_value_list(
        MDL::IValue_list* value_list, const mi::base::IInterface* owner) const;

    /// Creates a wrapper for the internal value list.
    const mi::neuraylib::IValue_list* create_value_list(
        const MDL::IValue_list* value_list, const mi::base::IInterface* owner) const;

    /// Returns the DB transaction corresponding to m_transaction.
    DB::Transaction* get_db_transaction() const;

private:
    const mi::base::Handle<mi::neuraylib::ITransaction> m_transaction;
    const mi::base::Handle<MDL::IValue_factory> m_vf;
    const mi::base::Handle<mi::neuraylib::IType_factory> m_tf;
};

} // namespace NEURAY

} // namespace MI

#endif // API_API_NEURAY_NEURAY_VALUE_IMPL_H
