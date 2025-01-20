/***************************************************************************************************
 * Copyright (c) 2010-2025, NVIDIA CORPORATION. All rights reserved.
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
 ** \brief Source for the IData_simple implementations.
 **/

#include "pch.h"

#include "idata_simple_impl.h"

#include <cstring>

#include <mi/base/config.h>
#include <base/system/main/i_assert.h>

#include "i_idata_factory.h"

// disable C4800: 'T' : forcing value to bool 'true' or 'false' (performance warning)
#ifdef MI_COMPILER_MSC
#pragma warning( disable : 4800 )
#endif

namespace MI {

namespace IDATA {

mi::base::IInterface* Enum_decl_impl::create_instance(
    const Factory* factory,
    DB::Transaction* transaction,
    mi::Uint32 argc,
    const mi::base::IInterface* argv[])
{
    if( argc != 0)
        return nullptr;
    return new Enum_decl_impl();
}

mi::Sint32 Enum_decl_impl::add_enumerator( const char* name, mi::Sint32 value)
{
    if( !name)
        return -1;

    for( auto& n : m_names)
        if( n == name)
            return -2;

    m_names.emplace_back( name);
    m_values.push_back( value);
    return 0;
}

mi::Sint32 Enum_decl_impl::remove_enumerator( const char* name)
{
    if( !name)
        return -1;

    for( mi::Size i = 0; i < m_names.size(); ++i)
        if( m_names[i] == name) {
            m_names.erase( m_names.begin() + i);
            m_values.erase( m_values.begin() + i);
            return 0;
        }

    return -2;
}

mi::Size Enum_decl_impl::get_length() const
{
    return m_names.size();
}

const char* Enum_decl_impl::get_name( mi::Size index) const
{
    if( index >= m_names.size())
        return nullptr;

    return m_names[index].c_str();
}

mi::Sint32 Enum_decl_impl::get_value( mi::Size index) const
{
    if( index >= m_values.size())
        return m_values[0];

    return m_values[index];
}

mi::Size Enum_decl_impl::get_index( const char* name) const
{
    if( !name)
        return 0;

    for( mi::Size i = 0; i < m_names.size(); ++i)
        if( m_names[i] == name)
            return i;

    return static_cast<mi::Size>( -1);
}

mi::Size Enum_decl_impl::get_index( mi::Sint32 value) const
{
    for( mi::Size i = 0; i < m_values.size(); ++i)
        if( m_values[i] == value)
            return i;

    return static_cast<mi::Size>( -1);
}

const char* Enum_decl_impl::get_enum_type_name() const
{
    if( m_enum_type_name.empty())
        return nullptr;

    return m_enum_type_name.c_str();
}

void Enum_decl_impl::set_enum_type_name( const char* enum_type_name)
{
    MI_ASSERT( enum_type_name);

    m_enum_type_name = enum_type_name;
}

mi::base::IInterface* Enum_impl::create_instance(
    const Factory* factory,
    DB::Transaction* transaction,
    mi::Uint32 argc,
    const mi::base::IInterface* argv[])
{
    if( argc != 2)
        return nullptr;

    mi::base::Handle enum_decl( argv[0]->get_interface<mi::IEnum_decl>());
    if( !enum_decl)
        return nullptr;

    mi::base::Handle type_name( argv[1]->get_interface<mi::IString>());
    if( !type_name)
        return nullptr;

    const char* type_name_cstr = type_name->get_c_str();
    return new Enum_impl( enum_decl.get(), type_name_cstr);
}

Enum_impl::Enum_impl( const mi::IEnum_decl* enum_decl, const char* type_name)
  : m_enum_decl( make_handle_dup( enum_decl)),
    m_type_name( type_name)
{
}

void Enum_impl::get_value( mi::Sint32& value) const
{
    value = m_enum_decl->get_value( m_storage);
}

const char* Enum_impl::get_value_by_name() const
{
    return m_enum_decl->get_name( m_storage);
}

mi::Sint32 Enum_impl::set_value( mi::Sint32 value)
{
    mi::Size index = m_enum_decl->get_index( value);
    if( index == static_cast<mi::Size>( -1))
        return -1;

    m_storage = static_cast<mi::Uint32>( index);
    return 0;
}

mi::Sint32 Enum_impl::set_value_by_name( const char* name)
{
    mi::Size index = m_enum_decl->get_index( name);
    if( index == static_cast<mi::Size>( -1))
        return -1;

    m_storage = static_cast<mi::Uint32>( index);
    return 0;
}

const mi::IEnum_decl* Enum_impl::get_enum_decl() const
{
    m_enum_decl->retain();
    return m_enum_decl.get();
}

mi::base::IInterface* Enum_impl_proxy::create_instance(
    const Factory* factory,
    DB::Transaction* transaction,
    mi::Uint32 argc,
    const mi::base::IInterface* argv[])
{
    if( argc != 2)
        return nullptr;

    mi::base::Handle enum_decl( argv[0]->get_interface<mi::IEnum_decl>());
    if( !enum_decl)
        return nullptr;

    mi::base::Handle type_name( argv[1]->get_interface<mi::IString>());
    if( !type_name)
        return nullptr;

    const char* type_name_cstr = type_name->get_c_str();
    return (new Enum_impl_proxy( enum_decl.get(), type_name_cstr))->cast_to_major();
}

Enum_impl_proxy::Enum_impl_proxy( const mi::IEnum_decl* enum_decl, const char* type_name)
  : m_enum_decl( make_handle_dup( enum_decl)),
    m_type_name( type_name)
{
}

void Enum_impl_proxy::get_value( mi::Sint32& value) const
{
    value = m_enum_decl->get_value( *m_pointer);
}

const char* Enum_impl_proxy::get_value_by_name() const
{
    return m_enum_decl->get_name( *m_pointer);
}

mi::Sint32 Enum_impl_proxy::set_value( mi::Sint32 value)
{
    mi::Size index = m_enum_decl->get_index( value);
    if( index == static_cast<mi::Size>( -1))
        return -1;

    *m_pointer = static_cast<mi::Uint32>( index);
    return 0;
}

mi::Sint32 Enum_impl_proxy::set_value_by_name( const char* name)
{
    mi::Size index = m_enum_decl->get_index( name);
    if( index == static_cast<mi::Size>( -1))
        return -1;

    *m_pointer = static_cast<mi::Uint32>( index);
    return 0;
}

const mi::IEnum_decl* Enum_impl_proxy::get_enum_decl() const
{
    m_enum_decl->retain();
    return m_enum_decl.get();
}

void Enum_impl_proxy::set_pointer_and_owner( void* pointer, const mi::base::IInterface* owner)
{
    m_pointer = static_cast<mi::Uint32*>( pointer);
    m_owner = make_handle_dup( owner);
}

void Enum_impl_proxy::release_referenced_memory()
{
    // nothing to do
}

template <typename I, typename T>
mi::base::IInterface* Number_impl<I, T>::create_instance(
    const Factory* factory,
    DB::Transaction* transaction,
    mi::Uint32 argc,
    const mi::base::IInterface* argv[])
{
    if( argc != 0)
        return nullptr;
    return new Number_impl<I, T>();
}

template <typename I, typename T>
const char* Number_impl<I, T>::get_type_name() const
{
    return Type_traits<T>::get_type_name();
}

template <>
const char* Number_impl<mi::ISize, mi::Size>::get_type_name() const
{
    return "Size";
}

template <>
const char* Number_impl<mi::IDifference, mi::Difference>::get_type_name() const
{
    return "Difference";
}

template <typename I, typename T>
void Number_impl<I, T>::get_value( bool& value) const
{
    value = static_cast<bool>( m_storage);
}

template <typename I, typename T>
void Number_impl<I, T>::set_value( bool value)
{
    m_storage = static_cast<T>( value);
}

template <typename I, typename T>
void Number_impl<I, T>::get_value( mi::Uint8& value) const
{
    value = static_cast<mi::Uint8>( m_storage);
}

template <typename I, typename T>
void Number_impl<I, T>::set_value( mi::Uint8 value)
{
    m_storage = static_cast<T>( value);
}

template <typename I, typename T>
void Number_impl<I, T>::get_value( mi::Uint16& value) const
{
    value = static_cast<mi::Uint16>( m_storage);
}

template <typename I, typename T>
void Number_impl<I, T>::set_value( mi::Uint16 value)
{
    m_storage = static_cast<T>( value);
}

template <typename I, typename T>
void Number_impl<I, T>::get_value( mi::Uint32& value) const
{
    value = static_cast<mi::Uint32>( m_storage);
}

template <typename I, typename T>
void Number_impl<I, T>::set_value( mi::Uint32 value)
{
    m_storage = static_cast<T>( value);
}

template <typename I, typename T>
void Number_impl<I, T>::get_value( mi::Uint64& value) const
{
    value = static_cast<mi::Uint64>( m_storage);
}

template <typename I, typename T>
void Number_impl<I, T>::set_value( mi::Uint64 value)
{
    m_storage = static_cast<T>( value);
}

template <typename I, typename T>
void Number_impl<I, T>::get_value( mi::Sint8& value) const
{
    value = static_cast<mi::Sint8>( m_storage);
}

template <typename I, typename T>
void Number_impl<I, T>::set_value( mi::Sint8 value)
{
    m_storage = static_cast<T>( value);
}

template <typename I, typename T>
void Number_impl<I, T>::get_value( mi::Sint16& value) const
{
    value = static_cast<mi::Sint16>( m_storage);
}

template <typename I, typename T>
void Number_impl<I, T>::set_value( mi::Sint16 value)
{
    m_storage = static_cast<T>( value);
}

template <typename I, typename T>
void Number_impl<I, T>::get_value( mi::Sint32& value) const
{
    value = static_cast<mi::Sint32>( m_storage);
}

template <typename I, typename T>
void Number_impl<I, T>::set_value( mi::Sint32 value)
{
    m_storage = static_cast<T>( value);
}

template <typename I, typename T>
void Number_impl<I, T>::get_value( mi::Sint64& value) const
{
    value = static_cast<mi::Sint64>( m_storage);
}

template <typename I, typename T>
void Number_impl<I, T>::set_value( mi::Sint64 value)
{
    m_storage = static_cast<T>( value);
}

template <typename I, typename T>
void Number_impl<I, T>::get_value( mi::Float32& value) const
{
    value = static_cast<mi::Float32>( m_storage);
}

template <typename I, typename T>
void Number_impl<I, T>::set_value( mi::Float32 value)
{
    m_storage = static_cast<T>( value);
}

template <typename I, typename T>
void Number_impl<I, T>::get_value( mi::Float64& value) const
{
    value = static_cast<mi::Float64>( m_storage);
}

template <typename I, typename T>
void Number_impl<I, T>::set_value( mi::Float64 value)
{
    m_storage = static_cast<T>( value);
}

template <typename I, typename T>
mi::base::IInterface* Number_impl_proxy<I, T>::create_instance(
    const Factory* factory,
    DB::Transaction* transaction,
    mi::Uint32 argc,
    const mi::base::IInterface* argv[])
{
    if( argc != 0)
        return nullptr;
    return (new Number_impl_proxy<I, T>())->cast_to_major();
}

template <typename I, typename T>
const char* Number_impl_proxy<I, T>::get_type_name() const
{
    return Type_traits<T>::get_type_name();
}

template <>
const char* Number_impl_proxy<mi::ISize, mi::Size>::get_type_name() const
{
    return "Size";
}

template <>
const char* Number_impl_proxy<mi::IDifference, mi::Difference>::get_type_name() const
{
    return "Difference";
}

template <typename I, typename T>
void Number_impl_proxy<I, T>::get_value( bool& value) const
{
    value = static_cast<bool>( *m_pointer);
}

template <typename I, typename T>
void Number_impl_proxy<I, T>::set_value( bool value)
{
    *m_pointer = static_cast<T>( value);
}

template <typename I, typename T>
void Number_impl_proxy<I, T>::get_value( mi::Uint8& value) const
{
    value = static_cast<mi::Uint8>( *m_pointer);
}

template <typename I, typename T>
void Number_impl_proxy<I, T>::set_value( mi::Uint8 value)
{
    *m_pointer = static_cast<T>( value);
}

template <typename I, typename T>
void Number_impl_proxy<I, T>::get_value( mi::Uint16& value) const
{
    value = static_cast<mi::Uint16>( *m_pointer);
}

template <typename I, typename T>
void Number_impl_proxy<I, T>::set_value( mi::Uint16 value)
{
    *m_pointer = static_cast<T>( value);
}

template <typename I, typename T>
void Number_impl_proxy<I, T>::get_value( mi::Uint32& value) const
{
    value = static_cast<mi::Uint32>( *m_pointer);
}

template <typename I, typename T>
void Number_impl_proxy<I, T>::set_value( mi::Uint32 value)
{
    *m_pointer = static_cast<T>( value);
}

template <typename I, typename T>
void Number_impl_proxy<I, T>::get_value( mi::Uint64& value) const
{
    value = static_cast<mi::Uint64>( *m_pointer);
}

template <typename I, typename T>
void Number_impl_proxy<I, T>::set_value( mi::Uint64 value)
{
    *m_pointer = static_cast<T>( value);
}

template <typename I, typename T>
void Number_impl_proxy<I, T>::get_value( mi::Sint8& value) const
{
    value = static_cast<mi::Sint8>( *m_pointer);
}

template <typename I, typename T>
void Number_impl_proxy<I, T>::set_value( mi::Sint8 value)
{
    *m_pointer = static_cast<T>( value);
}

template <typename I, typename T>
void Number_impl_proxy<I, T>::get_value( mi::Sint16& value) const
{
    value = static_cast<mi::Sint16>( *m_pointer);
}

template <typename I, typename T>
void Number_impl_proxy<I, T>::set_value( mi::Sint16 value)
{
    *m_pointer = static_cast<T>( value);
}

template <typename I, typename T>
void Number_impl_proxy<I, T>::get_value( mi::Sint32& value) const
{
    value = static_cast<mi::Sint32>( *m_pointer);
}

template <typename I, typename T>
void Number_impl_proxy<I, T>::set_value( mi::Sint32 value)
{
    *m_pointer = static_cast<T>( value);
}

template <typename I, typename T>
void Number_impl_proxy<I, T>::get_value( mi::Sint64& value) const
{
    value = static_cast<mi::Sint64>( *m_pointer);
}

template <typename I, typename T>
void Number_impl_proxy<I, T>::set_value( mi::Sint64 value)
{
    *m_pointer = static_cast<T>( value);
}

template <typename I, typename T>
void Number_impl_proxy<I, T>::get_value( mi::Float32& value) const
{
    value = static_cast<mi::Float32>( *m_pointer);
}

template <typename I, typename T>
void Number_impl_proxy<I, T>::set_value( mi::Float32 value)
{
    *m_pointer = static_cast<T>( value);
}

template <typename I, typename T>
void Number_impl_proxy<I, T>::get_value( mi::Float64& value) const
{
    value = static_cast<mi::Float64>( *m_pointer);
}

template <typename I, typename T>
void Number_impl_proxy<I, T>::set_value( mi::Float64 value)
{
    *m_pointer = static_cast<T>( value);
}

template <typename I, typename T>
void Number_impl_proxy<I, T>::set_pointer_and_owner(
    void* pointer, const mi::base::IInterface* owner)
{
    m_pointer = static_cast<T*>( pointer);
    m_owner = make_handle_dup( owner);
}

template <typename I, typename T>
void Number_impl_proxy<I, T>::release_referenced_memory()
{
    // nothing to do
}

template<> const char* Type_traits<bool       >::get_type_name() { return "Boolean"; }
template<> const char* Type_traits<mi::Sint8  >::get_type_name() { return "Sint8";   }
template<> const char* Type_traits<mi::Sint16 >::get_type_name() { return "Sint16";  }
template<> const char* Type_traits<mi::Sint32 >::get_type_name() { return "Sint32";  }
template<> const char* Type_traits<mi::Sint64 >::get_type_name() { return "Sint64";  }
template<> const char* Type_traits<mi::Uint8  >::get_type_name() { return "Uint8";   }
template<> const char* Type_traits<mi::Uint16 >::get_type_name() { return "Uint16";  }
template<> const char* Type_traits<mi::Uint32 >::get_type_name() { return "Uint32";  }
template<> const char* Type_traits<mi::Uint64 >::get_type_name() { return "Uint64";  }
template<> const char* Type_traits<mi::Float32>::get_type_name() { return "Float32"; }
template<> const char* Type_traits<mi::Float64>::get_type_name() { return "Float64"; }

// explicit template instantiation for Number_impl<I, T>
template class Number_impl<mi::IBoolean,    bool>;
template class Number_impl<mi::ISint8,      mi::Sint8>;
template class Number_impl<mi::ISint16,     mi::Sint16>;
template class Number_impl<mi::ISint32,     mi::Sint32>;
template class Number_impl<mi::ISint64,     mi::Sint64>;
template class Number_impl<mi::IUint8,      mi::Uint8>;
template class Number_impl<mi::IUint16,     mi::Uint16>;
template class Number_impl<mi::IUint32,     mi::Uint32>;
template class Number_impl<mi::IUint64,     mi::Uint64>;
template class Number_impl<mi::IFloat32,    mi::Float32>;
template class Number_impl<mi::IFloat64,    mi::Float64>;
template class Number_impl<mi::ISize,       mi::Size>;
template class Number_impl<mi::IDifference, mi::Difference>;

// explicit template instantiation for Number_impl_proxy<I, T>
template class Number_impl_proxy<mi::IBoolean,    bool>;
template class Number_impl_proxy<mi::ISint8,      mi::Sint8>;
template class Number_impl_proxy<mi::ISint16,     mi::Sint16>;
template class Number_impl_proxy<mi::ISint32,     mi::Sint32>;
template class Number_impl_proxy<mi::ISint64,     mi::Sint64>;
template class Number_impl_proxy<mi::IUint8,      mi::Uint8>;
template class Number_impl_proxy<mi::IUint16,     mi::Uint16>;
template class Number_impl_proxy<mi::IUint32,     mi::Uint32>;
template class Number_impl_proxy<mi::IUint64,     mi::Uint64>;
template class Number_impl_proxy<mi::IFloat32,    mi::Float32>;
template class Number_impl_proxy<mi::IFloat64,    mi::Float64>;

// explicit template instantiation for Type_traits<T>
template class Type_traits<mi::Sint8>;
template class Type_traits<mi::Sint16>;
template class Type_traits<mi::Sint32>;
template class Type_traits<mi::Sint64>;
template class Type_traits<mi::Uint8>;
template class Type_traits<mi::Uint16>;
template class Type_traits<mi::Uint32>;
template class Type_traits<mi::Uint64>;
template class Type_traits<mi::Float32>;
template class Type_traits<mi::Float64>;

mi::base::IInterface* Pointer_impl::create_instance(
    const Factory* factory,
    DB::Transaction* transaction,
    mi::Uint32 argc,
    const mi::base::IInterface* argv[])
{
    if( argc != 1)
        return nullptr;

    mi::base::Handle string( argv[0]->get_interface<mi::IString>());
    if( !string)
        return nullptr;

    const char* value_type_name = string->get_c_str();
    mi::base::Handle pointer( new Pointer_impl( factory, transaction, value_type_name));
    return pointer->successfully_constructed() ? pointer.extract() : nullptr;
}

Pointer_impl::Pointer_impl(
    const Factory* factory, DB::Transaction* transaction, const char* value_type_name)
  : m_transaction( transaction),
    m_value_type_name( value_type_name),
    m_type_name( "Pointer<" + m_value_type_name + '>')
{
    MI_ASSERT( value_type_name);

    std::string mangled_value_type_name
        = (m_value_type_name == "Interface") ? "Void" : m_value_type_name;
    mi::base::Handle<mi::base::IInterface> element( factory->create(
        transaction, mangled_value_type_name.c_str()));
    m_successfully_constructed = !!element;
}

mi::Sint32 Pointer_impl::set_pointer( mi::base::IInterface* pointer)
{
    if( !has_correct_value_type( pointer))
        return -1;

    m_pointer = make_handle_dup( pointer);
    return 0;
}

mi::base::IInterface* Pointer_impl::get_pointer() const
{
    mi::base::IInterface* pointer = m_pointer.get();
    if( pointer)
        pointer->retain();
    return pointer;
}

bool Pointer_impl::has_correct_value_type( const mi::base::IInterface* value) const
{
    if( !value)
        return true;

    if( m_value_type_name == "Interface")
        return true;

    mi::base::Handle data( value->get_interface<mi::IData>());
    if( !data)
        return false;

    const char* type_name = data->get_type_name();
    if( !type_name)
        return false;

    return m_value_type_name == type_name;
}

mi::base::IInterface* Const_pointer_impl::create_instance(
    const Factory* factory,
    DB::Transaction* transaction,
    mi::Uint32 argc,
    const mi::base::IInterface* argv[])
{
    if( argc != 1)
        return nullptr;

    mi::base::Handle string( argv[0]->get_interface<mi::IString>());
    if( !string)
        return nullptr;

    const char* value_type_name = string->get_c_str();
    mi::base::Handle const_pointer(
        new Const_pointer_impl( factory, transaction, value_type_name));
    return const_pointer->successfully_constructed() ? const_pointer.extract() : nullptr;
}

Const_pointer_impl::Const_pointer_impl(
    const Factory* factory, DB::Transaction* transaction, const char* value_type_name)
  : m_transaction( transaction),
    m_value_type_name( value_type_name),
    m_type_name( "Const_pointer<" + m_value_type_name + '>')
{
    MI_ASSERT( value_type_name);

    std::string mangled_value_type_name
        = (m_value_type_name == "Interface") ? "Void" : m_value_type_name;
    mi::base::Handle<mi::base::IInterface> element( factory->create(
        transaction, mangled_value_type_name.c_str()));
    m_successfully_constructed = !!element;
}

mi::Sint32 Const_pointer_impl::set_pointer( const mi::base::IInterface* pointer)
{
    if( !has_correct_value_type( pointer))
        return -1;

    m_pointer = make_handle_dup( pointer);
    return 0;
}

const mi::base::IInterface* Const_pointer_impl::get_pointer() const
{
    const mi::base::IInterface* pointer = m_pointer.get();
    if( pointer)
        pointer->retain();
    return pointer;
}

bool Const_pointer_impl::has_correct_value_type( const mi::base::IInterface* value) const
{
    if( !value)
        return true;

    if( m_value_type_name == "Interface")
        return true;

    mi::base::Handle data( value->get_interface<mi::IData>());
    if( !data)
        return false;

    const char* type_name = data->get_type_name();
    if( !type_name)
        return false;

    return m_value_type_name == type_name;
}

mi::base::IInterface* Ref_impl::create_instance(
    const Factory* factory,
    DB::Transaction* transaction,
    mi::Uint32 argc,
    const mi::base::IInterface* argv[])
{
    if( !transaction)
        return nullptr;
    if( argc != 0)
        return nullptr;
    return new Ref_impl( factory, transaction);
}

Ref_impl::Ref_impl( const Factory* factory, DB::Transaction* transaction)
  : m_factory( factory),
    m_transaction( transaction)
{
}

mi::Sint32 Ref_impl::set_reference( const IInterface* interface)
{
    if( !interface) {
        m_storage = DB::Tag();
        return 0;
    }

    mi::base::Handle<ITag_handler> tag_handler( m_factory->get_tag_handler());

    mi::Sint32 result = 0;
    DB::Tag tag;
    std::tie( tag, result) = tag_handler->get_tag( interface);
    if( result != 0)
        return result;

    m_storage = tag;
    return 0;
}

mi::Sint32 Ref_impl::set_reference( const char* name)
{
    if( !name) {
        m_storage = DB::Tag();
        return 0;
    }

    mi::base::Handle<ITag_handler> tag_handler( m_factory->get_tag_handler());
    DB::Tag tag = tag_handler->name_to_tag( m_transaction, name);
    if( !tag)
        return -2;

    m_storage = tag;
    return 0;
}

const mi::base::IInterface* Ref_impl::get_reference() const
{
    mi::base::Handle<ITag_handler> tag_handler( m_factory->get_tag_handler());
    return m_storage ? tag_handler->access_tag( m_transaction, m_storage) : nullptr;
}

mi::base::IInterface* Ref_impl::get_reference()
{
    mi::base::Handle<ITag_handler> tag_handler( m_factory->get_tag_handler());
    return m_storage ? tag_handler->edit_tag( m_transaction, m_storage) : nullptr;
}

const char* Ref_impl::get_reference_name() const
{
    mi::base::Handle<ITag_handler> tag_handler( m_factory->get_tag_handler());
    return m_storage ? tag_handler->tag_to_name( m_transaction, m_storage) : nullptr;
}

mi::base::IInterface* Ref_impl_proxy::create_instance(
    const Factory* factory,
    DB::Transaction* transaction,
    mi::Uint32 argc,
    const mi::base::IInterface* argv[])
{
    if( !transaction)
        return nullptr;
    if( argc != 0)
        return nullptr;
    return (new Ref_impl_proxy( factory, transaction))->cast_to_major();
}

Ref_impl_proxy::Ref_impl_proxy( const Factory* factory, DB::Transaction* transaction)
  : m_factory( factory),
    m_transaction( transaction)
{
    MI_ASSERT( transaction);
}

mi::Sint32 Ref_impl_proxy::set_reference( const IInterface* interface)
{
    if( !interface) {
        *m_pointer = DB::Tag();
        return 0;
    }

    mi::base::Handle<ITag_handler> tag_handler( m_factory->get_tag_handler());

    mi::Sint32 result = 0;
    DB::Tag tag;
    std::tie( tag, result) = tag_handler->get_tag( interface);
    if( result != 0)
        return result;

    *m_pointer = tag;
    return 0;
}

mi::Sint32 Ref_impl_proxy::set_reference( const char* name)
{
    if( !name) {
        *m_pointer = DB::Tag();
        return 0;
    }

    mi::base::Handle<ITag_handler> tag_handler( m_factory->get_tag_handler());
    DB::Tag tag = tag_handler->name_to_tag( m_transaction, name);
    if( !tag)
        return -2;

    mi::base::Handle attribute_context( m_owner->get_interface<IAttribute_context>());
    if( !attribute_context->can_reference_tag( tag))
        return -4;

    *m_pointer = tag;
    return 0;
}

const mi::base::IInterface* Ref_impl_proxy::get_reference() const
{
    mi::base::Handle<ITag_handler> tag_handler( m_factory->get_tag_handler());
    return *m_pointer ? tag_handler->access_tag( m_transaction, *m_pointer) : nullptr;
}

mi::base::IInterface* Ref_impl_proxy::get_reference()
{
    mi::base::Handle<ITag_handler> tag_handler( m_factory->get_tag_handler());
    return *m_pointer ? tag_handler->edit_tag( m_transaction, *m_pointer) : nullptr;
}

const char* Ref_impl_proxy::get_reference_name() const
{
    mi::base::Handle<ITag_handler> tag_handler( m_factory->get_tag_handler());
    return *m_pointer ? tag_handler->tag_to_name( m_transaction, *m_pointer) : nullptr;
}

void Ref_impl_proxy::set_pointer_and_owner(
    void* pointer, const mi::base::IInterface* owner)
{
    m_pointer = static_cast<DB::Tag*>( pointer);
    m_owner = make_handle_dup( owner);
}

void Ref_impl_proxy::release_referenced_memory()
{
    // nothing to do
}

mi::base::IInterface* String_impl::create_instance(
    const Factory* factory,
    DB::Transaction* transaction,
    mi::Uint32 argc,
    const mi::base::IInterface* argv[])
{
    if( argc != 0)
        return nullptr;
    return new String_impl();
}

mi::base::IInterface* String_impl_proxy::create_instance(
    const Factory* factory,
    DB::Transaction* transaction,
    mi::Uint32 argc,
    const mi::base::IInterface* argv[])
{
    if( argc != 0)
        return nullptr;
    return (new String_impl_proxy())->cast_to_major();
}

const char* String_impl_proxy::get_c_str() const
{
    // Note that the constructor of the proxy implementation does not call set_c_str( "").
    // Hence, we have to check for nullptr here.
    if( *m_pointer == nullptr)
        return "";
    return *m_pointer;
}

void String_impl_proxy::set_c_str( const char* str)
{
    // Defer deallocation of *m_pointer, str might be identical to *m_pointer.
    if( !str)
        str = "";
    const mi::Size len = strlen( str);
    char* const copy = new char[len+1];
    memcpy( copy, str, len+1);
    delete[] *m_pointer;
    *m_pointer = copy;
}

void String_impl_proxy::set_pointer_and_owner( void* pointer, const mi::base::IInterface* owner)
{
    m_pointer = static_cast<const char**>( pointer);
    m_owner = make_handle_dup( owner);
}

void String_impl_proxy::release_referenced_memory()
{
    delete[] *m_pointer;
    *m_pointer = nullptr;
}

mi::base::IInterface* Uuid_impl::create_instance(
    const Factory* factory,
    DB::Transaction* transaction,
    mi::Uint32 argc,
    const mi::base::IInterface* argv[])
{
    if( argc != 0)
        return nullptr;
    return new Uuid_impl();
}

mi::base::IInterface* Void_impl::create_instance(
    const Factory* factory,
    DB::Transaction* transaction,
    mi::Uint32 argc,
    const mi::base::IInterface* argv[])
{
    if( argc != 0)
        return nullptr;
    return new Void_impl();
}

} // namespace IDATA

} // namespace MI
