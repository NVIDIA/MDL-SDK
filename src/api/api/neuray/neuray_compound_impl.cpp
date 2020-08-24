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
 ** \brief Source for the ICompound implementation.
 **/

#include "pch.h"

#include "neuray_class_factory.h"
#include "neuray_compound_impl.h"
#include "neuray_number_impl.h"

#include <mi/base/config.h>
#include <mi/neuraylib/inumber.h>

#include <sstream>
#include <base/lib/log/i_log_assert.h>

// disable C4800: 'T' : forcing value to bool 'true' or 'false' (performance warning)
#ifdef MI_COMPILER_MSC
#pragma warning( disable : 4800 )
#endif

namespace MI {

namespace NEURAY {

template <typename I, typename T, mi::Size ROWS, mi::Size COLUMNS>
Compound_impl<I, T, ROWS, COLUMNS>::Compound_impl()
{
    std::ostringstream s;
    s << get_element_type_name() << "<" << ROWS;
    if( COLUMNS != 1)
        s << "," << COLUMNS;
    s << ">";
    m_type_name = s.str();

    m_storage = new T[ROWS*COLUMNS];
    for( mi::Size i = 0; i < ROWS*COLUMNS; ++i)
        m_storage[i] = static_cast<T>( 0);
}

template <typename I, typename T, mi::Size ROWS, mi::Size COLUMNS>
Compound_impl<I, T, ROWS, COLUMNS>::~Compound_impl()
{
    if( !m_owner.is_valid_interface())
        delete[] m_storage;
}

template <typename I, typename T, mi::Size ROWS, mi::Size COLUMNS>
const char* Compound_impl<I, T, ROWS, COLUMNS>::get_type_name() const
{
    return m_type_name.c_str();
}

template <typename I, typename T, mi::Size ROWS, mi::Size COLUMNS>
const char* Compound_impl<I, T, ROWS, COLUMNS>::get_key( mi::Size index) const
{
    std::string key;
    if( !index_to_key( index, key))
        return nullptr;

    m_cached_key = key;
    return m_cached_key.c_str();
}

template <typename I, typename T, mi::Size ROWS, mi::Size COLUMNS>
bool Compound_impl<I, T, ROWS, COLUMNS>::has_key( const char* key) const
{
    mi::Size index;
    return key_to_index( key, index);
}

template <typename I, typename T, mi::Size ROWS, mi::Size COLUMNS>
const mi::base::IInterface* Compound_impl<I, T, ROWS, COLUMNS>::get_value( const char* key) const
{
    mi::Size index;
    if( !key_to_index( key, index))
        return nullptr;

    return get_value( index);
}

template <typename I, typename T, mi::Size ROWS, mi::Size COLUMNS>
mi::base::IInterface* Compound_impl<I, T, ROWS, COLUMNS>::get_value( const char* key)
{
    mi::Size index;
    if( !key_to_index( key, index))
        return nullptr;

    return get_value( index);
}

template <typename I, typename T, mi::Size ROWS, mi::Size COLUMNS>
const mi::base::IInterface* Compound_impl<I, T, ROWS, COLUMNS>::get_value( mi::Size index) const
{
    if( index >= ROWS*COLUMNS)
        return nullptr;

    std::string element_proxy_class_name = "__";
    element_proxy_class_name += get_element_type_name();
    element_proxy_class_name += "_proxy";

    mi::INumber* value
        = s_class_factory->create_type_instance<mi::INumber>( nullptr, element_proxy_class_name.c_str());
    mi::base::Handle<IProxy> proxy( value->get_interface<IProxy>());
    proxy->set_pointer_and_owner( &m_storage[index], this->cast_to_major());
    return value;
}

template <typename I, typename T, mi::Size ROWS, mi::Size COLUMNS>
mi::base::IInterface* Compound_impl<I, T, ROWS, COLUMNS>::get_value( mi::Size index)
{
    if( index >= ROWS*COLUMNS)
        return nullptr;

    std::string element_proxy_class_name = "__";
    element_proxy_class_name += get_element_type_name();
    element_proxy_class_name += "_proxy";

    mi::INumber* value
        = s_class_factory->create_type_instance<mi::INumber>( nullptr, element_proxy_class_name.c_str());
    mi::base::Handle<IProxy> proxy( value->get_interface<IProxy>());
    proxy->set_pointer_and_owner( &m_storage[index], this->cast_to_major());
    return value;
}

template <typename I, typename T, mi::Size ROWS, mi::Size COLUMNS>
mi::Sint32 Compound_impl<I, T, ROWS, COLUMNS>::set_value(
    const char* key, mi::base::IInterface* value)
{
    if( !value)
        return -1;
    mi::Size index = 0; // avoid compiler warning
    if( !key_to_index( key, index))
        return -2;

    return set_value( index, value);
}

template <typename I, typename T, mi::Size ROWS, mi::Size COLUMNS>
mi::Sint32 Compound_impl<I, T, ROWS, COLUMNS>::set_value(
    mi::Size index, mi::base::IInterface* value)
{
    if( !value)
        return -1;
    if( index >= ROWS*COLUMNS)
        return -2;

    mi::base::Handle<mi::INumber> ivalue( value->get_interface<mi::INumber>());
    if( !ivalue.is_valid_interface())
        return -3;
    if( strcmp( ivalue->get_type_name(), get_element_type_name()) != 0)
        return -3;
    m_storage[index] = ivalue->get_value<T>();
    return 0;
}

template <typename I, typename T, mi::Size ROWS, mi::Size COLUMNS>
mi::Size Compound_impl<I, T, ROWS, COLUMNS>::get_number_of_rows() const
{
    return ROWS;
}

template <typename I, typename T, mi::Size ROWS, mi::Size COLUMNS>
mi::Size Compound_impl<I, T, ROWS, COLUMNS>::get_number_of_columns() const
{
    return COLUMNS;
}

template <typename I, typename T, mi::Size ROWS, mi::Size COLUMNS>
mi::Size Compound_impl<I, T, ROWS, COLUMNS>::get_length() const
{
    return ROWS * COLUMNS;
}

template <typename I, typename T, mi::Size ROWS, mi::Size COLUMNS>
const char* Compound_impl<I, T, ROWS, COLUMNS>::get_element_type_name() const
{
    return Type_traits<T>::get_type_name();
}

template <typename I, typename T, mi::Size ROWS, mi::Size COLUMNS>
bool Compound_impl<I, T, ROWS, COLUMNS>::get_value(
    mi::Size row, mi::Size column, bool& value) const
{
    if(( row >= ROWS) || ( column >= COLUMNS)) {
        value = false;
        return false;
    }
    value = static_cast<bool>( m_storage[ row*COLUMNS + column]);
    return true;
}

template <typename I, typename T, mi::Size ROWS, mi::Size COLUMNS>
bool Compound_impl<I, T, ROWS, COLUMNS>::get_value(
    mi::Size row, mi::Size column, mi::Sint32& value) const
{
    if(( row >= ROWS) || ( column >= COLUMNS)) {
        value = static_cast<mi::Sint32>( 0);
        return false;
    }
    value = static_cast<mi::Sint32>( m_storage[ row*COLUMNS + column]);
    return true;
}

template <typename I, typename T, mi::Size ROWS, mi::Size COLUMNS>
bool Compound_impl<I, T, ROWS, COLUMNS>::get_value(
    mi::Size row, mi::Size column, mi::Uint32& value) const
{
    if(( row >= ROWS) || ( column >= COLUMNS)) {
        value = static_cast<mi::Uint32>( 0);
        return false;
    }
    value = static_cast<mi::Uint32>( m_storage[ row*COLUMNS + column]);
    return true;
}

template <typename I, typename T, mi::Size ROWS, mi::Size COLUMNS>
bool Compound_impl<I, T, ROWS, COLUMNS>::get_value(
    mi::Size row, mi::Size column, mi::Float32& value) const
{
    if(( row >= ROWS) || ( column >= COLUMNS)) {
        value = static_cast<mi::Float32>( 0);
        return false;
    }
    value = static_cast<mi::Float32>( m_storage[ row*COLUMNS + column]);
    return true;
}

template <typename I, typename T, mi::Size ROWS, mi::Size COLUMNS>
bool Compound_impl<I, T, ROWS, COLUMNS>::get_value(
    mi::Size row, mi::Size column, mi::Float64& value) const
{
    if(( row >= ROWS) || ( column >= COLUMNS)) {
        value = static_cast<mi::Float64>( 0);
        return false;
    }
    value = static_cast<mi::Float64>( m_storage[ row*COLUMNS + column]);
    return true;
}

template <typename I, typename T, mi::Size ROWS, mi::Size COLUMNS>
bool Compound_impl<I, T, ROWS, COLUMNS>::set_value(
    mi::Size row, mi::Size column, bool value)
{
    if(( row >= ROWS) || ( column >= COLUMNS))
        return false;
    m_storage[ row*COLUMNS + column] = static_cast<T>( value);
    return true;
}

template <typename I, typename T, mi::Size ROWS, mi::Size COLUMNS>
bool Compound_impl<I, T, ROWS, COLUMNS>::set_value(
    mi::Size row, mi::Size column, mi::Sint32 value)
{
    if(( row >= ROWS) || ( column >= COLUMNS))
        return false;
    m_storage[ row*COLUMNS + column] = static_cast<T>( value);
    return true;
}

template <typename I, typename T, mi::Size ROWS, mi::Size COLUMNS>
bool Compound_impl<I, T, ROWS, COLUMNS>::set_value(
    mi::Size row, mi::Size column, mi::Uint32 value)
{
    if(( row >= ROWS) || ( column >= COLUMNS))
        return false;
    m_storage[ row*COLUMNS + column] = static_cast<T>( value);
    return true;
}

template <typename I, typename T, mi::Size ROWS, mi::Size COLUMNS>
bool Compound_impl<I, T, ROWS, COLUMNS>::set_value(
    mi::Size row, mi::Size column, mi::Float32 value)
{
    if(( row >= ROWS) || ( column >= COLUMNS))
        return false;
    m_storage[ row*COLUMNS + column] = static_cast<T>( value);
    return true;
}

template <typename I, typename T, mi::Size ROWS, mi::Size COLUMNS>
bool Compound_impl<I, T, ROWS, COLUMNS>::set_value(
    mi::Size row, mi::Size column, mi::Float64 value)
{
    if(( row >= ROWS) || ( column >= COLUMNS))
        return false;
    m_storage[ row*COLUMNS + column] = static_cast<T>( value);
    return true;
}

template <typename I, typename T, mi::Size ROWS, mi::Size COLUMNS>
void Compound_impl<I, T, ROWS, COLUMNS>::get_values( bool* values) const
{
    for( mi::Size i = 0; i < ROWS*COLUMNS; ++i)
        values[i] = static_cast<bool>( m_storage[i]);
}

template <typename I, typename T, mi::Size ROWS, mi::Size COLUMNS>
void Compound_impl<I, T, ROWS, COLUMNS>::get_values( mi::Sint32* values) const
{
    for( mi::Size i = 0; i < ROWS*COLUMNS; ++i)
        values[i] = static_cast<mi::Sint32>( m_storage[i]);
}

template <typename I, typename T, mi::Size ROWS, mi::Size COLUMNS>
void Compound_impl<I, T, ROWS, COLUMNS>::get_values( mi::Uint32* values) const
{
    for( mi::Size i = 0; i < ROWS*COLUMNS; ++i)
        values[i] = static_cast<mi::Uint32>( m_storage[i]);
}

template <typename I, typename T, mi::Size ROWS, mi::Size COLUMNS>
void Compound_impl<I, T, ROWS, COLUMNS>::get_values( mi::Float32* values) const
{
    for( mi::Size i = 0; i < ROWS*COLUMNS; ++i)
        values[i] = static_cast<mi::Float32>( m_storage[i]);
}

template <typename I, typename T, mi::Size ROWS, mi::Size COLUMNS>
void Compound_impl<I, T, ROWS, COLUMNS>::get_values( mi::Float64* values) const
{
    for( mi::Size i = 0; i < ROWS*COLUMNS; ++i)
        values[i] = static_cast<mi::Float64>( m_storage[i]);
}

template <typename I, typename T, mi::Size ROWS, mi::Size COLUMNS>
void Compound_impl<I, T, ROWS, COLUMNS>::set_values( const bool* values)
{
    for( mi::Size i = 0; i < ROWS*COLUMNS; ++i)
        m_storage[i] = static_cast<T>( values[i]);
}

template <typename I, typename T, mi::Size ROWS, mi::Size COLUMNS>
void Compound_impl<I, T, ROWS, COLUMNS>::set_values( const mi::Sint32* values)
{
    for( mi::Size i = 0; i < ROWS*COLUMNS; ++i)
        m_storage[i] = static_cast<T>( values[i]);
}

template <typename I, typename T, mi::Size ROWS, mi::Size COLUMNS>
void Compound_impl<I, T, ROWS, COLUMNS>::set_values( const mi::Uint32* values)
{
    for( mi::Size i = 0; i < ROWS*COLUMNS; ++i)
        m_storage[i] = static_cast<T>( values[i]);
}

template <typename I, typename T, mi::Size ROWS, mi::Size COLUMNS>
void Compound_impl<I, T, ROWS, COLUMNS>::set_values( const mi::Float32* values)
{
    for( mi::Size i = 0; i < ROWS*COLUMNS; ++i)
        m_storage[i] = static_cast<T>( values[i]);
}

template <typename I, typename T, mi::Size ROWS, mi::Size COLUMNS>
void Compound_impl<I, T, ROWS, COLUMNS>::set_values( const mi::Float64* values)
{
    for( mi::Size i = 0; i < ROWS*COLUMNS; ++i)
        m_storage[i] = static_cast<T>( values[i]);
}

template <typename I, typename T, mi::Size ROWS, mi::Size COLUMNS>
void Compound_impl<I, T, ROWS, COLUMNS>::set_pointer_and_owner(
    void* pointer, const mi::base::IInterface* owner)
{
    ASSERT( M_NEURAY_API, owner);
    if( !m_owner.is_valid_interface())
        delete[] m_storage;

    m_storage = static_cast<T*>( pointer);
    m_owner = make_handle_dup( owner);
}

template <typename I, typename T, mi::Size ROWS, mi::Size COLUMNS>
void Compound_impl<I, T, ROWS, COLUMNS>::release_referenced_memory()
{
    // nothing to do (as long as neither string nor collections can be elements of compounds)
}

template <typename I, typename T, mi::Size ROWS, mi::Size COLUMNS>
bool Compound_impl<I, T, ROWS, COLUMNS>::key_to_index( const char* key, mi::Size& index) const
{
    return false;
}

template <typename I, typename T, mi::Size ROWS, mi::Size COLUMNS>
bool Compound_impl<I, T, ROWS, COLUMNS>::index_to_key( mi::Size index, std::string& key) const
{
    return false;
}

template <typename I, typename T, mi::Size ROWS>
mi::base::IInterface* Vector_impl<I, T, ROWS>::create_api_class(
    mi::neuraylib::ITransaction* transaction,
    mi::Uint32 argc,
    const mi::base::IInterface* argv[])
{
    if( argc != 0)
        return nullptr;
    return (new Vector_impl<I,T,ROWS>())->cast_to_major();
}

template <typename I, typename T, mi::Size ROWS>
mi::math::Vector_struct<T,ROWS> Vector_impl<I,T,ROWS>::get_value() const
{
    mi::math::Vector_struct<T,ROWS> result;
    get_value( result);
    return result;
}

template <typename I, typename T, mi::Size ROWS>
void Vector_impl<I,T,ROWS>::get_value( mi::math::Vector_struct<T,ROWS>& value) const
{
    for( mi::Size i = 0; i < ROWS; ++i)
        (&value.x)[i] = this->m_storage[i];
}

template <typename I, typename T, mi::Size ROWS>
void Vector_impl<I,T,ROWS>::set_value( const mi::math::Vector_struct<T,ROWS>& value)
{
    for( mi::Size i = 0; i < ROWS; ++i)
        this->m_storage[i] = (&value.x)[i];
}

template <typename I, typename T, mi::Size ROWS>
bool Vector_impl<I,T,ROWS>::key_to_index( const char* key, mi::Size& index) const
{
    if( !key)
        return false;
    if( strlen( key) != 1)
        return false;

    if( (key[0] == 'x')               ) { index = 0; return true; }
    if( (key[0] == 'y') && (ROWS >= 2)) { index = 1; return true; }
    if( (key[0] == 'z') && (ROWS >= 3)) { index = 2; return true; }
    if( (key[0] == 'w') && (ROWS >= 4)) { index = 3; return true; }
    return false;
}

template <typename I, typename T, mi::Size ROWS>
bool Vector_impl<I,T,ROWS>::index_to_key( mi::Size index, std::string& key) const
{
    if( index >= ROWS)
        return false;

    key = (index == 3) ? 'w' : static_cast<char>( 'x' + index); //-V601 PVS
    return true;
}

template <typename I, typename T, mi::Size ROWS, mi::Size COLUMNS>
mi::base::IInterface* Matrix_impl<I, T, ROWS, COLUMNS>::create_api_class(
    mi::neuraylib::ITransaction* transaction,
    mi::Uint32 argc,
    const mi::base::IInterface* argv[])
{
    if( argc != 0)
        return nullptr;
    return (new Matrix_impl<I,T,ROWS,COLUMNS>())->cast_to_major();
}

template <typename I, typename T, mi::Size ROWS, mi::Size COLUMNS>
mi::math::Matrix_struct<T,ROWS,COLUMNS> Matrix_impl<I,T,ROWS,COLUMNS>::get_value() const
{
    mi::math::Matrix_struct<T,ROWS,COLUMNS> result;
    get_value( result);
    return result;
}

template <typename I, typename T, mi::Size ROWS, mi::Size COLUMNS>
void Matrix_impl<I,T,ROWS,COLUMNS>::get_value(
    mi::math::Matrix_struct<T,ROWS,COLUMNS>& value) const
{
    for( mi::Size i = 0; i < ROWS*COLUMNS; ++i)
        (&value.xx)[i] = this->m_storage[i];
}

template <typename I, typename T, mi::Size ROWS, mi::Size COLUMNS>
void Matrix_impl<I,T,ROWS,COLUMNS>::set_value(
    const mi::math::Matrix_struct<T,ROWS,COLUMNS>& value)
{
    for( mi::Size i = 0; i < ROWS*COLUMNS; ++i)
        this->m_storage[i] = (&value.xx)[i];
}

template <typename I, typename T, mi::Size ROWS, mi::Size COLUMNS>
bool Matrix_impl<I,T,ROWS,COLUMNS>::key_to_index( const char* key, mi::Size& index) const
{
    if( !key)
        return false;
    if( strlen( key) != 2)
        return false;

    int tmp = -1;
    if( (key[0] == 'x')               ) { tmp = 0*COLUMNS; }
    if( (key[0] == 'y') && (ROWS >= 2)) { tmp = 1*COLUMNS; }
    if( (key[0] == 'z') && (ROWS >= 3)) { tmp = 2*COLUMNS; }
    if( (key[0] == 'w') && (ROWS >= 4)) { tmp = 3*COLUMNS; }
    if( tmp == -1)
        return false;

    if( (key[1] == 'x')                  ) { index = tmp  ; return true; }
    if( (key[1] == 'y') && (COLUMNS >= 2)) { index = tmp+1; return true; }
    if( (key[1] == 'z') && (COLUMNS >= 3)) { index = tmp+2; return true; }
    if( (key[1] == 'w') && (COLUMNS >= 4)) { index = tmp+3; return true; }
    return false;
}

template <typename I, typename T, mi::Size ROWS, mi::Size COLUMNS>
bool Matrix_impl<I,T,ROWS,COLUMNS>::index_to_key( mi::Size index, std::string& key) const
{
    if( index >= ROWS*COLUMNS)
        return false;

    mi::Size row    = index / COLUMNS;
    mi::Size column = index % COLUMNS;
    key  = (row    == 3) ? 'w' : static_cast<char>( 'x' + row   ); //-V601 PVS
    key += (column == 3) ? 'w' : static_cast<char>( 'x' + column); //-V601 PVS
    return true;
}

mi::base::IInterface* Color_impl::create_api_class(
    mi::neuraylib::ITransaction* transaction,
    mi::Uint32 argc,
    const mi::base::IInterface* argv[])
{
    if( argc != 0)
        return nullptr;
    return (new Color_impl)->cast_to_major();
}

Color_impl::Color_impl()
{
    m_type_name = "Color";
}

bool Color_impl::key_to_index( const char* key, mi::Size& index) const
{
    if( !key)
        return false;

    std::string key_string( key);
    if( key_string == "r") { index = 0; return true; }
    if( key_string == "g") { index = 1; return true; }
    if( key_string == "b") { index = 2; return true; }
    if( key_string == "a") { index = 3; return true; }
    return false;
}

bool Color_impl::index_to_key( mi::Size index, std::string& key) const
{
    if( index == 0) { key = "r"; return true; }
    if( index == 1) { key = "g"; return true; }
    if( index == 2) { key = "b"; return true; }
    if( index == 3) { key = "a"; return true; }
    return false;
}

mi::Color_struct Color_impl::get_value() const
{
    mi::Color_struct result;
    get_value( result);
    return result;
}

void Color_impl::get_value( mi::Color_struct& value) const
{
    value.r = this->m_storage[0];
    value.g = this->m_storage[1];
    value.b = this->m_storage[2];
    value.a = this->m_storage[3];
}

void Color_impl::set_value( const mi::Color_struct& value)
{
    this->m_storage[0] = value.r;
    this->m_storage[1] = value.g;
    this->m_storage[2] = value.b;
    this->m_storage[3] = value.a;
}

mi::base::IInterface* Color3_impl::create_api_class(
    mi::neuraylib::ITransaction* transaction,
    mi::Uint32 argc,
    const mi::base::IInterface* argv[])
{
    if( argc != 0)
        return nullptr;
    return (new Color3_impl)->cast_to_major();
}

Color3_impl::Color3_impl()
{
    m_type_name = "Color3";
}

bool Color3_impl::key_to_index( const char* key, mi::Size& index) const
{
    if( !key)
        return false;

    std::string key_string( key);
    if( key_string == "r") { index = 0; return true; }
    if( key_string == "g") { index = 1; return true; }
    if( key_string == "b") { index = 2; return true; }
    return false;
}

bool Color3_impl::index_to_key( mi::Size index, std::string& key) const
{
    if( index == 0) { key = "r"; return true; }
    if( index == 1) { key = "g"; return true; }
    if( index == 2) { key = "b"; return true; }
    return false;
}

mi::Color_struct Color3_impl::get_value() const
{
    mi::Color_struct result;
    get_value( result);
    return result;
}

void Color3_impl::get_value( mi::Color_struct& value) const
{
    value.r = this->m_storage[0];
    value.g = this->m_storage[1];
    value.b = this->m_storage[2];
    value.a = 1.0f;
}

void Color3_impl::set_value( const mi::Color_struct& value)
{
    this->m_storage[0] = value.r;
    this->m_storage[1] = value.g;
    this->m_storage[2] = value.b;
}

mi::base::IInterface* Spectrum_impl::create_api_class(
    mi::neuraylib::ITransaction* transaction,
    mi::Uint32 argc,
    const mi::base::IInterface* argv[])
{
    if( argc != 0)
        return nullptr;
    return (new Spectrum_impl)->cast_to_major();
}

Spectrum_impl::Spectrum_impl()
{
    m_type_name = "Spectrum";
}

bool Spectrum_impl::key_to_index( const char* key, mi::Size& index) const
{
    if( !key)
        return false;

    std::string key_string( key);
    if( key_string == "0") { index = 0; return true; }
    if( key_string == "1") { index = 1; return true; }
    if( key_string == "2") { index = 2; return true; }
    return false;
}

bool Spectrum_impl::index_to_key( mi::Size index, std::string& key) const
{
    if( index == 0) { key = "0"; return true; }
    if( index == 1) { key = "1"; return true; }
    if( index == 2) { key = "2"; return true; }
    return false;
}

mi::Spectrum_struct Spectrum_impl::get_value() const
{
    mi::Spectrum_struct result;
    get_value( result);
    return result;
}

void Spectrum_impl::get_value( mi::Spectrum_struct& value) const
{
    value.c[0] = this->m_storage[0];
    value.c[1] = this->m_storage[1];
    value.c[2] = this->m_storage[2];
}

void Spectrum_impl::set_value( const mi::Spectrum_struct& value)
{
    this->m_storage[0] = value.c[0];
    this->m_storage[1] = value.c[1];
    this->m_storage[2] = value.c[2];
}

mi::base::IInterface* Bbox3_impl::create_api_class(
    mi::neuraylib::ITransaction* transaction,
    mi::Uint32 argc,
    const mi::base::IInterface* argv[])
{
    if( argc != 0)
        return nullptr;
    return (new Bbox3_impl)->cast_to_major();
}

Bbox3_impl::Bbox3_impl()
{
    m_type_name = "Bbox3";
}

bool Bbox3_impl::key_to_index( const char* key, mi::Size& index) const
{
    if( !key)
        return false;

    std::string key_string( key);
    if( key_string == "min_x") { index = 0; return true; }
    if( key_string == "min_y") { index = 1; return true; }
    if( key_string == "min_z") { index = 2; return true; }
    if( key_string == "max_x") { index = 3; return true; }
    if( key_string == "max_y") { index = 4; return true; }
    if( key_string == "max_z") { index = 5; return true; }
    return false;
}

bool Bbox3_impl::index_to_key( mi::Size index, std::string& key) const
{
    if( index == 0) { key = "min_x"; return true; }
    if( index == 1) { key = "min_y"; return true; }
    if( index == 2) { key = "min_z"; return true; }
    if( index == 3) { key = "max_x"; return true; }
    if( index == 4) { key = "max_y"; return true; }
    if( index == 5) { key = "max_z"; return true; }
    return false;
}

mi::Bbox3_struct Bbox3_impl::get_value() const
{
    mi::Bbox3_struct result;
    get_value( result);
    return result;
}

void Bbox3_impl::get_value( mi::Bbox3_struct& value) const
{
    value.min.x = this->m_storage[0];
    value.min.y = this->m_storage[1];
    value.min.z = this->m_storage[2];
    value.max.x = this->m_storage[3];
    value.max.y = this->m_storage[4];
    value.max.z = this->m_storage[5];
}

void Bbox3_impl::set_value( const mi::Bbox3_struct& value)
{
    this->m_storage[0] = value.min.x;
    this->m_storage[1] = value.min.y;
    this->m_storage[2] = value.min.z;
    this->m_storage[3] = value.max.x;
    this->m_storage[4] = value.max.y;
    this->m_storage[5] = value.max.z;
}


// explicit template instantiation for Vector_impl<I, T, ROWS>
template class Vector_impl<mi::IBoolean_2, bool,        2>;
template class Vector_impl<mi::IBoolean_3, bool,        3>;
template class Vector_impl<mi::IBoolean_4, bool,        4>;

template class Vector_impl<mi::ISint32_2,  mi::Sint32,  2>;
template class Vector_impl<mi::ISint32_3,  mi::Sint32,  3>;
template class Vector_impl<mi::ISint32_4,  mi::Sint32,  4>;

template class Vector_impl<mi::IUint32_2,  mi::Uint32,  2>;
template class Vector_impl<mi::IUint32_3,  mi::Uint32,  3>;
template class Vector_impl<mi::IUint32_4,  mi::Uint32,  4>;

template class Vector_impl<mi::IFloat32_2, mi::Float32, 2>;
template class Vector_impl<mi::IFloat32_3, mi::Float32, 3>;
template class Vector_impl<mi::IFloat32_4, mi::Float32, 4>;

template class Vector_impl<mi::IFloat64_2, mi::Float64, 2>;
template class Vector_impl<mi::IFloat64_3, mi::Float64, 3>;
template class Vector_impl<mi::IFloat64_4, mi::Float64, 4>;

// explicit template instantiation for Matrix_impl<I, T, ROWS, COLUMNS>
template class Matrix_impl<mi::IBoolean_2_2, bool,        2, 2>;
template class Matrix_impl<mi::IBoolean_2_3, bool,        2, 3>;
template class Matrix_impl<mi::IBoolean_2_4, bool,        2, 4>;
template class Matrix_impl<mi::IBoolean_3_2, bool,        3, 2>;
template class Matrix_impl<mi::IBoolean_3_3, bool,        3, 3>;
template class Matrix_impl<mi::IBoolean_3_4, bool,        3, 4>;
template class Matrix_impl<mi::IBoolean_4_2, bool,        4, 2>;
template class Matrix_impl<mi::IBoolean_4_3, bool,        4, 3>;
template class Matrix_impl<mi::IBoolean_4_4, bool,        4, 4>;

template class Matrix_impl<mi::ISint32_2_2,  mi::Sint32,  2, 2>;
template class Matrix_impl<mi::ISint32_2_3,  mi::Sint32,  2, 3>;
template class Matrix_impl<mi::ISint32_2_4,  mi::Sint32,  2, 4>;
template class Matrix_impl<mi::ISint32_3_2,  mi::Sint32,  3, 2>;
template class Matrix_impl<mi::ISint32_3_3,  mi::Sint32,  3, 3>;
template class Matrix_impl<mi::ISint32_3_4,  mi::Sint32,  3, 4>;
template class Matrix_impl<mi::ISint32_4_2,  mi::Sint32,  4, 2>;
template class Matrix_impl<mi::ISint32_4_3,  mi::Sint32,  4, 3>;
template class Matrix_impl<mi::ISint32_4_4,  mi::Sint32,  4, 4>;

template class Matrix_impl<mi::IUint32_2_2,  mi::Uint32,  2, 2>;
template class Matrix_impl<mi::IUint32_2_3,  mi::Uint32,  2, 3>;
template class Matrix_impl<mi::IUint32_2_4,  mi::Uint32,  2, 4>;
template class Matrix_impl<mi::IUint32_3_2,  mi::Uint32,  3, 2>;
template class Matrix_impl<mi::IUint32_3_3,  mi::Uint32,  3, 3>;
template class Matrix_impl<mi::IUint32_3_4,  mi::Uint32,  3, 4>;
template class Matrix_impl<mi::IUint32_4_2,  mi::Uint32,  4, 2>;
template class Matrix_impl<mi::IUint32_4_3,  mi::Uint32,  4, 3>;
template class Matrix_impl<mi::IUint32_4_4,  mi::Uint32,  4, 4>;

template class Matrix_impl<mi::IFloat32_2_2, mi::Float32, 2, 2>;
template class Matrix_impl<mi::IFloat32_2_3, mi::Float32, 2, 3>;
template class Matrix_impl<mi::IFloat32_2_4, mi::Float32, 2, 4>;
template class Matrix_impl<mi::IFloat32_3_2, mi::Float32, 3, 2>;
template class Matrix_impl<mi::IFloat32_3_3, mi::Float32, 3, 3>;
template class Matrix_impl<mi::IFloat32_3_4, mi::Float32, 3, 4>;
template class Matrix_impl<mi::IFloat32_4_2, mi::Float32, 4, 2>;
template class Matrix_impl<mi::IFloat32_4_3, mi::Float32, 4, 3>;
template class Matrix_impl<mi::IFloat32_4_4, mi::Float32, 4, 4>;

template class Matrix_impl<mi::IFloat64_2_2, mi::Float64, 2, 2>;
template class Matrix_impl<mi::IFloat64_2_3, mi::Float64, 2, 3>;
template class Matrix_impl<mi::IFloat64_2_4, mi::Float64, 2, 4>;
template class Matrix_impl<mi::IFloat64_3_2, mi::Float64, 3, 2>;
template class Matrix_impl<mi::IFloat64_3_3, mi::Float64, 3, 3>;
template class Matrix_impl<mi::IFloat64_3_4, mi::Float64, 3, 4>;
template class Matrix_impl<mi::IFloat64_4_2, mi::Float64, 4, 2>;
template class Matrix_impl<mi::IFloat64_4_3, mi::Float64, 4, 3>;
template class Matrix_impl<mi::IFloat64_4_4, mi::Float64, 4, 4>;

} // namespace NEURAY

} // namespace MI
