/***************************************************************************************************
 * Copyright (c) 2012-2025, NVIDIA CORPORATION. All rights reserved.
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
/// \brief Helper functions to set/get values of #mi::IData.

#ifndef MI_NEURAYLIB_SET_GET_H
#define MI_NEURAYLIB_SET_GET_H

#include <mi/base/config.h>
#include <mi/base/handle.h>
#include <mi/neuraylib/iarray.h>
#include <mi/neuraylib/iattribute_set.h>
#include <mi/neuraylib/ibbox.h>
#include <mi/neuraylib/icolor.h>
#include <mi/neuraylib/idynamic_array.h>
#include <mi/neuraylib/ienum.h>
#include <mi/neuraylib/imatrix.h>
#include <mi/neuraylib/inumber.h>
#include <mi/neuraylib/iref.h>
#include <mi/neuraylib/ispectrum.h>
#include <mi/neuraylib/istring.h>
#include <mi/neuraylib/iuuid.h>
#include <mi/neuraylib/ivector.h>
#include <mi/neuraylib/type_traits.h>

#include <string>
#include <vector>

namespace mi {

/** \addtogroup mi_neuray_types
@{
*/

/// Simplifies setting the value of #mi::IData from the corresponding classes from the %base and
/// %math API.
///
/// \param data            The instance of #mi::IData to modify.
/// \param value           The new value to be set.
/// \return
///                        -  0: Success.
///                        - -1: The dynamic type of \p data does not match the static type of
///                              \p value.
///                        - -2: The value of \p value is not valid.
///
/// This general template handles #mi::INumber and expects one of the following types as second
/// argument:
/// - \c bool,
/// - #mi::Sint8, #mi::Sint16, #mi::Sint32, #mi::Sint64,
/// - #mi::Uint8, #mi::Uint16, #mi::Uint32, #mi::Uint64,
/// - #mi::Size, #mi::Difference
/// - #mi::Float32, or #mi::Float64.
///
/// It also handles #mi::IEnum and expects an #mi::Sint32 argument in that case.
template<class T>
mi::Sint32 set_value( mi::IData* data, const T& value)
{
    mi::base::Handle<mi::INumber> v( data->get_interface<mi::INumber>());
    if( v) {
        v->set_value( value);
        return 0;
    }
    mi::base::Handle<mi::IEnum> e( data->get_interface<mi::IEnum>());
    if( e) {
        mi::Sint32 result = e->set_value( static_cast<mi::Sint32>( value));
        return result == 0 ? 0 : -2;
    }
    return -1;
}

/// This specialization handles #mi::IString and #mi::IRef.
///
/// It expects a \c const \c char* argument. See #mi::set_value() for details.
inline mi::Sint32 set_value( mi::IData* data, const char* value)
{
    mi::base::Handle<mi::IString> s( data->get_interface<mi::IString>());
    if( s) {
        s->set_c_str( value);
        return 0;
    }
    mi::base::Handle<mi::IRef> r( data->get_interface<mi::IRef>());
    if( r) {
        mi::Sint32 result = r->set_reference( value);
        return result == 0 ? 0 : -2;
    }
    mi::base::Handle<mi::IEnum> e( data->get_interface<mi::IEnum>());
    if( e) {
        mi::Sint32 result = e->set_value_by_name( value);
        return result == 0 ? 0 : -2;
    }
    return -1;
}

/// This specialization handles #mi::IString and #mi::IRef.
///
/// It expects a \c const std::string& argument. See #mi::set_value() for details.
inline mi::Sint32 set_value( mi::IData* data, const std::string& value)
{
    return set_value( data, value.c_str());
}

/// This specialization handles #mi::IUuid.
///
/// It expects an #mi::base::Uuid argument. See #mi::set_value() for details.
inline mi::Sint32 set_value( mi::IData* data, const mi::base::Uuid& value)
{
    mi::base::Handle<mi::IUuid> u( data->get_interface<mi::IUuid>());
    if( u) {
        u->set_uuid( value);
        return 0;
    }
    return -1;
}

/// This specialization handles the vector specializations of #mi::ICompound.
///
/// It expects one of the following specializations of #mi::math::Vector as its second argument:
/// - #mi::Boolean_2, #mi::Boolean_3, #mi::Boolean_4
/// - #mi::Sint32_2, #mi::Sint32_3, #mi::Sint32_4
/// - #mi::Uint32_2, #mi::Uint32_3, #mi::Uint32_4
/// - #mi::Float32_2, #mi::Float32_3, #mi::Float32_4
/// - #mi::Float64_2, #mi::Float64_3, #mi::Float64_4
///
/// See #mi::set_value() for details.
template <class T, Size DIM>
inline mi::Sint32 set_value( mi::IData* data, const mi::math::Vector<T,DIM>& value)
{
    typedef typename mi::Vector_type_traits<T,DIM>::Interface_type Vector_interface_type;
    mi::base::Handle<Vector_interface_type> v( data->get_interface<Vector_interface_type>());
    if( v) {
        v->set_value( value);
        return 0;
    }
    return -1;
}

/// This specialization handles the matrix specializations of #mi::ICompound.
///
/// It expects one of the following specializations of #mi::math::Matrix as its second argument:
/// - #mi::Boolean_2_2, #mi::Boolean_2_3, ..., #mi::Boolean_4_4
/// - #mi::Sint32_2_2, #mi::Sint32_2_3, ..., #mi::Sint32_4_4
/// - #mi::Uint32_2_2, #mi::Uint32_2_3, ..., #mi::Uint32_4_4
/// - #mi::Float32_2_2, #mi::Float32_2_3,...,  #mi::Float32_4_4
/// - #mi::Float64_2_2, #mi::Float64_2_3,...,  #mi::Float64_4_4
///
/// See #mi::set_value() for details.
template <class T, Size ROW, Size COL>
mi::Sint32 set_value( mi::IData* data, const mi::math::Matrix<T,ROW,COL>& value)
{
    typedef typename mi::Matrix_type_traits<T,ROW,COL>::Interface_type Matrix_interface_type;
    mi::base::Handle<Matrix_interface_type> m( data->get_interface<Matrix_interface_type>());
    if( m) {
        m->set_value( value);
        return 0;
    }
    return -1;
}

/// This specialization handles #mi::IColor and #mi::IColor3.
///
/// It expects an #mi::Color argument. See #mi::set_value() for details.
inline mi::Sint32 set_value( mi::IData* data, const mi::Color& value)
{
    mi::base::Handle<mi::IColor> c( data->get_interface<mi::IColor>());
    if( c) {
        c->set_value( value);
        return 0;
    }
    mi::base::Handle<mi::IColor3> c3( data->get_interface<mi::IColor3>());
    if( c3) {
        c3->set_value( value);
        return 0;
    }
    return -1;
}

/// This specialization handles #mi::ISpectrum.
///
/// It expects an #mi::Spectrum argument. See #mi::set_value() for details.
inline mi::Sint32 set_value( mi::IData* data, const mi::Spectrum& value)
{
    mi::base::Handle<mi::ISpectrum> s( data->get_interface<mi::ISpectrum>());
    if( s) {
        s->set_value( value);
        return 0;
    }
    return -1;
}

/// This specialization handles #mi::IBbox3.
///
/// It expects an #mi::Bbox3 argument. See #mi::set_value() for details.
inline mi::Sint32 set_value( mi::IData* data, const mi::Bbox3& value)
{
    mi::base::Handle<mi::IBbox3> b( data->get_interface<mi::IBbox3>());
    if( b) {
        b->set_value( value);
        return 0;
    }
    return -1;
}

/// This variant handles elements of collections identified via an additional index.
///
/// See also #mi::set_value(mi::IData*,const T*,mi::Size n) and
/// #mi::set_value(mi::IData*,const std::vector<T>&) to set entire arrays.
///
/// \param data            The instance of #mi::IData to modify.
/// \param index           The index of the affected collection element.
/// \param value           The new value to be set.
/// \return
///                        -  0: Success.
///                        - -1: The dynamic type of \p data does not match the static type of
///                              \p value.
///                        - -2: The value of \p value is not valid.
///                        - -3: The index or key is not valid for the collection.
template<class T>
mi::Sint32 set_value( mi::IData* data, mi::Size index, const T& value)
{
    mi::base::Handle<mi::IData_collection> c( data->get_interface<mi::IData_collection>());
    if( c) {
        mi::base::Handle<mi::IData> d( c->get_value<mi::IData>( index));
        if( !d)
            return -3;
        return set_value( d.get(), value);
    }
    return -1;
}

/// This variant handles elements of collections identified via an additional key.
///
/// See also #mi::set_value(mi::IData*,const T*,mi::Size n) and
/// #mi::set_value(mi::IData*,const std::vector<T>&) to set entire arrays.
///
/// \param data            The instance of #mi::IData to modify.
/// \param key             The key of the affected collection element.
/// \param value           The new value to be set.
/// \return
///                        -  0: Success.
///                        - -1: The dynamic type of \p data does not match the static type of
///                              \p value.
///                        - -2: The value of \p value is not valid.
///                        - -3: The key is not valid.
template<class T>
mi::Sint32 set_value( mi::IData* data, const char* key, const T& value)
{
    mi::base::Handle<mi::IData_collection> c( data->get_interface<mi::IData_collection>());
    if( c) {
        mi::base::Handle<mi::IData> d( c->get_value<mi::IData>( key));
        if( !d)
            return -3;
        return set_value( d.get(), value);
    }
    return -1;
}

/// This variant handles entire arrays (as C-like array).
///
/// See also #mi::set_value(mi::IData*,const std::vector<T>&) for std::vector support.
///
/// \param data            The instance of #mi::IData to modify.
/// \param values          The new values to be set (as pointer to a C-like array).
/// \param n               The size of the C-like array. If \p data is a dynamic array, then it is
///                        resized accordingly. If \p data is a static array, then \p n needs to
///                        match the size of \p data.
/// \return
///                        -  0: Success.
///                        - -1: The dynamic type of \p data is not an array, or the dynamic type
///                              of the array elements does not match the static type of \p values.
///                        - -2: At least one value of \p values is not valid (possibly incomplete
///                              operation).
///                        - -5: The array sizes do not match (if \p data is a static array).
template<class T>
mi::Sint32 set_value( mi::IData* data, const T* values, mi::Size n)
{
    mi::base::Handle<mi::IArray> a( data->get_interface<mi::IArray>());
    if( a) {

        mi::base::Handle<mi::IDynamic_array> da( data->get_interface<mi::IDynamic_array>());
        if( da)
            da->set_length( n);
        else if( a->get_length() != n)
            return -5;

        for( mi::Size i = 0; i < n; ++i) {
            mi::base::Handle<mi::IData> d( a->get_element<mi::IData>( i));
            if( !d)
                return -1;
            mi::Sint32 result = set_value( d.get(), values[i]);
            if( result != 0)
                return result;
        }
        return 0;
    }

    return -1;
}

/// This variant handles entire arrays (as std::vector).
///
/// See also #mi::set_value(mi::IData*,const T*,mi::Size) for C-like arrays.
///
/// \param data            The instance of #mi::IData to modify.
/// \param values          The new values to be set. If \p data is a dynamic array, then it is
///                        resized accordingly. If \p data is a static array, then its size needs to
///                        match the size of \p values.
/// \return
///                        -  0: Success.
///                        - -1: The dynamic type of \p data is not an array, or the dynamic type
///                              of the array elements does not match the static type of \p values.
///                        - -2: At least one value of \p values is not valid (possibly incomplete
///                              operation)
///                        - -5: The array sizes do not match (if \p data is a static array).
template<class T>
mi::Sint32 set_value( mi::IData* data, const std::vector<T>& values)
{
    return set_value( data, values.data(), static_cast<mi::Size>( values.size()));
}

/// Simplifies reading the value of #mi::IData into the corresponding classes from the %base and
/// %math API.
///
/// \param data            The instance of #mi::IData to read.
/// \param[out] value      The new value will be stored here.
/// \return
///                        -  0: Success.
///                        - -1: The dynamic type of \p data does not match the static type of
///                              \p value.
///
/// This general template handles #mi::INumber, and expects one of the following types as second
/// argument:
/// - \c bool,
/// - #mi::Sint8, #mi::Sint16, #mi::Sint32, #mi::Sint64,
/// - #mi::Uint8, #mi::Uint16, #mi::Uint32, #mi::Uint64,
/// - #mi::Size, #mi::Difference
/// - #mi::Float32, or #mi::Float64.
///
/// It also handles #mi::IEnum and expects an #mi::Sint32 argument in that case.
template<class T>
mi::Sint32 get_value( const mi::IData* data, T& value)
{
    mi::base::Handle<const mi::INumber> v( data->get_interface<mi::INumber>());
    if( v) {
        v->get_value( value);
        return 0;
    }
// disable C4800: 'mi::Sint32' : forcing value to bool 'true' or 'false' (performance warning)
// disable C4800: 'mi::Uint32' : forcing value to bool 'true' or 'false' (performance warning)
#ifdef MI_COMPILER_MSC
#pragma warning( push )
#pragma warning( disable : 4800 )
#endif
    mi::base::Handle<const mi::IEnum> e( data->get_interface<mi::IEnum>());
    if( e) {
        value = static_cast<T>( e->get_value());
        return 0;
    }
#ifdef MI_COMPILER_MSC
#pragma warning( pop )
#endif
    return -1;
}

/// This specialization handles #mi::IString and #mi::IRef.
///
/// It expects a \c const \c char* argument. See #mi::get_value() for details.
inline mi::Sint32 get_value( const mi::IData* data, const char*& value)
{
    mi::base::Handle<const mi::IString> i( data->get_interface<mi::IString>());
    if( i) {
        value = i->get_c_str();
        return 0;
    }
    mi::base::Handle<const mi::IRef> r( data->get_interface<mi::IRef>());
    if( r) {
        value = r->get_reference_name();
        return 0;
    }
    mi::base::Handle<const mi::IEnum> e( data->get_interface<mi::IEnum>());
    if( e) {
        value = e->get_value_by_name();
        return 0;
    }
    return -1;
}

/// This specialization handles #mi::IUuid.
///
/// It expects an #mi::base::Uuid argument. See #mi::get_value() for details.
inline mi::Sint32 get_value( const mi::IData* data, mi::base::Uuid& value)
{
    mi::base::Handle<const mi::IUuid> u( data->get_interface<mi::IUuid>());
    if( u) {
        value = u->get_uuid();
        return 0;
    }
    return -1;
}

/// This specialization handles the vector specializations of #mi::ICompound.
///
/// It expects one of the following specializations of #mi::math::Vector as its second argument:
/// - #mi::Boolean_2, #mi::Boolean_3, #mi::Boolean_4
/// - #mi::Sint32_2, #mi::Sint32_3, #mi::Sint32_4
/// - #mi::Uint32_2, #mi::Uint32_3, #mi::Uint32_4
/// - #mi::Float32_2, #mi::Float32_3, #mi::Float32_4
/// - #mi::Float64_2, #mi::Float64_3, #mi::Float64_4
///
/// See #mi::get_value() for details.
template <class T, Size DIM>
inline mi::Sint32 get_value( const mi::IData* data, mi::math::Vector<T,DIM>& value)
{
    typedef typename mi::Vector_type_traits<T,DIM>::Interface_type Vector_interface_type;
    mi::base::Handle<const Vector_interface_type> v( data->get_interface<Vector_interface_type>());
    if( v) {
        value = v->get_value();
        return 0;
    }
    return -1;
}

/// This specialization handles the matrix specializations of #mi::ICompound.
///
/// It expects one of the following specializations of #mi::math::Matrix as its second argument:
/// - #mi::Boolean_2_2, #mi::Boolean_2_3, ..., #mi::Boolean_4_4
/// - #mi::Sint32_2_2, #mi::Sint32_2_3, ..., #mi::Sint32_4_4
/// - #mi::Uint32_2_2, #mi::Uint32_2_3, ..., #mi::Uint32_4_4
/// - #mi::Float32_2_2, #mi::Float32_2_3,...,  #mi::Float32_4_4
/// - #mi::Float64_2_2, #mi::Float64_2_3,...,  #mi::Float64_4_4
///
/// See #mi::get_value() for details.
template <class T, Size ROW, Size COL>
mi::Sint32 get_value( const mi::IData* data, mi::math::Matrix<T,ROW,COL>& value)
{
    typedef typename mi::Matrix_type_traits<T,ROW,COL>::Interface_type Matrix_interface_type;
    mi::base::Handle<const Matrix_interface_type> m( data->get_interface<Matrix_interface_type>());
    if( m) {
        value = m->get_value();
        return 0;
    }
    return -1;
}

/// This specialization handles #mi::IColor and #mi::IColor3.
///
/// It expects an #mi::Color argument. See #mi::get_value() for details.
inline mi::Sint32 get_value( const mi::IData* data, mi::Color& value)
{
    mi::base::Handle<const mi::IColor> c( data->get_interface<mi::IColor>());
    if( c) {
        value = c->get_value();
        return 0;
    }
    mi::base::Handle<const mi::IColor3> c3( data->get_interface<mi::IColor3>());
    if( c3) {
        value = c3->get_value();
        return 0;
    }
    return -1;
}

/// This specialization handles #mi::ISpectrum.
///
/// It expects an #mi::Spectrum argument. See #mi::get_value() for details.
inline mi::Sint32 get_value( const mi::IData* data, mi::Spectrum& value)
{
    mi::base::Handle<const mi::ISpectrum> s( data->get_interface<mi::ISpectrum>());
    if( s) {
        value = s->get_value();
        return 0;
    }
    return -1;
}

/// This specialization handles #mi::IBbox3.
///
/// It expects an #mi::Bbox3 argument. See #mi::get_value() for details.
inline mi::Sint32 get_value( const mi::IData* data, mi::Bbox3& value)
{
    mi::base::Handle<const mi::IBbox3> b( data->get_interface<mi::IBbox3>());
    if( b) {
        value = b->get_value();
        return 0;
    }
    return -1;
}

/// This variant handles elements of collections identified via an additional index.
///
/// See also #mi::get_value(const mi::IData*,T*,mi::Size n) and
/// #mi::get_value(const mi::IData*,std::vector<T>&) to read entire arrays.
///
/// \param data            The instance of #mi::IData to read.
/// \param index           The index of the affected collection element.
/// \param value           The new value will be stored here.
/// \return
///                        -  0: Success.
///                        - -1: The dynamic type of \p data does not match the static type of
///                              \p value.
///                        - -3: The index is not valid.
template<class T>
mi::Sint32 get_value( const mi::IData* data, mi::Size index, T& value)
{
    mi::base::Handle<const mi::IData_collection> c( data->get_interface<mi::IData_collection>());
    if( c) {
        mi::base::Handle<const mi::IData> d( c->get_value<mi::IData>( index));
        if( !d)
            return -3;
        return get_value( d.get(), value);
    }
    return -1;
}

/// This variant handles elements of collections identified via an additional key.
///
/// See also #mi::get_value(const mi::IData*,T*,mi::Size n) and
/// #mi::get_value(const mi::IData*,std::vector<T>&) to read entire arrays.
///
/// \param data            The instance of #mi::IData to read.
/// \param key             The key of the affected collection element.
/// \param value           The new value will be stored here.
/// \return
///                        -  0: Success.
///                        - -1: The dynamic type of \p data does not match the static type of
///                              \p value.
///                        - -3: The key is not valid.
template<class T>
mi::Sint32 get_value( const mi::IData* data, const char* key, T& value)
{
    mi::base::Handle<const mi::IData_collection> c( data->get_interface<mi::IData_collection>());
    if( c) {
        mi::base::Handle<const mi::IData> d( c->get_value<mi::IData>( key));
        if( !d)
            return -3;
        return get_value( d.get(), value);
    }
    return -1;
}

/// This variant handles entire arrays (as C-like arrays).
///
/// See also #mi::get_value(const IData*,std::vector<T>&) for std::vector support.
///
/// \param data            The instance of #mi::IData to read.
/// \param values          The new values will be stored here (as pointer to a C-like array).
/// \param n               The size of the C-like array (needs to match the size of \p data).
/// \return
///                        -  0: Success.
///                        - -1: The dynamic type of \p data is not an array, or the dynamic type
///                              of the array elements does not match the static type of \p values.
///                        - -5: The array sizes do not match.
template<class T>
mi::Sint32 get_value( const mi::IData* data, T* values, mi::Size n)
{
    mi::base::Handle<const mi::IArray> a( data->get_interface<mi::IArray>());
    if( a) {

        if( a->get_length() != n)
            return -5;

        for( mi::Size i = 0; i < n; ++i) {
            mi::base::Handle<const mi::IData> d( a->get_element<mi::IData>( i));
            if( !d)
                return -1;
            mi::Sint32 result = get_value( d.get(), values[i]);
            if( result != 0)
                return result;
        }
        return 0;
    }

    return -1;
}

/// This variant handles entire arrays (as std::vector).
///
/// See also #mi::get_value(const IData*,T*,mi::Size) for C-like arrays.
///
/// \param data            The instance of #mi::IData to read.
/// \param values          The new values will be stored here. The vector will be resized
///                        accordingly.
/// \return
///                        -  0: Success.
///                        - -1: The dynamic type of \p data is not an array, or the dynamic type
///                              of the array elements does not match the static type of \p values.
template<class T>
mi::Sint32 get_value( const mi::IData* data, std::vector<T>& values)
{
    mi::base::Handle<const mi::IArray> a( data->get_interface<mi::IArray>());
    if( a)
        values.resize( a->get_length());

    return get_value( data, values.data(), static_cast<mi::Size>( values.size()));
}

/// This variant handles strings as std::string.
///
/// See also #mi::get_value(const IData*,const char*&) for C-strings.
///
/// \param data            The instance of #mi::IData to read.
/// \param value           The new values will be stored here.
/// \return
///                        -  0: Success.
///                        - -1: The dynamic type of \p data does not match.
inline mi::Sint32 get_value( const mi::IData* data, std::string& value)
{
    const char* c_str = 0;
    const mi::Sint32 res = get_value( data, c_str);
    if( c_str)
        value.assign( c_str);
    return res;
}

/**@}*/ // end group mi_neuray_types

/** \addtogroup mi_neuray_scene_element
@{
*/

/// Simplifies setting the value of an attribute from the corresponding classes from the %base and
/// %math API.
///
/// \param attribute_set   The affected attribute set.
/// \param name            The name of the attribute to modify.
/// \param value           The new value to be set.
/// \return
///                        -  0: Success.
///                        - -1: The dynamic type of the attribute does not match the static type of
///                              \p value.
///                        - -2: The value of \p value is not valid.
///                        - -4: The attribute \p name does not exist.
template<class T>
mi::Sint32 set_value(
    mi::neuraylib::IAttribute_set* attribute_set, const char* name, const T& value)
{
    mi::base::Handle<mi::IData> data( attribute_set->edit_attribute<mi::IData>( name));
    if( !data)
        return -4;
    return set_value( data.get(), value);
}

/// Simplifies setting the value of an attribute from the corresponding classes from the %base and
/// %math API (variant with an index for collections).
///
/// \param attribute_set   The affected attribute set.
/// \param name            The name of the attribute to modify.
/// \param index           The index of the affected collection element.
/// \param value           The new value to be set.
/// \return
///                        -  0: Success.
///                        - -1: The dynamic type of the attribute does not match the static type of
///                              \p value.
///                        - -2: The value of \p value is not valid.
///                        - -3: The index is not valid.
///                        - -4: The attribute \p name does not exist.
template<class T>
mi::Sint32 set_value(
    mi::neuraylib::IAttribute_set* attribute_set, const char* name, mi::Size index, const T& value)
{
    mi::base::Handle<mi::IData> data( attribute_set->edit_attribute<mi::IData>( name));
    if( !data)
        return -4;
    return set_value( data.get(), index, value);
}

/// Simplifies setting the value of an attribute from the corresponding classes from the %base and
/// %math API (variant with a key for collections).
///
/// \param attribute_set   The affected attribute set.
/// \param name            The name of the attribute to modify.
/// \param key             The key of the affected collection element.
/// \param value           The new value to be set.
/// \return
///                        -  0: Success.
///                        - -1: The dynamic type of the attribute does not match the static type of
///                              \p value.
///                        - -2: The value of \p value is not valid.
///                        - -3: The key is not valid.
///                        - -4: The attribute \p name does not exist.
template<class T>
mi::Sint32 set_value(
    mi::neuraylib::IAttribute_set* attribute_set, const char* name, const char* key, const T& value)
{
    mi::base::Handle<mi::IData> data( attribute_set->edit_attribute<mi::IData>( name));
    if( !data)
        return -4;
    return set_value( data.get(), key, value);
}

/// Simplifies setting the value of an attribute from the corresponding classes from the %base and
/// %math API (variant for entire arrays as C-like arrays).
///
/// See also #mi::set_value(mi::neuraylib::IAttribute_set*,const char*,const std::vector<T>&) for
/// std::vector support.
///
/// \param attribute_set   The affected attribute set.
/// \param name            The name of the attribute to modify.
/// \param values          The new values to be set (as pointer to a C-like array).
/// \param n               The size of the C-like array. If the attribute is a dynamic array, then
///                        it is resized accordingly. If the attribute is a static array, then its
///                        size needs to match \p n.
/// \return
///                        -  0: Success.
///                        - -1: The dynamic type of the attribute is not an array, or the dynamic
///                              type of the array elements does not match the static type of
///                              \p values.
///                        - -2: At least one value of \p values is not valid (possibly incomplete
///                              operation)
///                        - -4: The attribute \p name does not exist.
///                        - -5: The array sizes do not match (if the attribute is a static array).
template<class T>
mi::Sint32 set_value(
    mi::neuraylib::IAttribute_set* attribute_set, const char* name, const T* values, mi::Size n)
{
    mi::base::Handle<mi::IData> data( attribute_set->edit_attribute<mi::IData>( name));
    if( !data)
        return -4;
    return set_value( data.get(), values, n);
}

/// Simplifies setting the value of an attribute from the corresponding classes from the %base and
/// %math API (variant for entire arrays as std::vector).
///
/// See also #mi::set_value(mi::neuraylib::IAttribute_set*,const char*,const T*,mi::Size) for
/// C-like arrays.
///
/// \param attribute_set   The affected attribute set.
/// \param name            The name of the attribute to modify.
/// \param values          The new values to be set. If the attribute is a dynamic array, then it is
///                        resized accordingly. If the attribute is a static array, then its size
///                        needs to match the size of \p values.
/// \return
///                        -  0: Success.
///                        - -1: The dynamic type of the attribute is not an array, or the dynamic
///                              type of the array elements does not match the static type of
///                              \p values.
///                        - -2: At least one value of \p values is not valid (possibly incomplete
///                              operation)
///                        - -4: The attribute \p name does not exist.
///                        - -5: The array sizes do not match (if the attribute is a static array).
template<class T>
mi::Sint32 set_value(
    mi::neuraylib::IAttribute_set* attribute_set, const char* name, const std::vector<T>& values)
{
    mi::base::Handle<mi::IData> data( attribute_set->edit_attribute<mi::IData>( name));
    if( !data)
        return -4;
    return set_value( data.get(), values);
}

/// Simplifies reading the value of an attribute into the corresponding classes from the %base and
/// %math API.
///
/// \param attribute_set   The affected attribute set.
/// \param name            The name of the attribute to read.
/// \param value           The new value will be stored here.
/// \return
///                        -  0: Success.
///                        - -1: The dynamic type of the attribute does not match the static type of
///                              \p value.
///                        - -4: The attribute \p name does not exist.
template<class T>
mi::Sint32 get_value(
    const mi::neuraylib::IAttribute_set* attribute_set, const char* name, T& value)
{
    mi::base::Handle<const mi::IData> data( attribute_set->access_attribute<mi::IData>( name));
    if( !data)
        return -4;
    return get_value( data.get(), value);
}

/// Simplifies reading the value of an attribute into the corresponding classes from the %base and
/// %math API (variant with an index for collections).
///
/// \param attribute_set   The affected attribute set.
/// \param name            The name of the attribute to read.
/// \param index           The index of the affected collection element.
/// \param value           The new value will be stored here.
/// \return
///                        -  0: Success.
///                        - -1: The dynamic type of the attribute does not match the static type of
///                              \p value.
///                        - -3: The index is not valid.
///                        - -4: The attribute \p name does not exist.
template<class T>
mi::Sint32 get_value(
    const mi::neuraylib::IAttribute_set* attribute_set, const char* name, mi::Size index, T& value)
{
    mi::base::Handle<const mi::IData> data( attribute_set->access_attribute<mi::IData>( name));
    if( !data)
        return -4;
    return get_value( data.get(), index, value);
}

/// Simplifies reading the value of an attribute into the corresponding classes from the %base and
/// %math API (variant with a key for collections).
///
/// \param attribute_set   The affected attribute set.
/// \param name            The name of the attribute to read.
/// \param key             The key of the affected collection element.
/// \param value           The new value will be stored here.
/// \return
///                        -  0: Success.
///                        - -1: The dynamic type of the attribute does not match the static type of
///                              \p value.
///                        - -3: The key is not valid.
///                        - -4: The attribute \p name does not exist.
template<class T>
mi::Sint32 get_value(
    const mi::neuraylib::IAttribute_set* attribute_set, const char* name, const char* key, T& value)
{
    mi::base::Handle<const mi::IData> data( attribute_set->access_attribute<mi::IData>( name));
    if( !data)
        return -4;
    return get_value( data.get(), key, value);
}

/// Simplifies reading the value of an attribute into the corresponding classes from the %base and
/// %math API (variant for entire arrays as C-like array).
///
/// See also #mi::get_value(const mi::neuraylib::IAttribute_set*,const char*,std::vector<T>&) for
/// std::vector support.
///
/// \param attribute_set   The affected attribute set.
/// \param name            The name of the attribute to read.
/// \param values          The new values will be stored here (as pointer to a C-like array).
/// \param n               The size of the C-like array (needs to match the size of the attribute).
/// \return
///                        -  0: Success.
///                        - -1: The dynamic type of the attribute is not an array, or the dynamic
///                              type of the array elements does not match the static type of
///                              \p values.
///                        - -2: At least one value of \p values is not valid (possibly incomplete
///                              operation)
///                        - -4: The attribute \p name does not exist.
///                        - -5: The array sizes do not match.
template<class T>
mi::Sint32 get_value(
    const mi::neuraylib::IAttribute_set* attribute_set, const char* name, T* values, mi::Size n)
{
    mi::base::Handle<const mi::IData> data( attribute_set->access_attribute<mi::IData>( name));
    if( !data)
        return -4;
    return get_value( data.get(), values, n);
}

/// Simplifies reading the value of an attribute into the corresponding classes from the %base and
/// %math API (variant for entire arrays as std::vector).
///
/// See also #mi::get_value(const mi::neuraylib::IAttribute_set*,const char*,T*,mi::Size) for C-like
/// arrays.
///
/// \param attribute_set   The affected attribute set.
/// \param name            The name of the attribute to read.
/// \param values          The new values will be stored here. The vector will be resized
///                        accordingly.
/// \return
///                        -  0: Success.
///                        - -1: The dynamic type of the attribute is not an array, or the dynamic
///                              type of the array elements does not match the static type of
///                              \p values.
///                        - -2: At least one value of \p values is not valid (possibly incomplete
///                              operation)
///                        - -4: The attribute \p name does not exist.
template<class T>
mi::Sint32 get_value(
    const mi::neuraylib::IAttribute_set* attribute_set, const char* name, std::vector<T>& values)
{
    mi::base::Handle<const mi::IData> data( attribute_set->access_attribute<mi::IData>( name));
    if( !data)
        return -4;
    return get_value( data.get(), values);
}

/**@}*/ // end group mi_neuray_types

} // namespace mi

#endif // MI_NEURAYLIB_SET_GET_H
